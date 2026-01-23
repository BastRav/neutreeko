mod utils;
use burn_store::{BurnpackStore, ModuleSnapshot};
use utils::{moves_and_value_to_target, illegal_mask, opening, PolicyValueTarget, add_symmetries};

use super::{
    ANN, PolicyValueOutput,
    utils::board_to_input,
};
use std::collections::HashSet;
use burn::{
    module::Module,
    nn::loss::{MseLoss, Reduction},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor, decay::WeightDecayConfig, lr_scheduler::{LrScheduler, cosine::{CosineAnnealingLrScheduler, CosineAnnealingLrSchedulerConfig}}},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Device, Tensor, activation::log_softmax, backend::AutodiffBackend},
};
use crate::{
    ai::{AI, alphazeutreeko::AlphaZeutreeko},
    logic::{Board, Color},
    platform::NativePlatform,
};

pub struct ANNTrainer<B: AutodiffBackend, A: AI<NativePlatform>> {
    alphazeutreeko: AlphaZeutreeko<B, NativePlatform>,
    pub opponent: Option<A>,
    optimizer: OptimizerAdaptor<Adam, ANN<B>, B>,
    learning_rate_schedule: CosineAnnealingLrScheduler,
    device: Device<B>,
    recorder: BinFileRecorder<FullPrecisionSettings>,
}

impl<B: AutodiffBackend<FloatElem = f32>, A: AI<NativePlatform>> ANNTrainer<B, A> {
    pub fn new() -> Self {
        let device = B::Device::default();
        let learning_rate_schedule = CosineAnnealingLrSchedulerConfig::new(5e-4, 1000).with_min_lr(5e-5).init().unwrap();
        let optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1e-4))).init();
        let alphazeutreeko = AlphaZeutreeko::new(Color::Green, 6);
        let opponent = None;
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

        Self {
            alphazeutreeko,
            opponent,
            optimizer,
            learning_rate_schedule,
            device,
            recorder
        }
    }

    fn loss(&self, output: PolicyValueOutput<B>, target: PolicyValueTarget<B>, illegal_mask: Tensor<B, 4>) -> Tensor<B, 1> {
        let masked_probabilities = output.policy + illegal_mask;
        let flat_probas: Tensor<B, 2> = masked_probabilities.flatten(1, 3);
        let log_probabilities = log_softmax(flat_probas, 1);
        let flat_target = target.policy.flatten(1, 3);
        let policy_loss = -(flat_target * log_probabilities).sum_dim(1).mean();
        let value_loss = MseLoss::new().forward(output.value, target.value, Reduction::Mean);
        policy_loss + value_loss * 0.5
    }

    fn train_step(&mut self, input:Tensor<B, 4>, target: PolicyValueTarget<B>, illegal_mask: Tensor<B, 4>) -> Tensor<B, 1> {
        // Forward pass
        let output = self.alphazeutreeko.policy.ann.forward(input);

        let loss = self.loss(output, target, illegal_mask);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.alphazeutreeko.policy.ann);

        // Update self.alphazeutreeko.policy.ann parameters
        let lr = self.learning_rate_schedule.step();
        self.alphazeutreeko.policy.ann = self.optimizer.step(lr, self.alphazeutreeko.policy.ann.clone(), grads);
        loss
    }

    pub fn training_loop(&mut self, max_epoch: usize) {
        let mut victories = 0.0;
        let mut draws = 0.0;
        let has_opponent = self.opponent.is_some();
        for epoch in 1..=max_epoch {
            println!("Starting iteration {}", epoch);
            self.alphazeutreeko.clear_graph();
            let mut to_feed = vec![];
            let mut board = Board::random_board::<NativePlatform>();
            let mut board_hashes = HashSet::new();
            board_hashes.insert(board.get_hash());
            let mut number_moves = 0;
            while board.winner().is_none() {
                let alphazeutreeko_color = self.alphazeutreeko.color();
                println!("Current board");
                println!("{}", board.str_rep());
                let possible_moves;
                let best_move;
                if board.next_player == Some(alphazeutreeko_color.clone()) {
                    println!("AlphaZeutreeko is playing");
                    possible_moves = self.alphazeutreeko.give_all_options(&board, true);
                    best_move = self.alphazeutreeko.best_move_from_vec(&possible_moves.1, false);
                }
                else if !has_opponent {
                    println!("AlphaZeutreeko is playing against itself");
                    self.alphazeutreeko.set_color(alphazeutreeko_color.other_color());
                    possible_moves = self.alphazeutreeko.give_all_options(&board, true);
                    best_move = self.alphazeutreeko.best_move_from_vec(&possible_moves.1, false);
                }
                else {
                    println!("Opponent is playing");
                    possible_moves = self.opponent.as_mut().unwrap().give_all_options(&board, false);
                    best_move = self.opponent.as_mut().unwrap().best_move_from_vec(&possible_moves.1, false);
                }
                
                to_feed.push((board.clone(), possible_moves));
                let moved = board.move_pawn_until_blocked(best_move.0, &best_move.1);
                if !moved {
                    panic!("An invalid move was selected!!!");
                }
                number_moves += 1;
                if number_moves > 255 {
                    println!("Game taking too long, consider it a draw");
                    draws += 1.0;
                    break;
                }
                let new_hash = board.get_hash();
                if !board_hashes.insert(new_hash){
                    println!("Back to a previous board, break game to avoid loops, consider it a draw");
                    draws += 1.0;
                    break;
                }
            }
            let alphazeutreeko_color = self.alphazeutreeko.color().clone();
            if has_opponent && board.winner() == Some(alphazeutreeko_color.clone()) {
                victories += 1.0;
                println!("AlphaZeutreeko won!!!");
            }
            println!("Final board");
            println!("{}", board.str_rep());
            println!("Proceeding to learning");
            for (board_learn, (board_eval, moves_eval)) in to_feed.into_iter(){
                let input = board_to_input(&board_learn, &self.device);
                let target = moves_and_value_to_target(&board_learn, board_eval, &moves_eval, &self.device);
                let illegal_mask = illegal_mask(&board_learn, &self.device);
                for (input_iter, target_iter, illegal_mask_iter) in add_symmetries(input, target, illegal_mask).into_iter() {
                    self.train_step(input_iter, target_iter, illegal_mask_iter);
                }
            }
            if has_opponent {
                self.alphazeutreeko.set_color(alphazeutreeko_color.other_color());
                self.opponent.as_mut().unwrap().set_color(alphazeutreeko_color);
            }
        }
        println!("Victories: {:.1}%, Draws: {:.1}%", 100.0*victories/max_epoch as f32, 100.0*draws/max_epoch as f32);
    }

    pub fn train_opening(&mut self, number_passes: usize) {
        let opening_sequence = opening(&self.device);
        for iteration in 1..=number_passes {
            println!("Starting iteration {}/{}", iteration, number_passes);
            for (input, target, illegal_mask) in opening_sequence.iter() {
                self.train_step(input.clone(), target.clone(), illegal_mask.clone());
            }
        }
    }

    pub fn save(&self, filepath: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.alphazeutreeko.policy.ann.clone().save_file(filepath, &self.recorder)?;
        Ok(())
    }

    pub fn save_for_web(&self) {
        let mut store = BurnpackStore::from_file("assets/models/web/model");
        let _ = self.alphazeutreeko.policy.ann.save_into(&mut store);
    }

    pub fn load(&mut self, filepath: &str) -> Result<(), Box<dyn std::error::Error>> {
        let loaded_ann = self.alphazeutreeko.policy.ann.clone().load_file(filepath, &self.recorder, &self.device)?;
        self.alphazeutreeko.policy.ann = loaded_ann;
        Ok(())
    }
}
