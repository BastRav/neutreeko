use super::{
    ANN, PolicyValueOutput,
    inputouput::{board_to_input, moves_and_value_to_target, illegal_mask},
};
use burn::{
    module::Module,
    nn::loss::{MseLoss, Reduction},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Device, Tensor, activation::log_softmax, backend::AutodiffBackend}
};
use crate::{
    ai::{AI, alphazeutreeko::AlphaZeutreeko},
    logic::{Board, Color},
    platform::NativePlatform,
};

pub struct PolicyValueTarget<B: AutodiffBackend> {
    pub value: Tensor<B, 1>,
    pub policy: Tensor<B, 2>,
}

pub struct ANNTrainer<B: AutodiffBackend, A: AI<NativePlatform>> {
    pub alphazeutreeko: AlphaZeutreeko<B, NativePlatform>,
    pub opponent: A,
    pub optimizer: OptimizerAdaptor<Adam, ANN<B>, B>,
    pub device: Device<B>,
    pub recorder: BinFileRecorder<FullPrecisionSettings>,
}

impl<B: AutodiffBackend<FloatElem = f32>, A: AI<NativePlatform>> ANNTrainer<B, A> {
    pub fn new() -> Self {
        let device = B::Device::default();
        let optimizer = AdamConfig::new().init();
        let alphazeutreeko = AlphaZeutreeko::new(Color::Green, 3);
        let opponent = AI::new(Color::Yellow, 3);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

        Self {
            alphazeutreeko,
            opponent,
            optimizer,
            device,
            recorder
        }
    }

    fn loss(&self, output: PolicyValueOutput<B>, target: PolicyValueTarget<B>, illegal_mask: Tensor<B, 2>) -> Tensor<B, 1> {
        let masked_probabilities = output.policy + illegal_mask;
        let log_probabilities = log_softmax(masked_probabilities, 1);
        let policy_loss = -(target.policy * log_probabilities).sum_dim(1).mean();
        let value_loss = MseLoss::new().forward(output.value, target.value, Reduction::Mean);
        policy_loss + value_loss * 0.5
    }

    fn train_step(&mut self, input:Tensor<B, 3>, target: PolicyValueTarget<B>, illegal_mask: Tensor<B, 2>) -> Tensor<B, 1> {
        // Forward pass
        let output = self.alphazeutreeko.mcts.policy.ann.forward(input);

        let loss = self.loss(output, target, illegal_mask);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.alphazeutreeko.mcts.policy.ann);

        // Update self.alphazeutreeko.policy.ann parameters
        self.alphazeutreeko.mcts.policy.ann = self.optimizer.step(0.001, self.alphazeutreeko.mcts.policy.ann.clone(), grads);
        loss
    }

    pub fn training_loop(&mut self, max_epoch: usize) {
        let mut victories = 0.0;
        let mut draws = 0.0;
        for epoch in 1..=max_epoch {
            println!("Starting iteration {}", epoch);
            let mut to_feed = vec![];
            let mut board = Board::default_new();
            let mut number_moves = 0;
            let mut board_eval = 1.0;
            let alphazeutreeko_color = self.alphazeutreeko.color().clone();
            while board.winner().is_none() {
                let possible_moves;
                let best_move;
                if board.next_player == Some(alphazeutreeko_color.clone()) {
                    println!("AlphaZeutreeko is playing");
                    possible_moves = self.alphazeutreeko.give_all_options(&board);
                    best_move = self.alphazeutreeko.best_move_from_vec(&possible_moves);
                }
                else {
                    println!("Opponent is playing");
                    possible_moves = self.opponent.give_all_options(&board);
                    best_move = self.opponent.best_move_from_vec(&possible_moves);
                }
                
                to_feed.push((board.clone(), possible_moves.clone()));
                let moved = board.move_pawn_until_blocked(best_move.0, &best_move.1);
                if !moved {
                    panic!("An invalid move was selected!!!");
                }
                number_moves += 1;
                if number_moves > 255 {
                    println!("Game taking too long, consider it a draw");
                    board_eval = 0.5;
                    draws += 1.0;
                    break;
                }
            }
            if board.winner() == Some(alphazeutreeko_color.clone()) {
                victories += 1.0;
            }
            println!("Game done, proceeding to learning");
            for element in to_feed.into_iter(){
                let input = board_to_input(&board, &self.device);
                let target = moves_and_value_to_target(&element, board_eval, &self.device);
                let illegal_mask = illegal_mask(&board, &self.device);
                self.train_step(input, target, illegal_mask);
                board_eval = 1.0 - board_eval;
            }
            self.alphazeutreeko.set_color(alphazeutreeko_color.other_color());
            self.opponent.set_color(alphazeutreeko_color);
        }
        println!("Victories: {:.1}%, Draws: {:.1}%", 100.0*victories/max_epoch as f32, 100.0*draws/max_epoch as f32);
    }

    pub fn save(&self, filepath: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.alphazeutreeko.mcts.policy.ann.clone().save_file(filepath, &self.recorder)?;
        Ok(())
    }

    pub fn load(&mut self, filepath: &str) -> Result<(), Box<dyn std::error::Error>> {
        let loaded_ann = self.alphazeutreeko.mcts.policy.ann.clone().load_file(filepath, &self.recorder, &self.device)?;
        self.alphazeutreeko.mcts.policy.ann = loaded_ann;
        Ok(())
    }
}
