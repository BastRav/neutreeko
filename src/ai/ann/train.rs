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

pub struct ANNTrainer<B: AutodiffBackend> {
    pub alphazeutreeko: AlphaZeutreeko<B, NativePlatform>,
    pub optimizer: OptimizerAdaptor<Adam, ANN<B>, B>,
    pub device: Device<B>,
    pub recorder: BinFileRecorder<FullPrecisionSettings>,
}

impl<B: AutodiffBackend<FloatElem = f32>> ANNTrainer<B> {
    pub fn new() -> Self {
        let device = B::Device::default();
        let optimizer = AdamConfig::new().init();
        let alphazeutreeko = AlphaZeutreeko::new(Color::Green, 3);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

        Self {
            alphazeutreeko,
            optimizer,
            device,
            recorder
        }
    }

    pub fn loss(&self, output: PolicyValueOutput<B>, target: PolicyValueTarget<B>, illegal_mask: Tensor<B, 2>) -> Tensor<B, 1> {
        let masked_probabilities = output.policy + illegal_mask;
        let log_probabilities = log_softmax(masked_probabilities, 1);
        let policy_loss = -(target.policy * log_probabilities).sum_dim(1).mean();
        let value_loss = MseLoss::new().forward(output.value, target.value, Reduction::Mean);
        policy_loss + value_loss * 0.5
    }

    pub fn train_step(&mut self, input:Tensor<B, 3>, target: PolicyValueTarget<B>, illegal_mask: Tensor<B, 2>) -> Tensor<B, 1> {
        // Forward pass
        let output = self.alphazeutreeko.mcts.policy.ann.forward(input);

        let loss = self.loss(output, target, illegal_mask);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.alphazeutreeko.mcts.policy.ann);

        // Update self.alphazeutreeko.policy.ann parameters
        self.alphazeutreeko.mcts.policy.ann = self.optimizer.step(0.001, self.alphazeutreeko.mcts.policy.ann.clone(), grads);
        loss
    }

    pub fn training_loop(&mut self, max_epoch: i32) {
        for epoch in 1..=max_epoch {
            println!("Starting iteration {}", epoch);
            let mut to_feed = vec![];
            let mut board = Board::default_new();
            let mut number_moves = 0;
            let mut board_eval = 1.0;
            while board.winner().is_none() {
                self.alphazeutreeko.set_color(board.next_player.clone().unwrap());
                let possible_moves = self.alphazeutreeko.mcts.give_all_options(&board);
                to_feed.push((board.clone(), possible_moves.clone()));
                let best_move = possible_moves.iter().max_by(|a, b| b.0.partial_cmp(&a.0).unwrap()).unwrap();
                let moved = board.move_pawn_until_blocked(best_move.1, &best_move.2);
                if !moved {
                    panic!["An invalid move was selected!!!"];
                }
                number_moves += 1;
                if number_moves > 255 {
                    println!("Game taking too long, consider it a draw");
                    board_eval = 0.5;
                    break;
                }
            }
            println!("Game done, proceeding to learning");
            for element in to_feed.iter(){
                let input = board_to_input(&board, &self.device);
                let target = moves_and_value_to_target(element, board_eval, &self.device);
                let illegal_mask = illegal_mask(&board, &self.device);
                self.train_step(input, target, illegal_mask);
                board_eval = 1.0 - board_eval;
            }
        }
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
