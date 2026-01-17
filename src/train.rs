use neutreeko::{
    ai::ann::{
        inputouput::{board_to_input, illegal_mask},
        train::{ANNTrainer, PolicyValueTarget},
    },
    logic::Board,
};
use burn::{
    backend::{Autodiff, NdArray},
    tensor::Tensor,
};

#[cfg(feature = "train")]
fn main() {
    let mut trainer: ANNTrainer<Autodiff<NdArray<f32>>> = ANNTrainer::new();

    // Example training loop
    let board = Board::default_new();
    let value_target = 0.5; // Example target value
    let policy_target = [0.1, 0.2, 0.7, 0.0]; // Example move probabilities

    let value_target = Tensor::from_floats([value_target], &trainer.device);
    let policy_target = Tensor::from_floats(policy_target, &trainer.device);
    let target = PolicyValueTarget {
        value: value_target,
        policy: policy_target,
    };

    let input = board_to_input(&board, &trainer.device);
    let illegal_mask = illegal_mask(&board, &trainer.device);
    trainer.train_step(input, target, illegal_mask);
}