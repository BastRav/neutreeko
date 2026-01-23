use crate::{
    ai::{
        mcts::{MCTSGeneric, Policy},
        ann::{ANN, ANNConfig},
    },
    logic::{Board, Direction},
};
use burn::tensor::backend::Backend;

#[derive(Clone)]
pub struct ANNPolicy<B: Backend> {
    pub ann: ANN<B>,
}

impl<B: Backend> Policy for ANNPolicy<B> {
    const IS_TRIVIAL:bool = false;
    fn new() -> Self {
        Self {
            ann: ANNConfig::init_from_data(32, &B::Device::default()),
        }
    }

    fn predict(&self, board:&Board) -> (f32, Vec<(f32, usize, Direction, Board)>) {
        self.ann.predict(board)
    }
}

pub type AlphaZeutreeko <B, O> = MCTSGeneric<ANNPolicy<B>, O>;
