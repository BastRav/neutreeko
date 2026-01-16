use crate::ai::mcts::{MCTSGeneric, Policy};
use crate::ai::AI;
use crate::ai::ann::ANN;
use burn::tensor::backend::Backend;
use crate::logic::{Board, Color, Direction};

#[derive(Clone)]
pub struct ANNPolicy<B: Backend> {
    pub ann: ANN<B>,
}

impl<B: Backend> Policy for ANNPolicy<B> {
    fn new() -> Self {
        Self {
            ann: ANN::init(32, &B::Device::default()),
        }
    }

    fn predict(&self, board:&Board) -> (f32, Vec<(f32, usize, Direction, Board)>) {
        self.ann.predict(board)
    }
}

#[derive(Clone)]
pub struct AlphaZeutreeko <B: Backend> {
    pub mcts: MCTSGeneric<ANNPolicy<B>>,
}

impl<B: Backend> AI for AlphaZeutreeko<B> {
    fn color(&self) -> &Color {
        &self.mcts.color
    }

    fn new(color: Color, difficulty: usize) -> Self {
        let mcts = MCTSGeneric::new(color.clone(), difficulty);
        Self {
            mcts,
        }
    }

    fn best_move(&mut self, board:&Board) -> (usize, Direction) {
        self.mcts.best_move(board)
    }
}