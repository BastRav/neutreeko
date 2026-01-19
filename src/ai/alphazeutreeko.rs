use crate::{
    ai::{
        mcts::{MCTSGeneric, Policy},
        AI,
        ann::{ANN, ANNConfig},
    },
    logic::{Board, Color, Direction},
    platform::Platform,
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
            ann: ANNConfig::init(32, &B::Device::default()),
        }
    }

    fn predict(&self, board:&Board) -> (f32, Vec<(f32, usize, Direction, Board)>) {
        self.ann.predict(board)
    }
}

#[derive(Clone)]
pub struct AlphaZeutreeko <B: Backend, O: Platform> {
    pub mcts: MCTSGeneric<ANNPolicy<B>, O>,
}

impl<B: Backend, O: Platform> AI<O> for AlphaZeutreeko<B, O> {
    fn color(&self) -> &Color {
        &self.mcts.color
    }
    
    fn set_color(&mut self, color:Color){
        self.mcts.color = color;
    }

    fn new(color: Color, difficulty: usize) -> Self {
        let mcts = MCTSGeneric::new(color.clone(), difficulty);
        Self {
            mcts,
        }
    }

    fn give_all_options(&mut self, board:&Board, verbose:bool) -> (f32, Vec<(f32, usize, Direction)>) {
        self.mcts.give_all_options(board, verbose)
    }
}