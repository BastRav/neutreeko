pub mod minmax;
pub mod mcts;
use crate::logic::{Direction, Board, Color};

pub trait AI: Clone {
    fn color(&self) -> &Color;

    fn ai_play(&mut self, board:&Board) -> Option<(usize, Direction)> {
        if board.next_player != Some(self.color().clone()) {
            return None;
        }
        Some(self.best_move(board))
    }

    fn best_move(&mut self, board:&Board) -> (usize, Direction);
}