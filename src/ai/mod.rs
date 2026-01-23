pub mod minmax;
pub mod mcts;
pub mod ann;
pub mod alphazeutreeko;
use crate::{logic::{Board, Color, Direction}, platform::Platform};

pub trait AI<O: Platform>: Clone {
    fn color(&self) -> &Color;
    fn set_color(&mut self, color: Color);
    fn new(color: Color, depth: usize) -> Self;

    fn ai_play(&mut self, board:&Board, verbose: bool) -> Option<(usize, Direction)> {
        if board.next_player != Some(self.color().clone()) {
            return None;
        }
        Some(self.best_move(board, verbose))
    }

    fn best_move_from_vec(&mut self, moves: &Vec<(f32, usize, Direction)>, verbose: bool) -> (usize, Direction) {
        let mut best_moves_found = vec![];
        let mut best_score = 0.0;
        for option in moves {
            if verbose {
                O::print(&format!("Considering move {:?} with score {}", (option.1, option.2.clone()), option.0));
            }
            if option.0 > best_score {
                best_score = option.0;
                best_moves_found = vec![(option.1, &option.2)];
            } else if option.0 == best_score {
                best_moves_found.push((option.1, &option.2));
            }
        }
        let best_move_found = best_moves_found[(O::random() * best_moves_found.len() as f32).floor() as usize];
        if verbose {
            O::print(&format!("==Best move found: {:?} with score {}==", best_move_found, best_score));
        }
        (best_move_found.0, best_move_found.1.clone())
    }

    fn best_move(&mut self, board:&Board, verbose: bool) -> (usize, Direction) {
        let all_options = self.give_all_options(board, verbose);
        self.best_move_from_vec(&all_options.1, verbose)
    }

    fn give_all_options(&mut self, board:&Board, verbose: bool) -> (f32, Vec<(f32, usize, Direction)>);
}