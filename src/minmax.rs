use std::vec;

use crate::logic::{Board, Color, Direction};
use crate::ai::AI;
use wasm_bindgen::prelude::*;
use log::info;
use petgraph::Graph;
use petgraph::visit::EdgeRef;
use petgraph::prelude::NodeIndex;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Math)]
    fn random() -> f64;
}

fn get_random_f32() -> f32 {
    random() as f32
}

#[derive(Clone)]
pub struct BoardEvaluation {
    pub board: Board,
    pub color: Color,
    pub score: isize,
    pub depth: usize,
}

impl BoardEvaluation {
    pub fn new(board: Board, color: Color, depth: usize) -> Self {
        let mut a = Self { board: board, color: color, score: 0, depth: depth};
        a.score_board();
        a
    }

    fn score_board(&mut self) {
        match self.board.winner() {
            Some(winner_color) if winner_color == self.color => self.score = 100 - self.depth as isize,
            Some(_) => self.score = -100 + self.depth as isize,
            None => self.score = 0,
        }
    }
}

#[derive(Clone)]
pub struct MinMax {
    pub color: Color,
    pub depth: usize,
    pub graph: Graph<BoardEvaluation, (usize, Direction)>,
}

impl MinMax{
    pub fn new(color: Color, depth: usize) -> Self {
        Self {
            color,
            depth,
            graph: Graph::<BoardEvaluation, (usize, Direction)>::new(),
        }
    }

    fn minmax_score(&self, node_index: NodeIndex, depth_remaining: usize, mut alpha: isize, mut beta: isize, maximizing_player: bool) -> isize {
        if depth_remaining == 0 {
            return self.graph.node_weight(node_index).unwrap().score;
        }

        let mut value = if maximizing_player { isize::MIN } else { isize::MAX };
        let mut at_least_one_edge = false;
        for edge in self.graph.edges(node_index) {
            at_least_one_edge = true;
            let target_node_index = edge.target();
            let score = self.minmax_score(target_node_index, depth_remaining - 1, alpha, beta, !maximizing_player);

            if maximizing_player {
                value = value.max(score);
                alpha = alpha.max(value);
            } else {
                value = value.min(score);
                beta = beta.min(value);
            }

            if beta <= alpha {
                break; // Alpha-beta pruning
            }
        }
        if !at_least_one_edge {
            value = self.graph.node_weight(node_index).unwrap().score;
        }
        value
    }
}

impl AI for MinMax {
    fn color(&self) -> &Color {
        &self.color
    }

    fn best_move(&mut self, board:&Board) -> (usize, Direction) {
        self.graph.clear();
        let origin = self.graph.add_node(BoardEvaluation::new(board.clone(), self.color.clone(), 0));
        let mut to_explore = vec![origin];
        for current_depth in 0..self.depth {
            let color_at_this_depth = if current_depth % 2 == 0 {
                self.color.clone()
            } else {
                self.color.other_color()
            };
            let mut to_explore_next = Vec::new();
            for considered_node_index in to_explore.iter() {
                let considered_board = self.graph.node_weight(*considered_node_index).unwrap().board.clone();
                if considered_board.winner().is_some() {
                    continue;
                }
                for (pawn_index, pawn) in considered_board.pawns.iter().enumerate() {
                    if pawn.color != color_at_this_depth {
                        continue;
                    }
                    let directions = considered_board.get_valid_directions_and_resulting_boards(pawn_index);
                    for (direction, new_board) in directions {
                        let new_node_index = self.graph.add_node(BoardEvaluation::new(new_board.clone(), self.color.clone(), current_depth + 1));
                        self.graph.add_edge(*considered_node_index, new_node_index, (pawn_index, direction.clone()));
                        to_explore_next.push(new_node_index);
                    }
                }
            }
            to_explore = to_explore_next;
        }
        let mut best_score = isize::MIN;
        let mut best_moves_found = vec![];
        for edge in self.graph.edges(origin) {
            let target_node_index = edge.target();
            let minmax = self.minmax_score(target_node_index, self.depth - 1, isize::MIN, isize::MAX, false);
            info!("Considering move {:?} with minmax score {}", edge.weight(), minmax);
            if minmax > best_score {
                best_score = minmax;
                best_moves_found = vec![edge.weight().clone()];
            }
            if minmax == best_score {
                best_moves_found.push(edge.weight().clone());
            }
        }
        let best_move_found = best_moves_found[(get_random_f32() * best_moves_found.len() as f32).floor() as usize].clone();
        info!("==Best move found: {:?} with score {}==", best_move_found, best_score);
        best_move_found
    }
}
