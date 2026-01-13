use std::vec;

use crate::logic::{Board, Color, Direction};
use wasm_bindgen::prelude::*;
use log::info;
use petgraph::Graph;
use petgraph::visit::EdgeRef;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Math)]
    fn random() -> f64;
}

fn get_random_f32() -> f32 {
    random() as f32
}

pub struct BoardEvaluation {
    pub board: Board,
    pub color: Color,
    pub score: f32,
}

impl BoardEvaluation {
    pub fn new(board: Board, color: Color) -> Self {
        Self { board: board.clone(), color: color.clone(), score: Self::score_board(&board, &color) }
    }

    fn score_board(board: &Board, color: &Color) -> f32 {
        let random_addition = get_random_f32() * 2.0 - 1.0;
        match board.winner() {
            Some(winner_color) if winner_color == *color => 100.0 + random_addition,
            Some(_) => -100.0 + random_addition,
            None => random_addition,
        }
    }
}

pub struct AI {
    pub color: Color,
    pub board: Board,
    pub depth: usize,
}

impl AI {
    pub fn new(color: Color, board: Board, depth: usize) -> Self {
        Self {
            color,
            board,
            depth,
        }
    }

    pub fn ai_play(&mut self) -> Option<(usize, Direction)> {
        if self.board.next_player != Some(self.color.clone()) {
            return None;
        }
        Some(self.best_move())
    }

    fn best_move(&self) -> (usize, Direction) {
        let mut possible_boards = Graph::<BoardEvaluation, (usize, Direction)>::new();
        let origin = possible_boards.add_node(BoardEvaluation::new(self.board.clone(), self.color.clone()));
        let mut to_explore = vec![origin];
        for current_depth in 0..self.depth {
            let color_at_this_depth = if current_depth % 2 == 0 {
                self.color.clone()
            } else {
                match self.color {
                    Color::Green => Color::Yellow,
                    Color::Yellow => Color::Green,
                }
            };
            let mut to_explore_next = Vec::new();
            for considered_node_index in to_explore.iter() {
                let considered_board = possible_boards.node_weight(*considered_node_index).unwrap().board.clone();
                if considered_board.winner().is_some() {
                    continue;
                }
                for (pawn_index, pawn) in considered_board.pawns.iter().enumerate() {
                    if pawn.color != color_at_this_depth {
                        continue;
                    }
                    let directions = considered_board.get_valid_directions_and_resulting_boards(pawn_index);
                    for (direction, new_board) in directions {
                        let new_node_index = possible_boards.add_node(BoardEvaluation::new(new_board.clone(), self.color.clone()));
                        possible_boards.add_edge(*considered_node_index, new_node_index, (pawn_index, direction.clone()));
                        to_explore_next.push(new_node_index);
                    }
                }
            }
            to_explore = to_explore_next;
        }
        let mut best_score = f32::MIN;
        let mut best_move_found = possible_boards.edges(origin).next().unwrap().weight().clone();
        for edge in possible_boards.edges(origin) {
            let target_node_index = edge.target();
            let minmax = self.minmax_score(&possible_boards, target_node_index, self.depth - 1);
            info!("Considering move {:?} with minmax score {}", edge.weight(), minmax);
            if minmax > best_score {
                best_score = minmax;
                best_move_found = edge.weight().clone();
            }
        }
        info!("==Best move found: {:?} with score {}==", best_move_found, best_score);
        best_move_found
    }

    fn minmax_score(&self, graph: &Graph<BoardEvaluation, (usize, Direction)>, node_index: petgraph::prelude::NodeIndex, depth_remaining: usize) -> f32 {
        if depth_remaining == 0 {
            return graph.node_weight(node_index).unwrap().score;
        }
        let mut scores = Vec::new();
        for edge in graph.edges(node_index) {
            let target_node_index = edge.target();
            let score = self.minmax_score(graph, target_node_index, depth_remaining - 1);
            scores.push(score);
        }
        if scores.is_empty() {
            return graph.node_weight(node_index).unwrap().score;
        }
        if (self.depth - depth_remaining) % 2 == 0 {
            *scores.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        } else {
            *scores.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        }
    }
}
