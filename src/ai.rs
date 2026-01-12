use std::vec;
use std::collections::HashMap;

use crate::logic::{Board, Color, Direction};
use strum::IntoEnumIterator;
use wasm_bindgen::prelude::*;
use log::info;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Math)]
    fn random() -> f64;
}

fn get_random_f32() -> f32 {
    random() as f32
}

pub struct AI {
    pub color: Color,
    pub board: Board,
    pub depth: usize,
}

impl AI {
    pub fn new(color: Color, board: Board, depth: usize) -> Self {
        AI {
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

    fn score_board(&self) -> f32 {
        match self.board.winner() {
            Some(winner_color) => {
                if winner_color == self.color.clone() {
                    100.0
                } else {
                    -100.0
                }
            },
            None => {
                get_random_f32() * 2.0 - 1.0
            }
        }
    }

    fn best_move(&self) -> (usize, Direction) {
        let mut reached_boards: HashMap<(usize, Direction), HashMap<usize, Vec<Board>>> = HashMap::new();
        let mut next_reached_boards: HashMap<(usize, Direction), HashMap<usize, Vec<Board>>> = HashMap::new();
        for current_depth in 0..self.depth {
            let color_at_this_depth = if current_depth % 2 == 0 {
                self.color.clone()
            } else {
                match self.color {
                    Color::Green => Color::Yellow,
                    Color::Yellow => Color::Green,
                }
            };
            if current_depth == 0 {
                for (pawn_index, pawn) in self.board.clone().pawns.iter().enumerate() {
                    if pawn.color != color_at_this_depth {
                        continue;
                    }
                    let directions = Direction::iter();
                    for direction in directions {
                        let mut new_board = self.board.clone();
                        match new_board.move_pawn_until_blocked(pawn_index, &direction) {
                            true => {
                                let move_to_record = (pawn_index, direction);
                                let mut to_insert = HashMap::new();
                                to_insert.insert(1, vec![new_board]);
                                next_reached_boards.insert(move_to_record, to_insert);
                            }
                            false => ()
                        };
                    }
                }
            }
            else {
                for (initial_move, boards_at_depth) in reached_boards.iter() {
                    let mut boards_next = Vec::new();
                    if !boards_at_depth.contains_key(&current_depth) {
                        continue;
                    }
                    for considered_board in boards_at_depth[&current_depth].iter() {
                        if considered_board.winner().is_some() {
                            continue;
                        }
                        for (pawn_index, pawn) in considered_board.pawns.iter().enumerate() {
                            if pawn.color != color_at_this_depth {
                                continue;
                            }
                            let directions = Direction::iter();
                            for direction in directions {
                                let mut new_board = considered_board.clone();
                                match new_board.move_pawn_until_blocked(pawn_index, &direction) {
                                    true => boards_next.push(new_board),
                                    false => ()
                                };
                            }
                        }
                    }
                    if boards_next.len() > 0 {
                        let entry = next_reached_boards.entry(initial_move.clone()).or_insert(HashMap::new());
                        entry.insert(current_depth + 1, boards_next);
                    }
                }
            }
            reached_boards = next_reached_boards;
            next_reached_boards = reached_boards.clone();
        }
        let mut score_per_move: HashMap<(usize, Direction), f32> = HashMap::new();
        for (initial_move, boards_at_depth) in reached_boards.iter() {
            let mut total_score = 0.0;
            let mut win = false;
            let mut lose = false;
            for current_depth in 1..=self.depth {
                if !boards_at_depth.contains_key(&current_depth) {
                    continue;
                }
                let boards = &boards_at_depth[&current_depth];
                let mut score_at_this_depth = 0.0;
                let number_boards = boards.len() as f32;
                for board in boards.iter() {
                    let board_score = AI::new(self.color.clone(), board.clone(), self.depth).score_board();
                    if board_score.abs() > 90.0 {
                        score_at_this_depth = board_score;
                        if board_score > 0.0 {
                            win = true;
                        }
                        else {
                            lose = true;
                        }
                        break;
                    }
                    score_at_this_depth += board_score / number_boards;
                }
                let weight = 1.0 / ((current_depth + 1) as f32);
                if win && !lose {
                    // prioritize winning moves if no risk of losing before
                    total_score = 10.0 * score_at_this_depth * weight ;
                    break;
                }
                if current_depth <= 2 && lose {
                    // heavily penalize moves that lead to losing immediately
                    total_score = 10.0 * score_at_this_depth * weight ;
                    break;
                }
                total_score += score_at_this_depth * weight;
            }
            score_per_move.insert(initial_move.clone(), total_score);
        }
        let mut best_move = (0, Direction::Up);
        let mut best_score = f32::NEG_INFINITY;
        for (move_key, score) in score_per_move.iter() {
            let direction_string = format!("{:?}", move_key.1);
            info!("Moving pawn {} in direction {} has score {}", move_key.0, direction_string, score);
            if *score > best_score {
                best_score = *score;
                best_move = move_key.clone();
            }
        }
        best_move
    }
}
