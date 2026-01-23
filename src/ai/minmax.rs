use std::vec;
use std::marker::PhantomData;

use crate::{
    logic::{Board, Color, Direction},
    platform::Platform,
};
use super::AI;

use petgraph::Graph;
use petgraph::visit::EdgeRef;
use petgraph::prelude::NodeIndex;

#[derive(Clone)]
struct BoardEvaluation {
    board: Board,
    color: Color,
    score: usize,
    depth: usize,
}

impl BoardEvaluation {
    fn new(board: Board, color: Color, depth: usize) -> Self {
        let mut a = Self { board: board, color: color, score: 0, depth: depth};
        a.score_board();
        a
    }

    fn score_board(&mut self) {
        match self.board.winner() {
            Some(winner_color) if winner_color == self.color => self.score = 2000 - self.depth,
            Some(_) => self.score = self.depth + 1,
            None => self.score = 100,
        }
    }
}

#[derive(Clone)]
pub struct MinMax<O: Platform> {
    color: Color,
    depth: usize,
    graph: Graph<BoardEvaluation, (usize, Direction)>,
    _platform: PhantomData<O>,
}

impl <O: Platform> MinMax<O> {
    fn minmax_score(&self, node_index: NodeIndex, depth_remaining: usize, mut alpha: usize, mut beta: usize, maximizing_player: bool) -> usize {
        if depth_remaining == 0 {
            return self.graph.node_weight(node_index).unwrap().score;
        }

        let mut value = if maximizing_player { 0 } else { usize::MAX };
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

impl <O: Platform> AI<O> for MinMax<O> {
    fn new(color: Color, depth: usize) -> Self {
        Self {
            color,
            depth,
            graph: Graph::<BoardEvaluation, (usize, Direction)>::new(),
            _platform: PhantomData,
        }
    }

    fn color(&self) -> &Color {
        &self.color
    }

    fn set_color(&mut self, color: Color){
        self.color = color;
    }

    fn give_all_options(&mut self, board:&Board, _verbose: bool) -> (f32, Vec<(f32, usize, Direction)>) {
        self.graph.clear();
        let origin = self.graph.add_node(BoardEvaluation::new(board.clone(), self.color.clone(), 0));
        let mut to_explore = vec![origin];
        for current_depth in 0..self.depth {
            let color_at_this_depth = if current_depth % 2 == 0 {
                &self.color
            } else {
                &self.color.other_color()
            };
            let mut to_explore_next = Vec::new();
            for considered_node_index in to_explore.iter() {
                let considered_board = self.graph.node_weight(*considered_node_index).unwrap().board.clone();
                if considered_board.winner().is_some() {
                    continue;
                }
                for (pawn_index, pawn) in considered_board.pawns.iter().enumerate() {
                    if &pawn.color != color_at_this_depth {
                        continue;
                    }
                    let directions = considered_board.get_valid_directions_and_resulting_boards(pawn_index);
                    for (direction, new_board) in directions {
                        let new_node_index = self.graph.add_node(BoardEvaluation::new(new_board, self.color.clone(), current_depth + 1));
                        self.graph.add_edge(*considered_node_index, new_node_index, (pawn_index, direction));
                        to_explore_next.push(new_node_index);
                    }
                }
            }
            to_explore = to_explore_next;
        }
        let mut total = 0.0;
        let mut best_minmax = 0;
        let mut all_moves_found = vec![];
        for edge in self.graph.edges(origin) {
            let target_node_index = edge.target();
            let minmax = self.minmax_score(target_node_index, self.depth - 1, 0, usize::MAX, false);
            total += minmax as f32;
            let move_found = edge.weight().clone();
            all_moves_found.push((minmax as f32, move_found.0, move_found.1));
            if minmax > best_minmax {best_minmax = minmax;}
        }
        all_moves_found.iter_mut().for_each(|x| x.0 /= total);
        // indicates draw
        if best_minmax == 100 { best_minmax = 1000;}
        let board_eval = best_minmax as f32 / 2000.0;
        (board_eval, all_moves_found)
    }
}
