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
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Math)]
    fn random() -> f64;
}

fn get_random_f32() -> f32 {
    random() as f32
}

#[derive(Clone)]
pub struct MCTSNode {
    pub board: Board,
    pub color: Color,
    pub visits: usize,
    pub wins: usize,
    pub untried_actions: Vec<(usize, Direction, Board)>,
}

impl MCTSNode {
    pub fn new(board: Board, color: Color) -> Self {
        let untried_actions = board.get_all_valid_directions_and_resulting_boards();
        Self {
            board,
            color,
            visits: 0,
            wins: 0,
            untried_actions,
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.board.winner().is_some()
    }

    pub fn is_fully_expanded(&self) -> bool {
        self.untried_actions.len() == 0
    }
}

#[derive(Clone)]
pub struct MCTS {
    pub color: Color,
    pub time_allowed_ms: f64,
    pub graph: Graph<MCTSNode, (usize, Direction)>,
}

impl MCTS {
    pub fn new(color: Color, difficulty: usize) -> Self {
        Self {
            color,
            time_allowed_ms: (difficulty.pow(3)) as f64 * 0.05 * 1000.0,
            graph: Graph::<MCTSNode, (usize, Direction)>::new(),
        }
    }

    pub fn expand(&mut self, node_index: NodeIndex) -> NodeIndex{
        let node = self.graph.node_weight_mut(node_index).unwrap();
        let action = node.untried_actions.pop().unwrap();
        let child_color = node.color.other_color();
        let child = self.graph.add_node(MCTSNode::new(action.2, child_color));
        self.graph.add_edge(node_index, child, (action.0, action.1));
        child
    }

    pub fn rollout(&self, node_index: NodeIndex) -> Color {
        let mut current_board = self.graph.node_weight(node_index).unwrap().board.clone();
        while current_board.next_player.is_some() {
            let all_possible_moves = current_board.get_all_valid_directions_and_resulting_boards();
            let random_move_index = (get_random_f32() * all_possible_moves.len() as f32).floor() as usize;
            current_board = all_possible_moves[random_move_index].2.clone();
        }
        current_board.winner().unwrap()
    }

    pub fn backpropagate(&mut self, node_index:NodeIndex, winner:&Color) {
        let mut current_node_index = node_index;
        loop {
            let current_node = self.graph.node_weight_mut(current_node_index).unwrap();
            current_node.visits += 1;
            if current_node.color == *winner {
                current_node.wins += 1;
            }
            match self.graph.edges_directed(current_node_index, petgraph::Direction::Incoming).next() {
                None => break,
                Some(edge) => current_node_index = edge.source(),
            }
        }
    }

    pub fn best_child(&mut self, node_index: NodeIndex) -> NodeIndex {
        let mut best_score = 0.0;
        let mut best_child = self.graph.edges_directed(node_index, petgraph::Direction::Outgoing).next().unwrap().target();
        let parent_visits = self.graph.node_weight(node_index).unwrap().visits as f32;
        for edge in self.graph.edges_directed(node_index, petgraph::Direction::Outgoing) {
            let child_index = edge.target();
            let child = self.graph.node_weight(child_index).unwrap();
            if child.visits == 0 {
                return child_index;
            }
            let exploit = child.wins as f32 / child.visits as f32;
            let explore = 1.414 * (parent_visits.ln() / child.visits as f32).sqrt();
            let score = exploit + explore;
            if score > best_score {
                best_child = child_index;
                best_score = score;
            }
        }
        best_child
    }
}

impl AI for MCTS {
    fn color(&self) -> &Color {
        &self.color
    }

    fn best_move(&mut self, board:&Board) -> (usize, Direction) {
        self.graph.clear();
        let origin = self.graph.add_node(MCTSNode::new(board.clone(), self.color.other_color()));

        let start_time = now();
        let mut iterations = 0;
        while now() - start_time < self.time_allowed_ms {
            let mut node_index = origin;
            let mut node = self.graph.node_weight(node_index).unwrap();
            while !node.is_terminal() && node.is_fully_expanded() {
                node_index = self.best_child(node_index);
                node = self.graph.node_weight(node_index).unwrap();
            }
            if !node.is_terminal() && !node.is_fully_expanded() {
                node_index = self.expand(node_index);
                node = self.graph.node_weight(node_index).unwrap();
            }

            let winner = self.rollout(node_index);
            self.backpropagate(node_index, &winner);
            iterations += 1;
        }
        info!("MCTS completed {} iterations", iterations);

        let mut best_moves_found = vec![];
        let mut best_score = 0;
        for edge in self.graph.edges(origin) {
            let target_node_index = edge.target();
            let visits = self.graph.node_weight(target_node_index).unwrap().visits;
            info!("Considering move {:?} with MCTS score {}", edge.weight(), visits);
            if visits > best_score {
                best_score = visits;
                best_moves_found = vec![edge.weight().clone()];
            } else if visits == best_score {
                best_moves_found.push(edge.weight().clone());
            }
        }
        let best_move_found = best_moves_found[(get_random_f32() * best_moves_found.len() as f32).floor() as usize].clone();
        info!("==Best move found: {:?} with score {}==", best_move_found, best_score);
        best_move_found
    }
}
