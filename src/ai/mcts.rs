use std::vec;
use std::marker::PhantomData;

use crate::logic::{Board, Color, Direction};
use super::AI;

use wasm_bindgen::prelude::*;
use log::info;
use petgraph::Graph;
use petgraph::visit::EdgeRef;
use petgraph::prelude::NodeIndex;

pub trait Platform: Clone {
    fn now() -> f64;
    fn random() -> f32;
}

#[derive(Clone)]
pub struct WasmPlatform;

impl Platform for WasmPlatform {
    fn now() -> f64 {
        #[wasm_bindgen(js_namespace = performance)]
        extern "C" {
            fn now() -> f64;
        }
        now()
    }

    fn random() -> f32 {
        #[wasm_bindgen(js_namespace = Math)]
        extern "C" {
            fn random() -> f64;
        }
        random() as f32
    }
}

#[derive(Clone)]
pub struct NativePlatform;

impl Platform for NativePlatform {
    fn now() -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64() * 1000.0
    }

    fn random() -> f32 {
        use rand::Rng;
        rand::rng().random()
    }
}

#[derive(Clone)]
pub struct MCTSNode {
    pub board: Board,
    pub color: Color,
    pub visits: usize,
    pub wins: f32,
    pub untried_actions: Vec<(f32, usize, Direction, Board)>,
    pub board_eval: f32,
}

impl MCTSNode {
    pub fn new(board: Board, color: Color, untried_actions: Vec<(f32, usize, Direction, Board)>, board_eval: f32) -> Self {
        Self {
            board,
            color,
            visits: 0,
            wins: 0.0,
            untried_actions,
            board_eval,
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.board.winner().is_some()
    }

    pub fn is_fully_expanded(&self) -> bool {
        self.untried_actions.len() == 0
    }
}

pub trait Policy: Clone {
    const IS_TRIVIAL: bool;
    fn predict(&self, board:&Board) -> (f32, Vec<(f32, usize, Direction, Board)>) {
        (0.0, board.get_all_valid_directions_and_resulting_boards().into_iter().map(|(p, dir, b)| (0.0, p, dir, b)).collect())
    }
    fn new() -> Self;
}

#[derive(Clone)]
pub struct MCTSGeneric<P: Policy, O: Platform> {
    pub color: Color,
    pub time_allowed_ms: f64,
    pub graph: Graph<MCTSNode, (f32, usize, Direction)>,
    pub policy: P,
    pub platform: PhantomData<O>,
}

impl<P: Policy, O: Platform> MCTSGeneric<P, O> {
    pub fn iterate(&mut self, origin: NodeIndex) {
        let mut node_index = origin;
        let mut node = self.graph.node_weight(node_index).unwrap();
        while !node.is_terminal() && node.is_fully_expanded() {
            node_index = self.best_child(node_index);
            node = self.graph.node_weight(node_index).unwrap();
        }
        if !node.is_terminal() && !node.is_fully_expanded() {
            node_index = self.expand(node_index);
        }

        let winner = self.rollout(node_index);
        self.backpropagate(node_index, winner);
    }

    pub fn expand(&mut self, node_index: NodeIndex) -> NodeIndex{
        let node = self.graph.node_weight_mut(node_index).unwrap();
        let action = node.untried_actions.pop().unwrap();
        let child_color = node.color.other_color();
        let prediction = self.policy.predict(&action.3);
        let child = self.graph.add_node(MCTSNode::new(action.3, child_color, prediction.1, prediction.0));
        self.graph.add_edge(node_index, child, (action.0, action.1, action.2));
        child
    }

    pub fn random_rollout(&self, node: &MCTSNode) -> f32 {
        let mut current_board = node.board.clone();
        let player_color = node.color.clone();
        while current_board.next_player.is_some() {
            let all_possible_moves = current_board.get_all_valid_directions_and_resulting_boards();
            let random_move_index = (O::random() * all_possible_moves.len() as f32).floor() as usize;
            current_board = all_possible_moves[random_move_index].2.clone();
        }
        if current_board.winner().unwrap() == player_color {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    pub fn rollout(&self, node_index: NodeIndex) -> f32 {
        let node = self.graph.node_weight(node_index).unwrap();
        if P::IS_TRIVIAL {
            return self.random_rollout(node);
        } else {
            return node.board_eval;
        }
    }

    pub fn backpropagate(&mut self, node_index:NodeIndex, winner:f32) {
        let mut current_node_index = node_index;
        let mut to_add = winner;
        loop {
            let current_node = self.graph.node_weight_mut(current_node_index).unwrap();
            current_node.visits += 1;
            current_node.wins += to_add;
            to_add = 1.0 - to_add;
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
            let mut prior = edge.weight().0;
            if P::IS_TRIVIAL {
                prior = 1.0;
            }
            let exploit = child.wins / child.visits as f32;
            let explore = prior * 1.414 * (parent_visits.ln() / child.visits as f32).sqrt();
            let score = exploit + explore;
            if score > best_score {
                best_child = child_index;
                best_score = score;
            }
        }
        best_child
    }

    pub fn choose_final_move(&self, origin: NodeIndex) -> (usize, Direction) {
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
        let best_move_found = best_moves_found[(O::random() * best_moves_found.len() as f32).floor() as usize].clone();
        info!("==Best move found: {:?} with score {}==", best_move_found, best_score);
        (best_move_found.1, best_move_found.2)
    }

    pub fn choose_final_move_give_all_options(&self, origin: NodeIndex) -> Vec<(f32, usize, Direction)> {
        let mut moves_found = vec![];
        let mut total_visits = 0.0;
        for edge in self.graph.edges(origin) {
            let target_node_index = edge.target();
            let visits = self.graph.node_weight(target_node_index).unwrap().visits as f32;
            let move_with_policy = edge.weight().clone();
            moves_found.push((visits, move_with_policy.1, move_with_policy.2));
            total_visits += visits;
        }
        moves_found.iter_mut().for_each(|x| x.0 /= total_visits);
        moves_found
    }

    pub fn give_all_options(&mut self, board:&Board) -> Vec<(f32, usize, Direction)> {
        self.graph.clear();
        let first_prediction = self.policy.predict(board);
        let origin = self.graph.add_node(MCTSNode::new(board.clone(), self.color.other_color(), first_prediction.1, first_prediction.0));

        let start_time = O::now();
        while O::now() - start_time < self.time_allowed_ms {
            self.iterate(origin);
        }
        self.choose_final_move_give_all_options(origin)
    }
}

#[derive(Clone)]
pub struct TrivialPolicy;

impl Policy for TrivialPolicy {
    const IS_TRIVIAL:bool = true;
    fn new() -> Self {
        Self {}
    }
}

impl<P: Policy, O: Platform> AI for MCTSGeneric<P, O> {
    fn new(color: Color, difficulty: usize) -> Self {
        info!("Creating MCTS AI with trivial policy? {}", P::IS_TRIVIAL);
        Self {
            color,
            time_allowed_ms: (difficulty.pow(3)) as f64 * 0.05 * 1000.0,
            graph: Graph::<MCTSNode, (f32, usize, Direction)>::new(),
            policy: P::new(),
            platform: PhantomData,
        }
    }

    fn color(&self) -> &Color {
        &self.color
    }

    fn best_move(&mut self, board:&Board) -> (usize, Direction) {
        self.graph.clear();
        let first_prediction = self.policy.predict(board);
        let origin = self.graph.add_node(MCTSNode::new(board.clone(), self.color.other_color(), first_prediction.1, first_prediction.0));

        let start_time = O::now();
        let mut iterations = 0;
        while O::now() - start_time < self.time_allowed_ms {
            self.iterate(origin);
            iterations += 1;
        }
        info!("MCTS completed {} iterations", iterations);

        self.choose_final_move(origin)
    }
}

#[derive(Clone)]
pub struct MCTS<O: Platform> {
    pub mcts: MCTSGeneric<TrivialPolicy, O>,
}

impl<O: Platform> AI for MCTS<O> {
    fn new(color: Color, difficulty: usize) -> Self {
        Self {
            mcts: MCTSGeneric::new(color, difficulty),
        }
    }

    fn color(&self) -> &Color {
        self.mcts.color()
    }

    fn best_move(&mut self, board:&Board) -> (usize, Direction) {
        self.mcts.best_move(board)
    }
}
