use std::vec;
use std::marker::PhantomData;

use crate::{
    ai::alphazeutreeko::AlphaZeutreeko, logic::{Board, Color, Direction}, platform::Platform
};
use super::{AI, alphazeutreeko::ANNPolicy};

use petgraph::Graph;
use petgraph::visit::EdgeRef;
use petgraph::prelude::NodeIndex;
use burn::tensor::backend::Backend;


#[derive(Clone)]
struct MCTSNode {
    board_hash: u64,
    board: Board,
    color_next_player: Color, // the color of the next player (if no next player, not the color of the previous player)
    visits: usize,
    wins: f32,
    untried_actions: Vec<(f32, usize, Direction, Board)>,
    board_eval: f32,
}

impl MCTSNode {
    fn new(board: Board, color_next_player: Color, untried_actions: Vec<(f32, usize, Direction, Board)>, board_eval: f32) -> Self {
        Self {
            board_hash: board.get_hash(),
            board,
            color_next_player,
            visits: 0,
            wins: 0.0,
            untried_actions,
            board_eval,
        }
    }

    fn is_terminal(&self) -> bool {
        self.board.winner().is_some()
    }

    fn is_fully_expanded(&self) -> bool {
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
    color: Color,
    time_allowed_ms: f64,
    graph: Graph<MCTSNode, (f32, usize, Direction)>,
    pub policy: P,
    platform: PhantomData<O>,
}

impl<P: Policy, O: Platform> MCTSGeneric<P, O> {
    pub fn clear_graph(&mut self) {
        self.graph.clear();
    }

    fn iterate(&mut self, origin: NodeIndex) {
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

    fn expand(&mut self, node_index: NodeIndex) -> NodeIndex {
        let node = self.graph.node_weight_mut(node_index).unwrap();
        let action = node.untried_actions.pop().unwrap();
        let child_color = node.color_next_player.other_color();
        let prediction = self.policy.predict(&action.3);
        let child = self.graph.add_node(MCTSNode::new(action.3, child_color, prediction.1, prediction.0));
        self.graph.add_edge(node_index, child, (action.0, action.1, action.2));
        child
    }

    fn random_rollout(&self, node: &MCTSNode) -> f32 {
        let mut current_board = node.board.clone();
        while current_board.next_player.is_some() {
            let all_possible_moves = current_board.get_all_valid_directions_and_resulting_boards();
            let random_move_index = O::random_int(all_possible_moves.len());
            current_board = all_possible_moves[random_move_index].2.clone();
        }
        if current_board.winner().unwrap() == node.color_next_player {1.0} else {-1.0}
    }

    fn rollout(&self, node_index: NodeIndex) -> f32 {
        let node = self.graph.node_weight(node_index).unwrap();
        if node.board.winner().is_some() {
            -1.0 // cannot win because opponent made a move, this is a loss
        }
        else if P::IS_TRIVIAL {
            self.random_rollout(node)
        }
        else {
            node.board_eval
        }
    }

    fn backpropagate(&mut self, node_index:NodeIndex, winner:f32) {
        let mut current_node_index = node_index;
        let mut to_add = winner;
        loop {
            let current_node = self.graph.node_weight_mut(current_node_index).unwrap();
            current_node.visits += 1;
            current_node.wins += to_add;
            to_add = -to_add;
            match self.graph.edges_directed(current_node_index, petgraph::Direction::Incoming).next() {
                None => break,
                Some(edge) => current_node_index = edge.source(),
            }
        }
    }

    fn best_child(&mut self, node_index: NodeIndex) -> NodeIndex {
        let mut best_score = f32::MIN;
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
            let exploit = -child.wins / child.visits as f32;
            let explore = prior * 1.414 * (parent_visits.ln() / child.visits as f32).sqrt();
            let score = exploit + explore;
            if score > best_score {
                best_child = child_index;
                best_score = score;
            }
        }
        best_child
    }

    fn choose_final_move_give_all_options(&self, origin: NodeIndex) -> (f32, Vec<(f32, usize, Direction)>) {
        let origin_node = self.graph.node_weight(origin).unwrap();
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
        (origin_node.wins / origin_node.visits as f32 , moves_found)
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

impl<P: Policy, O: Platform> AI<O> for MCTSGeneric<P, O> {
    fn new(color: Color, difficulty: usize) -> Self {
        O::print(&format!("Creating MCTS AI with trivial policy? {}", P::IS_TRIVIAL));
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

    fn set_color(&mut self, color:Color){
        self.color = color;
        self.graph.clear();
    }

    fn give_all_options(&mut self, board:&Board, verbose: bool) -> (f32, Vec<(f32, usize, Direction)>) {
        // graph is no longer cleared by default, risk of high memory usage
        // self.graph.clear();
        let first_prediction = self.policy.predict(board);
        if verbose {
            O::print(&format!("Policy gives board eval {}", first_prediction.0));
            for element in first_prediction.1.iter() {
                O::print(&format!("Policy gives eval {} to move {:?}", element.0, (element.1, element.2.clone())));
            }
        }
        let board_hash = board.get_hash();
        let possible_origin = self.graph.node_indices().find(|index| {
            let index_hash = self.graph.node_weight(*index).unwrap().board_hash;
            board_hash == index_hash
        });
        let origin = possible_origin.unwrap_or_else(|| {
            self.graph.add_node(MCTSNode::new(board.clone(), self.color.clone(), first_prediction.1, first_prediction.0))
        });

        let start_time = O::now();
        while O::now() - start_time < self.time_allowed_ms {
            self.iterate(origin);
        }
        self.choose_final_move_give_all_options(origin)
    }
}

pub type MCTS<O> = MCTSGeneric<TrivialPolicy, O>;

impl<B: Backend, O: Platform> AlphaZeutreeko<B, O> {
    pub fn new_no_data(color: Color, difficulty: usize) -> Self {
        O::print(&format!("Creating MCTS AI with trivial policy? {}", false));
        Self {
            color,
            time_allowed_ms: (difficulty.pow(3)) as f64 * 0.05 * 1000.0,
            graph: Graph::<MCTSNode, (f32, usize, Direction)>::new(),
            policy: ANNPolicy::new_no_data(),
            platform: PhantomData,
        }
    }
}
