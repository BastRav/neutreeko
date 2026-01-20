use core::f32;

use crate::{
    ai::ann::{
        train::PolicyValueTarget,
        utils::{board_to_input, position_direction_to_index},
    },
    logic::{Board, Direction}
};

use burn::tensor::{backend::{Backend, AutodiffBackend}, Device, Tensor};

pub fn illegal_mask<B>(board: &Board, device: &Device<B>) -> Tensor<B, 2>
where B:Backend {
    let possible_moves = board.get_all_valid_directions_and_resulting_boards();
    let mut illegal_mask_array = [-1e6; 200];
    for possible_move in possible_moves.iter() {
        let pawn_position = board.pawns[possible_move.0].position.clone();
        let mask_index = position_direction_to_index((pawn_position.row, pawn_position.column),possible_move.1.clone());
        illegal_mask_array[mask_index] = 0.0;
    }
    Tensor::from_data([illegal_mask_array], device)
}

pub fn moves_and_value_to_target<B>(board: &Board, board_eval: f32, moves_eval: &Vec<(f32, usize, Direction)>, device: &Device<B>) -> PolicyValueTarget<B>
where B: AutodiffBackend {
    let value = Tensor::from_floats([board_eval], device);
    let mut policy_floats = [0.0; 200];
    for possible_move in moves_eval.iter() {
        let position = board.pawns[possible_move.1].position.clone();
        let index_to_consider = position_direction_to_index((position.row, position.column), possible_move.2.clone());
        policy_floats[index_to_consider] = possible_move.0
    }
    let policy = Tensor::from_floats( [policy_floats], device);
    PolicyValueTarget { value, policy }
}

pub fn opening<B>(device: &Device<B>) -> Vec<(Tensor<B, 3>, PolicyValueTarget<B>, Tensor<B, 2>)>
where B: AutodiffBackend {
    // first move
    let mut board = Board::default_new();
    let mut opening_moves = vec![(0.5, 0, Direction::Right), (0.5, 1, Direction::Left)];
    let board_eval = 0.5;
    let mut input = board_to_input(&board, device);
    let mut target = moves_and_value_to_target(&board, board_eval, &opening_moves, device);
    let mut illegal_m = illegal_mask(&board, device);
    let mut to_feed = vec![(input, target, illegal_m)];

    // second move
    let mut board_symmetric = board.clone();
    board.move_pawn_until_blocked(0, &Direction::Right);
    opening_moves = vec![(1.0, 4, Direction::Down)];
    input = board_to_input(&board, device);
    target = moves_and_value_to_target(&board, board_eval, &opening_moves, device);
    illegal_m = illegal_mask(&board, device);
    to_feed.push((input, target, illegal_m));

    board_symmetric.move_pawn_until_blocked(1, &Direction::Left);
    input = board_to_input(&board_symmetric, device);
    target = moves_and_value_to_target(&board_symmetric, board_eval, &opening_moves, device);
    illegal_m = illegal_mask(&board_symmetric, device);
    to_feed.push((input, target, illegal_m));

    to_feed
}
