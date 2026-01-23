use core::f32;

use crate::{
    ai::ann::{
        train::PolicyValueTarget,
        utils::board_to_input,
    },
    logic::{Board, Direction},
};
use strum::IntoEnumIterator;

use burn::tensor::{backend::{Backend, AutodiffBackend}, Device, Tensor, s};

pub fn illegal_mask<B>(board: &Board, device: &Device<B>) -> Tensor<B, 4>
where B:Backend {
    let possible_moves = board.get_all_valid_directions_and_resulting_boards();
    let mut illegal_mask_array = [[[[-1e6; 1]; 8]; 5]; 5];
    for (pawn_index, direction, _) in possible_moves.iter() {
        let pawn_position = &board.pawns[*pawn_index].position;
        illegal_mask_array[0][direction.clone() as usize][pawn_position.row as usize][pawn_position.column as usize] = 0.0;
    }
    Tensor::from_data([illegal_mask_array], device)
}

pub fn moves_and_value_to_target<B>(board: &Board, board_eval: f32, moves_eval: &Vec<(f32, usize, Direction)>, device: &Device<B>) -> PolicyValueTarget<B>
where B: AutodiffBackend {
    let value = Tensor::from_floats([[board_eval]], device);
    let mut policy_floats = [[[[0.0; 1]; 8]; 5]; 5];
    for (proba, pawn_index, direction) in moves_eval.iter() {
        let pawn_position = &board.pawns[*pawn_index].position;
        policy_floats[0][direction.clone() as usize][pawn_position.row as usize][pawn_position.column as usize] = *proba;
    }
    let policy = Tensor::from_floats( policy_floats, device);
    PolicyValueTarget { value, policy }
}

pub fn opening<B>(device: &Device<B>) -> Vec<(Tensor<B, 4>, PolicyValueTarget<B>, Tensor<B, 4>)>
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

pub fn add_symmetries<B>(input:Tensor<B, 4>, target: PolicyValueTarget<B>, illegal_mask: Tensor<B, 4>) -> Vec<(Tensor<B, 4>, PolicyValueTarget<B>, Tensor<B, 4>)>
where B: AutodiffBackend {
    let mut with_symmetries = vec![(input, target, illegal_mask)];
    // WIP
    with_symmetries
}

fn rotate_clockwise<B>(input: Tensor<B, 4>, quarter_turns: i32) -> Tensor<B, 4>
where B: AutodiffBackend {
    let mut output = input.clone(); 
    for _ in 0..quarter_turns.rem_euclid(4) {
        output = rotate_clockwise_once(output);
    }
    output
}

fn rotate_clockwise_once<B>(input: Tensor<B, 4>) -> Tensor<B, 4>
where B: AutodiffBackend {
    let rotated = input.flip([2, 3]).transpose();
    let directions_out: Vec<i32> = Direction::iter().map(|d| d.rotate_clockwise(1).clone() as i32).collect();
    let mut output = Tensor::zeros(rotated.shape(), &rotated.device());
    for (new_idx, &old_idx) in directions_out.iter().enumerate() {
        let src_slice = rotated.clone().slice(s![.., old_idx, .., ..]);
        output = output.slice_assign(s![.., new_idx, .., ..], src_slice);
    }
    output
}
