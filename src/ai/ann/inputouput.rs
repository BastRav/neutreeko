use core::f32;

use crate::{ai::ann::train::PolicyValueTarget, logic::{Board, Direction}};

use burn::tensor::{backend::{Backend, AutodiffBackend}, Device, Tensor};

pub fn board_to_input<B>(board: &Board, device: &Device<B>) -> Tensor<B, 3>
where B: Backend {
    // 2 channels: current player pawns, opponent pawns
    let mut input = [[[0.0; 5]; 5]; 2];

    for pawn in board.pawns.iter() {
        let channel = if Some(pawn.color.clone()) == board.next_player { 0 } else { 1 };
        input[channel][pawn.position.row as usize][pawn.position.column as usize] = 1.0;
    }

    Tensor::from_data(input, device)
}

fn position_direction_to_index(position: (u8, u8), direction: Direction) -> usize {
    let (row, col) = position;
    (row as usize * 5 + col as usize) * 8 + direction as usize
}

pub fn output_to_moves<B>(board: &Board, tensor: Tensor<B, 2>) -> Vec<(f32, usize, Direction, Board)>
where B: Backend {
    let possible_moves = board.get_all_valid_directions_and_resulting_boards();
    let tensor_data = tensor.to_data().into_vec().unwrap();

    let mut possible_moves_proba = vec![];
    let mut min_proba = f32::MIN;
    for possible_move in possible_moves.iter() {
        let pawn_position = board.pawns[possible_move.0].position.clone();
        let output_index = position_direction_to_index((pawn_position.row, pawn_position.column),possible_move.1.clone());
        let proba: f32 = tensor_data[output_index];
        if proba < min_proba {
            min_proba = proba;
        }
        possible_moves_proba.push((proba, possible_move.0, possible_move.1.clone(), possible_move.2.clone()));
    }

    // Normalize probabilities
    if min_proba < 0.0 { 
        possible_moves_proba.iter_mut().for_each(|x| x.0 += min_proba);
    }
    let total: f32 = possible_moves_proba.iter().map(|x| x.0).sum();
    possible_moves_proba.iter_mut().for_each(|x| x.0 /= total);

    // Sort by probability (descending)
    possible_moves_proba.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    possible_moves_proba
}

pub fn illegal_mask<B>(board: &Board, device: &Device<B>) -> Tensor<B, 2>
where B:Backend {
    let possible_moves = board.get_all_valid_directions_and_resulting_boards();
    let mut illegal_mask_array = [-1e9; 200];
    for possible_move in possible_moves.iter() {
        let pawn_position = board.pawns[possible_move.0].position.clone();
        let mask_index = position_direction_to_index((pawn_position.row, pawn_position.column),possible_move.1.clone());
        illegal_mask_array[mask_index] = 0.0;
    }
    Tensor::from_data(illegal_mask_array, device)
}

pub fn moves_and_value_to_target<B>(element:&(Board, Vec<(f32, usize, Direction)>), board_eval: f32, device: &Device<B>) -> PolicyValueTarget<B>
where B: AutodiffBackend {
    let value = Tensor::from_floats([board_eval], device);
    let mut policy_floats = [0.0; 200];
    let board = &element.0;
    for possible_move in element.1.iter() {
        let position = board.pawns[possible_move.1].position.clone();
        let index_to_consider = position_direction_to_index((position.row, position.column), possible_move.2.clone());
        policy_floats[index_to_consider] = possible_move.0
    }
    let policy = Tensor::from_floats( policy_floats, device);
    PolicyValueTarget { value, policy }
}
