use crate::logic::{Board, Direction};

use burn::tensor::{backend::Backend, Device, Tensor};

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

pub fn output_to_moves<B>(board: &Board, tensor: Tensor<B, 1>) -> Vec<(f32, usize, Direction, Board)>
where B: Backend {
    let possible_moves = board.get_all_valid_directions_and_resulting_boards();
    let tensor_data = tensor.to_data().into_vec().unwrap();

    let mut possible_moves_proba = vec![];
    for possible_move in possible_moves.iter() {
        // Calculate the index in the output tensor (from_position × 8 + direction)
        let from_index = possible_move.0;
        let direction_index = possible_move.1.clone() as usize;
        let output_index = from_index * 8 + direction_index;
        let proba: f32 = tensor_data[output_index];
        possible_moves_proba.push((proba, possible_move.0, possible_move.1.clone(), possible_move.2.clone()));
    }

    // Normalize probabilities
    let total: f32 = possible_moves_proba.iter().map(|x| x.0).sum();
    possible_moves_proba.iter_mut().for_each(|x| x.0 /= total);

    // Sort by probability (descending)
    possible_moves_proba.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    possible_moves_proba
}
