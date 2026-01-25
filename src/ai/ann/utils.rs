use core::f32;

use crate::logic::{Board, Direction};

use burn::tensor::{backend::Backend, Device, Tensor};

pub fn board_to_input<B>(board: &Board, device: &Device<B>) -> Tensor<B, 4>
where B: Backend {
    // 2 channels: current player pawns, opponent pawns
    let mut input = [[[[0.0; 5]; 5]; 2]; 1];

    for pawn in board.pawns.iter() {
        let channel = if Some(pawn.color.clone()) == board.next_player { 0 } else { 1 };
        input[0][channel][pawn.position.row as usize][pawn.position.column as usize] = 1.0;
    }

    Tensor::from_data(input, device)
}

pub fn output_to_moves<B>(board: &Board, tensor: Tensor<B, 4>) -> Vec<(f32, usize, Direction, Board)>
where B: Backend {
    let possible_moves = board.get_all_valid_directions_and_resulting_boards();
    let tensor_data: Vec<f32> = tensor.to_data().into_vec().unwrap();

    let mut possible_moves_proba = vec![];
    for (pawn_index, direction, board) in possible_moves.into_iter() {
        let pawn_position = &board.pawns[pawn_index].position;
        let index = (direction.clone() as usize) * 25 + (pawn_position.row as usize) * 5 + pawn_position.column as usize;
        let proba: f32 = tensor_data[index];
        possible_moves_proba.push((proba, pawn_index, direction, board));
    }
    // Softmax normalisation
    let max_proba = possible_moves_proba.iter().map(|x| x.0).fold(f32::MIN, f32::max);
    possible_moves_proba.iter_mut().for_each(|x| x.0 = (x.0 - max_proba).exp());
    let total: f32 = possible_moves_proba.iter().map(|x| x.0).sum();
    possible_moves_proba.iter_mut().for_each(|x| x.0 /= total);
    // sort in ascending proba order
    possible_moves_proba.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap());

    possible_moves_proba
}
