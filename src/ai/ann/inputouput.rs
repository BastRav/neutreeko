use crate::logic::{Board, Direction};

use burn::tensor::{backend::Backend, Device, Tensor};

pub fn board_to_input<B>(board: &Board, device:&Device<B>) -> Tensor<B, 3> 
where B: Backend {
    let mut input = [[[0; 5]; 5];2];
    for pawn in board.pawns.iter() {
        let mut layer = 0;
        if Some(pawn.color.clone()) == board.next_player {
            layer = 1;
        }
        input[layer][pawn.position.row as usize][pawn.position.column as usize] = 1;
    }
    Tensor::from_data(input, device)
}

pub fn output_to_moves<B>(board: &Board, tensor: Tensor<B, 1>) -> Vec<(f32, usize, Direction, Board)>
where B: Backend {
    let possible_moves = board.get_all_valid_directions_and_resulting_boards();
    let tensor_data = tensor.to_data().into_vec().unwrap();
    let mut possible_moves_proba = vec![];
    for possible_move in possible_moves.iter() {
        let proba:f32 = tensor_data[possible_move.0 * 8 + possible_move.1.clone() as usize];
        possible_moves_proba.push((proba, possible_move.0, possible_move.1.clone(), possible_move.2.clone()));
    }
    let total: f32 = possible_moves_proba.iter().map(|x| x.0).sum();
    possible_moves_proba.iter_mut().for_each(|x| x.0 /= total);

    possible_moves_proba
}
