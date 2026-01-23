use core::f32;

use crate::{
    ai::ann::utils::board_to_input,
    logic::{Board, Direction},
};
use strum::IntoEnumIterator;

use burn::tensor::{backend::{Backend, AutodiffBackend}, Device, Tensor, s};

#[derive(Clone)]
pub struct PolicyValueTarget<B: AutodiffBackend> {
    pub value: Tensor<B, 2>,
    pub policy: Tensor<B, 4>,
}

impl<B: AutodiffBackend> PolicyValueTarget<B> {
    fn rotate_clockwise_once(&self) -> Self {
        Self {
            value: self.value.clone(),
            policy: rotate_clockwise_once(self.policy.clone()),
        }
    }

    fn rotate_clockwise(&self, quarter_turns: i32) -> Self {
        Self {
            value: self.value.clone(),
            policy: rotate_clockwise(self.policy.clone(), quarter_turns),
        }
    }

    fn flip(&self, horizontal: bool, vertical: bool) -> Self {
        Self {
            value: self.value.clone(),
            policy: flip(self.policy.clone(), horizontal, vertical),
        }
    }

    fn flip_diagonal(&self, upleft_downright_diag: bool, upright_downleft_diag: bool) -> Self {
        Self {
            value: self.value.clone(),
            policy: flip_diagonal(self.policy.clone(), upleft_downright_diag, upright_downleft_diag),
        }
    }

}

pub fn illegal_mask<B>(board: &Board, device: &Device<B>) -> Tensor<B, 4>
where B:Backend {
    let possible_moves = board.get_all_valid_directions_and_resulting_boards();
    let mut illegal_mask_array = [[[[-1e6; 1]; 8]; 5]; 5];
    for (pawn_index, direction, _) in possible_moves.into_iter() {
        let pawn_position = &board.pawns[pawn_index].position;
        illegal_mask_array[0][direction as usize][pawn_position.row as usize][pawn_position.column as usize] = 0.0;
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
    let mut to_feed = add_symmetries(input, target, illegal_m);

    // second move
    board.move_pawn_until_blocked(0, &Direction::Right);
    opening_moves = vec![(1.0, 4, Direction::Down)];
    input = board_to_input(&board, device);
    target = moves_and_value_to_target(&board, board_eval, &opening_moves, device);
    illegal_m = illegal_mask(&board, device);
    to_feed.append(&mut add_symmetries(input, target, illegal_m));

    to_feed
}

pub fn add_symmetries<B>(input:Tensor<B, 4>, target: PolicyValueTarget<B>, illegal_mask: Tensor<B, 4>) -> Vec<(Tensor<B, 4>, PolicyValueTarget<B>, Tensor<B, 4>)>
where B: AutodiffBackend {
    let mut with_symmetries = vec![(input.clone(), target.clone(), illegal_mask.clone())];
    with_symmetries.push((rotate_clockwise_once(input.clone()), target.rotate_clockwise_once(), rotate_clockwise_once(illegal_mask.clone())));
    with_symmetries.push((rotate_clockwise(input.clone(), 2), target.rotate_clockwise(2), rotate_clockwise(illegal_mask.clone(), 2)));
    with_symmetries.push((rotate_clockwise(input.clone(), 3), target.rotate_clockwise(3), rotate_clockwise(illegal_mask.clone(), 3)));
    with_symmetries.push((flip(input.clone(), true, false), target.flip(true, false), flip(illegal_mask.clone(), true, false)));
    with_symmetries.push((flip(input.clone(), false, true), target.flip(false, true), flip(illegal_mask.clone(), false, true)));
    with_symmetries.push((flip_diagonal(input.clone(), true, false), target.flip_diagonal(true, false), flip_diagonal(illegal_mask.clone(), true, false)));
    with_symmetries.push((flip_diagonal(input.clone(), false, true), target.flip_diagonal(false, true), flip_diagonal(illegal_mask.clone(), false, true)));
    with_symmetries
}

fn swap_direction_indices<B>(input:Tensor<B, 4>, directions_out: Vec<i32>) -> Tensor<B, 4>
where B: AutodiffBackend {
    let mut output = Tensor::zeros(input.shape(), &input.device());
    for (new_idx, &old_idx) in directions_out.iter().enumerate() {
        let input_slice = input.clone().slice(s![.., old_idx, .., ..]);
        output = output.slice_assign(s![.., new_idx, .., ..], input_slice);
    }
    output
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
    swap_direction_indices(rotated, directions_out)
}

fn flip<B>(input: Tensor<B, 4>, horizontal: bool, vertical: bool) -> Tensor<B, 4>
where B: AutodiffBackend {
    let mut flipped = input.clone();
    if horizontal {
        flipped = flipped.flip([3]);
    }
    if vertical {
        flipped = flipped.flip([2]);
    }
    let directions_out: Vec<i32> = Direction::iter().map(|d| d.flip(horizontal, vertical).clone() as i32).collect();
    swap_direction_indices(flipped, directions_out)
}

fn flip_diagonal<B>(input: Tensor<B, 4>, upleft_downright_diag: bool, upright_downleft_diag: bool) -> Tensor<B, 4>
where B: AutodiffBackend {
    let mut flipped = input.clone();
    if upleft_downright_diag {
        flipped = flip(rotate_clockwise_once(flipped), false, true);
    }
    if upright_downleft_diag {
        flipped = flip(rotate_clockwise_once(flipped), true, false);
    }
    let directions_out: Vec<i32> = Direction::iter().map(|d| d.flip_diagonal(upleft_downright_diag, upright_downleft_diag).clone() as i32).collect();
    swap_direction_indices(flipped, directions_out)
}

