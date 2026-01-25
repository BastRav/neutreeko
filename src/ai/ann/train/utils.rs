use core::f32;

use crate::{
    ai::ann::utils::board_to_input,
    logic::{Board, Direction},
};
use strum::IntoEnumIterator;

use burn::tensor::{backend::{Backend, AutodiffBackend}, Device, Tensor, s};

#[derive(Clone, Debug)]
pub struct PolicyValueTarget<B: AutodiffBackend> {
    pub value: Tensor<B, 2>,
    pub policy: Tensor<B, 4>,
}

impl<B: AutodiffBackend> PolicyValueTarget<B> {
    fn rotate_clockwise(&self, quarter_turns: i32) -> Self {
        Self {
            value: self.value.clone(),
            policy: rotate_clockwise(self.policy.clone(), quarter_turns, true),
        }
    }

    fn flip(&self, horizontal: bool, vertical: bool) -> Self {
        Self {
            value: self.value.clone(),
            policy: flip(self.policy.clone(), horizontal, vertical, true),
        }
    }

    fn flip_diagonal(&self, upleft_downright_diag: bool, upright_downleft_diag: bool) -> Self {
        Self {
            value: self.value.clone(),
            policy: flip_diagonal(self.policy.clone(), upleft_downright_diag, upright_downleft_diag, true),
        }
    }
}

pub fn illegal_mask<B>(board: &Board, device: &Device<B>) -> Tensor<B, 4>
where B:Backend {
    let possible_moves = board.get_all_valid_directions_and_resulting_boards();
    let mut illegal_mask_array = [[[[-1e9; 5]; 5]; 8]; 1];
    for (pawn_index, direction, _) in possible_moves.into_iter() {
        let pawn_position = &board.pawns[pawn_index].position;
        illegal_mask_array[0][direction as usize][pawn_position.row as usize][pawn_position.column as usize] = 0.0;
    }
    Tensor::from_data(illegal_mask_array, device)
}

pub fn moves_and_value_to_target<B>(board: &Board, board_eval: f32, moves_eval: &Vec<(f32, usize, Direction)>, device: &Device<B>) -> PolicyValueTarget<B>
where B: AutodiffBackend {
    let value = Tensor::from_floats([[board_eval]], device);
    let mut policy_floats = [[[[0.0; 5]; 5]; 8]; 1];
    for (proba, pawn_index, direction) in moves_eval.iter() {
        let pawn_position = &board.pawns[*pawn_index].position;
        policy_floats[0][direction.clone() as usize][pawn_position.row as usize][pawn_position.column as usize] = *proba;
    }
    let policy = Tensor::from_floats( policy_floats, device);
    PolicyValueTarget { value, policy }
}

pub fn opening<B>(device: &Device<B>) -> Vec<(Tensor<B, 4>, PolicyValueTarget<B>, Tensor<B, 4>)>
where B: AutodiffBackend {
    // initial board
    let mut board = Board::default_new();
    let mut opening_moves = vec![(0.5, 0, Direction::Right), (0.5, 1, Direction::Left)];
    let board_eval = 0.0;
    let mut to_feed = move_to_learning_input(&board, &opening_moves, board_eval, device);
    // println!("Initial board");
    // println!("{}", board.str_rep());

    // 1st move b1-c1
    board.move_pawn_until_blocked(0, &Direction::Right);
    opening_moves = vec![(1.0, 3, Direction::Down)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 1");
    // println!("{}", board.str_rep());

    // 2nd move c2-c3
    board.move_pawn_until_blocked(3, &Direction::Down);
    opening_moves = vec![(1.0, 0, Direction::DownLeft)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 2");
    // println!("{}", board.str_rep());

    // 3rd move c1-a3
    board.move_pawn_until_blocked(0, &Direction::DownLeft);
    opening_moves = vec![(1.0, 4, Direction::Right)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 3");
    // println!("{}", board.str_rep());

    // 4th move b5-c5
    board.move_pawn_until_blocked(4, &Direction::Right);
    opening_moves = vec![(1.0, 1, Direction::Down)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 4");
    // println!("{}", board.str_rep());

    // 5th move d1-d4
    board.move_pawn_until_blocked(1, &Direction::Down);
    opening_moves = vec![(1.0, 4, Direction::UpLeft)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 5");
    // println!("{}", board.str_rep());

    // 6th move c5-b4
    board.move_pawn_until_blocked(4, &Direction::UpLeft);
    opening_moves = vec![(1.0, 1, Direction::DownLeft)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 6");
    // println!("{}", board.str_rep());

    // 7th move d4-c5
    board.move_pawn_until_blocked(1, &Direction::DownLeft);
    opening_moves = vec![(1.0, 5, Direction::Up)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 7");
    // println!("{}", board.str_rep());

    // 8th move d5-d1
    board.move_pawn_until_blocked(5, &Direction::Up);
    opening_moves = vec![(1.0, 0, Direction::Down)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 8");
    // println!("{}", board.str_rep());

    // 9th move a3-a5
    board.move_pawn_until_blocked(0, &Direction::Down);
    opening_moves = vec![(1.0, 4, Direction::Down)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 9");
    // println!("{}", board.str_rep());

    // 10th move b4-b5
    board.move_pawn_until_blocked(4, &Direction::Down);
    opening_moves = vec![(1.0, 1, Direction::UpLeft)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 10");
    // println!("{}", board.str_rep());

    // 11th move c5-a3
    board.move_pawn_until_blocked(1, &Direction::UpLeft);
    opening_moves = vec![(1.0, 3, Direction::DownLeft)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 11");
    // println!("{}", board.str_rep());

    // 12th move c3-b4
    board.move_pawn_until_blocked(3, &Direction::DownLeft);
    opening_moves = vec![(1.0, 2, Direction::UpLeft)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 12");
    // println!("{}", board.str_rep());

    // 13th move c4-a2
    board.move_pawn_until_blocked(2, &Direction::UpLeft);
    opening_moves = vec![(1.0, 5, Direction::DownLeft)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 13");
    // println!("{}", board.str_rep());

    // 14th move d1-a4
    board.move_pawn_until_blocked(5, &Direction::DownLeft);
    opening_moves = vec![(1.0, 2, Direction::DownRight)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 14");
    // println!("{}", board.str_rep());

    // 15th move a2-d5
    board.move_pawn_until_blocked(2, &Direction::DownRight);
    opening_moves = vec![(0.2, 5, Direction::UpRight), (0.2, 3, Direction::Up), (0.2, 3, Direction::Right), (0.2, 3, Direction::DownRight), (0.2, 4, Direction::Right)];
    to_feed.append(&mut move_to_learning_input(&board, &opening_moves, board_eval, device));
    // println!("Move 15");
    // println!("{}", board.str_rep());

    to_feed
}

fn move_to_learning_input<B>(board: &Board, opening_moves: &Vec<(f32, usize, Direction)>, board_eval: f32, device:&Device<B>) -> Vec<(Tensor<B, 4>, PolicyValueTarget<B>, Tensor<B, 4>)>
where B: AutodiffBackend {
    let input = board_to_input(&board, device);
    let target = moves_and_value_to_target(&board, board_eval, &opening_moves, device);
    let illegal_m = illegal_mask(&board, device);
    add_symmetries(input, target, illegal_m)
}

pub fn add_symmetries<B>(input:Tensor<B, 4>, target: PolicyValueTarget<B>, illegal_mask: Tensor<B, 4>) -> Vec<(Tensor<B, 4>, PolicyValueTarget<B>, Tensor<B, 4>)>
where B: AutodiffBackend {
    let mut with_symmetries = vec![(input.clone(), target.clone(), illegal_mask.clone())];
    with_symmetries.push((rotate_clockwise(input.clone(), 1, false), target.rotate_clockwise(1), rotate_clockwise(illegal_mask.clone(), 1, true)));
    with_symmetries.push((rotate_clockwise(input.clone(), 2, false), target.rotate_clockwise(2), rotate_clockwise(illegal_mask.clone(), 2, true)));
    with_symmetries.push((rotate_clockwise(input.clone(), 3, false), target.rotate_clockwise(3), rotate_clockwise(illegal_mask.clone(), 3, true)));
    with_symmetries.push((flip(input.clone(), true, false, false), target.flip(true, false), flip(illegal_mask.clone(), true, false, true)));
    with_symmetries.push((flip(input.clone(), false, true, false), target.flip(false, true), flip(illegal_mask.clone(), false, true, true)));
    with_symmetries.push((flip_diagonal(input.clone(), true, false, false), target.flip_diagonal(true, false), flip_diagonal(illegal_mask.clone(), true, false, true)));
    with_symmetries.push((flip_diagonal(input.clone(), false, true, false), target.flip_diagonal(false, true), flip_diagonal(illegal_mask.clone(), false, true, true)));
    with_symmetries
}

fn swap_direction_indices<B>(input:Tensor<B, 4>, directions_out: Vec<i32>) -> Tensor<B, 4>
where B: AutodiffBackend {
    let mut output = Tensor::zeros(input.shape(), &input.device());
    for (old_idx, &new_idx) in directions_out.iter().enumerate() {
        let input_slice = input.clone().slice(s![.., old_idx, .., ..]);
        output = output.slice_assign(s![.., new_idx, .., ..], input_slice);
    }
    output
}

fn rotate_clockwise<B>(input: Tensor<B, 4>, quarter_turns: i32, swap_directions: bool) -> Tensor<B, 4>
where B: AutodiffBackend {
    let mut output = input.clone(); 
    for _ in 0..quarter_turns.rem_euclid(4) {
        output = rotate_clockwise_once(output, swap_directions);
    }
    output
}

fn rotate_clockwise_once<B>(input: Tensor<B, 4>, swap_directions: bool) -> Tensor<B, 4>
where B: AutodiffBackend {
    let rotated = input.transpose().flip([3]);
    let directions_out: Vec<i32> = Direction::iter().map(|d| d.rotate_clockwise(1).clone() as i32).collect();
    if swap_directions {
        swap_direction_indices(rotated, directions_out)
    }
    else {
        rotated
    }
}

fn flip<B>(input: Tensor<B, 4>, horizontal: bool, vertical: bool, swap_directions: bool) -> Tensor<B, 4>
where B: AutodiffBackend {
    let mut flipped = input.clone();
    if horizontal {
        flipped = flipped.flip([2]);
    }
    if vertical {
        flipped = flipped.flip([3]);
    }
    let directions_out: Vec<i32> = Direction::iter().map(|d| d.flip(horizontal, vertical).clone() as i32).collect();
    if swap_directions {
        swap_direction_indices(flipped, directions_out)
    }
    else {
        flipped
    }
}

fn flip_diagonal<B>(input: Tensor<B, 4>, upleft_downright_diag: bool, upright_downleft_diag: bool, swap_directions: bool) -> Tensor<B, 4>
where B: AutodiffBackend {
    let mut flipped = input.clone();
    if upleft_downright_diag {
        flipped = flipped.transpose();
    }
    if upright_downleft_diag {
        flipped = flipped.transpose().flip([2, 3]);
    }
    let directions_out: Vec<i32> = Direction::iter().map(|d| d.flip_diagonal(upleft_downright_diag, upright_downleft_diag).clone() as i32).collect();
    if swap_directions {
        flipped = swap_direction_indices(flipped, directions_out);
    }
    flipped
}
