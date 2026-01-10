use core::panic;
use std::collections::HashSet;
use strum_macros::EnumIter;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum Color {
    Yellow,
    Green
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Pawn {
    pub color: Color,
    pub position: Position,
}

impl Pawn {
    pub fn new (color: Color, position: Position) -> Pawn {
        Pawn { color, position }
    }
}

#[derive(Clone, PartialEq)]
pub struct Board {
    pub number_of_rows: u8,
    pub number_of_columns: u8,
    pub pawns: Vec<Pawn>
}

#[derive(EnumIter, Clone, Debug)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
    UpLeft,
    UpRight,
    DownLeft,
    DownRight
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Position {
    pub row : u8,
    pub column: u8
}

pub fn aligned_positions (positions: &Vec<Position>) -> bool {
    if positions.len() != 3 {
        panic!("aligned_positions function requires exactly 3 positions");
    }

    let first_position = &positions[0];
    let second_position = &positions[1];
    let third_position = &positions[2];
    let mut columns = vec![first_position.column, second_position.column, third_position.column];
    columns.sort();
    let mut rows = vec![first_position.row, second_position.row, third_position.row];
    rows.sort();
    if rows[0] - rows[1] == 0 && rows[1] - rows[2] == 0 && columns[2] - columns[0] == 2 && columns[1] - columns[0] == 1 {
        return true;
    }
    if columns[0] - columns[1] == 0 && columns[1] - columns[2] == 0 && rows[2] - rows[0] == 2 && rows[1] - rows[0] == 1 {
        return true;
    }
    if rows[1] - rows[0] == 1 && rows[2] - rows[1] == 1 && columns[1] - columns[0] == 1 && columns[2] - columns[1] == 1{
        return true;
    }
    false
}

impl Board {
    pub fn new (number_of_rows: u8, number_of_columns: u8, pawns: Vec<Pawn>) -> Board {
        let board = Board { number_of_rows, number_of_columns, pawns };
        if board.is_valid() {
            board
        }
        else {
            panic!("Invalid board, pawns are on the same position or out of bounds")
        }
    }

    pub fn default_new () -> Board {
        let mut pawns = Vec::new();
        pawns.push(Pawn::new(Color::Green, Position { row: 0, column: 1 }));
        pawns.push(Pawn::new(Color::Green, Position { row: 0, column: 3 }));
        pawns.push(Pawn::new(Color::Green, Position { row: 3, column: 2 }));
        pawns.push(Pawn::new(Color::Yellow, Position { row: 1, column: 2 }));
        pawns.push(Pawn::new(Color::Yellow, Position { row: 4, column: 1 }));
        pawns.push(Pawn::new(Color::Yellow, Position { row: 4, column: 3 }));
        Board::new(5, 5, pawns)
    }

    pub fn get_pawns(&self) -> &Vec<Pawn> {
        &self.pawns
    }

    fn is_valid (&self) -> bool {
        let mut occupied_positions_values: HashSet<Position> = HashSet::new();
        for pawn in self.pawns.iter() {
            if pawn.position.row >= self.number_of_rows || pawn.position.column >= self.number_of_columns {
                return false
            }
            if !occupied_positions_values.insert(pawn.position.clone()) {
                return false
            }
        }
        true
    }

    pub fn winner(&self) -> Option<Color> {
        if !self.is_valid() {
            return None
        }

        let mut yellow_positions = Vec::new();
        let mut green_positions = Vec::new();

        for pawn in self.pawns.iter() {
            match pawn.color {
                Color::Green => {
                    green_positions.push(pawn.position.clone());
                }
                Color::Yellow => {
                    yellow_positions.push(pawn.position.clone());
                }
            }
        }
        if aligned_positions(&green_positions) {
            return Some(Color::Green)
        }
        if aligned_positions(&yellow_positions) {
            return Some(Color::Yellow)
        }
        None
    }

    pub fn move_pawn(&mut self, pawn_index: usize, direction: &Direction) -> Option<bool> {
        let init_position = &self.pawns.get(pawn_index)?.position.clone();
        
        let mut row_increment = 0;
        let mut column_increment = 0;
        match direction {
            Direction::Up => row_increment = -1,
            Direction::Down => row_increment = 1,
            Direction::Left => column_increment = -1,
            Direction::Right => column_increment = 1,
            Direction::UpLeft => {
                row_increment = -1;
                column_increment = -1;
            }
            Direction::UpRight => {
                row_increment = -1;
                column_increment = 1;
            }
            Direction::DownLeft => {
                row_increment = 1;
                column_increment = -1;
            }
            Direction::DownRight => {
                row_increment = 1;
                column_increment = 1;
            }
        }
        
        let final_row = i32::from(init_position.row) + row_increment;
        let final_column = i32::from(init_position.column) + column_increment;

        let mut valid_move = true;
        if final_row < 0 || final_row >= i32::from(self.number_of_rows) {
            valid_move = false;
        }
        if final_column < 0 || final_column >= i32::from(self.number_of_columns) {
            valid_move = false;
        }
        if !valid_move {
            return None;
        }
        let final_position = Position{
            row: u8::try_from(final_row).ok()?,
            column: u8::try_from(final_column).ok()?
        };
        
        self.pawns.get_mut(pawn_index)?.position = final_position;
        if self.is_valid() {
            self.move_pawn(pawn_index, direction);
            Some(true)
        }
        else {
            self.pawns.get_mut(pawn_index)?.position = init_position.clone();
            None 
        }
    }
}
