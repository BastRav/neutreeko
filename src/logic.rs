use std::collections::HashSet;
use strum_macros::EnumIter;
use strum::IntoEnumIterator;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum Color {
    Yellow,
    Green,
}

impl Color {
    pub fn other_color(&self) -> Color {
        match self {
            Color::Green => Color::Yellow,
            Color::Yellow => Color::Green,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct Pawn {
    pub color: Color,
    pub position: Position,
}

impl Pawn {
    pub fn new(color: Color, position: Position) -> Self {
        Self { color, position }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Board {
    pub number_of_rows: u8,
    pub number_of_columns: u8,
    pub pawns: Vec<Pawn>,
    pub next_player: Option<Color>,
}

#[derive(EnumIter, Clone, Debug, PartialEq, Eq, Hash)]
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

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct Position {
    pub row: u8,
    pub column: u8,
}

pub fn aligned_positions(positions: &mut Vec<Position>) -> bool {
    if positions.len() != 3 {
        panic!("aligned_positions function requires exactly 3 positions");
    }

    positions.sort_by(|a, b| a.row.cmp(&b.row));

    let first_position = &positions[0];
    let second_position = &positions[1];
    let third_position = &positions[2];
    let columns = vec![first_position.column as i8, second_position.column as i8, third_position.column as i8];
    let rows = vec![first_position.row as i8, second_position.row as i8, third_position.row as i8];
    let sorted_columns = {
        let mut cols = columns.clone();
        cols.sort();
        cols
    };
    let same_row = rows[0] == rows[1] && rows[1] == rows[2];
    let same_column = columns[0] == columns[1] && columns[1] == columns[2];
    let adjacent_rows = (rows[2] - rows[1] == 1) && (rows[1] - rows[0] == 1);
    let adjacent_columns = (sorted_columns[2] - sorted_columns[1] == 1) && (sorted_columns[1] - sorted_columns[0] == 1);
    let diagonal = adjacent_rows && ((columns[2] - columns[1] == 1 && columns[1] - columns[0] == 1) || (columns[2] - columns[1] == -1 && columns[1] - columns[0] == -1));
    (same_row && adjacent_columns) || (same_column && adjacent_rows) || diagonal
}

impl Board {
    pub fn new(number_of_rows: u8, number_of_columns: u8, pawns: Vec<Pawn>, next_player: Option<Color>) -> Self {
        let board = Self { number_of_rows, number_of_columns, pawns, next_player};
        if board.is_valid() {
            board
        } else {
            panic!("Invalid board, pawns are on the same position or out of bounds")
        }
    }

    pub fn default_new() -> Self {
        let mut pawns = Vec::new();
        pawns.push(Pawn::new(Color::Green, Position { row: 0, column: 1 }));
        pawns.push(Pawn::new(Color::Green, Position { row: 0, column: 3 }));
        pawns.push(Pawn::new(Color::Green, Position { row: 3, column: 2 }));
        pawns.push(Pawn::new(Color::Yellow, Position { row: 1, column: 2 }));
        pawns.push(Pawn::new(Color::Yellow, Position { row: 4, column: 1 }));
        pawns.push(Pawn::new(Color::Yellow, Position { row: 4, column: 3 }));
        Self::new(5, 5, pawns, Some(Color::Green))
    }

    pub fn str_rep(&self) -> String {
        let mut result = String::new();
        let mut grid = vec![vec![". ".to_string(); self.number_of_columns as usize]; self.number_of_rows as usize];

        for (index, pawn) in self.pawns.iter().enumerate() {
            let symbol = match pawn.color {
                Color::Green => format!("G{}", index),
                Color::Yellow => format!("Y{}", index),
            };
            grid[pawn.position.row as usize][pawn.position.column as usize] = symbol;
        }

        for row in &grid {
            for cell in row {
                result.push_str(&format!("{} ", cell));
            }
            result.push('\n');
        }

        if let Some(color) = &self.next_player {
            result.push_str(&format!("Next player: {:?}\n", color));
        } else {
            result.push_str("Game over\n");
        }
        result
    }

    fn is_valid(&self) -> bool {
        let mut occupied_positions_values: HashSet<Position> = HashSet::new();
        for pawn in self.pawns.iter() {
            if pawn.position.row >= self.number_of_rows || pawn.position.column >= self.number_of_columns {
                return false;
            }
            if !occupied_positions_values.insert(pawn.position.clone()) {
                return false;
            }
        }
        true
    }

    pub fn winner(&self) -> Option<Color> {
        if !self.is_valid() {
            return None;
        }

        let mut yellow_positions = Vec::new();
        let mut green_positions = Vec::new();

        for pawn in self.pawns.iter() {
            match pawn.color {
                Color::Green => green_positions.push(pawn.position.clone()),
                Color::Yellow => yellow_positions.push(pawn.position.clone()),
            }
        }
        if aligned_positions(&mut green_positions) {
            return Some(Color::Green);
        }
        if aligned_positions(&mut yellow_positions) {
            return Some(Color::Yellow);
        }
        None
    }

    pub fn move_pawn(&mut self, pawn_index: usize, row_increment: i32, column_increment: i32) -> bool {
        let init_position = self.pawns[pawn_index].position.clone();
        
        let final_row = i32::from(init_position.row) + row_increment;
        let final_column = i32::from(init_position.column) + column_increment;

        if final_row < 0 || final_row >= i32::from(self.number_of_rows)
            || final_column < 0 || final_column >= i32::from(self.number_of_columns) {
            return false;
        }
        let final_position = Position{
            row: u8::try_from(final_row).unwrap(),
            column: u8::try_from(final_column).unwrap()
        };
        
        self.pawns[pawn_index].position = final_position;
        if self.is_valid() {
            true
        } else {
            self.pawns[pawn_index].position = init_position;
            false
        }
    }

    pub fn move_pawn_until_blocked(&mut self, pawn_index: usize, direction: &Direction) -> bool {
        let mut has_moved = false;
        match &self.next_player {
            None => return false,
            Some(color) => {
                if self.pawns[pawn_index].color != *color {
                    return false;
                }
            }
        };
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
        loop {
            if !self.move_pawn(pawn_index, row_increment, column_increment) {
                break;
            }
            has_moved = true;
        }
        if has_moved {
            if self.winner().is_some() {
                self.next_player = None;
                return has_moved;
            }
            self.next_player = match &self.next_player {
                Some(color) => Some(color.other_color()),
                None => None,
            };
        }
        has_moved
    }

    pub fn get_valid_directions(&self, pawn_index: usize) -> Vec<Direction> {
        let mut valid_directions = vec![];
        let directions = Direction::iter();
        for direction in directions {
            let mut new_board = self.clone();
            if new_board.move_pawn_until_blocked(pawn_index, &direction) {
                valid_directions.push(direction);
            }
        }
        valid_directions
    }

    pub fn get_valid_directions_and_resulting_boards(&self, pawn_index: usize) -> Vec<(Direction, Board)> {
        let mut valid_directions = Vec::with_capacity(8);
        let directions = Direction::iter();
        for direction in directions {
            let mut new_board = self.clone();
            if new_board.move_pawn_until_blocked(pawn_index, &direction) {
                valid_directions.push((direction, new_board));
            }
        }
        valid_directions
    }

    pub fn get_all_valid_directions_and_resulting_boards(&self) -> Vec<(usize, Direction, Board)> {
        let mut valid_directions = Vec::with_capacity(24);
        for (pawn_index, pawn) in self.pawns.iter().enumerate() {
            if Some(pawn.color.clone()) != self.next_player {
                continue;
            }
            let directions = self.get_valid_directions_and_resulting_boards(pawn_index);
            for (direction, new_board) in directions {
                valid_directions.push((pawn_index, direction, new_board))
            }
        }
        valid_directions
    }
}
