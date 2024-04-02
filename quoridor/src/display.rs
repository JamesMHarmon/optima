use std::fmt::{self, Display, Formatter};

use crate::{
    board::{map_board_to_arr_rotatable, BoardType},
    GameState, ASCII_LETTER_A, BOARD_HEIGHT, BOARD_WIDTH, NUM_WALLS_PER_PLAYER,
};

impl Display for GameState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let p1_board = map_board_to_arr_rotatable(self.p1_pawn_board, BoardType::Pawn, false);
        let p2_board = map_board_to_arr_rotatable(self.p2_pawn_board, BoardType::Pawn, false);
        let horizontal_wall_placement =
            map_board_to_arr_rotatable(self.horizontal_wall_board, BoardType::Pawn, false);
        let vertical_wall_placement =
            map_board_to_arr_rotatable(self.vertical_wall_board, BoardType::Pawn, false);
        let horizontal_wall_board = map_board_to_arr_rotatable(
            self.horizontal_wall_board,
            BoardType::HorizontalWall,
            false,
        );
        let vertical_wall_board =
            map_board_to_arr_rotatable(self.vertical_wall_board, BoardType::VerticalWall, false);

        writeln!(f)?;

        for y in 0..BOARD_HEIGHT {
            for x in 0..BOARD_WIDTH {
                if x == 0 {
                    write!(f, "  +")?;
                }
                let idx = y * BOARD_WIDTH + x;
                let w = if horizontal_wall_board[idx] != 0.0 {
                    "■■■"
                } else {
                    "---"
                };
                let c = if horizontal_wall_placement[idx] != 0.0 {
                    "■"
                } else if vertical_wall_placement[idx] != 0.0 {
                    "█"
                } else {
                    "+"
                };
                write!(f, "{}{}", w, c)?;
            }

            writeln!(f)?;

            for x in 0..BOARD_WIDTH {
                let idx = y * BOARD_WIDTH + x;
                if x == 0 {
                    write!(f, "{} |", BOARD_HEIGHT - y)?;
                }
                let p = if p1_board[idx] != 0.0 {
                    "1"
                } else if p2_board[idx] != 0.0 {
                    "2"
                } else {
                    " "
                };
                let w = if vertical_wall_board[idx] != 0.0 {
                    "█"
                } else {
                    "|"
                };
                write!(f, " {} {}", p, w)?;
            }

            writeln!(f)?;
        }

        for x in 0..BOARD_WIDTH {
            if x == 0 {
                write!(f, "  +")?;
            }
            write!(f, "---+")?;
        }

        writeln!(f)?;

        for x in 0..BOARD_WIDTH {
            if x == 0 {
                write!(f, "   ")?;
            }
            let col_letter = (ASCII_LETTER_A + x as u8) as char;
            write!(f, " {}  ", col_letter)?;
        }

        writeln!(f)?;
        writeln!(f)?;
        writeln!(
            f,
            "  P1: {}  P2: {}",
            NUM_WALLS_PER_PLAYER - self.p1_num_walls_placed,
            NUM_WALLS_PER_PLAYER - self.p2_num_walls_placed
        )?;

        Ok(())
    }
}
