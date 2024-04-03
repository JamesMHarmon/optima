use std::{
    collections::HashSet,
    fmt::{self, Display, Formatter},
};

use crate::{Coordinate, GameState, ASCII_LETTER_A, BOARD_HEIGHT, BOARD_WIDTH};

impl Display for GameState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let player_1_info = self.player_info(1);
        let player_2_info = self.player_info(2);

        let player_1_pawn = player_1_info.pawn();
        let player_2_pawn = player_2_info.pawn();

        let vertical_walls = self.vertical_walls().collect::<HashSet<_>>();
        let vertical_walls_splayed = vertical_walls
            .iter()
            .flat_map(|c| [*c, Coordinate::new(c.col(), c.row() - 1)])
            .collect::<HashSet<_>>();

        let horizontal_walls = self.horizontal_walls().collect::<HashSet<_>>();
        let horizontal_walls_splayed = horizontal_walls
            .iter()
            .flat_map(|c| [*c, Coordinate::new((c.col() as u8 + 1) as char, c.row())])
            .collect::<HashSet<_>>();

        writeln!(f)?;

        for x in 0..BOARD_WIDTH {
            if x == 0 {
                write!(f, "  +")?;
            }
            write!(f, "---+")?;
        }

        writeln!(f)?;

        for y in 0..BOARD_HEIGHT {
            for x in 0..BOARD_WIDTH {
                let coord = Coordinate::from_index(y * BOARD_WIDTH + x);
                if x == 0 {
                    write!(f, "{} |", BOARD_HEIGHT - y)?;
                }
                let p = if player_1_pawn == coord {
                    "1"
                } else if player_2_pawn == coord {
                    "2"
                } else {
                    " "
                };
                let w = if vertical_walls_splayed.contains(&coord) {
                    "█"
                } else {
                    "|"
                };
                write!(f, " {} {}", p, w)?;
            }

            writeln!(f)?;

            for x in 0..BOARD_WIDTH {
                if x == 0 {
                    write!(f, "  +")?;
                }
                let coord = Coordinate::from_index(y * BOARD_WIDTH + x);
                let w = if horizontal_walls_splayed.contains(&coord) {
                    "■■■"
                } else {
                    "---"
                };
                let c = if horizontal_walls.contains(&coord) {
                    "■"
                } else if vertical_walls.contains(&coord) {
                    "█"
                } else {
                    "+"
                };
                write!(f, "{}{}", w, c)?;
            }

            writeln!(f)?;
        }

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
            self.player_info(1).num_walls(),
            self.player_info(2).num_walls()
        )?;

        Ok(())
    }
}
