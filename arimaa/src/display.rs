use std::str::FromStr;
use std::fmt::{self,Display,Formatter};
use anyhow::Result;

use common::linked_list::List;
use super::engine::{GameState,PlayPhase,Phase,PieceBoard,PieceBoardState};
use super::action::{Piece,Square};
use super::constants::{BOARD_WIDTH,BOARD_HEIGHT};
use super::zobrist::Zobrist;

impl Display for GameState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let move_number = self.get_move_number();
        let curr_player = if self.is_p1_turn_to_move() { "g" } else { "s" };
        let piece_board = &self.get_piece_board();
        writeln!(f, "{}{}", move_number, curr_player)?;

        writeln!(f, " +-----------------+")?;

        for row_idx in 0..BOARD_HEIGHT {
            write!(f, "{}|", BOARD_HEIGHT - row_idx)?;
            for col_idx in 0..BOARD_WIDTH {
                let idx = (row_idx * BOARD_WIDTH + col_idx) as u8;
                let square = Square::from_index(idx);
                let letter = if let Some(piece) = piece_board.get_piece_type_at_square(&square) {
                    let is_p1_piece = is_p1_piece(square.as_bit_board(), piece_board);
                    convert_piece_to_letter(&piece, is_p1_piece)
                } else if idx == 18 || idx == 21 || idx == 42 || idx == 45 { 
                    "x".to_string()
                } else {
                    " ".to_string()
                };

                write!(f, " {}", letter)?;
            }
            writeln!(f, " |")?;
        }

        writeln!(f, " +-----------------+")?;
        writeln!(f, "   a b c d e f g h")?;

        Ok(())
    }
}

impl FromStr for GameState {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let lines: Vec<_> = s.split("|").enumerate().filter(|(i, _)| i % 2 == 1).map(|(_, s)| s).collect();
        let regex = regex::Regex::new(r"^\s*(\d+)([gswb])").unwrap();

        let (move_number, p1_turn_to_move) = regex.captures(s.split("|").find(|_| true).unwrap()).map_or((2, true), |c| (
            c.get(1).unwrap().as_str().parse().unwrap(),
            c.get(2).unwrap().as_str() != "s" && c.get(2).unwrap().as_str() != "b"
        ));

        let mut p1_pieces = 0; let mut elephants = 0; let mut camels = 0; let mut horses = 0; let mut dogs = 0; let mut cats = 0; let mut rabbits = 0;

        for (row_idx, line) in lines.iter().enumerate() {
            for (col_idx, charr) in line.chars().enumerate().filter(|(i, _)| i % 2 == 1).map(|(_, s)| s).enumerate() {
                let idx = (row_idx * BOARD_WIDTH + col_idx) as u8;
                let square = Square::from_index(idx);
                if let Some((piece, is_p1)) = convert_char_to_piece(charr) {
                    let square_bit = square.as_bit_board();

                    match piece {
                        Piece::Elephant => elephants |= square_bit,
                        Piece::Camel => camels |= square_bit,
                        Piece::Horse => horses |= square_bit,
                        Piece::Dog => dogs |= square_bit,
                        Piece::Cat => cats |= square_bit,
                        Piece::Rabbit => rabbits |= square_bit,
                    }

                    if is_p1 {
                        p1_pieces |= square_bit;
                    }
                }
            }
        }

        let piece_board = PieceBoard::new(p1_pieces, elephants, camels, horses, dogs, cats, rabbits);
        let hash = Zobrist::from_piece_board(piece_board.get_piece_board(), p1_turn_to_move, 0);
        let hash_history = List::new();
        let hash_history = hash_history.append(hash);

        Ok(GameState::new(
            p1_turn_to_move,
            move_number,
            Phase::PlayPhase(PlayPhase::initial(hash, hash_history)),
            piece_board,
            hash
        ))
    }
}

fn convert_char_to_piece(c: char) -> Option<(Piece, bool)> {
    let is_p1 = c.is_uppercase();

    let piece = match c {
        'E' | 'e' => Some(Piece::Elephant),
        'M' | 'm' => Some(Piece::Camel),
        'H' | 'h' => Some(Piece::Horse),
        'D' | 'd' => Some(Piece::Dog),
        'C' | 'c' => Some(Piece::Cat),
        'R' | 'r' => Some(Piece::Rabbit),
        _ => None
    };

    piece.map(|p| (p, is_p1))
}

fn is_p1_piece(square_bit: u64, piece_board: &PieceBoardState) -> bool {
    let p1_piece_mask = piece_board.get_player_piece_mask(true);
    (square_bit & p1_piece_mask) != 0
}

pub fn convert_piece_to_letter(piece: &Piece, is_p1: bool) -> String {
    let letter = match piece {
        Piece::Elephant => "E",
        Piece::Camel => "M",
        Piece::Horse => "H",
        Piece::Dog => "D",
        Piece::Cat => "C",
        Piece::Rabbit => "R"
    };

    if is_p1 { letter.to_string()} else { letter.to_lowercase() }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use super::super::action::{Action,Piece};
    use engine::game_state::{GameState as GameStateTrait};

    fn place_major_pieces(game_state: &GameState) -> GameState {
        let game_state = game_state.take_action(&Action::Place(Piece::Horse));
        let game_state = game_state.take_action(&Action::Place(Piece::Cat));
        let game_state = game_state.take_action(&Action::Place(Piece::Dog));
        let game_state = game_state.take_action(&Action::Place(Piece::Camel));
        let game_state = game_state.take_action(&Action::Place(Piece::Elephant));
        let game_state = game_state.take_action(&Action::Place(Piece::Dog));
        let game_state = game_state.take_action(&Action::Place(Piece::Cat));
        game_state.take_action(&Action::Place(Piece::Horse))
    }

    fn place_8_rabbits(game_state: &GameState) -> GameState {
        let mut game_state = game_state.take_action(&Action::Place(Piece::Rabbit));
        for _ in 0..7 {
            game_state = game_state.take_action(&Action::Place(Piece::Rabbit));
        }
        game_state
    }

    fn initial_play_state() -> GameState {
        let game_state = GameState::initial();
        let game_state = place_8_rabbits(&game_state);
        let game_state = place_major_pieces(&game_state);

        let game_state = place_major_pieces(&game_state);
        place_8_rabbits(&game_state)
    }

    #[test]
    fn test_gamestate_fromstr() {
        let game_state: GameState = "
            +-----------------+
            8|   r   r r   r   |
            7| m   h     e   c |
            6|   r x r r x r   |
            5| h   d     c   d |
            4| E   H         M |
            3|   R x R R H R   |
            2| D   C     C   D |
            1|   R   R R   R   |
            +-----------------+
               a b c d e f g h"
            .parse().unwrap();

        let piece_board = game_state.get_piece_board();
        assert_eq!(piece_board.get_bits_for_piece(Piece::Rabbit, true),    0b__01011010__00000000__01011010__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(piece_board.get_bits_for_piece(Piece::Cat, true),       0b__00000000__00100100__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(piece_board.get_bits_for_piece(Piece::Dog, true),       0b__00000000__10000001__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(piece_board.get_bits_for_piece(Piece::Horse, true),     0b__00000000__00000000__00100000__00000100__00000000__00000000__00000000__00000000);
        assert_eq!(piece_board.get_bits_for_piece(Piece::Camel, true),     0b__00000000__00000000__00000000__10000000__00000000__00000000__00000000__00000000);
        assert_eq!(piece_board.get_bits_for_piece(Piece::Elephant, true),  0b__00000000__00000000__00000000__00000001__00000000__00000000__00000000__00000000);

        assert_eq!(piece_board.get_bits_for_piece(Piece::Rabbit, false),   0b__00000000__00000000__00000000__00000000__00000000__01011010__00000000__01011010);
        assert_eq!(piece_board.get_bits_for_piece(Piece::Cat, false),      0b__00000000__00000000__00000000__00000000__00100000__00000000__10000000__00000000);
        assert_eq!(piece_board.get_bits_for_piece(Piece::Dog, false),      0b__00000000__00000000__00000000__00000000__10000100__00000000__00000000__00000000);
        assert_eq!(piece_board.get_bits_for_piece(Piece::Horse, false),    0b__00000000__00000000__00000000__00000000__00000001__00000000__00000100__00000000);
        assert_eq!(piece_board.get_bits_for_piece(Piece::Camel, false),    0b__00000000__00000000__00000000__00000000__00000000__00000000__00000001__00000000);
        assert_eq!(piece_board.get_bits_for_piece(Piece::Elephant, false), 0b__00000000__00000000__00000000__00000000__00000000__00000000__00100000__00000000);
    }

    #[test]
    fn test_gamestate_fromstr_move_number_default() {
        let game_state: GameState = "
            +-----------------+
            8|   r   r r   r   |
            7| m   h     e   c |
            6|   r x r r x r   |
            5| h   d     c   d |
            4| E   H         M |
            3|   R x R R H R   |
            2| D   C     C   D |
            1|   R   R R   R   |
            +-----------------+
               a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.get_move_number(), 2);
        assert_eq!(game_state.is_p1_turn_to_move(), true);
    }

    #[test]
    fn test_gamestate_fromstr_move_number() {
        let game_state: GameState = "
            5s
            +-----------------+
            8|   r   r r   r   |
            7| m   h     e   c |
            6|   r x r r x r   |
            5| h   d     c   d |
            4| E   H         M |
            3|   R x R R H R   |
            2| D   C     C   D |
            1|   R   R R   R   |
            +-----------------+
               a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.get_move_number(), 5);
        assert_eq!(game_state.is_p1_turn_to_move(), false);
    }

    #[test]
    fn test_gamestate_fromstr_player() {
        let game_state: GameState = "
            176b
            +-----------------+
            8|   r   r r   r   |
            7| m   h     e   c |
            6|   r x r r x r   |
            5| h   d     c   d |
            4| E   H         M |
            3|   R x R R H R   |
            2| D   C     C   D |
            1|   R   R R   R   |
            +-----------------+
               a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.get_move_number(), 176);
        assert_eq!(game_state.is_p1_turn_to_move(), false);
    }

    #[test]
    fn test_gamestate_fromstr_player_w() {
        let game_state: GameState = "
            13w
            +-----------------+
            8|   r   r r   r   |
            7| m   h     e   c |
            6|   r x r r x r   |
            5| h   d     c   d |
            4| E   H         M |
            3|   R x R R H R   |
            2| D   C     C   D |
            1|   R   R R   R   |
            +-----------------+
               a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.get_move_number(), 13);
        assert_eq!(game_state.is_p1_turn_to_move(), true);
    }

    #[test]
    fn test_gamestate_to_str() {
        let expected_output = "2g
 +-----------------+
8| h c d m e d c h |
7| r r r r r r r r |
6|     x     x     |
5|                 |
4|                 |
3|     x     x     |
2| R R R R R R R R |
1| H C D M E D C H |
 +-----------------+
   a b c d e f g h
";

        assert_eq!(format!("{}", initial_play_state()), expected_output);
    }

    #[test]
    fn test_gamestate_from_str_and_to_str() {
        let orig_str = "14g
 +-----------------+
8|   r   r r   r   |
7| m   h     e   c |
6|   r x r r x r   |
5| h   d     c   d |
4| E   H         M |
3|   R x R R x R   |
2| D   C     C   D |
1|   R   R R   R   |
 +-----------------+
   a b c d e f g h
";

        let game_state: GameState = orig_str.parse().unwrap();
        let new_str = format!("{}", game_state);

        assert_eq!(new_str, orig_str);
    }
}
