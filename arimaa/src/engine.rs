use std::str::FromStr;
use std::fmt::{self,Display,Formatter};
use super::constants::{BOARD_WIDTH,BOARD_HEIGHT,MAX_NUMBER_OF_MOVES};
use super::action::{Action,Direction,Piece,Square};
use super::value::Value;
use engine::engine::GameEngine;
use engine::game_state::{GameState as GameStateTrait};
use common::bits::first_set_bit;
use failure::Error;

const LEFT_COLUMN_MASK: u64 =       0b__00000001__00000001__00000001__00000001__00000001__00000001__00000001__00000001;
const RIGHT_COLUMN_MASK: u64 =      0b__10000000__10000000__10000000__10000000__10000000__10000000__10000000__10000000;

const TOP_ROW_MASK: u64 =           0b__00000000__00000000__00000000__00000000__00000000__00000000__00000000__11111111;
const BOTTOM_ROW_MASK: u64 =        0b__11111111__00000000__00000000__00000000__00000000__00000000__00000000__00000000;

const P1_OBJECTIVE_MASK: u64 =      TOP_ROW_MASK;
const P2_OBJECTIVE_MASK: u64 =      BOTTOM_ROW_MASK;

const P1_PLACEMENT_MASK: u64 =      0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000;
const P2_PLACEMENT_MASK: u64 =      0b__00000000__00000000__00000000__00000000__00000000__00000000__11111111__11111111;
const LAST_P1_PLACEMENT_MASK: u64 = 0b__10000000__00000000__00000000__00000000__00000000__00000000__00000000__00000000;
const LAST_P2_PLACEMENT_MASK: u64 = 0b__00000000__00000000__00000000__00000000__00000000__00000000__10000000__00000000;

const TRAP_MASK: u64 =              0b__00000000__00000000__00100100__00000000__00000000__00100100__00000000__00000000;

/*
Layers:
In:
1 player
6 curr piece boards
6 opp piece boards
6 curr pieces remaining temp board
Where next piece is placed
Out:
6 pieces

6 curr piece boards
6 opp piece boards
3 num action boards
1 must push board?
1 enemy piece which can be moved through push/pull?
Out:
4 directional boards (substract irrelevant squares)
1 pass bit
*/


// @TODO: Catch issue where pulling a piece and losing puller in trap on same move.

// @TODO: Ensure pass can't happen from the MustPush state.
// @TODO: Ensure pass can't happen from the 0 actions state.PLACEMENT_MASK
// @TODO: Don't push and pull on the same move.


#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub enum PushPullState {
    /*
        Either:
        - The turn started
        - The last move was the completion of a pull.
        - The last move was the completion of a push.
    */
    None,
    // Means that the currently player's piece was moved last action, next action can possibly be a pull.
    PossiblePull(Square, Piece),
    // Means that we pushed an opponent's piece last move. Follow up must be taking the empty square.
    MustPush(Square, Piece)
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct PlayPhase {
    actions_this_turn: usize,
    push_pull_state: PushPullState
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub enum Phase {
    PlacePhase,
    PlayPhase(PlayPhase)
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct GameState {
    pub p1_turn_to_move: bool,
    pub move_number: usize,
    pub p1_piece_board: u64,
    pub elephant_board: u64,
    pub camel_board: u64,
    pub horse_board: u64,
    pub dog_board: u64,
    pub cat_board: u64,
    pub rabbit_board: u64,
    pub phase: Phase
}

impl Display for GameState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let move_number = self.move_number;
        let player_to_move = if self.p1_turn_to_move { "g" } else { "s" };
        writeln!(f, "{}{}", move_number, player_to_move)?;

        writeln!(f, " +-----------------+")?;

        for row_idx in 0..BOARD_HEIGHT {
            write!(f, "{}|", BOARD_HEIGHT - row_idx)?;
            for col_idx in 0..BOARD_WIDTH {
                let idx = (row_idx * BOARD_WIDTH + col_idx) as u8;
                let square = Square::from_index(idx);
                let letter = if let Some(piece) = self.get_piece_type_at_square(&square) {
                    let is_p1_piece = self.is_p1_piece(square.as_bit_board());
                    convert_piece_to_letter(piece, is_p1_piece)
                } else if idx == 18 || idx == 21 || idx == 42 || idx == 45 { 
                    "X".to_string()
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

impl GameState {
    pub fn take_action(&self, action: &Action) -> Self {
        match action {
            Action::Pass => self.pass(),
            Action::Place(piece) => self.place(piece),
            Action::Move(square,direction) => self.move_piece(square,direction)
        }
    }

    pub fn is_terminal(&self) -> Option<Value> {
        panic!("TODO")
    }

    fn place(&self, piece: &Piece) -> Self {
        let placement_bit = self.get_placement_bit();
        let mut new_state = self.clone();

        match piece {
            Piece::Elephant => new_state.elephant_board |= placement_bit,
            Piece::Camel => new_state.camel_board |= placement_bit,
            Piece::Horse => new_state.horse_board |= placement_bit,
            Piece::Dog => new_state.dog_board |= placement_bit,
            Piece::Cat => new_state.cat_board |= placement_bit,
            Piece::Rabbit => new_state.rabbit_board |= placement_bit,
        }

        if self.p1_turn_to_move {
            new_state.p1_piece_board |= placement_bit;
        }

        if placement_bit == LAST_P1_PLACEMENT_MASK {
            new_state.p1_turn_to_move = false;
        } else if placement_bit == LAST_P2_PLACEMENT_MASK {
            new_state.p1_turn_to_move = true;
            new_state.phase = Phase::PlayPhase(PlayPhase::initial());
        }

        new_state
    }

    fn pass(&self) -> Self {
        GameState {
            phase: Phase::PlayPhase(PlayPhase::initial()),
            p1_turn_to_move: !self.p1_turn_to_move,
            ..*self
        }
    }

    fn move_piece(&self, square: &Square, direction: &Direction) -> Self {
        let source_square_bit = square.as_bit_board();

        let new_elephant_board = shift_piece_in_direction(self.elephant_board, source_square_bit, direction);
        let new_camel_board = shift_piece_in_direction(self.camel_board, source_square_bit, direction);
        let new_horse_board = shift_piece_in_direction(self.horse_board, source_square_bit, direction);
        let new_dog_board = shift_piece_in_direction(self.dog_board, source_square_bit, direction);
        let new_cat_board = shift_piece_in_direction(self.cat_board, source_square_bit, direction);
        let new_rabbit_board = shift_piece_in_direction(self.rabbit_board, source_square_bit, direction);
        let new_p1_piece_board = shift_piece_in_direction(self.p1_piece_board, source_square_bit, direction);

        let new_play_phase = self.get_new_phase(square, direction);
        let new_p1_turn_to_move = if new_play_phase.actions_this_turn == 0 { !self.p1_turn_to_move } else { self.p1_turn_to_move };
        let new_move_number = self.move_number + if new_play_phase.actions_this_turn == 0 && new_p1_turn_to_move { 1 } else { 0 };

        let mut new_game_state = GameState {
            p1_turn_to_move: new_p1_turn_to_move,
            move_number: new_move_number,
            p1_piece_board: new_p1_piece_board,
            elephant_board: new_elephant_board,
            camel_board: new_camel_board,
            horse_board: new_horse_board,
            dog_board: new_dog_board,
            cat_board: new_cat_board,
            rabbit_board: new_rabbit_board,
            phase: Phase::PlayPhase(new_play_phase)
        };

        new_game_state.remove_unprotected_pieces();

        new_game_state
    }

    fn play_phase(&self) -> &PlayPhase {
        match &self.phase {
            Phase::PlayPhase(play_phase) => play_phase,
            _ => panic!("Wrong phase")
        }
    }

    fn get_new_phase(&self, square: &Square, direction: &Direction) -> PlayPhase {
        let play_phase = self.play_phase();
        let actions_this_turn = play_phase.actions_this_turn;
        let switch_players = actions_this_turn >= 3;

        if switch_players {
            PlayPhase::initial()
        } else {
            let source_square_bit = square.as_bit_board();
            let is_opponent_piece = self.is_their_piece(source_square_bit);

            // Check if previous move can count as a pull, if so, do that.
            // Otherwise state that it must be followed with a push.
            let push_pull_state = if is_opponent_piece && !self.move_can_be_counted_as_pull(source_square_bit, direction) {
                PushPullState::MustPush(square.clone(), self.get_piece_type_at_bit(source_square_bit))
            } else if !is_opponent_piece && !play_phase.must_push() {
                PushPullState::PossiblePull(square.clone(), self.get_piece_type_at_bit(source_square_bit))
            } else {
                PushPullState::None
            };

            PlayPhase {
                actions_this_turn: play_phase.actions_this_turn + 1,
                push_pull_state
            }
        }
    }

    fn remove_unprotected_pieces(&mut self) {
        let all_piece_bits = self.get_all_piece_bits();
        let animal_is_on_trap = animal_is_on_trap(all_piece_bits);

        if animal_is_on_trap {
            let unsupported_piece_bits = self.get_unsupported_piece_bits(all_piece_bits);
            let trapped_animal_bits = unsupported_piece_bits & TRAP_MASK;
            if trapped_animal_bits != 0 {
                let untrapped_animal_bits = !trapped_animal_bits;
                self.elephant_board &= untrapped_animal_bits;
                self.camel_board &= untrapped_animal_bits;
                self.horse_board &= untrapped_animal_bits;
                self.dog_board &= untrapped_animal_bits;
                self.cat_board &= untrapped_animal_bits;
                self.rabbit_board &= untrapped_animal_bits;
                self.p1_piece_board &= untrapped_animal_bits;
            }
        }
    }

    fn move_can_be_counted_as_pull(&self, new_move_square_bit: u64, direction: &Direction) -> bool {
        let play_phase = self.play_phase();
        if let PushPullState::PossiblePull(prev_move_square, my_piece) = &play_phase.push_pull_state {
            if prev_move_square.as_bit_board() == shift_in_direction(new_move_square_bit, direction) {
                let their_piece = self.get_piece_type_at_bit(new_move_square_bit);
                if my_piece > &their_piece {
                    return true;
                }
            }
        }

        false
    }

    fn get_piece_type_at_square(&self, square: &Square) -> Option<Piece> {
        let square_bit = square.as_bit_board();
        if square_bit & self.get_all_piece_bits() != 0 {
            Some(self.get_piece_type_at_bit(square_bit))
        } else {
            None
        }
    }

    fn get_piece_type_at_bit(&self, square_bit: u64) -> Piece {
        if self.rabbit_board & square_bit != 0 {
            Piece::Rabbit
        } else if self.elephant_board & square_bit != 0 {
            Piece::Elephant
        } else if self.camel_board & square_bit != 0 {
            Piece::Camel
        } else if self.horse_board & square_bit != 0 {
            Piece::Horse
        } else if self.dog_board & square_bit != 0 {
            Piece::Dog
        } else {
            Piece::Cat
        }
    }

    fn is_p1_piece(&self, square_bit: u64) -> bool {
        (square_bit & self.p1_piece_board) != 0
    }

    fn is_their_piece(&self, square_bit: u64) -> bool {
        let is_p1_piece = square_bit & self.p1_piece_board != 0;
        self.p1_turn_to_move ^ is_p1_piece
    }

    fn get_placement_bit(&self) -> u64 {
        let placed_pieces = self.get_all_piece_bits();

        let placement_mask = if self.p1_turn_to_move { P1_PLACEMENT_MASK } else { P2_PLACEMENT_MASK };
        let squares_to_place = !placed_pieces & placement_mask;

        first_set_bit(squares_to_place)
    }

    fn get_all_piece_bits(&self) -> u64 {
        self.elephant_board
        | self.camel_board
        | self.horse_board
        | self.dog_board
        | self.cat_board
        | self.rabbit_board
    }

    fn get_unsupported_piece_bits(&self, all_piece_bits: u64) -> u64 {
        all_piece_bits & !self.get_supported_piece_bits(all_piece_bits)
    }

    fn get_supported_piece_bits(&self, all_piece_bits: u64) -> u64 {
        let p1_pieces = self.p1_piece_board;
        let p2_pieces = all_piece_bits & !p1_pieces;

        let up_supported_pieces = p1_pieces & shift_up!(p1_pieces & !TOP_ROW_MASK) | p2_pieces & shift_up!(p2_pieces & !TOP_ROW_MASK);
        let right_supported_pieces = p1_pieces & shift_right!(p1_pieces & !RIGHT_COLUMN_MASK) | p2_pieces & shift_right!(p2_pieces & !RIGHT_COLUMN_MASK);
        let down_supported_pieces = p1_pieces & shift_down!(p1_pieces & !BOTTOM_ROW_MASK) | p2_pieces & shift_down!(p2_pieces & !BOTTOM_ROW_MASK);
        let left_supported_pieces = p1_pieces & shift_left!(p1_pieces & !LEFT_COLUMN_MASK) | p2_pieces & shift_left!(p2_pieces & !LEFT_COLUMN_MASK);

        up_supported_pieces | right_supported_pieces | down_supported_pieces | left_supported_pieces
    }
}

fn animal_is_on_trap(all_piece_bits: u64) -> bool {
    (all_piece_bits & TRAP_MASK) != 0
}

fn shift_piece_in_direction(piece_board: u64, source_square_bit: u64, direction: &Direction) -> u64 {
    shift_in_direction(piece_board & source_square_bit, direction) | piece_board & !source_square_bit
}

fn shift_in_direction(bits: u64, direction: &Direction) -> u64 {
    match direction {
        Direction::Up => shift_up!(bits),
        Direction::Right => shift_right!(bits),
        Direction::Down => shift_down!(bits),
        Direction::Left => shift_left!(bits)
    }
}

impl GameStateTrait for GameState {
    fn initial() -> Self {
        GameState {
            p1_turn_to_move: true,
            move_number: 1,
            p1_piece_board: 0,
            elephant_board: 0,
            camel_board: 0,
            horse_board: 0,
            dog_board: 0,
            cat_board: 0,
            rabbit_board: 0,
            phase: Phase::PlacePhase
        }
    }
}

impl FromStr for GameState {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // @TODO: Introduce correct push/pull state
        // @TODO: Introduce correct num_actions

        let mut game_state = GameState::initial();
        game_state.phase = Phase::PlayPhase(PlayPhase {
            actions_this_turn: 0,
            push_pull_state: PushPullState::None
        });

        let lines: Vec<_> = s.split("|").enumerate().filter(|(i, _)| i % 2 == 1).map(|(_, s)| s).collect();
        let p1_to_move = !lines[0].contains("s") && !lines[0].contains("b");
        game_state.p1_turn_to_move = p1_to_move;

        for (row_idx, line) in lines.iter().enumerate() {
            for (col_idx, charr) in line.chars().enumerate().filter(|(i, _)| i % 2 == 1).map(|(_, s)| s).enumerate() {
                let idx = (row_idx * BOARD_WIDTH + col_idx) as u8;
                let square = Square::from_index(idx);
                if let Some((piece, is_p1)) = convert_char_to_piece(charr) {
                    insert_piece_to_state(&mut game_state, &square, piece, is_p1);
                }
            }
        }

        Ok(game_state)
    }
}

fn insert_piece_to_state(game_state: &mut GameState, square: &Square, piece: Piece, p1_piece: bool) {
    let square_bit = square.as_bit_board();

    match piece {
        Piece::Elephant => game_state.elephant_board |= square_bit,
        Piece::Camel => game_state.camel_board |= square_bit,
        Piece::Horse => game_state.horse_board |= square_bit,
        Piece::Dog => game_state.dog_board |= square_bit,
        Piece::Cat => game_state.cat_board |= square_bit,
        Piece::Rabbit => game_state.rabbit_board |= square_bit,
    }

    if p1_piece {
        game_state.p1_piece_board |= square_bit;
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

fn convert_piece_to_letter(piece: Piece, is_p1: bool) -> String {
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

impl PlayPhase {
    fn initial() -> Self {
        PlayPhase {
            actions_this_turn: 0,
            push_pull_state: PushPullState::None
        }
    }

    fn must_push(&self) -> bool {
        match self.push_pull_state {
            PushPullState::MustPush(_, _) => true,
            _ => false
        }
    }
}

pub struct Engine {}

impl Engine {
    pub fn new() -> Self { Self {} }
}

impl GameEngine for Engine {
    type Action = Action;
    type State = GameState;
    type Value = Value;

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State {
        game_state.take_action(action)
    }

    fn is_terminal_state(&self, game_state: &Self::State) -> Option<Self::Value> {
        game_state.is_terminal()
    }

    fn get_player_to_move(&self, game_state: &Self::State) -> usize {
        if game_state.p1_turn_to_move { 1 } else { 2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::action::{Action,Piece,Square};
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
    fn test_action_placing_pieces() {
        let game_state = initial_play_state();

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00000000__11111111__00000000);
        assert_eq!(game_state.cat_board,        0b__01000010__00000000__00000000__00000000__00000000__00000000__00000000__01000010);
        assert_eq!(game_state.dog_board,        0b__00100100__00000000__00000000__00000000__00000000__00000000__00000000__00100100);
        assert_eq!(game_state.horse_board,      0b__10000001__00000000__00000000__00000000__00000000__00000000__00000000__10000001);
        assert_eq!(game_state.camel_board,      0b__00001000__00000000__00000000__00000000__00000000__00000000__00000000__00001000);
        assert_eq!(game_state.elephant_board,   0b__00010000__00000000__00000000__00000000__00000000__00000000__00000000__00010000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);

        assert_eq!(game_state.p1_turn_to_move,  true);
        assert_eq!(game_state.play_phase().actions_this_turn, 0);
        assert_eq!(game_state.play_phase().push_pull_state, PushPullState::None);
    }

    #[test]
    fn test_action_move_up() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Up));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111110__00000001__00000000__00000000__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111110__00000001__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 3), Direction::Up));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111110__00000000__00000001__00000000__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111110__00000000__00000001__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 2);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 4), Direction::Up));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111110__00000000__00000000__00000001__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111110__00000000__00000000__00000001__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 3);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Up));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111100__00000010__00000000__00000001__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111100__00000010__00000000__00000001__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_down() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('d', 7), Direction::Down));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00001000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 6), Direction::Down));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00001000__00000000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 2);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Down));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00001000__00000000__00000000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 3);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Down));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00001000__00000000__00000000__00000000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_left() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('d', 7), Direction::Down));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00001000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 6), Direction::Left));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00000100__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 2);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('c', 6), Direction::Left));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00000010__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 3);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('b', 6), Direction::Left));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00000001__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_right() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('d', 7), Direction::Down));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00001000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 6), Direction::Right));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00010000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 2);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('e', 6), Direction::Right));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00100000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 3);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('f', 6), Direction::Right));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__01000000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_trap_unsupported() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 2), Direction::Up));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111011__00000000__00000000__00000000__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111011__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 1);
        assert_eq!(game_state.p1_turn_to_move, true);
    }

    #[test]
    fn test_action_move_trap_supported_right() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 2), Direction::Up));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111001__00000110__00000000__00000000__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111001__00000110__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 2);
        assert_eq!(game_state.p1_turn_to_move, true);
    }

    #[test]
    fn test_action_move_trap_supported_left() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('d', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 2), Direction::Up));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11110011__00001100__00000000__00000000__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11110011__00001100__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 2);
        assert_eq!(game_state.p1_turn_to_move, true);
    }

    #[test]
    fn test_action_move_trap_supported_top() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 3), Direction::Right));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111101__00000100__00000000__00000000__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111101__00000100__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 2);
        assert_eq!(game_state.p1_turn_to_move, true);
    }

    #[test]
    fn test_action_move_trap_supported_bottom() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 3), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 4), Direction::Right));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 2), Direction::Up));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111001__00000100__00000100__00000000__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111001__00000100__00000100__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }


    #[test]
    fn test_action_move_trap_adjacent_opp_unsupported() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 3), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 4), Direction::Up));

        let game_state = game_state.take_action(&Action::Move(Square::new('c', 7), Direction::Down));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111001__00000010__00000000__00000100__00000000__11111011__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111001__00000010__00000000__00000100__00000000__00000000__00000000);
        assert_eq!(game_state.play_phase().actions_this_turn, 1);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_push_must_push_rabbit() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 7), Direction::Down));

        assert_eq!(game_state.play_phase().push_pull_state, PushPullState::MustPush(Square::new('b', 7), Piece::Rabbit));
    }

    #[test]
    fn test_action_move_push_must_push_elephant() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Pass);
        let game_state = game_state.take_action(&Action::Move(Square::new('e', 7), Direction::Down));
        let game_state = game_state.take_action(&Action::Pass);
        let game_state = game_state.take_action(&Action::Move(Square::new('e', 8), Direction::Down));

        assert_eq!(game_state.play_phase().push_pull_state, PushPullState::MustPush(Square::new('e', 8), Piece::Elephant));
    }

    #[test]
    fn test_action_move_possible_pull_with_valid_pull() {
        let game_state: GameState = "
              +-----------------+
             8|   r   r r   r   |
             7|                 |
             6|     x     x     |
             5|                 |
             4|     E m         |
             3|     x     x     |
             2|                 |
             1|   R   R R   R   |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('c', 4), Direction::Up));
        assert_eq!(game_state.play_phase().push_pull_state, PushPullState::PossiblePull(Square::new('c', 4), Piece::Elephant));

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Left));
        assert_eq!(game_state.play_phase().push_pull_state, PushPullState::None);
    }

    #[test]
    fn test_action_move_pull_into_trap() {
        let game_state: GameState = "
              +-----------------+
             8|   r   r r   r   |
             7|                 |
             6|     x     x     |
             5|                 |
             4|     E m         |
             3|     x     x     |
             2|                 |
             1|   R   R R   R   |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('c', 4), Direction::Down));
        assert_eq!(game_state.play_phase().push_pull_state, PushPullState::PossiblePull(Square::new('c', 4), Piece::Elephant));

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Left));
        assert_eq!(game_state.play_phase().push_pull_state, PushPullState::None);
    }

    #[test]
    fn test_action_move_possible_pull_with_invalid_pull() {
        let game_state: GameState = "
              +-----------------+
             8|   r   r r   r   |
             7|                 |
             6|     x     x     |
             5|                 |
             4|     E e         |
             3|     x     x     |
             2|                 |
             1|   R   R R   R   |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('c', 4), Direction::Up));
        assert_eq!(game_state.play_phase().push_pull_state, PushPullState::PossiblePull(Square::new('c', 4), Piece::Elephant));

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Left));
        assert_eq!(game_state.play_phase().push_pull_state, PushPullState::MustPush(Square::new('d', 4), Piece::Elephant));
    }

    #[test]
    fn test_action_move_possible_pull_with_invalid_pull_2() {
        let game_state: GameState = "
              +-----------------+
             8|   r   r r   r   |
             7|                 |
             6|     x     x     |
             5|                 |
             4|     M e         |
             3|     x     x     |
             2|                 |
             1|   R   R R   R   |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('c', 4), Direction::Up));
        assert_eq!(game_state.play_phase().push_pull_state, PushPullState::PossiblePull(Square::new('c', 4), Piece::Camel));

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Left));
        assert_eq!(game_state.play_phase().push_pull_state, PushPullState::MustPush(Square::new('d', 4), Piece::Elephant));
    }

    #[test]
    fn test_action_pass() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('d', 7), Direction::Down));

        assert_eq!(game_state.play_phase().actions_this_turn, 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Pass);

        assert_eq!(game_state.play_phase().actions_this_turn, 0);
        assert_eq!(game_state.p1_turn_to_move, false);
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

        assert_eq!(game_state.rabbit_board,     0b__01011010__00000000__01011010__00000000__00000000__01011010__00000000__01011010);
        assert_eq!(game_state.cat_board,        0b__00000000__00100100__00000000__00000000__00100000__00000000__10000000__00000000);
        assert_eq!(game_state.dog_board,        0b__00000000__10000001__00000000__00000000__10000100__00000000__00000000__00000000);
        assert_eq!(game_state.horse_board,      0b__00000000__00000000__00100000__00000100__00000001__00000000__00000100__00000000);
        assert_eq!(game_state.camel_board,      0b__00000000__00000000__00000000__10000000__00000000__00000000__00000001__00000000);
        assert_eq!(game_state.elephant_board,   0b__00000000__00000000__00000000__00000001__00000000__00000000__00100000__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__01011010__10100101__01111010__10000101__00000000__00000000__00000000__00000000);
    }

    #[test]
    fn test_gamestate_to_str() {
        let game_state: GameState = "
              +-----------------+
             8|   r   r r   r   |
             7| m   h     e   c |
             6|   r x r r x r   |
             5| h   d     c   d |
             4| E   H         M |
             3|   R x R R   R   |
             2| D   C     C   D |
             1|   R   R R   R   |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        println!("{}", game_state);

        assert_eq!(format!("{}", game_state), "");
    }
}
