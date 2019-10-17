use std::fmt::{self,Display,Formatter};
use super::constants::{BOARD_WIDTH,BOARD_HEIGHT,MAX_NUMBER_OF_MOVES};
use super::action::{Action,Direction,Piece,Square};
use super::value::Value;
use engine::engine::GameEngine;
use engine::game_state;
use common::bits::first_set_bit;

const LEFT_COLUMN_MASK: u64 =   0b__00000001__00000001__00000001__00000001__00000001__00000001__00000001__00000001;
const RIGHT_COLUMN_MASK: u64 =  0b__10000000__10000000__10000000__10000000__10000000__10000000__10000000__10000000;

const P1_OBJECTIVE_MASK: u64 =  0b__00000000__00000000__00000000__00000000__00000000__00000000__00000000__11111111;
const P2_OBJECTIVE_MASK: u64 =  0b__11111111__00000000__00000000__00000000__00000000__00000000__00000000__00000000;

const PLACEMENT_MASK: u64 =     0b__11111111__11111111__00000000__00000000__00000000__00000000__11111111__11111111;
const P2_PLACEMENT_MASK: u64 =  0b__00000000__00000000__00000000__00000000__00000000__00000000__11111111__11111111;

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
        // @TODO: Display game state

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
            Piece::Elephant => new_state.elephant_board &= placement_bit,
            Piece::Camel => new_state.camel_board &= placement_bit,
            Piece::Horse => new_state.horse_board &= placement_bit,
            Piece::Dog => new_state.dog_board &= placement_bit,
            Piece::Cat => new_state.cat_board &= placement_bit,
            Piece::Rabbit => new_state.rabbit_board &= placement_bit,
        }

        if self.p1_turn_to_move {
            new_state.p1_piece_board &= placement_bit;
        }

        let new_placement_bit = new_state.get_placement_bit();

        if new_placement_bit == 0 {
            new_state.p1_turn_to_move = true;
            new_state.phase = Phase::PlayPhase(PlayPhase::initial());
        } else if new_placement_bit & P2_PLACEMENT_MASK != 0 {
            new_state.p1_turn_to_move = false;
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

        let new_play_phase = self.get_new_phase(square);
        let new_p1_turn_to_move = if new_play_phase.actions_this_turn == 0 { !self.p1_turn_to_move } else { self.p1_turn_to_move };

        let mut new_game_state = GameState {
            p1_turn_to_move: new_p1_turn_to_move,
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

    fn get_new_phase(&self, square: &Square) -> PlayPhase {
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
            let push_pull_state = if is_opponent_piece && !self.move_can_be_counted_as_pull(square) {
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
        panic!("TODO")
    }

    fn move_can_be_counted_as_pull(&self, new_move_square: &Square) -> bool {
        let play_phase = self.play_phase();
        if let PushPullState::PossiblePull(prev_move_square, my_piece) = &play_phase.push_pull_state {
            if prev_move_square == new_move_square {
                let their_piece = self.get_piece_type_at_square(new_move_square);
                if my_piece > &their_piece {
                    return true;
                }
            }
        }

        false
    }

    fn get_piece_type_at_square(&self, square: &Square) -> Piece {
        self.get_piece_type_at_bit(square.as_bit_board())
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
        let placed_pieces = self.elephant_board
            | self.camel_board
            | self.horse_board
            | self.dog_board
            | self.cat_board
            | self.rabbit_board;

        let squares_to_place = !placed_pieces & PLACEMENT_MASK;
        first_set_bit(squares_to_place)
    }
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

impl game_state::GameState for GameState {
    fn initial() -> Self {
        GameState {
            p1_turn_to_move: true,
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

// #[cfg(test)]
// mod tests {
//     use super::GameState;
//     use super::super::action::{Action,Coordinate};
//     use engine::game_state::{GameState as GameStateTrait};

//     fn intersects(actions: &Vec<Action>, exclusions: &Vec<Action>) -> bool {
//         actions.iter().any(|a| exclusions.iter().any(|a2| a == a2))
//     }

//     #[test]
//     fn test_get_valid_pawn_move_actions_p1() {
//         let game_state = GameState::initial();
//         let valid_actions = game_state.get_valid_pawn_move_actions();

//         assert_eq!(valid_actions, vec!(
//             Action::MovePawn(Coordinate::new('f', 1)),
//             Action::MovePawn(Coordinate::new('d', 1)),
//             Action::MovePawn(Coordinate::new('e', 2))
//         ));
//     }

//     #[test]
//     fn test_get_valid_pawn_move_actions_p2() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',1)));
//         let valid_actions = game_state.get_valid_pawn_move_actions();

//         assert_eq!(valid_actions, vec!(
//             Action::MovePawn(Coordinate::new('e', 8)),
//             Action::MovePawn(Coordinate::new('f', 9)),
//             Action::MovePawn(Coordinate::new('d', 9))
//         ));
//     }

//     #[test]
//     fn test_get_valid_pawn_move_actions_vertical_wall() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('d',1)));
//         let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',1)));
//         let valid_actions = game_state.get_valid_pawn_move_actions();

//         assert_eq!(valid_actions, vec!(
//             Action::MovePawn(Coordinate::new('e', 2))
//         ));
//     }

//     #[test]
//     fn test_get_valid_pawn_move_actions_vertical_wall_top() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',2)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',7)));
//         let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('d',1)));
//         let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',1)));
//         let valid_actions = game_state.get_valid_pawn_move_actions();

//         assert_eq!(valid_actions, vec!(
//             Action::MovePawn(Coordinate::new('e', 1)),
//             Action::MovePawn(Coordinate::new('e', 3))
//         ));
//     }

//     #[test]
//     fn test_get_valid_pawn_move_actions_horizontal_wall() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('d',8)));
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e',1)));
//         let valid_actions = game_state.get_valid_pawn_move_actions();

//         assert_eq!(valid_actions, vec!(
//             Action::MovePawn(Coordinate::new('f', 1)),
//             Action::MovePawn(Coordinate::new('d', 1))
//         ));

//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',1)));
//         let valid_actions = game_state.get_valid_pawn_move_actions();

//         assert_eq!(valid_actions, vec!(
//             Action::MovePawn(Coordinate::new('f', 9)),
//             Action::MovePawn(Coordinate::new('d', 9))
//         ));

//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',9)));
//         let valid_actions = game_state.get_valid_pawn_move_actions();

//         assert_eq!(valid_actions, vec!(
//             Action::MovePawn(Coordinate::new('g', 1)),
//             Action::MovePawn(Coordinate::new('e', 1))
//         ));
//     }

//     #[test]
//     fn test_get_valid_pawn_move_actions_blocked() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',2)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',8)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',3)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',7)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',4)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',6)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',5)));

//         let valid_actions = game_state.get_valid_pawn_move_actions();
//         assert_eq!(valid_actions, vec!(
//             Action::MovePawn(Coordinate::new('e',4)),
//             Action::MovePawn(Coordinate::new('f',6)),
//             Action::MovePawn(Coordinate::new('d',6)),
//             Action::MovePawn(Coordinate::new('e',7))
//         ));

//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e',4)));
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('a',1)));
//         let valid_actions = game_state.get_valid_pawn_move_actions();

//         assert_eq!(valid_actions, vec!(
//             Action::MovePawn(Coordinate::new('f',5)),
//             Action::MovePawn(Coordinate::new('d',5)),
//             Action::MovePawn(Coordinate::new('f',6)),
//             Action::MovePawn(Coordinate::new('d',6)),
//             Action::MovePawn(Coordinate::new('e',7))
//         ));

//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e',6)));
//         let valid_actions = game_state.get_valid_pawn_move_actions();

//         assert_eq!(valid_actions, vec!(
//             Action::MovePawn(Coordinate::new('f',5)),
//             Action::MovePawn(Coordinate::new('d',5)),
//             Action::MovePawn(Coordinate::new('f',6)),
//             Action::MovePawn(Coordinate::new('d',6))
//         ));
//     }

//     #[test]
//     fn test_get_valid_horizontal_wall_actions_initial() {
//         let game_state = GameState::initial();
//         let valid_actions = game_state.get_valid_horizontal_wall_actions();

//         let mut cols = ['a','b','c','d','e','f','g','h'];
//         let rows = [1,2,3,4,5,6,7,8];
//         cols.reverse();

//         let mut actions = Vec::new();

//         for row in rows.into_iter() {
//             for col in cols.into_iter() {
//                 actions.push(Action::PlaceHorizontalWall(Coordinate::new(*col, *row)));
//             }
//         }

//         assert_eq!(valid_actions.len(), actions.len());
//         assert_eq!(valid_actions, actions);
//     }

//     #[test]
//     fn test_get_valid_horizontal_wall_actions_on_horizontal_wall() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('d',1)));

//         let valid_actions = game_state.get_valid_horizontal_wall_actions();
//         let excludes_actions = vec!(
//             Action::PlaceHorizontalWall(Coordinate::new('c', 1)),
//             Action::PlaceHorizontalWall(Coordinate::new('d', 1)),
//             Action::PlaceHorizontalWall(Coordinate::new('e', 1))
//         );
//         let intersects = intersects(&valid_actions, &excludes_actions);

//         assert_eq!(intersects, false);
//         assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
//     }

//     #[test]
//     fn test_get_valid_horizontal_wall_actions_on_vertical_wall() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',5)));

//         let valid_actions = game_state.get_valid_horizontal_wall_actions();
//         let excludes_actions = vec!(
//             Action::PlaceHorizontalWall(Coordinate::new('e', 5))
//         );
//         let intersects = intersects(&valid_actions, &excludes_actions);

//         assert_eq!(intersects, false);
//         assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
//     }

//     #[test]
//     fn test_get_valid_horizontal_wall_actions_blocking_path() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('c',1)));
//         let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',1)));

//         let valid_actions = game_state.get_valid_horizontal_wall_actions();
//         let excludes_actions = vec!(
//             Action::PlaceHorizontalWall(Coordinate::new('c', 1)),
//             Action::PlaceHorizontalWall(Coordinate::new('e', 1)),
//             Action::PlaceHorizontalWall(Coordinate::new('d', 1)),
//             Action::PlaceHorizontalWall(Coordinate::new('d', 2))
//         );
//         let intersects = intersects(&valid_actions, &excludes_actions);

//         assert_eq!(intersects, false);
//         assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
//     }

//     #[test]
//     fn test_get_valid_horizontal_wall_actions_blocking_path_other_player() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('c',1)));
//         let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',1)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',2)));

//         let valid_actions = game_state.get_valid_horizontal_wall_actions();
//         let excludes_actions = vec!(
//             Action::PlaceHorizontalWall(Coordinate::new('c', 1)),
//             Action::PlaceHorizontalWall(Coordinate::new('e', 1)),
//             Action::PlaceHorizontalWall(Coordinate::new('d', 2))
//         );
//         let intersects = intersects(&valid_actions, &excludes_actions);

//         assert_eq!(intersects, false);
//         assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
//     }

//     #[test]
//     fn test_get_valid_horizontal_wall_actions_blocking_path_vert_horz() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('c',1)));
//         let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',1)));
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('c',2)));

//         let valid_actions = game_state.get_valid_horizontal_wall_actions();
//         let excludes_actions = vec!(
//             Action::PlaceHorizontalWall(Coordinate::new('c', 1)),
//             Action::PlaceHorizontalWall(Coordinate::new('e', 1)),
//             Action::PlaceHorizontalWall(Coordinate::new('d', 2)),
//             Action::PlaceHorizontalWall(Coordinate::new('b', 2)),
//             Action::PlaceHorizontalWall(Coordinate::new('c', 2)),
//             Action::PlaceHorizontalWall(Coordinate::new('d', 2)),
//             Action::PlaceHorizontalWall(Coordinate::new('e', 2)),
//         );
//         let intersects = intersects(&valid_actions, &excludes_actions);

//         assert_eq!(intersects, false);
//         assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
//     }

//     #[test]
//     fn test_get_valid_horizontal_wall_actions_blocking_path_edge() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',1)));
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e',2)));
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('c',2)));
//         let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('b',3)));

//         let valid_actions = game_state.get_valid_horizontal_wall_actions();
//         let excludes_actions = vec!(
//             Action::PlaceHorizontalWall(Coordinate::new('a', 2)),
//             Action::PlaceHorizontalWall(Coordinate::new('a', 3)),
//             Action::PlaceHorizontalWall(Coordinate::new('a', 4)),
//             Action::PlaceHorizontalWall(Coordinate::new('b', 2)),
//             Action::PlaceHorizontalWall(Coordinate::new('b', 3)),
//             Action::PlaceHorizontalWall(Coordinate::new('c', 2)),
//             Action::PlaceHorizontalWall(Coordinate::new('d', 2)),
//             Action::PlaceHorizontalWall(Coordinate::new('e', 2)),
//             Action::PlaceHorizontalWall(Coordinate::new('e', 1)),
//             Action::PlaceHorizontalWall(Coordinate::new('f', 2)),
//         );
//         let intersects = intersects(&valid_actions, &excludes_actions);

//         assert_eq!(intersects, false);
//         assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
//     }

//     #[test]
//     fn test_get_valid_vertical_wall_actions_initial() {
//         let game_state = GameState::initial();
//         let valid_actions = game_state.get_valid_vertical_wall_actions();

//         let mut cols = ['a','b','c','d','e','f','g','h'];
//         let rows = [1,2,3,4,5,6,7,8];
//         cols.reverse();

//         let mut actions = Vec::new();

//         for row in rows.into_iter() {
//             for col in cols.into_iter() {
//                 actions.push(Action::PlaceVerticalWall(Coordinate::new(*col, *row)));
//             }
//         }

//         assert_eq!(valid_actions.len(), actions.len());
//         assert_eq!(valid_actions, actions);
//     }

//     #[test]
//     fn test_get_valid_vertical_wall_actions_on_vertical_wall() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',5)));

//         let valid_actions = game_state.get_valid_vertical_wall_actions();
//         let excludes_actions = vec!(
//             Action::PlaceVerticalWall(Coordinate::new('e', 4)),
//             Action::PlaceVerticalWall(Coordinate::new('e', 5)),
//             Action::PlaceVerticalWall(Coordinate::new('e', 6))
//         );
//         let intersects = intersects(&valid_actions, &excludes_actions);

//         assert_eq!(intersects, false);
//         assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
//     }

//     #[test]
//     fn test_get_valid_vertical_wall_actions_on_horizontal_wall() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e',5)));

//         let valid_actions = game_state.get_valid_vertical_wall_actions();
//         let excludes_actions = vec!(
//             Action::PlaceVerticalWall(Coordinate::new('e', 5))
//         );
//         let intersects = intersects(&valid_actions, &excludes_actions);

//         assert_eq!(intersects, false);
//         assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
//     }

//     #[test]
//     fn test_get_valid_wall_actions_on_all_walls_placed() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('a',1)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',9)));
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('c',1)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',9)));
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e',1)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',9)));
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('g',1)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',9)));
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('a',2)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',9)));
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('c',2)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',9)));
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e',2)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',9)));
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('g',2)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',9)));
//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('a',3)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',9)));

//         // 9 walls placed
//         let valid_actions = game_state.get_valid_horizontal_wall_actions();
//         assert_eq!(valid_actions.len(), 46);

//         let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('c',3)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',9)));

//         // 10 walls placed so we shouldn't be able to place anymore, horizontal or vertical
//         let valid_horizontal_actions = game_state.get_valid_horizontal_wall_actions();
//         assert_eq!(valid_horizontal_actions.len(), 0);

//         let valid_vertical_actions = game_state.get_valid_vertical_wall_actions();
//         assert_eq!(valid_vertical_actions.len(), 0);
//     }

//     #[test]
//     fn test_is_terminal_p1() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',2)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',8)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',3)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',7)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',4)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',6)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',5)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',4)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',6)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',3)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',7)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',2)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',8)));

//         let is_terminal = game_state.is_terminal();
//         assert_eq!(is_terminal, None);

//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',1)));

//         let is_terminal = game_state.is_terminal();
//         assert_eq!(is_terminal, Some(-1.0));
//     }

//     #[test]
//     fn test_is_terminal_p2() {
//         let game_state = GameState::initial();
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',2)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',8)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',3)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',7)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',4)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',6)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',5)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',4)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',6)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',3)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',7)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',2)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',8)));

//         let is_terminal = game_state.is_terminal();
//         assert_eq!(is_terminal, None);

//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',3)));
//         let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',9)));

//         let is_terminal = game_state.is_terminal();
//         assert_eq!(is_terminal, Some(-1.0));
//     }
// }
