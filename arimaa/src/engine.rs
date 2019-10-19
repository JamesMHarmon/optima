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

// @TODO: ... but may not pass the whole turn or make a move equivalent to passing the whole turn.

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
    MustCompletePush(Square, Piece)
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
        let curr_player = if self.p1_turn_to_move { "g" } else { "s" };
        writeln!(f, "{}{}", move_number, curr_player)?;

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

impl GameState {
    pub fn take_action(&self, action: &Action) -> Self {
        match action {
            Action::Pass => self.pass(),
            Action::Place(piece) => self.place(piece),
            Action::Move(square,direction) => self.move_piece(square,direction)
        }
    }

    pub fn is_terminal(&self) -> Option<Value> {
        self.as_play_phase().and_then(|play_phase| {
            // The order of checking for win/lose conditions is as follows assuming player A just made the move and player B now needs to move:
            if play_phase.actions_this_turn > 0 {
                return None;
            }

            // Check if a rabbit of player A reached goal. If so player A wins.
            // Check if a rabbit of player B reached goal. If so player B wins.
            self.rabbit_at_goal()
            // Check if player B lost all rabbits. If so player A wins.
            // Check if player A lost all rabbits. If so player B wins.
            .or_else(|| self.lost_all_rabbits())
            // Check if player B has no possible move (all pieces are frozen or have no place to move). If so player A wins.
            // @TODO
            //.or_else(|| self.has_move())
            // Check if the only moves player B has are 3rd time repetitions. If so player A wins.
            // @TODO
        })
    }

    pub fn valid_actions(&self) -> Vec<Action> {
        if let Phase::PlayPhase(play_phase) = &self.phase {
            let all_piece_bits = self.get_all_piece_bits();
            if play_phase.push_pull_state.is_must_complete_push() {
                self.get_must_complete_push_actions(all_piece_bits)
            } else {
                let mut valid_actions = Vec::with_capacity(50);
                self.extend_with_valid_curr_player_piece_moves(&mut valid_actions, all_piece_bits);       
                self.extend_with_pull_piece_actions(&mut valid_actions, all_piece_bits);
                self.extend_with_push_piece_actions(&mut valid_actions, all_piece_bits);

                valid_actions
            }
        } else {
            self.valid_placement()
        }
    }

    fn can_pass(&self) -> bool {
        self.as_play_phase().map_or(false, |play_phase| play_phase.actions_this_turn >= 1 && !play_phase.push_pull_state.is_must_complete_push())
    }

    fn valid_placement(&self) -> Vec<Action> {
        let mut actions = Vec::with_capacity(6);
        let curr_player_pieces = self.get_curr_player_piece_mask(self.get_all_piece_bits());

        if self.elephant_board & curr_player_pieces == 0 { actions.push(Action::Place(Piece::Elephant)); }
        if self.camel_board & curr_player_pieces == 0 { actions.push(Action::Place(Piece::Camel)); }
        if (self.horse_board & curr_player_pieces).count_ones() < 2 { actions.push(Action::Place(Piece::Horse)); }
        if (self.dog_board & curr_player_pieces).count_ones() < 2 { actions.push(Action::Place(Piece::Dog)); }
        if (self.cat_board & curr_player_pieces).count_ones() < 2 { actions.push(Action::Place(Piece::Cat)); }
        if (self.rabbit_board & curr_player_pieces).count_ones() < 8 { actions.push(Action::Place(Piece::Rabbit)); }

        actions
    }

    fn extend_with_valid_curr_player_piece_moves(&self, valid_actions: &mut Vec<Action>, all_piece_bits: u64) {
        let non_frozen_pieces = self.get_curr_player_non_frozen_pieces(all_piece_bits);

        for direction in [Direction::Up, Direction::Right, Direction::Down, Direction::Left].iter() {
            let unoccupied_directions = can_move_in_direction(all_piece_bits, direction);
            let invalid_rabbit_moves = self.get_invalid_rabbit_moves(direction);
            let valid_curr_piece_moves = unoccupied_directions & non_frozen_pieces & !invalid_rabbit_moves;

            let squares = map_bit_board_to_squares(valid_curr_piece_moves);
            valid_actions.extend(squares.into_iter().map(|s| Action::Move(s, *direction)));
        }

        if self.can_pass() {
            valid_actions.push(Action::Pass);
        }
    }

    fn extend_with_pull_piece_actions(&self, valid_actions: &mut Vec<Action>, all_piece_bits: u64) {
        if let Some(play_phase) = self.as_play_phase() {
            if let Some((square, piece)) = play_phase.push_pull_state.as_possible_pull() {
                let opp_piece_mask = self.get_opponent_piece_mask(all_piece_bits);
                let lesser_opp_pieces = self.get_lesser_pieces(piece) & opp_piece_mask;
                let square_bit = square.as_bit_board();

                for direction in [Direction::Up, Direction::Right, Direction::Down, Direction::Left].iter() {
                    if shift_in_direction(lesser_opp_pieces, &direction) & square_bit != 0 {
                        valid_actions.push(Action::Move(*square, *direction));
                    }
                }
            }
        }
    }

    fn extend_with_push_piece_actions(&self, valid_actions: &mut Vec<Action>, all_piece_bits: u64) {
        if let Some(play_phase) = self.as_play_phase() {
            if play_phase.push_pull_state.can_push() {
                let opp_piece_mask = self.get_opponent_piece_mask(all_piece_bits);
                let predator_piece_mask = !opp_piece_mask & all_piece_bits;
                let opp_threatened_pieces = self.get_threatened_pieces(predator_piece_mask, opp_piece_mask);

                for direction in [Direction::Up, Direction::Right, Direction::Down, Direction::Left].iter() {
                    let unoccupied_directions = can_move_in_direction(all_piece_bits, direction);
                    let valid_push_moves = unoccupied_directions & opp_threatened_pieces;

                    let squares = map_bit_board_to_squares(valid_push_moves);
                    valid_actions.extend(squares.into_iter().map(|s| Action::Move(s, *direction)));
                }
            }
        }
    }

    fn get_must_complete_push_actions(&self, all_piece_bits: u64) -> Vec<Action> {
        let play_phase = self.unwrap_play_phase();
        let (square, pushed_piece) = play_phase.push_pull_state.unwrap_must_complete_push();

        let curr_player_piece_mask = self.get_curr_player_piece_mask(all_piece_bits);
        let square_bit = square.as_bit_board();

        let mut valid_actions = vec![];
        for direction in [Direction::Up, Direction::Right, Direction::Down, Direction::Left].iter() {
            let pushing_piece_bit = shift_pieces_in_opp_direction(square_bit, &direction) & curr_player_piece_mask;
            if pushing_piece_bit != 0 && self.get_piece_type_at_bit(pushing_piece_bit) > *pushed_piece {
                valid_actions.push(Action::Move(Square::from_bit_board(pushing_piece_bit), *direction));
            }
        }

        valid_actions
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

        new_game_state.remove_trapped_pieces();

        new_game_state
    }

    fn unwrap_play_phase(&self) -> &PlayPhase {
        self.as_play_phase().expect("Expected phase to be PlayPhase")
    }

    fn as_play_phase(&self) -> Option<&PlayPhase> {
        match &self.phase {
            Phase::PlayPhase(play_phase) => Some(play_phase),
            _ => None
        }
    }

    fn get_new_phase(&self, square: &Square, direction: &Direction) -> PlayPhase {
        let play_phase = self.unwrap_play_phase();
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
                PushPullState::MustCompletePush(*square, self.get_piece_type_at_bit(source_square_bit))
            } else if !is_opponent_piece && !play_phase.push_pull_state.is_must_complete_push() {
                PushPullState::PossiblePull(*square, self.get_piece_type_at_bit(source_square_bit))
            } else {
                PushPullState::None
            };

            PlayPhase {
                actions_this_turn: play_phase.actions_this_turn + 1,
                push_pull_state
            }
        }
    }

    fn remove_trapped_pieces(&mut self) {
        let all_piece_bits = self.get_all_piece_bits();
        let animal_is_on_trap = animal_is_on_trap(all_piece_bits);

        if animal_is_on_trap {
            let unsupported_piece_bits = self.get_both_player_unsupported_piece_bits(all_piece_bits);
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
        let play_phase = self.unwrap_play_phase();
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

    fn get_both_player_unsupported_piece_bits(&self, all_piece_bits: u64) -> u64 {
        all_piece_bits & !self.get_both_player_supported_pieces(all_piece_bits)
    }

    fn get_both_player_supported_pieces(&self, all_piece_bits: u64) -> u64 {
        let p1_pieces = self.p1_piece_board;
        let p2_pieces = all_piece_bits & !p1_pieces;

        get_supported_pieces(p1_pieces) | get_supported_pieces(p2_pieces)
    }

    fn get_curr_player_non_frozen_pieces(&self, all_piece_bits: u64) -> u64 {
        let opp_piece_mask = self.get_opponent_piece_mask(all_piece_bits);
        let curr_player_piece_mask = !opp_piece_mask & all_piece_bits;
        let threatened_pieces = self.get_threatened_pieces(opp_piece_mask, curr_player_piece_mask);

        curr_player_piece_mask & (!threatened_pieces | get_supported_pieces(curr_player_piece_mask))
    }

    fn get_threatened_pieces(&self, predator_piece_mask: u64, prey_piece_mask: u64) -> u64 {
        let predator_elephant_influence = get_influenced_squares(self.elephant_board & predator_piece_mask);
        let predator_camel_influence = get_influenced_squares(self.camel_board & predator_piece_mask);
        let predator_horse_influence = get_influenced_squares(self.horse_board & predator_piece_mask);
        let predator_dog_influence = get_influenced_squares(self.dog_board & predator_piece_mask);
        let predator_cat_influence = get_influenced_squares(self.cat_board & predator_piece_mask);

        let camel_threats = predator_elephant_influence;
        let horse_threats = camel_threats | predator_camel_influence;
        let dog_threats = horse_threats | predator_horse_influence;
        let cat_threats = dog_threats | predator_dog_influence;
        let rabbit_threats = cat_threats | predator_cat_influence;

        let threatened_pieces =
            (self.camel_board & camel_threats)
            | (self.horse_board & horse_threats)
            | (self.dog_board & dog_threats)
            | (self.cat_board & cat_threats)
            | (self.rabbit_board & rabbit_threats);

        threatened_pieces & prey_piece_mask
    }

    fn get_curr_player_piece_mask(&self, all_piece_bits: u64) -> u64 {
        if self.p1_turn_to_move { self.p1_piece_board } else { !self.p1_piece_board & all_piece_bits }
    }

    fn get_opponent_piece_mask(&self, all_piece_bits: u64) -> u64 {
        if self.p1_turn_to_move { !self.p1_piece_board & all_piece_bits } else { self.p1_piece_board }
    }

    fn rabbit_at_goal(&self) -> Option<Value> {
        let p1_objective_met = self.p1_piece_board & self.rabbit_board & P1_OBJECTIVE_MASK != 0;
        let p2_objective_met = !self.p1_piece_board & self.rabbit_board & P2_OBJECTIVE_MASK != 0;

        if p1_objective_met || p2_objective_met {
            // Objective is opposite of the player to move since we are checking if there is a winner after the turn is complete.
            // Logic should include the condition of if both players have a rabbit at the goal. In that case the player who was last to move wins.
            let last_to_move_is_p1 = !self.p1_turn_to_move;
            let last_to_move_objective_met = if last_to_move_is_p1 { p1_objective_met } else { p2_objective_met };
            let p1_won = !(last_to_move_is_p1 ^ last_to_move_objective_met);
            Some(if p1_won { Value([1.0, 0.0]) } else { Value([0.0, 1.0]) })
        } else {
            None
        }
    }

    fn lost_all_rabbits(&self) -> Option<Value> {
        let p1_lost_rabbits = self.p1_piece_board & self.rabbit_board == 0;
        let p2_lost_rabbits = !self.p1_piece_board & self.rabbit_board == 0;

        // Check if player B lost all rabbits. If so player A wins.
        // Check if player A lost all rabbits. If so player B wins.

        if p1_lost_rabbits || p2_lost_rabbits {
            // Objective is opposite of the player to move since we are checking if there is a winner after the turn is complete.
            // Logic should include the condition of if both players lost their rabbits. In that case the player who was last to move wins.
            let last_to_move_is_p1 = !self.p1_turn_to_move;
            let last_to_move_objective_met = if last_to_move_is_p1 { p2_lost_rabbits } else { p1_lost_rabbits };
            let p1_won = !(last_to_move_is_p1 ^ last_to_move_objective_met);
            Some(if p1_won { Value([1.0, 0.0]) } else { Value([0.0, 1.0]) })
        } else {
            None
        }
    }

    fn get_invalid_rabbit_moves(&self, direction: &Direction) -> u64 {
        let backward_direction = if self.p1_turn_to_move { Direction::Down } else { Direction::Up };

        if *direction == backward_direction {
            let players_rabbits = if self.p1_turn_to_move { self.p1_piece_board } else { !self.p1_piece_board } & self.rabbit_board;
            players_rabbits
        } else {
            0
        }
    }

    fn get_lesser_pieces(&self, piece: &Piece) -> u64 {
        match piece {
            Piece::Rabbit => 0,
            Piece::Cat => self.rabbit_board,
            Piece::Dog => self.rabbit_board | self.cat_board,
            Piece::Horse => self.rabbit_board | self.cat_board | self.dog_board,
            Piece::Camel => self.rabbit_board | self.cat_board | self.dog_board | self.horse_board,
            Piece::Elephant => self.rabbit_board | self.cat_board | self.dog_board | self.horse_board | self.camel_board,
        }
    } 
}

fn animal_is_on_trap(all_piece_bits: u64) -> bool {
    (all_piece_bits & TRAP_MASK) != 0
}

fn get_influenced_squares(piece_board: u64) -> u64 {
    shift_pieces_up!(piece_board) | shift_pieces_right!(piece_board) | shift_pieces_down!(piece_board) | shift_pieces_left!(piece_board)
}

fn get_supported_pieces(piece_bits: u64) -> u64 {
    let up_supported_pieces = piece_bits & shift_pieces_up!(piece_bits);
    let right_supported_pieces = piece_bits & shift_pieces_right!(piece_bits);
    let down_supported_pieces = piece_bits & shift_pieces_down!(piece_bits);
    let left_supported_pieces = piece_bits & shift_pieces_left!(piece_bits);

    up_supported_pieces | right_supported_pieces | down_supported_pieces | left_supported_pieces
}

fn can_move_in_direction(all_piece_bits: u64, direction: &Direction) -> u64 {
    let empty_squares = !all_piece_bits;
    shift_pieces_in_opp_direction(empty_squares, direction)
}

fn shift_piece_in_direction(piece_board: u64, source_square_bit: u64, direction: &Direction) -> u64 {
    shift_in_direction(piece_board & source_square_bit, direction) | piece_board & !source_square_bit
}

fn shift_pieces_in_opp_direction(bits: u64, direction: &Direction) -> u64 {
    match direction {
        Direction::Up => shift_pieces_down!(bits),
        Direction::Right => shift_pieces_left!(bits),
        Direction::Down => shift_pieces_up!(bits),
        Direction::Left => shift_pieces_right!(bits)
    }
}

fn shift_in_direction(bits: u64, direction: &Direction) -> u64 {
    match direction {
        Direction::Up => shift_up!(bits),
        Direction::Right => shift_right!(bits),
        Direction::Down => shift_down!(bits),
        Direction::Left => shift_left!(bits)
    }
}

fn map_bit_board_to_squares(board: u64) -> Vec<Square> {
    let mut board = board;
    let mut squares = Vec::new();

    while board != 0 {
        let board_without_first_bit = board & (board - 1);
        let removed_bit = board_without_first_bit ^ board;
        let square = Square::from_bit_board(removed_bit);
        squares.push(square);

        board = board_without_first_bit;
    }

    squares
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

impl PushPullState {
    fn is_must_complete_push(&self) -> bool {
        match self {
            PushPullState::MustCompletePush(_,_) => true,
            _ => false
        }
    }

    fn unwrap_must_complete_push(&self) -> (&Square, &Piece) {
        match self {
            PushPullState::MustCompletePush(square, piece) => (square, piece),
            _ => panic!("Expected PushPullState to be MustCompletePush")
        }
    }

    fn as_possible_pull(&self) -> Option<(&Square, &Piece)> {
        match self {
            PushPullState::PossiblePull(square, piece) => Some((square, piece)),
            _ => None
        }
    }

    fn can_push(&self) -> bool {
        match self {
            // We can't push another piece if we are already obligated to push another
            PushPullState::MustCompletePush(_,_) => false,
            _ => true
        }
    }
}

impl FromStr for GameState {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut game_state = GameState::initial();
        game_state.phase = Phase::PlayPhase(PlayPhase {
            actions_this_turn: 0,
            push_pull_state: PushPullState::None
        });

        let lines: Vec<_> = s.split("|").enumerate().filter(|(i, _)| i % 2 == 1).map(|(_, s)| s).collect();
        let regex = regex::Regex::new(r"^\s*(\d+)([gswb])").unwrap();

        let (move_number, p1_to_move) = regex.captures(s.split("|").find(|_| true).unwrap()).map_or((1, true), |c| (
            c.get(1).unwrap().as_str().parse().unwrap(),
            c.get(2).unwrap().as_str() != "s" && c.get(2).unwrap().as_str() != "b"
        ));

        game_state.move_number = move_number;
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
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 0);
        assert_eq!(game_state.unwrap_play_phase().push_pull_state, PushPullState::None);
    }

    #[test]
    fn test_action_move_up() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Up));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111110__00000001__00000000__00000000__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111110__00000001__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 3), Direction::Up));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111110__00000000__00000001__00000000__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111110__00000000__00000001__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 2);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 4), Direction::Up));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111110__00000000__00000000__00000001__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111110__00000000__00000000__00000001__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 3);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Up));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111100__00000010__00000000__00000001__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111100__00000010__00000000__00000001__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_down() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('d', 7), Direction::Down));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00001000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 6), Direction::Down));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00001000__00000000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 2);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Down));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00001000__00000000__00000000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 3);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Down));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00001000__00000000__00000000__00000000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_left() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('d', 7), Direction::Down));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00001000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 6), Direction::Left));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00000100__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 2);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('c', 6), Direction::Left));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00000010__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 3);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('b', 6), Direction::Left));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00000001__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_right() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('d', 7), Direction::Down));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00001000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 6), Direction::Right));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00010000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 2);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('e', 6), Direction::Right));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__00100000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 3);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('f', 6), Direction::Right));
        assert_eq!(game_state.rabbit_board,     0b__00000000__11111111__00000000__00000000__00000000__01000000__11110111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_trap_unsupported() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 2), Direction::Up));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111011__00000000__00000000__00000000__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111011__00000000__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 1);
        assert_eq!(game_state.p1_turn_to_move, true);
    }

    #[test]
    fn test_action_move_trap_supported_right() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 2), Direction::Up));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111001__00000110__00000000__00000000__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111001__00000110__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 2);
        assert_eq!(game_state.p1_turn_to_move, true);
    }

    #[test]
    fn test_action_move_trap_supported_left() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('d', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 2), Direction::Up));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11110011__00001100__00000000__00000000__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11110011__00001100__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 2);
        assert_eq!(game_state.p1_turn_to_move, true);
    }

    #[test]
    fn test_action_move_trap_supported_top() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 3), Direction::Right));

        assert_eq!(game_state.rabbit_board,     0b__00000000__11111101__00000100__00000000__00000000__00000000__11111111__00000000);
        assert_eq!(game_state.p1_piece_board,   0b__11111111__11111101__00000100__00000000__00000000__00000000__00000000__00000000);
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 2);
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
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 0);
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
        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 1);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_push_must_push_rabbit() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 7), Direction::Down));

        assert_eq!(game_state.unwrap_play_phase().push_pull_state, PushPullState::MustCompletePush(Square::new('b', 7), Piece::Rabbit));
    }

    #[test]
    fn test_action_move_push_must_push_elephant() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Pass);
        let game_state = game_state.take_action(&Action::Move(Square::new('e', 7), Direction::Down));
        let game_state = game_state.take_action(&Action::Pass);
        let game_state = game_state.take_action(&Action::Move(Square::new('e', 8), Direction::Down));

        assert_eq!(game_state.unwrap_play_phase().push_pull_state, PushPullState::MustCompletePush(Square::new('e', 8), Piece::Elephant));
    }

    #[test]
    fn test_action_place_initial() {
        let game_state = GameState::initial();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[e, m, h, d, c, r]");
    }

    #[test]
    fn test_action_place_elephant() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::Place(Piece::Elephant));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[m, h, d, c, r]");
    }

    #[test]
    fn test_action_place_camel() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::Place(Piece::Camel));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[e, h, d, c, r]");
    }

    #[test]
    fn test_action_place_horse() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::Place(Piece::Horse));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[e, m, h, d, c, r]");

        let game_state = game_state.take_action(&Action::Place(Piece::Horse));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[e, m, d, c, r]");
    }

    #[test]
    fn test_action_place_dog() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::Place(Piece::Dog));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[e, m, h, d, c, r]");

        let game_state = game_state.take_action(&Action::Place(Piece::Dog));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[e, m, h, c, r]");
    }

    #[test]
    fn test_action_place_rabbits() {
        let game_state = GameState::initial();

        let game_state = place_8_rabbits(&game_state);

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[e, m, h, d, c]");
    }

    #[test]
    fn test_action_place_majors() {
        let game_state = GameState::initial();

        let game_state = place_major_pieces(&game_state);

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[r]");
    }

    #[test]
    fn test_action_place_p2_initial() {
        let game_state = GameState::initial();

        let game_state = place_major_pieces(&game_state);
        let game_state = place_8_rabbits(&game_state);

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[e, m, h, d, c, r]");
    }

    #[test]
    fn test_action_place_p2_camel() {
        let game_state = GameState::initial();

        let game_state = place_major_pieces(&game_state);
        let game_state = place_8_rabbits(&game_state);
        let game_state = game_state.take_action(&Action::Place(Piece::Camel));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[e, h, d, c, r]");
    }

    #[test]
    fn test_can_pass_first_move() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|   r   r r   r   |
             7|                 |
             6|     x     x     |
             5|     E r         |
             4|                 |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.can_pass(), false);
    }

    #[test]
    fn test_can_pass_during_possible_pull() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|   r   r r   r   |
             7|                 |
             6|     x     x     |
             5|     E r         |
             4|                 |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('c', 5), Direction::Down));
        assert_eq!(game_state.can_pass(), true);
    }

    #[test]
    fn test_can_pass_during_must_push() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|   r   r r   r   |
             7|                 |
             6|     x     x     |
             5|     E r         |
             4|                 |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Down));
        assert_eq!(game_state.can_pass(), false);
    }

    #[test]
    fn test_can_pass_during_place_phase() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|   r   r r   r   |
             7|                 |
             6|     x     x     |
             5|     E r         |
             4|                 |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.can_pass(), false);

        let game_state = game_state.take_action(&Action::Place(Piece::Elephant));
        assert_eq!(game_state.can_pass(), false);
    }

    #[test]
    fn test_is_terminal_no_winner() {
        let game_state: GameState = "
             1g
              +-----------------+
             8| M r   r r   r   |
             7|                 |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|                 |
             1| m R             |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.is_terminal(), None);
    }

    #[test]
    fn test_is_terminal_mid_turn() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|   r   r r   r   |
             7| R               |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|                 |
             1| m               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 7), Direction::Up));
        assert_eq!(game_state.is_terminal(), None);

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 8), Direction::Down));
        assert_eq!(game_state.is_terminal(), None);

        let game_state = game_state.take_action(&Action::Pass);
        assert_eq!(game_state.is_terminal(), None);
    }

    #[test]
    fn test_is_terminal_p1_winner_as_p1() {
        let game_state: GameState = "
             1s
              +-----------------+
             8| R r   r r   r   |
             7|                 |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|                 |
             1|   R             |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.is_terminal(), Some(Value([1.0, 0.0])));
    }

    #[test]
    fn test_is_terminal_p2_winner_as_p1() {
        let game_state: GameState = "
             1s
              +-----------------+
             8|   r   r r   r   |
             7|                 |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|                 |
             1| r R             |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.is_terminal(), Some(Value([0.0, 1.0])));
    }

    #[test]
    fn test_is_terminal_p1_winner_as_p2() {
        let game_state: GameState = "
             1g
              +-----------------+
             8| R r   r r   r   |
             7|                 |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|                 |
             1|   R             |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.is_terminal(), Some(Value([1.0, 0.0])));
    }

    #[test]
    fn test_is_terminal_p2_winner_as_p2() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|   r   r r   r   |
             7|                 |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|                 |
             1| r R             |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.is_terminal(), Some(Value([0.0, 1.0])));
    }

    #[test]
    fn test_is_terminal_p1_and_p2_met_as_p1() {
        let game_state: GameState = "
             1s
              +-----------------+
             8| R r   r r   r   |
             7|                 |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|                 |
             1| r R             |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.is_terminal(), Some(Value([1.0, 0.0])));
    }

    #[test]
    fn test_is_terminal_p1_and_p2_met_as_p2() {
        let game_state: GameState = "
             1g
              +-----------------+
             8| R r   r r   r   |
             7|                 |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|                 |
             1| r R             |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.is_terminal(), Some(Value([0.0, 1.0])));
    }

    #[test]
    fn test_is_terminal_p2_lost_rabbits_as_p1() {
        let game_state: GameState = "
             1s
              +-----------------+
             8|                 |
             7|   e             |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|   E R           |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.is_terminal(), Some(Value([1.0, 0.0])));
    }

    #[test]
    fn test_is_terminal_p1_lost_rabbits_as_p2() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|   r             |
             7|   e             |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|   E             |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.is_terminal(), Some(Value([0.0, 1.0])));
    }

    #[test]
    fn test_is_terminal_p1_lost_rabbits_as_p1() {
        let game_state: GameState = "
             1s
              +-----------------+
             8|   r             |
             7|   e             |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|   E             |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.is_terminal(), Some(Value([0.0, 1.0])));
    }

    #[test]
    fn test_is_terminal_p2_lost_rabbits_as_p2() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|                 |
             7|   e             |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|   E R           |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.is_terminal(), Some(Value([1.0, 0.0])));
    }

    #[test]
    fn test_is_terminal_p1_and_p2_lost_rabbits_as_p1() {
         let game_state: GameState = "
             1s
              +-----------------+
             8|                 |
             7|   e             |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|   E             |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.is_terminal(), Some(Value([1.0, 0.0])));
    }

    #[test]
    fn test_is_terminal_p1_and_p2_lost_rabbits_as_p2() {
        let game_state: GameState = "
            1g
              +-----------------+
             8|                 |
             7|   e             |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|   E             |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(game_state.is_terminal(), Some(Value([0.0, 1.0])));
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
        assert_eq!(game_state.unwrap_play_phase().push_pull_state, PushPullState::PossiblePull(Square::new('c', 4), Piece::Elephant));

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Left));
        assert_eq!(game_state.unwrap_play_phase().push_pull_state, PushPullState::None);
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
        assert_eq!(game_state.unwrap_play_phase().push_pull_state, PushPullState::PossiblePull(Square::new('c', 4), Piece::Elephant));

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Left));
        assert_eq!(game_state.unwrap_play_phase().push_pull_state, PushPullState::None);
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
        assert_eq!(game_state.unwrap_play_phase().push_pull_state, PushPullState::PossiblePull(Square::new('c', 4), Piece::Elephant));

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Left));
        assert_eq!(game_state.unwrap_play_phase().push_pull_state, PushPullState::MustCompletePush(Square::new('d', 4), Piece::Elephant));
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
        assert_eq!(game_state.unwrap_play_phase().push_pull_state, PushPullState::PossiblePull(Square::new('c', 4), Piece::Camel));

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Left));
        assert_eq!(game_state.unwrap_play_phase().push_pull_state, PushPullState::MustCompletePush(Square::new('d', 4), Piece::Elephant));
    }

    #[test]
    fn test_action_pass() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('d', 7), Direction::Down));

        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Pass);

        assert_eq!(game_state.unwrap_play_phase().actions_this_turn, 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_valid_actions() {
        let game_state = initial_play_state();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[a2n, b2n, c2n, d2n, e2n, f2n, g2n, h2n]");
    }

    #[test]
    fn test_valid_actions_p2() {
        let game_state: GameState = "
             1s
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
            ".parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[a7s, b7s, c7s, d7s, e7s, f7s, g7s, h7s]");
    }

    #[test]
    fn test_valid_actions_2() {
        let game_state: GameState = "
              +-----------------+
             8| h c d m e d c h |
             7| r r r r r r r r |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x   R x     |
             2| R R R R   R R R |
             1| H C D M E D C H |
              +-----------------+
                a b c d e f g h
            ".parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[e3n, a2n, b2n, c2n, d2n, f2n, g2n, h2n, e1n, e3e, d2e, e3w, f2w]");
    }

    #[test]
    fn test_valid_actions_3() {
        let game_state: GameState = "
             1s
              +-----------------+
             8| h c d m e d c h |
             7| r r r r   r r   |
             6|     x   r x   r |
             5|                 |
             4|                 |
             3|     x     x     |
             2| R R R R R R R R |
             1| H C D M E D C H |
              +-----------------+
                a b c d e f g h
            ".parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d7e, g7e, e6e, e8s, h8s, a7s, b7s, c7s, d7s, f7s, g7s, e6s, h6s, f7w, e6w, h6w]");
    }

    #[test]
    fn test_valid_actions_4() {
        let game_state: GameState = "
             1g
              +-----------------+
             8| h c d m e d c h |
             7| r r r r r r r r |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x   E x     |
             2| R R R R   R R R |
             1| H C D M R D C H |
              +-----------------+
                a b c d e f g h
            ".parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[e3n, a2n, b2n, c2n, d2n, f2n, g2n, h2n, e1n, e3e, d2e, e3s, e3w, f2w]");
    }

    #[test]
    fn test_valid_actions_5() {
        let game_state: GameState = "
             1s
              +-----------------+
             8| h c d m r d c h |
             7| r r r r   r r r |
             6|     x   e x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2| R R R R E R R R |
             1| H C D M R D C H |
              +-----------------+
                a b c d e f g h
            ".parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[e6n, d7e, e6e, e8s, a7s, b7s, c7s, d7s, f7s, g7s, h7s, e6s, f7w, e6w]");
    }

    #[test]
    fn test_valid_actions_6() {
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

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[b3n, d3n, e3n, f3n, g3n, a2n, c2n, h2n, b1n, d1n, e1n, g1n, a4e, c4e, b3e, g3e, a2e, c2e, f2e, b1e, e1e, g1e, a4s, c4s, h4s, a2s, c2s, f2s, h2s, c4w, h4w, b3w, d3w, c2w, f2w, h2w, b1w, d1w, g1w, a5n, c5n, h5n, a5e, c5e, c5w, h5w]");
    }

    #[test]
    fn test_valid_actions_7() {
        let game_state: GameState = "
             1s
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

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[a7n, c7n, f7n, h7n, f5n, b8e, e8e, g8e, a7e, c7e, f7e, b6e, e6e, g6e, f5e, b8s, d8s, e8s, g8s, a7s, c7s, f7s, h7s, b6s, d6s, e6s, g6s, f5s, b8w, d8w, g8w, c7w, f7w, h7w, b6w, d6w, g6w, f5w]");
    }

    #[test]
    fn test_valid_actions_frozen_piece_p1() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|                 |
             7|                 |
             6|     x     x     |
             5|       e         |
             4|       M         |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[a1n, a1e]");
    }

    #[test]
    fn test_valid_actions_frozen_piece_p1_2() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|                 |
             7|                 |
             6|     x     x     |
             5|       m         |
             4|       M         |
             3|     x     x     |
             2|                 |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d4e, d4s, d4w]");
    }

    #[test]
    fn test_valid_actions_frozen_piece_p2() {
        let game_state: GameState = "
             1s
              +-----------------+
             8|                 |
             7|                 |
             6|     x     x     |
             5|       m         |
             4|       M         |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d5n, d5e, d5w]");
    }

    #[test]
    fn test_valid_actions_frozen_piece_p2_2() {
        let game_state: GameState = "
             1s
              +-----------------+
             8| r               |
             7|                 |
             6|     x     x     |
             5|       h         |
             4|       M         |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[a8e, a8s]");
    }

    #[test]
    fn test_valid_actions_frozen_piece_p2_3() {
        let game_state: GameState = "
             1s
              +-----------------+
             8| r               |
             7|                 |
             6|     x     x     |
             5|       r         |
             4|       M         |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[a8e, a8s]");
    }

    #[test]
    fn test_valid_actions_frozen_piece_p2_4() {
        let game_state: GameState = "
             1s
              +-----------------+
             8|                 |
             7|                 |
             6|     x     x     |
             5|       e         |
             4|       E         |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d5n, d5e, d5w]");
    }

    #[test]
    fn test_valid_actions_frozen_push_p2_1() {
        let game_state: GameState = "
             1s
              +-----------------+
             8|                 |
             7|                 |
             6|     x     x     |
             5|       e         |
             4|       M         |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d5n, d5e, d5w, d4e, d4s, d4w]");
    }


    #[test]
    fn test_valid_actions_frozen_push_p2_2() {
        let game_state: GameState = "
             1s
              +-----------------+
             8| r               |
             7|                 |
             6|     x     x     |
             5|       e         |
             4|       M         |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d5n, a8e, d5e, a8s, d5w, d4e, d4s, d4w]");

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Right));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d5s]");

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Down));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d4n, a8e, a8s, d4s, d4w, p, e4n, e4e, e4s]");
    }

    #[test]
    fn test_valid_actions_frozen_push_p2_3() {
        let game_state: GameState = "
             1s
              +-----------------+
             8| r               |
             7|                 |
             6|     x     x     |
             5|       e         |
             4|       R         |
             3|     x c   x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d5n, a8e, d5e, d3e, a8s, d3s, d5w, d3w, d4e, d4w]");

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Right));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d3n, d5s]");

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Down));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d4n, a8e, d3e, a8s, d3s, d4w, d3w, p, e4n, e4e, e4s]");
    }

    #[test]
    fn test_valid_actions_frozen_push_p2_4() {
        let game_state: GameState = "
             1s
              +-----------------+
             8| r               |
             7|                 |
             6|     x     x     |
             5|       e         |
             4|     E R         |
             3|     x c   x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d5n, a8e, d5e, d3e, a8s, d3s, d5w, d3w, d4e]");

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Right));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d3n, d5s]");

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Down));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d4n, a8e, d3e, a8s, d3s, d3w, p, e4n, e4e, e4s]");
    }

    #[test]
    fn test_valid_actions_supported_piece_p2_1() {
        let game_state: GameState = "
             1s
              +-----------------+
             8|                 |
             7|                 |
             6|     x     x     |
             5|     r m         |
             4|     C E         |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d5n, d5e, c5w]");
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

        assert_eq!(game_state.move_number, 1);
        assert_eq!(game_state.p1_turn_to_move, true);
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

        assert_eq!(game_state.move_number, 5);
        assert_eq!(game_state.p1_turn_to_move, false);
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

        assert_eq!(game_state.move_number, 176);
        assert_eq!(game_state.p1_turn_to_move, false);
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

        assert_eq!(game_state.move_number, 13);
        assert_eq!(game_state.p1_turn_to_move, true);
    }

    #[test]
    fn test_gamestate_to_str() {
        let expected_output = "1g
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
