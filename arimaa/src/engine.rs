use super::action::{map_bit_board_to_squares, Action, Direction, Piece, Square};
use super::constants::{BOARD_WIDTH, MAX_NUMBER_OF_MOVES};
use super::value::Value;
use super::zobrist::Zobrist;
use common::bits::first_set_bit;
use common::linked_list::List;
use engine::engine::GameEngine;
use engine::game_state::GameState as GameStateTrait;
use std::hash::Hash;
use std::hash::Hasher;

const LEFT_COLUMN_MASK: u64 =
    0b__00000001__00000001__00000001__00000001__00000001__00000001__00000001__00000001;
const RIGHT_COLUMN_MASK: u64 =
    0b__10000000__10000000__10000000__10000000__10000000__10000000__10000000__10000000;

const TOP_ROW_MASK: u64 =
    0b__00000000__00000000__00000000__00000000__00000000__00000000__00000000__11111111;
const BOTTOM_ROW_MASK: u64 =
    0b__11111111__00000000__00000000__00000000__00000000__00000000__00000000__00000000;

const P1_OBJECTIVE_MASK: u64 = TOP_ROW_MASK;
const P2_OBJECTIVE_MASK: u64 = BOTTOM_ROW_MASK;

const P1_PLACEMENT_MASK: u64 =
    0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000;
const P2_PLACEMENT_MASK: u64 =
    0b__00000000__00000000__00000000__00000000__00000000__00000000__11111111__11111111;
const LAST_P1_PLACEMENT_MASK: u64 =
    0b__10000000__00000000__00000000__00000000__00000000__00000000__00000000__00000000;
const LAST_P2_PLACEMENT_MASK: u64 =
    0b__00000000__00000000__00000000__00000000__00000000__00000000__10000000__00000000;

const TRAP_MASK: u64 =
    0b__00000000__00000000__00100100__00000000__00000000__00100100__00000000__00000000;

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
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
    MustCompletePush(Square, Piece),
}

#[derive(Clone, Debug)]
pub struct PlayPhase {
    previous_piece_boards_this_move: Vec<PieceBoard>,
    push_pull_state: PushPullState,
    initial_hash_of_move: Zobrist,
    hash_history: List<Zobrist>,
    piece_trapped_this_turn: bool,
}

#[derive(Clone, Debug)]
pub enum Phase {
    PlacePhase,
    PlayPhase(PlayPhase),
}

#[derive(Clone, Debug)]
pub struct PieceBoardState {
    p1_pieces: u64,
    all_pieces: u64,
    elephants: u64,
    camels: u64,
    horses: u64,
    dogs: u64,
    cats: u64,
    rabbits: u64,
}

impl PieceBoardState {
    pub fn get_bits_for_piece(&self, piece: Piece, p1_pieces: bool) -> u64 {
        let player_piece_mask = if p1_pieces {
            self.p1_pieces
        } else {
            !self.p1_pieces & self.all_pieces
        };
        self.get_bits_by_piece_type(piece) & player_piece_mask
    }

    pub fn get_player_piece_mask(&self, p1_pieces: bool) -> u64 {
        if p1_pieces {
            self.p1_pieces
        } else {
            !self.p1_pieces & self.all_pieces
        }
    }

    pub fn get_bits_by_piece_type(&self, piece: Piece) -> u64 {
        match piece {
            Piece::Elephant => self.elephants,
            Piece::Camel => self.camels,
            Piece::Horse => self.horses,
            Piece::Dog => self.dogs,
            Piece::Cat => self.cats,
            Piece::Rabbit => self.rabbits,
        }
    }

    pub fn get_placement_bit(&self) -> u64 {
        let placement_mask = if self.p1_pieces & P1_PLACEMENT_MASK == P1_PLACEMENT_MASK {
            P2_PLACEMENT_MASK
        } else {
            P1_PLACEMENT_MASK
        };
        let squares_to_place = !self.all_pieces & placement_mask;

        first_set_bit(squares_to_place)
    }

    pub fn get_trapped_piece_bits(&self) -> u64 {
        let animal_is_on_trap = animal_is_on_trap(self);

        if animal_is_on_trap {
            let unsupported_piece_bits = get_both_player_unsupported_piece_bits(self);
            unsupported_piece_bits & TRAP_MASK
        } else {
            0
        }
    }

    pub fn get_piece_type_at_square(&self, square: &Square) -> Option<Piece> {
        let square_bit = square.as_bit_board();
        if square_bit & self.all_pieces != 0 {
            Some(get_piece_type_at_bit(square_bit, self))
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct PieceBoard(PieceBoardState);

impl PieceBoard {
    pub fn initial() -> Self {
        Self(PieceBoardState {
            p1_pieces: 0,
            all_pieces: 0,
            elephants: 0,
            camels: 0,
            horses: 0,
            dogs: 0,
            cats: 0,
            rabbits: 0,
        })
    }

    pub fn new(
        p1_pieces: u64,
        elephants: u64,
        camels: u64,
        horses: u64,
        dogs: u64,
        cats: u64,
        rabbits: u64,
    ) -> Self {
        Self(PieceBoardState {
            p1_pieces,
            elephants,
            camels,
            horses,
            dogs,
            cats,
            rabbits,
            all_pieces: elephants | camels | horses | dogs | cats | rabbits,
        })
    }

    pub fn get_piece_board(&self) -> &PieceBoardState {
        &self.0
    }

    fn take_action(&self, move_action: &Action) -> (PieceBoardState, bool) {
        let mut piece_board_state = self.0.clone();
        let mut animal_was_trapped = false;

        if let Action::Move(square, direction) = move_action {
            Self::move_piece(&mut piece_board_state, square, direction);
            animal_was_trapped = Self::remove_trapped_pieces(&mut piece_board_state);
        }

        (piece_board_state, animal_was_trapped)
    }

    fn move_piece(piece_board_state: &mut PieceBoardState, square: &Square, direction: &Direction) {
        let source_square_bit = square.as_bit_board();
        let piece_board = piece_board_state;

        piece_board.elephants =
            shift_piece_in_direction(piece_board.elephants, source_square_bit, direction);
        piece_board.camels =
            shift_piece_in_direction(piece_board.camels, source_square_bit, direction);
        piece_board.horses =
            shift_piece_in_direction(piece_board.horses, source_square_bit, direction);
        piece_board.dogs = shift_piece_in_direction(piece_board.dogs, source_square_bit, direction);
        piece_board.cats = shift_piece_in_direction(piece_board.cats, source_square_bit, direction);
        piece_board.rabbits =
            shift_piece_in_direction(piece_board.rabbits, source_square_bit, direction);
        piece_board.p1_pieces =
            shift_piece_in_direction(piece_board.p1_pieces, source_square_bit, direction);
        piece_board.all_pieces =
            shift_piece_in_direction(piece_board.all_pieces, source_square_bit, direction);
    }

    fn remove_trapped_pieces(piece_board_state: &mut PieceBoardState) -> bool {
        let trapped_animal_bits = piece_board_state.get_trapped_piece_bits();
        let animal_is_trapped = trapped_animal_bits != 0;

        if animal_is_trapped {
            let untrapped_animal_bits = !trapped_animal_bits;
            piece_board_state.elephants &= untrapped_animal_bits;
            piece_board_state.camels &= untrapped_animal_bits;
            piece_board_state.horses &= untrapped_animal_bits;
            piece_board_state.dogs &= untrapped_animal_bits;
            piece_board_state.cats &= untrapped_animal_bits;
            piece_board_state.rabbits &= untrapped_animal_bits;
            piece_board_state.p1_pieces &= untrapped_animal_bits;
            piece_board_state.all_pieces &= untrapped_animal_bits;
        }

        animal_is_trapped
    }
}

#[derive(Clone, Debug)]
pub struct GameState {
    p1_turn_to_move: bool,
    move_number: usize,
    phase: Phase,
    piece_board: PieceBoard,
    hash: Zobrist,
}

impl Hash for GameState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash.board_state_hash());
        state.finish();
    }
}

impl PartialEq for GameState {
    fn eq(&self, other: &GameState) -> bool {
        self.hash.board_state_hash() == other.hash.board_state_hash()
    }
}

impl Eq for GameState {}

impl GameState {
    pub fn new(
        p1_turn_to_move: bool,
        move_number: usize,
        phase: Phase,
        piece_board: PieceBoard,
        hash: Zobrist,
    ) -> Self {
        GameState {
            p1_turn_to_move,
            move_number,
            phase,
            piece_board,
            hash,
        }
    }

    pub fn take_action(&self, action: &Action) -> Self {
        match action {
            Action::Pass => self.pass(),
            Action::Place(piece) => self.place(*piece),
            Action::Move(square, direction) => self.move_piece(square, direction),
        }
    }

    pub fn is_terminal(&self) -> Option<Value> {
        self.as_play_phase().and_then(|play_phase| {
            let piece_board = &self.get_piece_board();

            // The order of checking for win/lose conditions is as follows assuming player A just made the move and player B now needs to move:
            if play_phase.get_step() > 0 {
                return self.has_move(piece_board);
            }

            // Check if a rabbit of player A reached goal. If so player A wins.
            // Check if a rabbit of player B reached goal. If so player B wins.
            self.rabbit_at_goal(piece_board)
                // Check if player B lost all rabbits. If so player A wins.
                // Check if player A lost all rabbits. If so player B wins.
                .or_else(|| self.lost_all_rabbits(piece_board))
                // Check if player B has no possible move (all pieces are frozen or have no place to move). If so player A wins.
                // Check if the only moves player B has are 3rd time repetitions. If so player A wins.
                .or_else(|| self.has_move(piece_board))
                .or_else(|| self.max_moves_reached())
        })
    }

    #[allow(
        clippy::blocks_in_if_conditions,
        clippy::if_same_then_else,
        clippy::needless_bool
    )]
    pub fn has_move(&self, piece_board: &PieceBoardState) -> Option<Value> {
        let has_move = if let Phase::PlayPhase(play_phase) = &self.phase {
            if play_phase.push_pull_state.is_must_complete_push() {
                let valid_actions = self.get_must_complete_push_actions(piece_board);
                self.has_non_passing_like_action(valid_actions)
            } else if self.can_pass(true) {
                true
            } else if {
                let mut valid_actions = Vec::new();
                self.extend_with_valid_curr_player_piece_moves(&mut valid_actions, piece_board);
                self.has_non_passing_like_action(valid_actions)
            } {
                true
            } else if {
                let mut valid_actions = Vec::new();
                self.extend_with_pull_piece_actions(&mut valid_actions, piece_board);
                self.has_non_passing_like_action(valid_actions)
            } {
                true
            } else if {
                let mut valid_actions = Vec::new();
                self.extend_with_push_piece_actions(&mut valid_actions, piece_board);
                self.has_non_passing_like_action(valid_actions)
            } {
                true
            } else {
                false
            }
        } else {
            true
        };

        if has_move {
            None
        } else if self.p1_turn_to_move {
            Some(Value([0.0, 1.0]))
        } else {
            Some(Value([1.0, 0.0]))
        }
    }

    pub fn max_moves_reached(&self) -> Option<Value> {
        if self.move_number > MAX_NUMBER_OF_MOVES {
            Some(Value([0.0, 0.0]))
        } else {
            None
        }
    }

    pub fn valid_actions(&self) -> Vec<Action> {
        self.valid_actions_(true)
    }

    pub fn valid_actions_no_exclusions(&self) -> Vec<Action> {
        self.valid_actions_(false)
    }

    pub fn get_piece_board_for_step(&self, step: usize) -> &PieceBoardState {
        if step == self.get_current_step() {
            return self.piece_board.get_piece_board();
        }

        let previous_piece_board = &self.unwrap_play_phase().previous_piece_boards_this_move[step];

        previous_piece_board.get_piece_board()
    }

    pub fn get_piece_board(&self) -> &PieceBoardState {
        self.piece_board.get_piece_board()
    }

    pub fn is_p1_turn_to_move(&self) -> bool {
        self.p1_turn_to_move
    }

    pub fn get_move_number(&self) -> usize {
        self.move_number
    }

    pub fn get_current_step(&self) -> usize {
        self.unwrap_play_phase().get_step()
    }

    pub fn is_play_phase(&self) -> bool {
        matches!(&self.phase, Phase::PlayPhase(_))
    }

    pub fn get_trapped_animal_for_action(&self, action: &Action) -> Option<(Square, Piece, bool)> {
        let mut piece_board_state = self.get_piece_board().clone();
        if let Action::Move(square, direction) = action {
            PieceBoard::move_piece(&mut piece_board_state, square, direction);
            let trapped_animal_bits = piece_board_state.get_trapped_piece_bits();
            if trapped_animal_bits != 0 {
                let square = Square::from_bit_board(trapped_animal_bits);
                let piece = piece_board_state.get_piece_type_at_square(&square).unwrap();
                let is_p1_piece =
                    piece_board_state.get_bits_for_piece(piece, true) & square.as_bit_board() != 0;

                return Some((square, piece, is_p1_piece));
            }
        }

        None
    }

    pub fn get_transposition_hash(&self) -> u64 {
        match &self.phase {
            Phase::PlayPhase(play_phase) => self
                .hash
                .board_state_hash_with_push_pull_state(play_phase.push_pull_state),
            Phase::PlacePhase => self.hash.board_state_hash(),
        }
    }

    pub fn valid_actions_(&self, check_repititions: bool) -> Vec<Action> {
        if let Phase::PlayPhase(play_phase) = &self.phase {
            let piece_board = self.get_piece_board();
            let mut valid_actions = if play_phase.push_pull_state.is_must_complete_push() {
                self.get_must_complete_push_actions(piece_board)
            } else {
                let mut valid_actions = Vec::with_capacity(50);
                self.extend_with_push_piece_actions(&mut valid_actions, piece_board);
                self.extend_with_pull_piece_actions(&mut valid_actions, piece_board);
                self.extend_with_valid_curr_player_piece_moves(&mut valid_actions, piece_board);

                if self.can_pass(check_repititions) {
                    valid_actions.push(Action::Pass);
                }

                valid_actions
            };

            if check_repititions {
                self.remove_passing_like_actions(&mut valid_actions);
            }

            valid_actions
        } else {
            self.valid_placement()
        }
    }

    fn can_pass(&self, check_repititions: bool) -> bool {
        self.as_play_phase().map_or(false, |play_phase| {
            play_phase.get_step() >= 1
                && !play_phase.push_pull_state.is_must_complete_push()
                && (!check_repititions
                    || ((self.unwrap_play_phase().initial_hash_of_move
                        != self.hash.exclude_step(play_phase.get_step()))
                        && !hash_history_contains_hash_twice(
                            &play_phase.hash_history,
                            &self.hash.pass(play_phase.get_step()),
                        )))
        })
    }

    fn valid_placement(&self) -> Vec<Action> {
        let mut actions = Vec::with_capacity(6);
        let piece_board = &self.get_piece_board();
        let curr_player_pieces = self.get_curr_player_piece_mask(piece_board);

        if piece_board.elephants & curr_player_pieces == 0 {
            actions.push(Action::Place(Piece::Elephant));
        }
        if piece_board.camels & curr_player_pieces == 0 {
            actions.push(Action::Place(Piece::Camel));
        }
        if (piece_board.horses & curr_player_pieces).count_ones() < 2 {
            actions.push(Action::Place(Piece::Horse));
        }
        if (piece_board.dogs & curr_player_pieces).count_ones() < 2 {
            actions.push(Action::Place(Piece::Dog));
        }
        if (piece_board.cats & curr_player_pieces).count_ones() < 2 {
            actions.push(Action::Place(Piece::Cat));
        }
        if (piece_board.rabbits & curr_player_pieces).count_ones() < 8 {
            actions.push(Action::Place(Piece::Rabbit));
        }

        actions
    }

    fn extend_with_valid_curr_player_piece_moves(
        &self,
        valid_actions: &mut Vec<Action>,
        piece_board: &PieceBoardState,
    ) {
        let non_frozen_pieces = self.get_curr_player_non_frozen_pieces(piece_board);

        for direction in [
            Direction::Up,
            Direction::Right,
            Direction::Down,
            Direction::Left,
        ]
        .iter()
        {
            let unoccupied_directions = can_move_in_direction(direction, piece_board);
            let invalid_rabbit_moves = self.get_invalid_rabbit_moves(direction, piece_board);
            let valid_curr_piece_moves =
                unoccupied_directions & non_frozen_pieces & !invalid_rabbit_moves;

            if valid_curr_piece_moves != 0 {
                let squares = map_bit_board_to_squares(valid_curr_piece_moves);
                valid_actions.extend(squares.into_iter().map(|s| Action::Move(s, *direction)));
            }
        }
    }

    fn extend_with_pull_piece_actions(
        &self,
        valid_actions: &mut Vec<Action>,
        piece_board: &PieceBoardState,
    ) {
        if let Some(play_phase) = self.as_play_phase() {
            if let Some((square, piece)) = play_phase.push_pull_state.as_possible_pull() {
                let opp_piece_mask = self.get_opponent_piece_mask(piece_board);
                let lesser_opp_pieces = self.get_lesser_pieces(piece, piece_board) & opp_piece_mask;
                let square_bit = square.as_bit_board();

                for direction in [
                    Direction::Up,
                    Direction::Right,
                    Direction::Down,
                    Direction::Left,
                ]
                .iter()
                {
                    if shift_pieces_in_direction(lesser_opp_pieces, &direction) & square_bit != 0 {
                        let source_opp_piece_square = Square::from_bit_board(
                            shift_pieces_in_opp_direction(square_bit, direction),
                        );
                        let action = Action::Move(source_opp_piece_square, *direction);
                        if !valid_actions.contains(&action) {
                            valid_actions.push(action);
                        }
                    }
                }
            }
        }
    }

    fn extend_with_push_piece_actions(
        &self,
        valid_actions: &mut Vec<Action>,
        piece_board: &PieceBoardState,
    ) {
        if let Some(play_phase) = self.as_play_phase() {
            if play_phase.push_pull_state.can_push() && play_phase.get_step() < 3 {
                let predator_piece_mask = self.get_curr_player_non_frozen_pieces(piece_board);
                let opp_piece_mask = self.get_opponent_piece_mask(piece_board);
                let opp_threatened_pieces =
                    self.get_threatened_pieces(predator_piece_mask, opp_piece_mask, piece_board);

                if opp_threatened_pieces != 0 {
                    for direction in [
                        Direction::Up,
                        Direction::Right,
                        Direction::Down,
                        Direction::Left,
                    ]
                    .iter()
                    {
                        let unoccupied_directions = can_move_in_direction(direction, piece_board);
                        let valid_push_moves = unoccupied_directions & opp_threatened_pieces;

                        if valid_push_moves != 0 {
                            let squares = map_bit_board_to_squares(valid_push_moves);
                            valid_actions
                                .extend(squares.into_iter().map(|s| Action::Move(s, *direction)));
                        }
                    }
                }
            }
        }
    }

    fn get_must_complete_push_actions(&self, piece_board: &PieceBoardState) -> Vec<Action> {
        let play_phase = self.unwrap_play_phase();
        let (square, pushed_piece) = play_phase.push_pull_state.unwrap_must_complete_push();

        let curr_player_non_frozen_piece_mask = self.get_curr_player_non_frozen_pieces(piece_board);
        let square_bit = square.as_bit_board();

        let mut valid_actions = vec![];
        for direction in [
            Direction::Up,
            Direction::Right,
            Direction::Down,
            Direction::Left,
        ]
        .iter()
        {
            let pushing_piece_bit = shift_pieces_in_opp_direction(square_bit, &direction)
                & curr_player_non_frozen_piece_mask;
            if pushing_piece_bit != 0
                && get_piece_type_at_bit(pushing_piece_bit, piece_board) > pushed_piece
            {
                valid_actions.push(Action::Move(
                    Square::from_bit_board(pushing_piece_bit),
                    *direction,
                ));
            }
        }

        valid_actions
    }

    fn place(&self, piece: Piece) -> Self {
        let piece_board = &self.get_piece_board();
        let placement_bit = piece_board.get_placement_bit();

        let mut new_elephants = piece_board.elephants;
        let mut new_camels = piece_board.camels;
        let mut new_horses = piece_board.horses;
        let mut new_dogs = piece_board.dogs;
        let mut new_cats = piece_board.cats;
        let mut new_rabbits = piece_board.rabbits;

        match piece {
            Piece::Elephant => new_elephants |= placement_bit,
            Piece::Camel => new_camels |= placement_bit,
            Piece::Horse => new_horses |= placement_bit,
            Piece::Dog => new_dogs |= placement_bit,
            Piece::Cat => new_cats |= placement_bit,
            Piece::Rabbit => new_rabbits |= placement_bit,
        }

        let new_p1_pieces = piece_board.p1_pieces
            | if self.p1_turn_to_move {
                placement_bit
            } else {
                0
            };
        let new_piece_board = PieceBoard::new(
            new_p1_pieces,
            new_elephants,
            new_camels,
            new_horses,
            new_dogs,
            new_cats,
            new_rabbits,
        );

        let switch_players = placement_bit == LAST_P1_PLACEMENT_MASK;
        let switch_phases = placement_bit == LAST_P2_PLACEMENT_MASK;
        let new_p1_turn_to_move = if switch_players {
            false
        } else if switch_phases {
            true
        } else {
            self.p1_turn_to_move
        };
        let new_hash = self.hash.place_piece(
            piece,
            Square::from_bit_board(placement_bit),
            self.p1_turn_to_move,
            switch_players,
            switch_phases,
        );
        let new_phase = if switch_phases {
            let hash_history = List::new();
            let hash_history = hash_history.append(new_hash);
            Phase::PlayPhase(PlayPhase::initial(new_hash, hash_history))
        } else {
            Phase::PlacePhase
        };

        let new_move_number = if switch_phases { 2 } else { 1 };

        Self {
            p1_turn_to_move: new_p1_turn_to_move,
            phase: new_phase,
            piece_board: new_piece_board,
            move_number: new_move_number,
            hash: new_hash,
        }
    }

    fn pass(&self) -> Self {
        let hash = self.hash.pass(self.get_current_step());
        let play_phase = self.unwrap_play_phase();
        let new_hash_history = if play_phase.piece_trapped_this_turn {
            List::new()
        } else {
            play_phase.hash_history.clone()
        };
        let new_hash_history = new_hash_history.append(hash);

        GameState {
            phase: Phase::PlayPhase(PlayPhase::initial(hash, new_hash_history)),
            p1_turn_to_move: !self.p1_turn_to_move,
            move_number: self.move_number + if self.p1_turn_to_move { 0 } else { 1 },
            piece_board: self.piece_board.clone(),
            hash,
        }
    }

    fn move_piece(&self, square: &Square, direction: &Direction) -> Self {
        let curr_play_phase = self.unwrap_play_phase();
        let curr_step = self.get_current_step();
        let is_last_step = curr_step >= 3;
        let new_action = Action::Move(*square, *direction);
        let (new_piece_board_state, new_animal_was_trapped) =
            self.piece_board.take_action(&new_action);
        let new_p1_turn_to_move = if is_last_step {
            !self.p1_turn_to_move
        } else {
            self.p1_turn_to_move
        };
        let new_step = if is_last_step { 0 } else { curr_step + 1 };
        let new_move_number = self.move_number
            + if is_last_step && new_p1_turn_to_move {
                1
            } else {
                0
            };
        let new_hash =
            self.hash
                .move_piece(self, &new_piece_board_state, new_step, new_p1_turn_to_move);
        let new_hash_history = if new_animal_was_trapped {
            List::new()
        } else {
            curr_play_phase.hash_history.clone()
        };

        let new_play_phase = if is_last_step {
            let hash_history = new_hash_history.append(new_hash);
            PlayPhase::initial(new_hash, hash_history)
        } else {
            let new_previous_piece_boards_this_move = self.get_next_piece_boards_this_move();
            let new_push_pull_state = self.get_next_push_pull_state(square, direction);

            PlayPhase {
                initial_hash_of_move: curr_play_phase.initial_hash_of_move,
                push_pull_state: new_push_pull_state,
                hash_history: new_hash_history,
                previous_piece_boards_this_move: new_previous_piece_boards_this_move,
                piece_trapped_this_turn: curr_play_phase.piece_trapped_this_turn
                    | new_animal_was_trapped,
            }
        };

        GameState {
            p1_turn_to_move: new_p1_turn_to_move,
            move_number: new_move_number,
            phase: Phase::PlayPhase(new_play_phase),
            piece_board: PieceBoard(new_piece_board_state),
            hash: new_hash,
        }
    }

    fn unwrap_play_phase(&self) -> &PlayPhase {
        self.as_play_phase()
            .expect("Expected phase to be PlayPhase")
    }

    pub fn as_play_phase(&self) -> Option<&PlayPhase> {
        match &self.phase {
            Phase::PlayPhase(play_phase) => Some(play_phase),
            _ => None,
        }
    }

    fn get_next_piece_boards_this_move(&self) -> Vec<PieceBoard> {
        let play_phase = self.unwrap_play_phase();
        let step = play_phase.get_step();

        let mut previous_piece_boards_this_move = Vec::with_capacity(step + 1);
        play_phase
            .previous_piece_boards_this_move
            .clone_into(&mut previous_piece_boards_this_move);
        previous_piece_boards_this_move.push(self.piece_board.clone());
        previous_piece_boards_this_move
    }

    fn get_next_push_pull_state(&self, square: &Square, direction: &Direction) -> PushPullState {
        let source_square_bit = square.as_bit_board();
        let piece_board = &self.get_piece_board();
        let is_opponent_piece = self.is_their_piece(source_square_bit, piece_board);
        let play_phase = self.unwrap_play_phase();
        let piece_type_at_bit = get_piece_type_at_bit(source_square_bit, piece_board);

        // Check if previous move can count as a pull, if so, do that.
        // Otherwise state that it must be followed with a push.
        if is_opponent_piece
            && !self.move_can_be_counted_as_pull(source_square_bit, direction, piece_board)
        {
            PushPullState::MustCompletePush(*square, piece_type_at_bit)
        } else if !is_opponent_piece
            && !play_phase.push_pull_state.is_must_complete_push()
            && piece_type_at_bit != Piece::Rabbit
        {
            PushPullState::PossiblePull(*square, piece_type_at_bit)
        } else {
            PushPullState::None
        }
    }

    fn move_can_be_counted_as_pull(
        &self,
        new_move_square_bit: u64,
        direction: &Direction,
        piece_board: &PieceBoardState,
    ) -> bool {
        let play_phase = self.unwrap_play_phase();
        if let PushPullState::PossiblePull(prev_move_square, my_piece) = &play_phase.push_pull_state
        {
            if prev_move_square.as_bit_board() == shift_in_direction(new_move_square_bit, direction)
            {
                let their_piece = get_piece_type_at_bit(new_move_square_bit, piece_board);
                if my_piece > &their_piece {
                    return true;
                }
            }
        }

        false
    }

    fn is_their_piece(&self, square_bit: u64, piece_board: &PieceBoardState) -> bool {
        let is_p1_piece = square_bit & piece_board.p1_pieces != 0;
        self.p1_turn_to_move ^ is_p1_piece
    }

    fn get_curr_player_non_frozen_pieces(&self, piece_board: &PieceBoardState) -> u64 {
        let opp_piece_mask = self.get_opponent_piece_mask(piece_board);
        let curr_player_piece_mask = !opp_piece_mask & piece_board.all_pieces;
        let threatened_pieces =
            self.get_threatened_pieces(opp_piece_mask, curr_player_piece_mask, piece_board);

        curr_player_piece_mask & (!threatened_pieces | get_supported_pieces(curr_player_piece_mask))
    }

    fn get_threatened_pieces(
        &self,
        predator_piece_mask: u64,
        prey_piece_mask: u64,
        piece_board: &PieceBoardState,
    ) -> u64 {
        let predator_elephant_influence =
            get_influenced_squares(piece_board.elephants & predator_piece_mask);
        let predator_camel_influence =
            get_influenced_squares(piece_board.camels & predator_piece_mask);
        let predator_horse_influence =
            get_influenced_squares(piece_board.horses & predator_piece_mask);
        let predator_dog_influence = get_influenced_squares(piece_board.dogs & predator_piece_mask);
        let predator_cat_influence = get_influenced_squares(piece_board.cats & predator_piece_mask);

        let camel_threats = predator_elephant_influence;
        let horse_threats = camel_threats | predator_camel_influence;
        let dog_threats = horse_threats | predator_horse_influence;
        let cat_threats = dog_threats | predator_dog_influence;
        let rabbit_threats = cat_threats | predator_cat_influence;

        let threatened_pieces = (piece_board.camels & camel_threats)
            | (piece_board.horses & horse_threats)
            | (piece_board.dogs & dog_threats)
            | (piece_board.cats & cat_threats)
            | (piece_board.rabbits & rabbit_threats);

        threatened_pieces & prey_piece_mask
    }

    fn get_curr_player_piece_mask(&self, piece_board: &PieceBoardState) -> u64 {
        if self.p1_turn_to_move {
            piece_board.p1_pieces
        } else {
            !piece_board.p1_pieces & piece_board.all_pieces
        }
    }

    fn get_opponent_piece_mask(&self, piece_board: &PieceBoardState) -> u64 {
        if self.p1_turn_to_move {
            !piece_board.p1_pieces & piece_board.all_pieces
        } else {
            piece_board.p1_pieces
        }
    }

    fn rabbit_at_goal(&self, piece_board: &PieceBoardState) -> Option<Value> {
        let p1_objective_met = piece_board.p1_pieces & piece_board.rabbits & P1_OBJECTIVE_MASK != 0;
        let p2_objective_met =
            !piece_board.p1_pieces & piece_board.rabbits & P2_OBJECTIVE_MASK != 0;

        if p1_objective_met || p2_objective_met {
            // Objective is opposite of the player to move since we are checking if there is a winner after the turn is complete.
            // Logic should include the condition of if both players have a rabbit at the goal. In that case the player who was last to move wins.
            let last_to_move_is_p1 = !self.p1_turn_to_move;
            let last_to_move_objective_met = if last_to_move_is_p1 {
                p1_objective_met
            } else {
                p2_objective_met
            };
            let p1_won = !(last_to_move_is_p1 ^ last_to_move_objective_met);
            Some(if p1_won {
                Value([1.0, 0.0])
            } else {
                Value([0.0, 1.0])
            })
        } else {
            None
        }
    }

    fn lost_all_rabbits(&self, piece_board: &PieceBoardState) -> Option<Value> {
        let p1_lost_rabbits = piece_board.p1_pieces & piece_board.rabbits == 0;
        let p2_lost_rabbits = !piece_board.p1_pieces & piece_board.rabbits == 0;

        // Check if player B lost all rabbits. If so player A wins.
        // Check if player A lost all rabbits. If so player B wins.

        if p1_lost_rabbits || p2_lost_rabbits {
            // Objective is opposite of the player to move since we are checking if there is a winner after the turn is complete.
            // Logic should include the condition of if both players lost their rabbits. In that case the player who was last to move wins.
            let last_to_move_is_p1 = !self.p1_turn_to_move;
            let last_to_move_objective_met = if last_to_move_is_p1 {
                p2_lost_rabbits
            } else {
                p1_lost_rabbits
            };
            let p1_won = !(last_to_move_is_p1 ^ last_to_move_objective_met);
            Some(if p1_won {
                Value([1.0, 0.0])
            } else {
                Value([0.0, 1.0])
            })
        } else {
            None
        }
    }

    #[allow(clippy::let_and_return)]
    fn get_invalid_rabbit_moves(
        &self,
        direction: &Direction,
        piece_board: &PieceBoardState,
    ) -> u64 {
        let backward_direction = if self.p1_turn_to_move {
            Direction::Down
        } else {
            Direction::Up
        };

        if *direction == backward_direction {
            let players_rabbits = if self.p1_turn_to_move {
                piece_board.p1_pieces
            } else {
                !piece_board.p1_pieces
            } & piece_board.rabbits;
            players_rabbits
        } else {
            0
        }
    }

    fn get_lesser_pieces(&self, piece: Piece, piece_board: &PieceBoardState) -> u64 {
        match piece {
            Piece::Rabbit => 0,
            Piece::Cat => piece_board.rabbits,
            Piece::Dog => piece_board.rabbits | piece_board.cats,
            Piece::Horse => piece_board.rabbits | piece_board.cats | piece_board.dogs,
            Piece::Camel => {
                piece_board.rabbits | piece_board.cats | piece_board.dogs | piece_board.horses
            }
            Piece::Elephant => {
                piece_board.rabbits
                    | piece_board.cats
                    | piece_board.dogs
                    | piece_board.horses
                    | piece_board.camels
            }
        }
    }

    fn remove_passing_like_actions(&self, valid_actions: &mut Vec<Action>) {
        let play_phase = self.unwrap_play_phase();
        if play_phase.get_step() == 3 && !play_phase.piece_trapped_this_turn {
            valid_actions.retain(|action| !self.is_passing_like_action(action));
        }
    }

    fn is_passing_like_action(&self, action: &Action) -> bool {
        let play_phase = self.unwrap_play_phase();
        let initial_hash_of_move = play_phase.initial_hash_of_move;
        let hash_history = &play_phase.hash_history;

        if let Action::Move(_, _) = action {
            let new_piece_board = &self.piece_board.take_action(&action).0;
            let new_hash_no_player_switch =
                self.hash
                    .move_piece(self, new_piece_board, 0, self.is_p1_turn_to_move());
            let new_hash_switch_players =
                self.hash
                    .move_piece(self, new_piece_board, 0, !self.is_p1_turn_to_move());

            if new_hash_no_player_switch == initial_hash_of_move
                || hash_history_contains_hash_twice(hash_history, &new_hash_switch_players)
            {
                return true;
            }
        }

        false
    }

    fn has_non_passing_like_action(&self, valid_actions: Vec<Action>) -> bool {
        if valid_actions.is_empty() {
            return false;
        }

        let play_phase = self.unwrap_play_phase();
        if play_phase.get_step() < 3 || play_phase.piece_trapped_this_turn {
            return true;
        }

        for action in valid_actions.iter() {
            if !self.is_passing_like_action(action) {
                return true;
            }
        }

        false
    }
}

fn get_piece_type_at_bit(square_bit: u64, piece_board: &PieceBoardState) -> Piece {
    if piece_board.rabbits & square_bit != 0 {
        Piece::Rabbit
    } else if piece_board.elephants & square_bit != 0 {
        Piece::Elephant
    } else if piece_board.camels & square_bit != 0 {
        Piece::Camel
    } else if piece_board.horses & square_bit != 0 {
        Piece::Horse
    } else if piece_board.dogs & square_bit != 0 {
        Piece::Dog
    } else {
        Piece::Cat
    }
}

fn animal_is_on_trap(piece_board: &PieceBoardState) -> bool {
    (piece_board.all_pieces & TRAP_MASK) != 0
}

fn get_influenced_squares(piece_board: u64) -> u64 {
    shift_pieces_up!(piece_board)
        | shift_pieces_right!(piece_board)
        | shift_pieces_down!(piece_board)
        | shift_pieces_left!(piece_board)
}

fn get_both_player_unsupported_piece_bits(piece_board: &PieceBoardState) -> u64 {
    piece_board.all_pieces & !get_both_player_supported_pieces(piece_board)
}

fn get_both_player_supported_pieces(piece_board: &PieceBoardState) -> u64 {
    let p1_pieces = piece_board.p1_pieces;
    let p2_pieces = piece_board.all_pieces & !p1_pieces;

    get_supported_pieces(p1_pieces) | get_supported_pieces(p2_pieces)
}

fn get_supported_pieces(piece_bits: u64) -> u64 {
    let up_supported_pieces = piece_bits & shift_pieces_up!(piece_bits);
    let right_supported_pieces = piece_bits & shift_pieces_right!(piece_bits);
    let down_supported_pieces = piece_bits & shift_pieces_down!(piece_bits);
    let left_supported_pieces = piece_bits & shift_pieces_left!(piece_bits);

    up_supported_pieces | right_supported_pieces | down_supported_pieces | left_supported_pieces
}

fn can_move_in_direction(direction: &Direction, piece_board: &PieceBoardState) -> u64 {
    let empty_squares = !piece_board.all_pieces;
    shift_pieces_in_opp_direction(empty_squares, direction)
}

fn shift_piece_in_direction(
    piece_board: u64,
    source_square_bit: u64,
    direction: &Direction,
) -> u64 {
    shift_in_direction(piece_board & source_square_bit, direction)
        | piece_board & !source_square_bit
}

fn shift_pieces_in_opp_direction(bits: u64, direction: &Direction) -> u64 {
    match direction {
        Direction::Up => shift_pieces_down!(bits),
        Direction::Right => shift_pieces_left!(bits),
        Direction::Down => shift_pieces_up!(bits),
        Direction::Left => shift_pieces_right!(bits),
    }
}

fn shift_pieces_in_direction(bits: u64, direction: &Direction) -> u64 {
    match direction {
        Direction::Up => shift_pieces_up!(bits),
        Direction::Right => shift_pieces_right!(bits),
        Direction::Down => shift_pieces_down!(bits),
        Direction::Left => shift_pieces_left!(bits),
    }
}

fn shift_in_direction(bits: u64, direction: &Direction) -> u64 {
    match direction {
        Direction::Up => shift_up!(bits),
        Direction::Right => shift_right!(bits),
        Direction::Down => shift_down!(bits),
        Direction::Left => shift_left!(bits),
    }
}

fn hash_history_contains_hash_twice(hash_history: &List<Zobrist>, hash: &Zobrist) -> bool {
    hash_history.iter().filter(|h| *h == hash).count() >= 2
}

impl GameStateTrait for GameState {
    fn initial() -> Self {
        GameState {
            p1_turn_to_move: true,
            move_number: 1,
            piece_board: PieceBoard::initial(),
            phase: Phase::PlacePhase,
            hash: Zobrist::initial(),
        }
    }
}

impl PushPullState {
    fn is_must_complete_push(&self) -> bool {
        matches!(self, PushPullState::MustCompletePush(_, _))
    }

    fn unwrap_must_complete_push(&self) -> (Square, Piece) {
        match self {
            PushPullState::MustCompletePush(square, piece) => (*square, *piece),
            _ => panic!("Expected PushPullState to be MustCompletePush"),
        }
    }

    fn as_possible_pull(&self) -> Option<(Square, Piece)> {
        match self {
            PushPullState::PossiblePull(square, piece) => Some((*square, *piece)),
            _ => None,
        }
    }

    fn can_push(&self) -> bool {
        // We can't push another piece if we are already obligated to push another
        !matches!(self, PushPullState::MustCompletePush(_, _))
    }
}

impl PlayPhase {
    pub fn initial(initial_hash_of_move: Zobrist, hash_history: List<Zobrist>) -> Self {
        PlayPhase {
            previous_piece_boards_this_move: Vec::with_capacity(0),
            push_pull_state: PushPullState::None,
            initial_hash_of_move,
            hash_history,
            piece_trapped_this_turn: false,
        }
    }

    pub fn new(
        initial_hash_of_move: Zobrist,
        hash_history: List<Zobrist>,
        previous_piece_boards_this_move: Vec<PieceBoard>,
        push_pull_state: PushPullState,
        piece_trapped_this_turn: bool,
    ) -> Self {
        PlayPhase {
            previous_piece_boards_this_move,
            push_pull_state,
            initial_hash_of_move,
            hash_history,
            piece_trapped_this_turn,
        }
    }

    pub fn get_previous_piece_boards(&self) -> &[PieceBoard] {
        &self.previous_piece_boards_this_move
    }

    pub fn get_push_pull_state(&self) -> PushPullState {
        self.push_pull_state
    }

    pub fn get_piece_trapped_this_turn(&self) -> bool {
        self.piece_trapped_this_turn
    }

    fn get_step(&self) -> usize {
        self.previous_piece_boards_this_move.len()
    }
}

#[derive(Default)]
pub struct Engine {}

impl Engine {
    pub fn new() -> Self {
        Self {}
    }
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
        if game_state.p1_turn_to_move {
            1
        } else {
            2
        }
    }

    fn get_move_number(&self, game_state: &Self::State) -> usize {
        game_state.move_number
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::super::action::{Action, Piece, Square};
    use super::*;
    use engine::game_state::GameState as GameStateTrait;
    use test::Bencher;

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
        let piece_board = game_state.get_piece_board();

        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111111__00000000__00000000__00000000__00000000__11111111__00000000
        );
        assert_eq!(
            piece_board.cats,
            0b__01000010__00000000__00000000__00000000__00000000__00000000__00000000__01000010
        );
        assert_eq!(
            piece_board.dogs,
            0b__00100100__00000000__00000000__00000000__00000000__00000000__00000000__00100100
        );
        assert_eq!(
            piece_board.horses,
            0b__10000001__00000000__00000000__00000000__00000000__00000000__00000000__10000001
        );
        assert_eq!(
            piece_board.camels,
            0b__00001000__00000000__00000000__00000000__00000000__00000000__00000000__00001000
        );
        assert_eq!(
            piece_board.elephants,
            0b__00010000__00000000__00000000__00000000__00000000__00000000__00000000__00010000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000
        );

        assert_eq!(game_state.p1_turn_to_move, true);
        assert_eq!(game_state.unwrap_play_phase().get_step(), 0);
        assert_eq!(
            game_state.unwrap_play_phase().push_pull_state,
            PushPullState::None
        );
    }

    #[test]
    fn test_action_move_up() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Up));
        let piece_board = game_state.get_piece_board();

        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111110__00000001__00000000__00000000__00000000__11111111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111110__00000001__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 3), Direction::Up));
        let piece_board = game_state.get_piece_board();
        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111110__00000000__00000001__00000000__00000000__11111111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111110__00000000__00000001__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 2);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 4), Direction::Up));
        let piece_board = game_state.get_piece_board();
        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111110__00000000__00000000__00000001__00000000__11111111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111110__00000000__00000000__00000001__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 3);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Up));
        let piece_board = game_state.get_piece_board();
        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111100__00000010__00000000__00000001__00000000__11111111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111100__00000010__00000000__00000001__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_down() {
        let game_state = initial_play_state();
        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 7), Direction::Down));
        let piece_board = game_state.get_piece_board();

        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111111__00000000__00000000__00000000__00001000__11110111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 6), Direction::Down));
        let piece_board = game_state.get_piece_board();
        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111111__00000000__00000000__00001000__00000000__11110111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 2);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Down));
        let piece_board = game_state.get_piece_board();
        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111111__00000000__00001000__00000000__00000000__11110111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 3);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Down));
        let piece_board = game_state.get_piece_board();
        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111111__00001000__00000000__00000000__00000000__11110111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_left() {
        let game_state = initial_play_state();
        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 7), Direction::Down));
        let piece_board = game_state.get_piece_board();

        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111111__00000000__00000000__00000000__00001000__11110111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 6), Direction::Left));
        let piece_board = game_state.get_piece_board();
        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111111__00000000__00000000__00000000__00000100__11110111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 2);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('c', 6), Direction::Left));
        let piece_board = game_state.get_piece_board();
        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111111__00000000__00000000__00000000__00000010__11110111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 3);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('b', 6), Direction::Left));
        let piece_board = game_state.get_piece_board();
        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111111__00000000__00000000__00000000__00000001__11110111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_right() {
        let game_state = initial_play_state();
        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 7), Direction::Down));
        let piece_board = game_state.get_piece_board();

        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111111__00000000__00000000__00000000__00001000__11110111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 6), Direction::Right));
        let piece_board = game_state.get_piece_board();
        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111111__00000000__00000000__00000000__00010000__11110111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 2);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('e', 6), Direction::Right));
        let piece_board = game_state.get_piece_board();
        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111111__00000000__00000000__00000000__00100000__11110111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 3);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('f', 6), Direction::Right));
        let piece_board = game_state.get_piece_board();
        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111111__00000000__00000000__00000000__01000000__11110111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111111__00000000__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_trap_unsupported() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 2), Direction::Up));
        let piece_board = game_state.get_piece_board();

        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111011__00000000__00000000__00000000__00000000__11111111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111011__00000000__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 1);
        assert_eq!(game_state.p1_turn_to_move, true);
    }

    #[test]
    fn test_action_move_trap_supported_right() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 2), Direction::Up));
        let piece_board = game_state.get_piece_board();

        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111001__00000110__00000000__00000000__00000000__11111111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111001__00000110__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 2);
        assert_eq!(game_state.p1_turn_to_move, true);
    }

    #[test]
    fn test_action_move_trap_supported_left() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('d', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 2), Direction::Up));
        let piece_board = game_state.get_piece_board();

        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11110011__00001100__00000000__00000000__00000000__11111111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11110011__00001100__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 2);
        assert_eq!(game_state.p1_turn_to_move, true);
    }

    #[test]
    fn test_action_move_trap_supported_top() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Up));
        let game_state =
            game_state.take_action(&Action::Move(Square::new('b', 3), Direction::Right));
        let piece_board = game_state.get_piece_board();

        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111101__00000100__00000000__00000000__00000000__11111111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111101__00000100__00000000__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 2);
        assert_eq!(game_state.p1_turn_to_move, true);
    }

    #[test]
    fn test_action_move_trap_supported_bottom() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 3), Direction::Up));
        let game_state =
            game_state.take_action(&Action::Move(Square::new('b', 4), Direction::Right));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 2), Direction::Up));
        let piece_board = game_state.get_piece_board();

        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111001__00000100__00000100__00000000__00000000__11111111__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111001__00000100__00000100__00000000__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_trap_adjacent_opp_unsupported() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 3), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 4), Direction::Up));

        let game_state =
            game_state.take_action(&Action::Move(Square::new('c', 7), Direction::Down));
        let piece_board = game_state.get_piece_board();

        assert_eq!(
            piece_board.rabbits,
            0b__00000000__11111001__00000010__00000000__00000100__00000000__11111011__00000000
        );
        assert_eq!(
            piece_board.p1_pieces,
            0b__11111111__11111001__00000010__00000000__00000100__00000000__00000000__00000000
        );
        assert_eq!(game_state.unwrap_play_phase().get_step(), 1);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_action_move_push_must_push_rabbit() {
        let game_state = initial_play_state();
        let game_state =
            game_state.take_action(&Action::Move(Square::new('b', 7), Direction::Down));

        assert_eq!(
            game_state.unwrap_play_phase().push_pull_state,
            PushPullState::MustCompletePush(Square::new('b', 7), Piece::Rabbit)
        );
    }

    #[test]
    fn test_action_move_push_must_push_elephant() {
        let game_state = initial_play_state();
        let game_state = game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Pass);
        let game_state =
            game_state.take_action(&Action::Move(Square::new('e', 7), Direction::Down));
        let game_state = game_state.take_action(&Action::Pass);
        let game_state =
            game_state.take_action(&Action::Move(Square::new('e', 8), Direction::Down));

        assert_eq!(
            game_state.unwrap_play_phase().push_pull_state,
            PushPullState::MustCompletePush(Square::new('e', 8), Piece::Elephant)
        );
    }

    #[test]
    fn test_action_place_initial() {
        let game_state = GameState::initial();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e, m, h, d, c, r]"
        );
    }

    #[test]
    fn test_action_place_elephant() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::Place(Piece::Elephant));

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[m, h, d, c, r]"
        );
    }

    #[test]
    fn test_action_place_camel() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::Place(Piece::Camel));

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e, h, d, c, r]"
        );
    }

    #[test]
    fn test_action_place_horse() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::Place(Piece::Horse));

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e, m, h, d, c, r]"
        );

        let game_state = game_state.take_action(&Action::Place(Piece::Horse));

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e, m, d, c, r]"
        );
    }

    #[test]
    fn test_action_place_dog() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::Place(Piece::Dog));

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e, m, h, d, c, r]"
        );

        let game_state = game_state.take_action(&Action::Place(Piece::Dog));

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e, m, h, c, r]"
        );
    }

    #[test]
    fn test_action_place_rabbits() {
        let game_state = GameState::initial();

        let game_state = place_8_rabbits(&game_state);

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e, m, h, d, c]"
        );
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

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e, m, h, d, c, r]"
        );
    }

    #[test]
    fn test_action_place_p2_camel() {
        let game_state = GameState::initial();

        let game_state = place_major_pieces(&game_state);
        let game_state = place_8_rabbits(&game_state);
        let game_state = game_state.take_action(&Action::Place(Piece::Camel));

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e, h, d, c, r]"
        );
    }

    #[test]
    fn test_action_correct_state_after_4_steps() {
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
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 1), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('a', 3), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('a', 4), Direction::Up));

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 8), Direction::Down));

        assert_eq!(
            game_state.to_string(),
            "1s
 +-----------------+
8|   r     r   r   |
7|       r         |
6|     x     x     |
5| R   E r         |
4|                 |
3|     x     x     |
2|                 |
1|                 |
 +-----------------+
   a b c d e f g h
"
        );
    }

    #[test]
    fn test_action_correct_state_after_pull_with_trap() {
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
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('c', 5), Direction::Up));

        assert_eq!(
            game_state
                .unwrap_play_phase()
                .push_pull_state
                .as_possible_pull()
                .unwrap(),
            (Square::new('c', 5), Piece::Elephant)
        );
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d5w, a1n, a1e, p]"
        );
        assert_eq!(
            game_state.to_string(),
            "1g
 +-----------------+
8|   r   r r   r   |
7|                 |
6|     x     x     |
5|       r         |
4|                 |
3|     x     x     |
2|                 |
1| R               |
 +-----------------+
   a b c d e f g h
"
        );

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 1), Direction::Up));
        assert_eq!(format!("{:?}", game_state.valid_actions()), "[a2n, a2e, p]");
        assert_eq!(
            game_state.to_string(),
            "1g
 +-----------------+
8|   r   r r   r   |
7|                 |
6|     x     x     |
5|       r         |
4|                 |
3|     x     x     |
2| R               |
1|                 |
 +-----------------+
   a b c d e f g h
"
        );

        let game_state = game_state.take_action(&Action::Pass);
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[b8e, e8e, g8e, d5e, b8s, d8s, e8s, g8s, d5s, b8w, d8w, g8w, d5w]"
        );
        assert_eq!(
            game_state.to_string(),
            "1s
 +-----------------+
8|   r   r r   r   |
7|                 |
6|     x     x     |
5|       r         |
4|                 |
3|     x     x     |
2| R               |
1|                 |
 +-----------------+
   a b c d e f g h
"
        );
    }

    #[test]
    fn test_action_correct_state_after_pull_with_trap_accepted() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|   r   r r   r   |
             7|                 |
             6|     x     x     |
             5|     E r         |
             4|                 |
             3|     x     x     |
             2| r               |
             1| D               |
              +-----------------+
                a b c d e f g h"
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('c', 5), Direction::Up));

        assert_eq!(
            game_state
                .unwrap_play_phase()
                .push_pull_state
                .as_possible_pull()
                .unwrap(),
            (Square::new('c', 5), Piece::Elephant)
        );
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[a2n, a2e, d5w, a1e, p]"
        );
        assert_eq!(
            game_state.to_string(),
            "1g
 +-----------------+
8|   r   r r   r   |
7|                 |
6|     x     x     |
5|       r         |
4|                 |
3|     x     x     |
2| r               |
1| D               |
 +-----------------+
   a b c d e f g h
"
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Left));
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[a2n, a2e, a1e, p]"
        );
        assert_eq!(
            game_state.to_string(),
            "1g
 +-----------------+
8|   r   r r   r   |
7|                 |
6|     x     x     |
5|     r           |
4|                 |
3|     x     x     |
2| r               |
1| D               |
 +-----------------+
   a b c d e f g h
"
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Right));
        assert_eq!(format!("{:?}", game_state.valid_actions()), "[a1n]");
        assert_eq!(
            game_state.to_string(),
            "1g
 +-----------------+
8|   r   r r   r   |
7|                 |
6|     x     x     |
5|     r           |
4|                 |
3|     x     x     |
2|   r             |
1| D               |
 +-----------------+
   a b c d e f g h
"
        );

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 1), Direction::Up));
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[b8e, e8e, g8e, c5e, b8s, d8s, e8s, g8s, c5s, b8w, d8w, g8w, c5w]"
        );
        assert_eq!(
            game_state.to_string(),
            "1s
 +-----------------+
8|   r   r r   r   |
7|                 |
6|     x     x     |
5|     r           |
4|                 |
3|     x     x     |
2| D r             |
1|                 |
 +-----------------+
   a b c d e f g h
"
        );
    }

    #[test]
    fn test_action_cant_push_on_last_step() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|   r   r r   r   |
             7|                 |
             6|     x     x     |
             5|     E r         |
             4|                 |
             3|     x     x     |
             2| r         D     |
             1| D               |
              +-----------------+
                a b c d e f g h"
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('f', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('c', 5), Direction::Up));

        assert_eq!(
            game_state
                .unwrap_play_phase()
                .push_pull_state
                .as_possible_pull()
                .unwrap(),
            (Square::new('c', 5), Piece::Elephant)
        );
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[a2n, a2e, d5w, a1e, p]"
        );
        assert_eq!(
            game_state.to_string(),
            "1g
 +-----------------+
8|   r   r r   r   |
7|                 |
6|     x     x     |
5|       r         |
4|                 |
3|     x     x     |
2| r               |
1| D               |
 +-----------------+
   a b c d e f g h
"
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Left));
        assert_eq!(format!("{:?}", game_state.valid_actions()), "[a1e, p]");
        assert_eq!(
            game_state.to_string(),
            "1g
 +-----------------+
8|   r   r r   r   |
7|                 |
6|     x     x     |
5|     r           |
4|                 |
3|     x     x     |
2| r               |
1| D               |
 +-----------------+
   a b c d e f g h
"
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('a', 1), Direction::Right));
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[b8e, e8e, g8e, c5e, a2e, b8s, d8s, e8s, g8s, c5s, a2s, b8w, d8w, g8w, c5w]"
        );
        assert_eq!(
            game_state.to_string(),
            "1s
 +-----------------+
8|   r   r r   r   |
7|                 |
6|     x     x     |
5|     r           |
4|                 |
3|     x     x     |
2| r               |
1|   D             |
 +-----------------+
   a b c d e f g h
"
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Down));
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[b8e, e8e, g8e, c5e, b8s, d8s, e8s, g8s, c5s, b8w, d8w, g8w, c5w, p]"
        );
        assert_eq!(
            game_state.to_string(),
            "1s
 +-----------------+
8|   r   r r   r   |
7|                 |
6|     x     x     |
5|     r           |
4|                 |
3|     x     x     |
2|                 |
1| r D             |
 +-----------------+
   a b c d e f g h
"
        );

        assert_eq!(game_state.is_terminal(), None);

        let game_state = game_state.take_action(&Action::Pass);
        assert_eq!(game_state.is_terminal(), Some(Value([0.0, 1.0])));
    }

    #[test]
    fn test_action_cant_push_and_pull_simultaneously() {
        let game_state: GameState = "
             1s
              +-----------------+
             8|                 |
             7|                 |
             6|     x     x     |
             5|                 |
             4| R               |
             3| c   x     x     |
             2| R               |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse()
            .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[a4n, a4e, a2e, a2s, a3e]"
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Right));
        assert_eq!(format!("{:?}", game_state.valid_actions()), "[a3s]");
        assert_eq!(
            game_state.to_string(),
            "1s
 +-----------------+
8|                 |
7|                 |
6|     x     x     |
5|                 |
4| R               |
3| c   x     x     |
2|   R             |
1|                 |
 +-----------------+
   a b c d e f g h
"
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('a', 3), Direction::Down));
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[b2n, b2e, b2s, a2n, a2s, p]"
        );
        assert_eq!(
            game_state.to_string(),
            "1s
 +-----------------+
8|                 |
7|                 |
6|     x     x     |
5|                 |
4| R               |
3|     x     x     |
2| c R             |
1|                 |
 +-----------------+
   a b c d e f g h
"
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Down));
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[b2w, a1n, a1e, p]"
        );
        assert_eq!(
            game_state.to_string(),
            "1s
 +-----------------+
8|                 |
7|                 |
6|     x     x     |
5|                 |
4| R               |
3|     x     x     |
2|   R             |
1| c               |
 +-----------------+
   a b c d e f g h
"
        );

        let game_state = game_state.take_action(&Action::Pass);
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[a4n, b2n, a4e, b2e, b2w]"
        );
        assert_eq!(
            game_state.to_string(),
            "2g
 +-----------------+
8|                 |
7|                 |
6|     x     x     |
5|                 |
4| R               |
3|     x     x     |
2|   R             |
1| c               |
 +-----------------+
   a b c d e f g h
"
        );
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
            .parse()
            .unwrap();

        assert_eq!(game_state.can_pass(true), false);
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
            .parse()
            .unwrap();

        let game_state =
            game_state.take_action(&Action::Move(Square::new('c', 5), Direction::Down));
        assert_eq!(game_state.can_pass(true), true);
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
            .parse()
            .unwrap();

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Down));
        assert_eq!(game_state.can_pass(true), false);
    }

    #[test]
    fn test_can_pass_during_place_phase() {
        let game_state = GameState::initial();

        assert_eq!(game_state.can_pass(true), false);

        let game_state = game_state.take_action(&Action::Place(Piece::Elephant));
        assert_eq!(game_state.can_pass(true), false);
    }

    #[test]
    fn test_can_pass_false_if_same_as_start_of_move_state() {
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
            .parse()
            .unwrap();

        assert_eq!(game_state.can_pass(true), false);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('c', 5), Direction::Down));
        assert_eq!(game_state.can_pass(true), true);

        let game_state = game_state.take_action(&Action::Move(Square::new('c', 4), Direction::Up));
        assert_eq!(game_state.can_pass(true), false);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('c', 5), Direction::Down));
        assert_eq!(game_state.can_pass(true), true);
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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 7), Direction::Up));
        assert_eq!(game_state.is_terminal(), None);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('a', 8), Direction::Down));
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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

        assert_eq!(game_state.is_terminal(), Some(Value([0.0, 1.0])));
    }

    #[test]
    fn test_action_cant_push_while_frozen() {
        let game_state: GameState = "
              +-----------------+
             8|                 |
             7|                 |
             6|     x     x     |
             5|       e         |
             4|       M r       |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse()
            .unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[a1n, a1e]");
    }

    #[test]
    fn test_action_cant_push_while_frozen_2() {
        let game_state: GameState = "
            35s
             +-----------------+
            8|     d           |
            7| r e r r h c r r |
            6| H R m r r X c R |
            5| d R E R R h R r |
            4| R         M   C |
            3|     X     X     |
            2|   D       D   H |
            1|             C   |
             +-----------------+
               a b c d e f g h
            "
        .parse()
        .unwrap();

        let game_state =
            game_state.take_action(&Action::Move(Square::new('c', 8), Direction::Left));
        let game_state =
            game_state.take_action(&Action::Move(Square::new('g', 5), Direction::Down));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[g6s]");
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
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('c', 4), Direction::Up));
        assert_eq!(
            game_state.unwrap_play_phase().push_pull_state,
            PushPullState::PossiblePull(Square::new('c', 4), Piece::Elephant)
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Left));
        assert_eq!(
            game_state.unwrap_play_phase().push_pull_state,
            PushPullState::None
        );
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
            .parse()
            .unwrap();

        let game_state =
            game_state.take_action(&Action::Move(Square::new('c', 4), Direction::Down));
        assert_eq!(
            game_state.unwrap_play_phase().push_pull_state,
            PushPullState::PossiblePull(Square::new('c', 4), Piece::Elephant)
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Left));
        assert_eq!(
            game_state.unwrap_play_phase().push_pull_state,
            PushPullState::None
        );
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
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('c', 4), Direction::Up));
        assert_eq!(
            game_state.unwrap_play_phase().push_pull_state,
            PushPullState::PossiblePull(Square::new('c', 4), Piece::Elephant)
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Left));
        assert_eq!(
            game_state.unwrap_play_phase().push_pull_state,
            PushPullState::MustCompletePush(Square::new('d', 4), Piece::Elephant)
        );
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
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('c', 4), Direction::Up));
        assert_eq!(
            game_state.unwrap_play_phase().push_pull_state,
            PushPullState::PossiblePull(Square::new('c', 4), Piece::Camel)
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Left));
        assert_eq!(
            game_state.unwrap_play_phase().push_pull_state,
            PushPullState::MustCompletePush(Square::new('d', 4), Piece::Elephant)
        );
    }

    #[test]
    fn test_action_pass() {
        let game_state = initial_play_state();
        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 7), Direction::Down));

        assert_eq!(game_state.unwrap_play_phase().get_step(), 1);
        assert_eq!(game_state.p1_turn_to_move, true);

        let game_state = game_state.take_action(&Action::Pass);

        assert_eq!(game_state.unwrap_play_phase().get_step(), 0);
        assert_eq!(game_state.p1_turn_to_move, false);
    }

    #[test]
    fn test_valid_actions() {
        let game_state = initial_play_state();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[a2n, b2n, c2n, d2n, e2n, f2n, g2n, h2n]"
        );
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
            "
        .parse()
        .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[a7s, b7s, c7s, d7s, e7s, f7s, g7s, h7s]"
        );
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
            "
        .parse()
        .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e3n, a2n, b2n, c2n, d2n, f2n, g2n, h2n, e1n, e3e, d2e, e3w, f2w]"
        );
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
            "
        .parse()
        .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d7e, g7e, e6e, e8s, h8s, a7s, b7s, c7s, d7s, f7s, g7s, e6s, h6s, f7w, e6w, h6w]"
        );
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
            "
        .parse()
        .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e3n, a2n, b2n, c2n, d2n, f2n, g2n, h2n, e1n, e3e, d2e, e3s, e3w, f2w]"
        );
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
            "
        .parse()
        .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e6n, d7e, e6e, e8s, a7s, b7s, c7s, d7s, f7s, g7s, h7s, e6s, f7w, e6w]"
        );
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
            .parse()
            .unwrap();

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[a5n, c5n, h5n, a5e, c5e, c5w, h5w, b3n, d3n, e3n, f3n, g3n, a2n, c2n, h2n, b1n, d1n, e1n, g1n, a4e, c4e, b3e, g3e, a2e, c2e, f2e, b1e, e1e, g1e, a4s, c4s, h4s, a2s, c2s, f2s, h2s, c4w, h4w, b3w, d3w, c2w, f2w, h2w, b1w, d1w, g1w]");
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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d4e, d4s, d4w]"
        );
    }

    #[test]
    fn test_valid_actions_frozen_piece_p1_mid_move() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|                 |
             7|                 |
             6|     x     x     |
             5|       e         |
             4|     M           |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse()
            .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[c4n, a1n, c4e, a1e, c4s, c4w]"
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('c', 4), Direction::Right));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[a1n, a1e, p]");
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
            .parse()
            .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d5n, d5e, d5w]"
        );
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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

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
            .parse()
            .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d5n, d5e, d5w]"
        );
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
            .parse()
            .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d4e, d4s, d4w, d5n, d5e, d5w]"
        );
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
            .parse()
            .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d4e, d4s, d4w, d5n, a8e, d5e, a8s, d5w]"
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Right));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d5s]");

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Down));

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e4n, e4e, e4s, d4n, a8e, a8s, d4s, d4w, p]"
        );
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
            .parse()
            .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d4e, d4w, d5n, a8e, d5e, d3e, a8s, d3s, d5w, d3w]"
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Right));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d3n, d5s]");

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Down));

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e4n, e4e, e4s, d4n, a8e, d3e, a8s, d3s, d4w, d3w, p]"
        );
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
            .parse()
            .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d4e, d5n, a8e, d5e, d3e, a8s, d3s, d5w, d3w]"
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Right));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[d3n, d5s]");

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Down));

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e4n, e4e, e4s, d4n, a8e, d3e, a8s, d3s, d3w, p]"
        );
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
            .parse()
            .unwrap();

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d5n, d5e, c5w]"
        );
    }

    #[test]
    fn test_valid_actions_matches_first_step() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|                 |
             7|                 |
             6|     x     x     |
             5|       m         |
             4|       E         |
             3|     x     x     |
             2|                 |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse()
            .unwrap();

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Right));
        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Down));

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Up));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[]");
    }

    #[test]
    fn test_valid_actions_matches_first_step_2() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|               r |
             7|             d C |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2| E             h |
             1|   e           R |
              +-----------------+
                a b c d e f g h"
            .parse()
            .unwrap();

        let game_state =
            game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Down));
        let game_state = game_state.take_action(&Action::Move(Square::new('a', 1), Direction::Up));

        let game_state =
            game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Down));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[p]");
    }

    #[test]
    fn test_valid_actions_matches_is_terminal_if_only_passing() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|                 |
             7|                 |
             6|     x     x     |
             5|       m         |
             4|       E         |
             3|     x     x     |
             2|                 |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse()
            .unwrap();

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Right));
        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 5), Direction::Down));

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Up));

        assert_eq!(game_state.is_terminal(), Some(Value([0.0, 1.0])));

        assert_eq!(format!("{:?}", game_state.valid_actions()), "[]");
    }

    #[test]
    fn test_valid_actions_repeated_positions() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|               r |
             7|               d |
             6|     x     x     |
             5|       m         |
             4|                 |
             3|     x E   x     |
             2|                 |
             1|               R |
              +-----------------+
                a b c d e f g h"
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 3), Direction::Up));
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d5n, d5e, d5w, h1n, d4e, d4s, d4w, h1w, p]"
        );
        let game_state = game_state.take_action(&Action::Pass);

        // First occurance of position

        let game_state =
            game_state.take_action(&Action::Move(Square::new('h', 7), Direction::Down));
        let game_state = game_state.take_action(&Action::Pass);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Down));
        let game_state = game_state.take_action(&Action::Pass);

        let game_state = game_state.take_action(&Action::Move(Square::new('h', 6), Direction::Up));
        let game_state = game_state.take_action(&Action::Pass);

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d3n, h1n, d3e, d3s, d3w, h1w]"
        );

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 3), Direction::Up));
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d5n, d5e, d5w, h1n, d4e, d4s, d4w, h1w, p]"
        );
        let game_state = game_state.take_action(&Action::Pass);

        // Second occurance of position

        let game_state =
            game_state.take_action(&Action::Move(Square::new('h', 7), Direction::Down));
        let game_state = game_state.take_action(&Action::Pass);

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d5n, d5e, d5w, h1n, d4e, d4s, d4w, h1w]"
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Down));
        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 3), Direction::Down));
        let game_state = game_state.take_action(&Action::Pass);

        let game_state = game_state.take_action(&Action::Move(Square::new('h', 6), Direction::Up));
        let game_state = game_state.take_action(&Action::Pass);

        // Would be third occurance of position

        let game_state = game_state.take_action(&Action::Move(Square::new('d', 2), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('d', 3), Direction::Up));

        // Should not allow pass here
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d5n, d5e, d5w, h1n, d4e, d4s, d4w, h1w]"
        );

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 4), Direction::Down));

        // Should not allow last move to be the final repeat
        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[d5s, h1n, d3e, d3w, h1w, p]"
        );
    }

    #[test]
    fn test_valid_actions_does_not_duplicate_piece_that_can_be_both_pushed_and_pull() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|                 |
             7|                 |
             6|     x     x     |
             5|                 |
             4|       E c H     |
             3|     x     x     |
             2|                 |
             1|   e             |
              +-----------------+
                a b c d e f g h"
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('f', 4), Direction::Up));

        assert_eq!(
            format!("{:?}", game_state.valid_actions()),
            "[e4n, e4e, e4s, f5n, d4n, f5e, f5s, d4s, f5w, d4w, p]"
        );
    }

    #[test]
    fn test_hash_initial_play_state() {
        let game_state: GameState = "
             1g
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
            "
        .parse()
        .unwrap();

        let game_state_2 = initial_play_state();

        assert_eq!(game_state.hash, game_state_2.hash);
    }

    #[test]
    fn test_hash_should_account_for_player_not_equal() {
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
            "
        .parse()
        .unwrap();

        let game_state_2 = initial_play_state();

        assert_ne!(game_state.hash, game_state_2.hash);
    }

    #[test]
    fn test_hash_should_account_for_step_not_equal() {
        let game_state: GameState = "
             1g
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
            "
        .parse()
        .unwrap();

        let game_state_2 = initial_play_state();
        let game_state_2 =
            game_state_2.take_action(&Action::Move(Square::new('e', 2), Direction::Up));

        assert_ne!(game_state.hash, game_state_2.hash);
    }

    #[test]
    fn test_hash_should_account_for_step_equal() {
        let game_state: GameState = "
             1g
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
            "
        .parse()
        .unwrap();

        let game_state_2 = initial_play_state();
        let game_state_2 =
            game_state_2.take_action(&Action::Move(Square::new('e', 2), Direction::Up));
        assert_ne!(game_state.hash, game_state_2.hash);

        let game_state_2 = game_state_2.take_action(&Action::Pass);
        let game_state_2 = game_state_2.take_action(&Action::Pass);

        assert_eq!(game_state.hash, game_state_2.hash);
    }

    #[test]
    fn test_hash_should_allow_for_pass_to_equal() {
        let game_state: GameState = "
             1g
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
            "
        .parse()
        .unwrap();

        let game_state_2 = initial_play_state();

        let game_state_2 = game_state_2.take_action(&Action::Pass);
        assert_ne!(game_state.hash, game_state_2.hash);

        let game_state_2 = game_state_2.take_action(&Action::Pass);
        assert_eq!(game_state.hash, game_state_2.hash);
    }

    #[test]
    fn test_hash_should_switch_player_on_pass_for_step() {
        let game_state: GameState = "
             1g
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
            "
        .parse()
        .unwrap();

        let game_state_2 = initial_play_state();
        let game_state_2 =
            game_state_2.take_action(&Action::Move(Square::new('e', 2), Direction::Up));

        assert_ne!(game_state.hash, game_state_2.hash);
    }

    #[test]
    fn test_hash_should_account_for_trapped_pieces() {
        let game_state: GameState = "
             1g
              +-----------------+
             8| h c d m e d c h |
             7| r r r r     r r |
             6|     x E r x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2| R R R R   R R R |
             1| H C D M   D C H |
              +-----------------+
                a b c d e f g h
            "
        .parse()
        .unwrap();

        let game_state_final: GameState = "
             1s
              +-----------------+
             8| h c d m e d c h |
             7| r r r r     r r |
             6|     x   E x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2| R R R R   R R R |
             1| H C D M   D C H |
              +-----------------+
                a b c d e f g h
            "
        .parse()
        .unwrap();

        let game_state =
            game_state.take_action(&Action::Move(Square::new('e', 6), Direction::Right));
        assert_ne!(game_state.hash, game_state_final.hash);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 6), Direction::Right));
        assert_ne!(game_state.hash, game_state_final.hash);

        let game_state = game_state.take_action(&Action::Pass);
        assert_eq!(game_state.hash, game_state_final.hash);
    }

    #[test]
    fn test_hash_should_account_for_four_actions() {
        let game_state: GameState = "
             1g
              +-----------------+
             8| h c d m e d c h |
             7| r r r r     r r |
             6|     x E r x     |
             5|         r       |
             4|                 |
             3|     x     x     |
             2| R R R R   R R R |
             1| H C D M   D C H |
              +-----------------+
                a b c d e f g h
            "
        .parse()
        .unwrap();

        let game_state_final: GameState = "
             1s
              +-----------------+
             8| h c d m e d c h |
             7| r r r r E   r r |
             6|     x   r x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2| R R R R   R R R |
             1| H C D M   D C H |
              +-----------------+
                a b c d e f g h
            "
        .parse()
        .unwrap();

        let game_state =
            game_state.take_action(&Action::Move(Square::new('e', 6), Direction::Right));
        assert_ne!(game_state.hash, game_state_final.hash);

        let game_state =
            game_state.take_action(&Action::Move(Square::new('d', 6), Direction::Right));
        assert_ne!(game_state.hash, game_state_final.hash);

        let game_state = game_state.take_action(&Action::Move(Square::new('e', 6), Direction::Up));
        assert_ne!(game_state.hash, game_state_final.hash);

        let game_state = game_state.take_action(&Action::Move(Square::new('e', 5), Direction::Up));
        assert_eq!(game_state.hash, game_state_final.hash);
    }

    #[bench]
    fn bench_valid_actions(b: &mut Bencher) {
        let game_state: GameState = "
             1g
              +-----------------+
             8| h c d m r d c h |
             7|   r r r e r r r |
             6|     x     x     |
             5|                 |
             4| r C             |
             3|     x     x     |
             2| R R R R E R R R |
             1| H   D M R D C H |
              +-----------------+
                a b c d e f g h
            "
        .parse()
        .unwrap();

        b.iter(|| {
            let game_state = game_state.clone();
            let game_state =
                game_state.take_action(&Action::Move(Square::new('b', 4), Direction::Up));
            let valid_actions = game_state.valid_actions();
            if valid_actions.is_empty() {
                panic!();
            }

            let game_state =
                game_state.take_action(&Action::Move(Square::new('a', 4), Direction::Right));
            let valid_actions = game_state.valid_actions();
            if valid_actions.is_empty() {
                panic!();
            }

            let game_state =
                game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Up));
            let valid_actions = game_state.valid_actions();
            if valid_actions.is_empty() {
                panic!();
            }

            let game_state =
                game_state.take_action(&Action::Move(Square::new('f', 2), Direction::Up));
            let valid_actions = game_state.valid_actions();
            if valid_actions.is_empty() {
                panic!();
            }

            let game_state =
                game_state.take_action(&Action::Move(Square::new('d', 7), Direction::Down));
            let valid_actions = game_state.valid_actions();
            if valid_actions.is_empty() {
                panic!();
            }

            let game_state =
                game_state.take_action(&Action::Move(Square::new('e', 7), Direction::Down));
            let valid_actions = game_state.valid_actions();
            if valid_actions.is_empty() {
                panic!();
            }

            let game_state =
                game_state.take_action(&Action::Move(Square::new('f', 7), Direction::Down));
            let valid_actions = game_state.valid_actions();
            if valid_actions.is_empty() {
                panic!();
            }

            let game_state =
                game_state.take_action(&Action::Move(Square::new('g', 7), Direction::Down));
            game_state.valid_actions()
        });
    }
}
