use common::linked_list::List;

use super::zobrist::Zobrist;
use super::engine::{GameState,Phase,PlayPhase,PushPullState,PieceBoard,PieceBoardState};
use super::action::{Piece,map_bit_board_to_squares};

pub fn get_symmetries(game_state: GameState) -> Vec<GameState> {
    match game_state.as_play_phase() {
        Some(_) => {
            let transposition = get_horizontal_symmetry(&game_state);
            vec!(game_state, transposition)
        },
        None => vec!(game_state)
    }
}

// The hash history produced for Zobrist are not accurate, this assumes that the hashes do not matter.
fn get_horizontal_symmetry(game_state: &GameState) -> GameState {
    if let Some(play_phase) = game_state.as_play_phase() {
        let previous_piece_boards_this_move = play_phase.get_previous_piece_boards().iter()
            .map(|b| b.get_piece_board())
            .map(get_piece_board_horizontal_symmetry)
            .collect::<Vec<_>>();

        let p1_turn_to_move = game_state.is_p1_turn_to_move();
        let step_num = game_state.get_current_step();
        let piece_board = get_piece_board_horizontal_symmetry(game_state.get_piece_board());
        let hash = Zobrist::from_piece_board(piece_board.get_piece_board(), p1_turn_to_move, step_num);
        let push_pull_state = play_phase.get_push_pull_state();
        let initial_hash_of_move = previous_piece_boards_this_move.first()
            .map(|b| Zobrist::from_piece_board(b.get_piece_board(), p1_turn_to_move, 0))
            .unwrap_or(hash);

        let play_phase = PlayPhase::new(
            initial_hash_of_move,
            List::new(),
            previous_piece_boards_this_move,
            get_push_pull_state_horizontal_symmetry(push_pull_state),
            play_phase.get_piece_trapped_this_turn()
        );

        GameState::new(
            p1_turn_to_move, game_state.get_move_number(),
            Phase::PlayPhase(play_phase),
            piece_board,
            hash
        )
    } else {
        panic!("Cannot create vertical symmetry for place phase");
    }
}

fn get_push_pull_state_horizontal_symmetry(push_pull_state: PushPullState) -> PushPullState {
    match push_pull_state {
        PushPullState::MustCompletePush(square, piece) => PushPullState::MustCompletePush(square.invert_horizontal(), piece),
        PushPullState::PossiblePull(square, piece) => PushPullState::PossiblePull(square.invert_horizontal(), piece),
        PushPullState::None => PushPullState::None,
    }
}

fn get_piece_board_horizontal_symmetry(piece_board: &PieceBoardState) -> PieceBoard {
    PieceBoard::new(
        get_bit_board_horizontal_symmetry(piece_board.get_player_piece_mask(true)),
        get_bit_board_horizontal_symmetry(piece_board.get_bits_by_piece_type(Piece::Elephant)),
        get_bit_board_horizontal_symmetry(piece_board.get_bits_by_piece_type(Piece::Camel)),
        get_bit_board_horizontal_symmetry(piece_board.get_bits_by_piece_type(Piece::Horse)),
        get_bit_board_horizontal_symmetry(piece_board.get_bits_by_piece_type(Piece::Dog)),
        get_bit_board_horizontal_symmetry(piece_board.get_bits_by_piece_type(Piece::Cat)),
        get_bit_board_horizontal_symmetry(piece_board.get_bits_by_piece_type(Piece::Rabbit)),
    )
}

fn get_bit_board_horizontal_symmetry(bit_board: u64) -> u64 {
    map_bit_board_to_squares(bit_board).iter()
        .map(|s| s.invert_horizontal())
        .fold(0, |r, s| r | s.as_bit_board())
}
