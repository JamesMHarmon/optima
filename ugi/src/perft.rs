use engine::{GameEngine, GameState, ValidActions};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use std::sync::atomic::{AtomicU64, Ordering};

pub fn run_perft<S, A, E>(depth: usize, engine: &E) -> u64
where
    S: GameState + Send + Sync + 'static,
    A: Send + Sync + 'static,
    E: GameEngine<State = S, Action = A>
        + ValidActions<State = S, Action = A>
        + Send
        + Sync
        + 'static,
{
    let game_state = S::initial();
    count_moves_par(&game_state, engine, depth)
}

fn count_moves_par<S, A, E>(game_state: &S, engine: &E, depth: usize) -> u64
where
    S: GameState + Sync + 'static,
    A: Sync + 'static,
    E: GameEngine<State = S, Action = A> + ValidActions<State = S, Action = A> + Sync + 'static,
{    
    if depth <= 2 {
        return count_moves(game_state, engine, depth);
    }

    let num_moves = AtomicU64::new(0);

    engine
        .valid_actions(game_state)
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|action| {
            let next_game_state = engine.take_action(game_state, action);
            let count = count_moves_par(&next_game_state, engine, depth - 1);

            num_moves.fetch_add(count, Ordering::Relaxed);
        });

    num_moves.load(Ordering::Relaxed)
}

fn count_moves<S, A, E>(game_state: &S, engine: &E, depth: usize) -> u64
where
    S: GameState,
    E: GameEngine<State = S, Action = A> + ValidActions<State = S, Action = A>,
{
    if depth == 0 {
        return 0;
    }

    if depth == 1 {
        return engine.valid_actions(game_state).count() as u64;
    };

    engine
        .valid_actions(game_state)
        .map(|a| {
            let next_game_state = engine.take_action(game_state, &a);
            count_moves(&next_game_state, engine, depth - 1)
        })
        .sum()
}
