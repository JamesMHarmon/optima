use model::position_metrics::PositionMetrics;
use model::model::TrainOptions;
use rand::seq::IteratorRandom;

use std::fmt::Debug;
use serde::Serialize;
use serde::de::DeserializeOwned;
use failure::Error;

use model::analytics::GameAnalyzer;
use engine::engine::GameEngine;
use engine::game_state::GameState;
use model::model::Model;

use super::self_learn::SelfLearnOptions;
use super::self_play_persistance::{SelfPlayPersistance};

pub fn train_model<S, A, E, M, T>(
    model: &M,
    self_play_persistance: &SelfPlayPersistance,
    game_engine: &E,
    options: &SelfLearnOptions
) -> Result<M, Error>
where
    S: GameState,
    A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
    E: GameEngine<State=S,Action=A> + Sync,
    M: Model<State=S,Action=A,Analyzer=T>,
    T: GameAnalyzer<Action=A,State=S> + Send
{
    let source_model_info = model.get_model_info();
    let new_model_info = source_model_info.get_next_model_info();
    let metric_iter = self_play_persistance.read_all_reverse_iter::<A>()?;

    println!("Loading positions for training...");

    let run_num = source_model_info.get_run_num();
    let num_games_since_beginning = run_num * options.number_of_games_per_net;
    let num_max_moving_window_percentage_games = (num_games_since_beginning as f64 * options.max_moving_window_percentage) as usize;
    let num_games = std::cmp::min(num_max_moving_window_percentage_games, options.moving_window_size);

    let positions_metrics: Vec<_> = metric_iter
        .take(num_games)
        .flat_map(|m| {
            let score = m.score();
            let analysis = m.take_analysis();

            let (_, positions_metrics) = analysis.into_iter().enumerate().fold(
                (S::initial(), Vec::new()),
                |(prev_game_state,mut samples), (i, (action, metrics))| {
                    let sample_is_p1 = i % 2 == 0;
                    let score = score * if sample_is_p1 { 1.0 } else { -1.0 };
                    let game_state = game_engine.take_action(&prev_game_state, &action);

                    samples.push(PositionMetrics {
                        game_state: prev_game_state,
                        score,
                        policy: metrics
                    });

                    (game_state, samples)
                }
            );

            positions_metrics
        }).collect();

    let num_positions = positions_metrics.len();
    let position_sample_percentage = options.position_sample_percentage;
    let num_samples = ((num_positions as f64) * position_sample_percentage) as usize;
    println!("Sampling {}% for a total of {} training positions.", position_sample_percentage * 100.0, num_samples);

    let mut rng = rand::thread_rng();
    let positions_metrics = positions_metrics.into_iter().choose_multiple(
        &mut rng,
        num_samples
    );

    Ok(model.train(
        new_model_info,
        &positions_metrics,
        &TrainOptions {
            train_ratio: options.train_ratio,
            train_batch_size: options.train_batch_size,
            epochs: options.epochs,
            learning_rate: options.learning_rate,
            policy_loss_weight: options.policy_loss_weight,
            value_loss_weight: options.value_loss_weight
        }
    ))
}