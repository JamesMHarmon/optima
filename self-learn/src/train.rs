use model::model_info::ModelInfo;
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
) -> Result<ModelInfo, Error>
where
    S: GameState,
    A: Clone + Eq + DeserializeOwned + Serialize + Debug + Send,
    E: GameEngine<State=S,Action=A> + Sync,
    M: Model<State=S,Action=A,Analyzer=T>,
    T: GameAnalyzer<Action=A,State=S> + Send
{
    let source_model_info = model.get_model_info();
    let new_model_info = source_model_info.get_next_model_info();
    let metric_iter = self_play_persistance.read_all_reverse_iter::<A>()?;

    println!("Loading positions for training...");

    let run_num = source_model_info.get_run_num();
    let number_of_games_per_net = options.number_of_games_per_net;
    let num_games_since_beginning = run_num * number_of_games_per_net;
    let num_max_moving_window_percentage_games = number_of_games_per_net + (num_games_since_beginning as f32 * options.max_moving_window_percentage) as usize;
    let num_games = std::cmp::min(num_max_moving_window_percentage_games, options.moving_window_size);
    let position_sample_percentage = options.position_sample_percentage;
    let exclude_drawn_games = options.exclude_drawn_games;
    let mut rng = rand::thread_rng();

    let positions_metrics = metric_iter
        .filter(|m| !exclude_drawn_games || m.score() != 0.0)
        .take(num_games)
        .flat_map(|m| {
            let score = m.score();
            let analysis = m.take_analysis();

            let num_positions = analysis.len();
            let num_samples = ((num_positions as f32) * position_sample_percentage).ceil() as usize;
            let samples = analysis.into_iter().choose_multiple(&mut rng, num_samples);
            let samples_len = samples.len();
            let (_, positions_metrics) = samples.into_iter().enumerate().fold(
                (S::initial(), Vec::with_capacity(samples_len)),
                |(prev_game_state, mut samples), (i, (action, metrics))| {
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
        });

    model.train(
        &new_model_info,
        positions_metrics,
        &TrainOptions {
            train_ratio: options.train_ratio,
            train_batch_size: options.train_batch_size,
            epochs: options.epochs,
            learning_rate: options.learning_rate,
            policy_loss_weight: options.policy_loss_weight,
            value_loss_weight: options.value_loss_weight
        }
    )?;

    Ok(new_model_info)
}
