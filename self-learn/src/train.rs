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
use engine::value::Value;
use model::model::Model;

use super::self_learn::SelfLearnOptions;
use super::self_play_persistance::{SelfPlayPersistance};

pub fn train_model<S, A, V, E, M, T>(
    model: &M,
    self_play_persistance: &SelfPlayPersistance,
    game_engine: &E,
    options: &SelfLearnOptions
) -> Result<ModelInfo, Error>
where
    S: GameState,
    A: Clone + Eq + DeserializeOwned + Serialize + Debug + Send,
    V: Value + Clone + DeserializeOwned,
    E: GameEngine<State=S,Action=A,Value=V> + Sync,
    M: Model<State=S,Action=A,Value=V,Analyzer=T>,
    T: GameAnalyzer<Action=A,State=S,Value=V> + Send
{
    let source_model_info = model.get_model_info();
    let new_model_info = source_model_info.get_next_model_info();
    let metric_iter = self_play_persistance.read_all_reverse_iter::<A,V>()?;

    println!("Loading positions for training...");

    let model_num = source_model_info.get_model_num();
    let number_of_games_per_net = options.number_of_games_per_net;
    let num_games_since_beginning = model_num * number_of_games_per_net;
    let num_max_moving_window_percentage_games = number_of_games_per_net + (num_games_since_beginning as f32 * options.max_moving_window_percentage) as usize;
    let num_games = std::cmp::min(num_max_moving_window_percentage_games, options.moving_window_size);
    let position_sample_percentage = options.position_sample_percentage;
    let mut rng = rand::thread_rng();

    let positions_metrics = metric_iter
        .take(num_games)
        .flat_map(|m| {
            let (analysis, score) = m.take();
            let mut num_positions = 0;

            let (_, positions_metrics, max_move_number) = analysis.into_iter().fold(
                (S::initial(), vec!(), 0),
                |(prev_game_state, mut samples, max_move_number), (action, metrics)| {
                    let next_game_state = game_engine.take_action(&prev_game_state, &action);
                    let move_number = game_engine.get_move_number(&prev_game_state);

                    if metrics.visits > options.fast_visits {
                        samples.push((PositionMetrics {
                            game_state: prev_game_state,
                            score: score.clone(),
                            policy: metrics,
                            moves_left: 0
                        }, move_number));
                    }

                    num_positions += 1;

                    (next_game_state, samples, max_move_number.max(move_number))
                }
            );

            let num_samples = ((num_positions as f32) * position_sample_percentage).ceil() as usize;
            let num_samples = num_samples.min(positions_metrics.len());
            positions_metrics.into_iter().choose_multiple(&mut rng, num_samples).into_iter().map(move |(mut sample, move_number)| {
                let moves_left = max_move_number - move_number + 1;
                sample.moves_left = moves_left;

                sample
            })
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
