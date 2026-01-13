use crate::UGIOptions;
use engine::GameState;
use std::time::Duration;

/// Trait for calculating search duration based on game options and state
pub trait TimeStrategy<S: GameState> {
    /// Calculate the duration for the search based on the current game state and options
    fn search_duration(
        &self,
        options: &UGIOptions,
        game_state: &S,
        current_player: usize,
    ) -> Duration;
}

/// Base timing strategy implementation that uses standard time controls
pub struct BaseTimeStrategy;

impl BaseTimeStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BaseTimeStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: GameState> TimeStrategy<S> for BaseTimeStrategy {
    fn search_duration(
        &self,
        options: &UGIOptions,
        _game_state: &S,
        current_player: usize,
    ) -> Duration {
        let current_g_reserve_time = options.current_g_reserve_time;
        let current_s_reserve_time = options.current_s_reserve_time;
        let reserve_time_to_use = options.reserve_time_to_use;
        let time_per_move = options.time_per_move;
        let fixed_time = options.fixed_time;
        let time_buffer = options.time_buffer;

        let reserve_time: f32 = if current_player == 1 {
            current_g_reserve_time
        } else {
            current_s_reserve_time
        };

        let reserve_time: f32 = (reserve_time - time_buffer).max(0.0);
        let search_time: f32 = reserve_time * reserve_time_to_use + time_per_move;
        let search_time = search_time - time_per_move * 0.05;
        let search_time: f32 = fixed_time.unwrap_or(search_time);

        Duration::from_secs_f32(0f32.max(search_time))
    }
}
