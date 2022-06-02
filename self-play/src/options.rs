use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfPlayOptions {
    pub play_options: PlayOptions,
    pub visits: usize,
    pub fast_visits: usize,
    pub full_visits_probability: f32,
    pub epsilon: f32,
    pub self_play_batch_size: usize,
    pub self_play_parallelism: usize
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PlayOptions {
    pub temperature: f32,
    pub temperature_max_moves: usize,
    pub temperature_post_max_moves: f32,
    pub temperature_visit_offset: f32,
    pub parallelism: usize,
    pub fpu: f32,
    pub fpu_root: f32,
    pub logit_q: bool,
    pub cpuct_base: f32,
    pub cpuct_init: f32,
    pub cpuct_root_scaling: f32,
    pub moves_left_threshold: f32,
    pub moves_left_scale: f32,
    pub moves_left_factor: f32,
}
