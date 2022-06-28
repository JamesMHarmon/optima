use anyhow::Result;
use common::Config;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfPlayOptions {
    pub play_options: PlayOptions,
    pub visits: usize,
    pub fast_visits: usize,
    pub full_visits_probability: f32,
    pub epsilon: f32,
    pub self_play_batch_size: usize,
    pub self_play_parallelism: usize,
}

impl Config for SelfPlayOptions {
    fn load(config: &common::ConfigLoader) -> Result<Self> {
        Ok(Self {
            play_options: PlayOptions::load(config)?,
            visits: config
                .get("visits")
                .and_then(|v| v.as_usize())
                .unwrap_or(800),
            fast_visits: config
                .get("fast_visits")
                .and_then(|v| v.as_usize())
                .unwrap_or(150),
            full_visits_probability: config
                .get("full_visits_probability")
                .and_then(|v| v.as_f32())
                .unwrap_or(0.25),
            epsilon: config
                .get("epsilon")
                .and_then(|v| v.as_f32())
                .unwrap_or(0.25),
            self_play_batch_size: config
                .get("self_play_batch_size")
                .and_then(|v| v.as_usize())
                .unwrap_or(512),
            self_play_parallelism: config
                .get("self_play_parallelism")
                .and_then(|v| v.as_usize())
                .unwrap_or(10),
        })
    }
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

impl Config for PlayOptions {
    fn load(config: &common::ConfigLoader) -> Result<Self> {
        Ok(Self {
            temperature: config
                .get("temperature")
                .and_then(|v| v.as_f32())
                .unwrap_or(1.1),
            temperature_max_moves: config
                .get("temperature_max_moves")
                .and_then(|v| v.as_usize())
                .unwrap_or(15),
            temperature_post_max_moves: config
                .get("temperature_post_max_moves")
                .and_then(|v| v.as_f32())
                .unwrap_or(0.45),
            temperature_visit_offset: config
                .get("temperature_visit_offset")
                .and_then(|v| v.as_f32())
                .unwrap_or(-0.9),
            parallelism: config
                .get("parallelism")
                .and_then(|v| v.as_usize())
                .unwrap_or(4),
            fpu: config.get("fpu").and_then(|v| v.as_f32()).unwrap_or(0.0),
            fpu_root: config
                .get("fpu_root")
                .and_then(|v| v.as_f32())
                .unwrap_or(1.0),
            logit_q: false,
            cpuct_base: config
                .get("cpuct_base")
                .and_then(|v| v.as_f32())
                .unwrap_or(19652.0),
            cpuct_init: config
                .get("cpuct_init")
                .and_then(|v| v.as_f32())
                .unwrap_or(1.25),
            cpuct_root_scaling: config
                .get("cpuct_root_scaling")
                .and_then(|v| v.as_f32())
                .unwrap_or(1.0),
            moves_left_threshold: config
                .get("moves_left_threshold")
                .and_then(|v| v.as_f32())
                .unwrap_or(0.95),
            moves_left_scale: config
                .get("moves_left_scale")
                .and_then(|v| v.as_f32())
                .unwrap_or(10.0),
            moves_left_factor: config
                .get("moves_left_factor")
                .and_then(|v| v.as_f32())
                .unwrap_or(0.05),
        })
    }
}
