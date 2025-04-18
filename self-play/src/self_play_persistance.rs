use anyhow::Result;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fs;
use std::fs::File;
use std::path::PathBuf;
use uuid_b64::UuidB64;

use super::SelfPlayMetrics;
use model::ModelInfo;

pub struct SelfPlayPersistance {
    game_directory: PathBuf,
}

impl SelfPlayPersistance {
    pub fn new(game_directory: PathBuf) -> Result<Self> {
        fs::create_dir_all(&game_directory)?;

        Ok(Self { game_directory })
    }

    pub fn write<A: Serialize, P: Serialize, PV: Serialize>(
        &mut self,
        self_play_metrics: &SelfPlayMetrics<A, P, PV>,
        model_info: &ModelInfo,
    ) -> Result<()> {
        let file_path = self.generate_file_path_for_game(model_info);

        fs::create_dir_all(
            file_path
                .parent()
                .expect("Path should always have a parent"),
        )?;

        let file = File::create(file_path)?;
        let compressor = GzEncoder::new(file, Compression::default());
        serde_json::to_writer(compressor, self_play_metrics)?;

        Ok(())
    }

    pub fn get_games_for_model(
        &self,
        model_info: &ModelInfo,
    ) -> Result<impl Iterator<Item = PathBuf>> {
        let res = fs::read_dir(self.get_game_dir_for_model(model_info))?
            .flatten()
            .filter(|p| p.file_type().is_ok_and(|p| p.is_file()))
            .map(|p| p.path());

        Ok(res)
    }

    pub fn read<A: DeserializeOwned, V: DeserializeOwned, PV: DeserializeOwned>(
        path: &PathBuf,
    ) -> Result<SelfPlayMetrics<A, V, PV>> {
        let file = File::open(path)?;
        let content = GzDecoder::new(file);
        let metrics = serde_json::from_reader(content)?;
        Ok(metrics)
    }

    fn get_game_dir_for_model(&self, model_info: &ModelInfo) -> PathBuf {
        self.game_directory.join(model_info.model_name_w_num())
    }

    fn generate_file_path_for_game(&self, model_info: &ModelInfo) -> PathBuf {
        self.get_game_dir_for_model(model_info).join(format!(
            "{}_{}.gz",
            model_info.model_name_w_num(),
            UuidB64::new()
                .to_string()
                .replace(|c: char| !c.is_alphanumeric(), "")
        ))
    }
}
