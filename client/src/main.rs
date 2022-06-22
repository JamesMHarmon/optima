mod cli;

use anyhow::{anyhow, Result};
use clap::Parser;
use cli::{Cli, Commands};
use common::{ConfigLoader, FsExt};
use dotenv::dotenv;
use env_logger::Env;
use self_play::{play_self, SelfPlayPersistance};
use std::path::PathBuf;

fn main() -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async_main())?;

    Ok(())
}

async fn async_main() -> Result<()> {
    dotenv().ok();
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::SelfPlay(self_play_args) => {
            let config_path = self_play_args.config.relative_to_cwd()?;
            let config = ConfigLoader::new(config_path, "self_play".to_string())?;

            let self_play_options = config.load()?;

            let games_dir = config.get_relative_path("games_dir")?;

            let model_dir = config.get_relative_path("model_dir")?;

            if !PathBuf::from(&games_dir).is_dir() {
                return Err(anyhow!("Path {:?} is not a valid directory", games_dir));
            }

            if !PathBuf::from(&model_dir).is_dir() {
                return Err(anyhow!("Path {:?} is not a valid directory", model_dir));
            }

            let model_factory = arimaa::model::ModelFactory::new(model_dir);
            let engine = arimaa::Engine::new();

            let mut self_play_persistance = SelfPlayPersistance::new(games_dir)?;

            play_self(
                &model_factory,
                &engine,
                &mut self_play_persistance,
                &self_play_options,
            )?
        }
        Commands::Arena(_self_play_args) => {
            todo!();
        }
        _ => {}
    }

    Ok(())
}
