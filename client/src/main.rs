mod cli;

use anyhow::{anyhow, Result};
use clap::Parser;
use cli::{Cli, Commands};
use common::{get_env_usize, ConfigLoader, FsExt};
use dotenv::dotenv;
use env_logger::Env;
use log::info;
use model::Load;
use quoridor::ModelRef;
use self_play::{play_self, SelfPlayPersistance};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use ugi::run_ugi;

fn main() -> Result<()> {
    let cli = Cli::parse();

    dotenv().ok();

    let log_level = Env::default().default_filter_or("warn");

    env_logger::Builder::from_env(log_level).init();

    let mut builder = tokio::runtime::Builder::new_multi_thread();

    builder.enable_all();

    if let Some(worker_threads) = get_env_usize("TOKIO_THREADS") {
        builder.worker_threads(worker_threads);
    }

    info!("{:?}", builder);

    builder.build().unwrap().block_on(async_main(cli))?;

    Ok(())
}

async fn async_main(cli: Cli) -> Result<()> {
    match &cli.command {
        Commands::SelfPlay(self_play_args) => {
            let config_path = self_play_args.config.relative_to_cwd()?;
            let config = ConfigLoader::new(config_path, "self_play".to_string())?;

            let self_play_options = config.load()?;

            let games_dir = config.get_relative_path("games_dir")?;
            let model_dir = config.get_relative_path("model_dir")?;

            assert_dir_exists(&games_dir)?;
            assert_dir_exists(&model_dir)?;

            let model_factory = quoridor::ModelFactory::new(model_dir);
            let engine = quoridor::Engine::new();

            let mut self_play_persistance = SelfPlayPersistance::new(games_dir)?;

            play_self(
                &model_factory,
                &engine,
                &mut self_play_persistance,
                &self_play_options,
            )?
        }
        Commands::Arena(arena_args) => {
            let config_path = arena_args.config.relative_to_cwd()?;
            let config = ConfigLoader::new(config_path, "arena".to_string())?;

            let arena_options = config.load()?;

            let champions_dir = config.get_relative_path("champions_dir")?;
            let candidates_dir = config.get_relative_path("candidates_dir")?;
            let certified_dir = config.get_relative_path("certified_dir")?;
            let evaluated_dir = config.get_relative_path("evaluated_dir")?;

            assert_dir_exists(&champions_dir)?;
            assert_dir_exists(&candidates_dir)?;
            assert_dir_exists(&certified_dir)?;
            assert_dir_exists(&evaluated_dir)?;

            let champion_factory = quoridor::ModelFactory::new(champions_dir.clone());
            let candidate_factory = quoridor::ModelFactory::new(candidates_dir);
            let engine: quoridor::Engine = quoridor::Engine::new();

            arena::championship(
                &champion_factory,
                &champions_dir,
                &candidate_factory,
                &certified_dir,
                &evaluated_dir,
                &engine,
                &"./".relative_to_cwd()?,
                &arena_options,
            )?
        }
        Commands::Ugi(ugi_args) => {
            let model_dir = std::env::var("BOT_MODEL_DIR")
                .ok()
                .as_ref()
                .or(ugi_args.dir.as_ref())
                .map(|dir| dir.relative_to_cwd())
                .transpose()?
                .map(PathBuf::from)
                .or_else(|| std::env::current_dir().ok())
                .expect("Could not determine model directory");

            let model_name = std::env::var("BOT_MODEL_NAME")
                .map(Cow::Owned)
                .ok()
                .or_else(|| ugi_args.model.as_ref().map(Cow::Borrowed))
                .expect("Model name was not provided");

            let model_path = model_dir.join(&*model_name);

            assert!(model_path.is_file(), "Model not found. {:?}", &model_path);

            let ugi = quoridor::UGI::new();
            let model_factory = quoridor::ModelFactory::new(model_dir);
            let model = model_factory.load(&ModelRef::new(model_path))?;
            let engine = quoridor::Engine::new();

            run_ugi(ugi, engine, model).await?
        }
    }

    Ok(())
}

fn assert_dir_exists<P: AsRef<Path>>(dir: P) -> Result<()> {
    if dir.as_ref().is_dir() {
        Ok(())
    } else {
        Err(anyhow!("{:?} is not a valid directory", dir.as_ref()))
    }
}
