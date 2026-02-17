mod cli;
mod game;

use anyhow::{Result, anyhow};
use arena::ArenaOptions;
use clap::Parser;
use cli::{Cli, Commands};
use common::{ConfigLoader, DynamicCPUCT, FsExt, get_env_usize};
use dotenv::dotenv;
use env_logger::Env;
use game::{Engine, ModelFactory, ModelRef, TimeStrategy, UGI};
use log::info;
use model::Load;
#[cfg(any(feature = "connect4", feature = "arimaa"))]
use puct::{MovesLeftSelectionPolicy, MovesLeftStrategyOptions, MovesLeftValueModel};

#[cfg(feature = "quoridor")]
use puct::{VictoryMarginSelectionPolicy, VictoryMarginStrategyOptions, VictoryMarginValueModel};
use self_play::{SelfPlayOptions, SelfPlayPersistance, play_self};
use std::borrow::Cow;
use std::path::Path;
use ugi::{UGIOptions, run_perft, run_ugi};

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

            let self_play_options: SelfPlayOptions = config.load()?;

            let games_dir = config.get_relative_path("games_dir")?;
            let model_dir = config.get_relative_path("model_dir")?;

            assert_dir_exists(&games_dir)?;
            assert_dir_exists(&model_dir)?;

            let model_factory = ModelFactory::new(model_dir);
            let engine = Engine::new();

            let mut self_play_persistance = SelfPlayPersistance::new(games_dir)?;

            let play_options = &self_play_options.play_options;
            let cpuct = DynamicCPUCT::<_>::new(
                play_options.cpuct_base,
                play_options.cpuct_init,
                1.0,
                play_options.cpuct_root_scaling,
            );

            #[cfg(any(feature = "connect4", feature = "arimaa"))]
            let (value_model, selection_policy) = (
                MovesLeftValueModel::<_, _, _>::new(),
                MovesLeftSelectionPolicy::new(
                    cpuct,
                    MovesLeftStrategyOptions {
                        fpu: play_options.fpu,
                        fpu_root: play_options.fpu_root,
                        moves_left_threshold: play_options.moves_left_threshold,
                        moves_left_scale: play_options.moves_left_scale,
                        moves_left_factor: play_options.moves_left_factor,
                    },
                ),
            );

            #[cfg(feature = "quoridor")]
            let (value_model, selection_policy) = (
                VictoryMarginValueModel::<_, _, _>::new(),
                VictoryMarginSelectionPolicy::new(
                    cpuct,
                    VictoryMarginStrategyOptions {
                        fpu: play_options.fpu,
                        fpu_root: play_options.fpu_root,
                        victory_margin_threshold: play_options.victory_margin_threshold,
                        victory_margin_factor: play_options.victory_margin_factor,
                    },
                ),
            );

            play_self(
                &model_factory,
                &engine,
                &value_model,
                &selection_policy,
                &mut self_play_persistance,
                &self_play_options,
            )?
        }
        Commands::Arena(arena_args) => {
            let config_path = arena_args.config.relative_to_cwd()?;
            let config = ConfigLoader::new(config_path, "arena".to_string())?;

            let arena_options: ArenaOptions = config.load()?;
            let play_options = &arena_options.play_options;

            let champions_dir = config.get_relative_path("champions_dir")?;
            let candidates_dir = config.get_relative_path("candidates_dir")?;
            let certified_dir = config.get_relative_path("certified_dir")?;
            let evaluated_dir = config.get_relative_path("evaluated_dir")?;

            assert_dir_exists(&champions_dir)?;
            assert_dir_exists(&candidates_dir)?;
            assert_dir_exists(&certified_dir)?;
            assert_dir_exists(&evaluated_dir)?;

            let cpuct = DynamicCPUCT::<_>::new(
                play_options.cpuct_base,
                play_options.cpuct_init,
                1.0,
                play_options.cpuct_root_scaling,
            );

            let champion_factory = ModelFactory::new(champions_dir.clone());
            let candidate_factory = ModelFactory::new(candidates_dir);
            let engine = Engine::new();

            #[cfg(any(feature = "connect4", feature = "arimaa"))]
            let (value_model, selection_policy) = (
                MovesLeftValueModel::<_, _, _>::new(),
                MovesLeftSelectionPolicy::new(
                    cpuct,
                    MovesLeftStrategyOptions {
                        fpu: play_options.fpu,
                        fpu_root: play_options.fpu_root,
                        moves_left_threshold: play_options.moves_left_threshold,
                        moves_left_scale: play_options.moves_left_scale,
                        moves_left_factor: play_options.moves_left_factor,
                    },
                ),
            );

            #[cfg(feature = "quoridor")]
            let (value_model, selection_policy) = (
                VictoryMarginValueModel::<_, _, _>::new(),
                VictoryMarginSelectionPolicy::new(
                    cpuct,
                    VictoryMarginStrategyOptions {
                        fpu: play_options.fpu,
                        fpu_root: play_options.fpu_root,
                        victory_margin_threshold: play_options.victory_margin_threshold,
                        victory_margin_factor: play_options.victory_margin_factor,
                    },
                ),
            );

            arena::championship(
                &champion_factory,
                &champions_dir,
                &candidate_factory,
                &certified_dir,
                &evaluated_dir,
                &engine,
                &value_model,
                &selection_policy,
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
                .or_else(|| std::env::current_dir().ok())
                .expect("Could not determine model directory");

            let model_name = std::env::var("BOT_MODEL_NAME")
                .map(Cow::Owned)
                .ok()
                .or_else(|| ugi_args.model.as_ref().map(Cow::Borrowed))
                .expect("Model name was not provided");

            let model_path = model_dir.join(&*model_name);

            assert!(model_path.is_file(), "Model not found. {:?}", &model_path);

            let ugi = UGI::new();
            let model_factory = ModelFactory::new(model_dir);
            let model = model_factory.load(&ModelRef::new(model_path))?;
            let engine = Engine::new();

            let cpuct = |options: &UGIOptions| {
                DynamicCPUCT::<_>::new(
                    options.cpuct_base,
                    options.cpuct_init,
                    options.cpuct_factor,
                    options.cpuct_root_scaling,
                )
            };

            #[cfg(any(feature = "connect4", feature = "arimaa"))]
            let selection_strategy_opts = |options: &UGIOptions| MovesLeftStrategyOptions {
                fpu: options.fpu,
                fpu_root: options.fpu_root,
                moves_left_threshold: options.moves_left_threshold,
                moves_left_scale: options.moves_left_scale,
                moves_left_factor: options.moves_left_factor,
            };

            #[cfg(any(feature = "connect4", feature = "arimaa"))]
            let value_model = move |_options: &UGIOptions| MovesLeftValueModel::<_, _, _>::new();

            #[cfg(any(feature = "connect4", feature = "arimaa"))]
            let selection_strategy = move |options: &UGIOptions| {
                MovesLeftSelectionPolicy::new(cpuct(options), selection_strategy_opts(options))
            };

            #[cfg(feature = "quoridor")]
            let selection_strategy_opts = |options: &UGIOptions| VictoryMarginStrategyOptions {
                fpu: options.fpu,
                fpu_root: options.fpu_root,
                victory_margin_threshold: options.victory_margin_threshold,
                victory_margin_factor: options.victory_margin_factor,
            };

            #[cfg(feature = "quoridor")]
            let value_model =
                move |_options: &UGIOptions| VictoryMarginValueModel::<_, _, _>::new();

            #[cfg(feature = "quoridor")]
            let selection_strategy = move |options: &UGIOptions| {
                VictoryMarginSelectionPolicy::new(cpuct(options), selection_strategy_opts(options))
            };

            let time_strategy = TimeStrategy::new();

            run_ugi(
                ugi,
                engine,
                model,
                value_model,
                selection_strategy,
                time_strategy,
            )
            .await?
        }
        Commands::Perft(perft_args) => {
            let engine = Engine::new();

            let count = run_perft(perft_args.depth, &engine);
            println!(
                "Depth {depth}: {count}",
                depth = perft_args.depth,
                count = count
            );
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
