use anyhow::Result;
use engine::{GameEngine, GameState, ValidActions};
use env_logger::Env;
use model::Analyzer;

use std::fmt::Debug;
use std::io::stdin;

use crate::{log_debug, log_warning, output_ugi_cmd, output_ugi_info, Output};
use crate::{ActionsToMoveString, InitialGameState, MoveStringToActions, ParseGameState};
use crate::{GameManager, InputParser};

pub async fn run_ugi<M, E, T, S, A, V, U>(ugi_mapper: U, engine: E, model: M) -> Result<()>
where
    S: GameState + Clone + Send,
    A: Debug + Eq + Clone + Send,
    U: MoveStringToActions<Action = A>
        + ParseGameState<State = S>
        + InitialGameState<State = S>
        + ActionsToMoveString<State = S, Action = A>
        + Clone
        + Sync,
    E: GameEngine<State = S, Action = A> + ValidActions<State = S, Action = A> + Sync,
    M: Analyzer<State = S, Action = A, Value = E::Value> + Sync,
{
    env_logger::Builder::from_env(Env::default().default_filter_or("warn")).init();

    let (game_manager, mut output_rx) = GameManager::new(ugi_mapper.clone(), engine, model);
    let input_parser = InputParser::new(ugi_mapper.clone());

    tokio::spawn(async move {
        while let Some(output) = output_rx.recv().await {
            match output {
                Output::Command(cmd, value) => output_ugi_cmd(&cmd, &value),
                Output::Info(msg) => output_ugi_info(&msg),
                Output::Debug(msg) => log_debug(&msg),
            }
        }
    });

    loop {
        let mut buffer = String::new();
        stdin().read_line(&mut buffer)?;
        let buffer = buffer.trim();
        let command = input_parser.parse_line(buffer);

        match command {
            Ok(command) => game_manager.command(command).await,
            Err(e) => {
                log_warning(&e.to_string());
            }
        }
    }
}
