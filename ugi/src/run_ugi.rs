use anyhow::Result;
use engine::{GameEngine, GameState, ValidActions};
use model::Analyzer;

use std::fmt::{Debug, Display};
use std::io::stdin;
use std::sync::Arc;

use crate::{log_debug, log_warning, output_ugi_cmd, output_ugi_info, Output};
use crate::{ActionsToMoveString, InitialGameState, MoveStringToActions, ParseGameState};
use crate::{GameManager, InputParser};

pub async fn run_ugi<M, E, S, A, U>(ugi_mapper: U, engine: E, model: M) -> Result<()>
where
    S: GameState + Clone + Display + Send + 'static,
    A: Debug + Eq + Clone + Send + 'static,
    U: MoveStringToActions<Action = A>
        + ParseGameState<State = S>
        + InitialGameState<State = S>
        + ActionsToMoveString<State = S, Action = A>
        + Send
        + Sync
        + 'static,
    E: GameEngine<State = S, Action = A> + ValidActions<State = S, Action = A> + Send + 'static,
    M: Analyzer<State = S, Action = A, Value = E::Value> + Send + 'static,
    M::Analyzer: Send,
{
    let ugi_mapper = Arc::new(ugi_mapper);

    let (game_manager, mut output_rx) = GameManager::new(ugi_mapper.clone(), engine, model);
    let input_parser = InputParser::new(&*ugi_mapper);

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
