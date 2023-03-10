use anyhow::Result;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::fmt::Display;
use std::str::FromStr;

use engine::engine::GameEngine;
use engine::game_state::GameState;
use mcts::mcts::{MCTSOptions, MCTS};
use model::analytics::GameAnalyzer;
use model::model::Model;

#[derive(Serialize, Deserialize, Debug)]
pub struct PlayOptions {
    pub visits: usize,
    pub parallelism: usize,
    pub fpu: f32,
    pub fpu_root: f32,
    pub cpuct_base: f32,
    pub cpuct_init: f32,
    pub cpuct_root_scaling: f32,
    pub moves_left_threshold: f32,
    pub moves_left_scale: f32,
    pub moves_left_factor: f32,
}

pub struct Play {}

impl Play {
    #[allow(non_snake_case)]
    pub async fn play<S, A, E, M, T>(
        model: &M,
        game_engine: &E,
        options: &PlayOptions,
    ) -> Result<()>
    where
        S: GameState + Display,
        A: FromStr + Display + Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin,
        E: GameEngine<State = S, Action = A, Value = M::Value>,
        M: Model<State = S, Action = A, Analyzer = T>,
        T: GameAnalyzer<Action = A, State = S, Value = M::Value> + Send,
    {
        let cpuct_base = options.cpuct_base;
        let cpuct_init = options.cpuct_init;
        let cpuct_root_scaling = options.cpuct_root_scaling;
        let visits = options.visits;
        let analyzer = model.get_game_state_analyzer();
        let mut actions: Vec<A> = vec![];

        'outer: loop {
            let mut state: S = S::initial();
            let mut total_visits = 0;

            let mut mcts = MCTS::with_capacity(
                S::initial(),
                game_engine,
                &analyzer,
                MCTSOptions::<S, _, _>::new(
                    None,
                    options.fpu,
                    options.fpu_root,
                    |_, Nsb, is_root| {
                        (((Nsb as f32 + cpuct_base + 1.0) / cpuct_base).ln() + cpuct_init)
                            * if is_root { cpuct_root_scaling } else { 1.0 }
                    },
                    |_| 0.0,
                    0.0,
                    options.moves_left_threshold,
                    options.moves_left_scale,
                    options.moves_left_factor,
                    options.parallelism,
                ),
                visits,
            );

            for action in actions.iter() {
                if mcts.advance_to_action_retain(action.clone()).await.is_err() {
                    println!("Illegal Action: {:?}", &action);
                    continue;
                }
                state = game_engine.take_action(&state, action);
            }

            while game_engine.is_terminal_state(&state).is_none() {
                println!("{}", state);

                println!("Input action or Enter to play");
                let reader = std::io::stdin();
                let mut input = String::new();
                reader.read_line(&mut input)?;
                let input = input.trim();
                println!("Read: {}", input);

                if input.is_empty() {
                    println!("PLAYING: {}", visits);
                    total_visits += visits;
                    mcts.search_visits(total_visits).await?;
                    println!("{:?}", mcts.get_focus_node_details()?);
                    let pvs: Vec<_> = mcts
                        .get_principal_variation()?
                        .iter()
                        .map(|n| format!("\n\t{:?}", n))
                        .collect();
                    println!("{}", pvs.join(""));
                    continue;
                }

                let inputs = input.split(&[',', ' '][..]).filter_map(|v| {
                    let trimmed = v.trim();
                    if !trimmed.is_empty() {
                        Some(trimmed)
                    } else {
                        None
                    }
                });

                if input == "help" {
                    println!("help: Displays the available commands.");
                    println!("moves: Lists the previous moves to get to this state.");
                    println!("undo: Undoes the last action.");
                    println!("{{A}}: Takes the actions.");
                    println!("{{A,A,..}}: Takes a comma deliminated list of actions.");
                }

                if input == "undo" {
                    actions.pop();
                    continue 'outer;
                }

                if input == "moves" {
                    println!(
                        "{}",
                        actions
                            .iter()
                            .map(|a| a.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                    continue;
                }

                for input in inputs {
                    let action = input.parse::<A>();

                    match action {
                        Ok(action) => {
                            if mcts.advance_to_action_retain(action.clone()).await.is_err() {
                                println!("Illegal Action: {:?}", &action);
                                continue;
                            }

                            println!("Taking Action: {:?}", &action);
                            state = game_engine.take_action(&state, &action);
                            actions.push(action);
                            total_visits = 0;
                        }
                        Err(_) => println!("Error parsing action"),
                    }
                }
            }

            println!("{}", state);

            break 'outer;
        }

        Ok(())
    }
}
