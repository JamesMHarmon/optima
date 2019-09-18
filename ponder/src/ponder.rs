use std::str::FromStr;
use std::fmt::Display;
use std::fmt::Debug;
use serde::{Serialize};
use serde::de::DeserializeOwned;
use uuid::Uuid;
use failure::Error;

use common::linked_list::List;
use common::rng;
use mcts::mcts::{MCTS,MCTSOptions};
use model::analytics::GameAnalyzer;
use engine::engine::GameEngine;
use engine::game_state::GameState;
use model::model::Model;

#[derive(Debug)]
pub struct PonderOptions {
    pub visits: usize,
    pub cpuct_base: f32,
    pub cpuct_init: f32,
    pub cpuct_root_scaling: f32
}

pub struct Ponder {}

impl Ponder
{
    #[allow(non_snake_case)]
    pub async fn ponder<S, A, E, M, T>(
        model: &M,
        game_engine: &E,
        options: &PonderOptions
    ) -> Result<(), Error> 
    where
        S: GameState + Display,
        A: FromStr + Display + Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin,
        E: GameEngine<State=S,Action=A>,
        M: Model<State=S,Action=A,Analyzer=T>,
        T: GameAnalyzer<Action=A,State=S> + Send
    {
        let uuid = Uuid::new_v4();
        let cpuct_base = options.cpuct_base;
        let cpuct_init = options.cpuct_init;
        let cpuct_root_scaling = options.cpuct_root_scaling;
        let visits = options.visits;
        let analyzer = model.get_game_state_analyzer();

        let mut mcts_1 = MCTS::new(
            S::initial(),
            List::new(),
            game_engine,
            &analyzer,
            MCTSOptions::<S,A,_,_,_>::new(
                None,
                0.0,
                1.0,
                |_,_,_,Nsb,is_root| (((Nsb as f32 + cpuct_base + 1.0) / cpuct_base).ln() + cpuct_init) * if is_root { cpuct_root_scaling } else { 1.0 },
                |_,_| 0.0,
                rng::create_rng_from_uuid(uuid),
            )
        );

        let mut state: S = S::initial();
        let mut total_visits = 0;

        while game_engine.is_terminal_state(&state) == None {
            println!("Input action or Enter to ponder");
            let reader = std::io::stdin();
            let mut input = String::new();
            reader.read_line(&mut input)?;
            let input = input.trim();
            println!("Read: {}", input);

            if input == "" {
                println!("PONDERING: {}", visits);
                total_visits += visits;
                mcts_1.search(total_visits).await?;
                println!("{}", mcts_1.get_root_node_details()?);
                continue;
            }

            let action = input.parse::<A>();

            match action {
                Ok(action) => {
                    println!("Taking Action: {:?}", action);
                    state = game_engine.take_action(&state, &action);
                    mcts_1.advance_to_action(action).await?;
                    total_visits = 0;
                    println!("New Game State: {}", state);
                },
                Err(_) => println!("{}", "Error parsing action")
            }
        };

        Ok(())
    }
}
