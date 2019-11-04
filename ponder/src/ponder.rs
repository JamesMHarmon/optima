use std::str::FromStr;
use std::fmt::Display;
use std::fmt::Debug;
use serde::{Serialize, Deserialize};
use serde::de::DeserializeOwned;
use failure::Error;

use mcts::mcts::{MCTS,MCTSOptions};
use model::analytics::GameAnalyzer;
use engine::engine::GameEngine;
use engine::game_state::GameState;
use model::model::Model;

#[derive(Serialize, Deserialize, Debug)]
pub struct PonderOptions {
    pub visits: usize,
    pub parallelism: usize,
    pub fpu: f32,
    pub fpu_root: f32,
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
        E: GameEngine<State=S,Action=A,Value=M::Value>,
        M: Model<State=S,Action=A,Analyzer=T>,
        T: GameAnalyzer<Action=A,State=S,Value=M::Value> + Send
    {
        let fpu = options.fpu;
        let fpu_root = options.fpu_root;
        let cpuct_base = options.cpuct_base;
        let cpuct_init = options.cpuct_init;
        let cpuct_root_scaling = options.cpuct_root_scaling;
        let visits = options.visits;
        let analyzer = model.get_game_state_analyzer();

        let mut mcts = MCTS::with_capacity(
            S::initial(),
            0,
            game_engine,
            &analyzer,
            MCTSOptions::<S,_,_>::new(
                None,
                fpu,
                fpu_root,
                |_,_,Nsb,is_root| (((Nsb as f32 + cpuct_base + 1.0) / cpuct_base).ln() + cpuct_init) * if is_root { cpuct_root_scaling } else { 1.0 },
                |_,_| 0.0,
                options.parallelism
            ),
            visits
        );

        let mut state: S = S::initial();
        let mut total_visits = 0;

        while game_engine.is_terminal_state(&state).is_none() {
            println!("{}", state);

            println!("Input action or Enter to ponder");
            let reader = std::io::stdin();
            let mut input = String::new();
            reader.read_line(&mut input)?;
            let input = input.trim();
            println!("Read: {}", input);

            if input == "" {
                println!("PONDERING: {}", visits);
                total_visits += visits;
                mcts.search(total_visits).await?;
                println!("{}", mcts.get_root_node_details().await?);
                let pvs: Vec<_> = mcts.get_principal_variation().await?.iter().map(|n| format!("\n\t{:?}", n)).collect();
                println!("{}", pvs.join(""));
                continue;
            }

            let action = input.parse::<A>();

            match action {
                Ok(action) => {
                    if let Err(_) = mcts.advance_to_action(action.clone()).await {
                        println!("Illegal Action: {:?}", &action);
                        continue;
                    }

                    println!("Taking Action: {:?}", &action);
                    state = game_engine.take_action(&state, &action);
                    total_visits = 0;
                },
                Err(_) => println!("{}", "Error parsing action")
            }
        };

        println!("{}", state);

        Ok(())
    }
}
