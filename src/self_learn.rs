use std::io::Write;
use std::fs::{create_dir_all, OpenOptions};
use std::env;
use std::path::Path;
use serde::{Serialize, Deserialize};

// game/run/iteration/
//                  ./games
//                  ./nets
pub struct SelfLearn {
    options: SelfLearnOptions
    // game_engine: E,
}

#[derive(Serialize, Deserialize)]
pub struct SelfLearnOptions {
    pub run_name: String,
    pub number_of_games_per_net: usize,
    pub moving_window_size: usize,
    pub train_ratio: f64,
    pub train_batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub policy_loss_weight: f64,
    pub value_loss_weight: f64,
    pub temperature: f64,
    pub visits: usize,
    pub cpuct: f64,
    pub number_of_filters: usize,
    pub number_of_residual_blocks: usize
}

impl SelfLearn {
    pub fn new(game_name: String, options: SelfLearnOptions) -> Result<(), &'static str> {
        Ok(SelfLearn::initialize_directories_and_files(game_name, &options)?)
    }

    fn initialize_directories_and_files(game_name: String, options: &SelfLearnOptions) -> Result<(), &'static str> {
        let directory_name = format!("./{}_runs/{}", game_name, options.run_name);
        
        create_dir_all(&directory_name).expect("Run already exists or unable to create directories");

        let config_path_name = format!("{}/config.json", directory_name);
        let config_path = Path::new(&config_path_name);

        if config_path.exists() {
            return Err("Run already exists");
        }

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(config_path)
            .expect("Couldn't open or create the config file");

        let serialized_options = serde_json::to_string_pretty(options).expect("Unable to serialize options");
        file.write(serialized_options.as_bytes()).expect("Unable to write options to file");

        Ok(())
    }
}

//     set_python_paths();
//     create_model();

//     let game_engine = Connect4Engine::new();
//     let mut file = OpenOptions::new()
//         .append(true)
//         .create(true)
//         .open("results.txt")
//         .expect("Couldn't open or create the results.txt file");

//     loop {
//         let mut analysis_cache = AnalysisCache::new();

//         let now = Instant::now();
//         let self_play_metrics = self_play::self_play(&game_engine, &mut analysis_cache)?;
//         let time = now.elapsed().as_millis();

//         let serialized = serde_json::to_string(&self_play_metrics).expect("Failed to serialize results");

//         writeln!(file, "{}", serialized).expect("File to write to results.txt.");

//         println!("{:?}", self_play_metrics);
//         println!("TIME: {}",time);
//     }


// // @TODO: Improve error handling
// fn set_python_paths() {
//     let gil = Python::acquire_gil();
//     let py = gil.python();

//     let current_dir_result = env::current_dir().unwrap();
//     let env_path = current_dir_result.to_str().ok_or("Path not valid").unwrap();
//     println!("Env Path: {}", env_path);

//     let sys = py.import("sys").unwrap();
//     let path = sys.get("path").unwrap().downcast_ref::<PyList>().unwrap();

//     path.call_method("append", (env_path.to_owned(), ), None).unwrap();
//     path.call_method("append", ("/anaconda3/lib/python3.6".to_owned(), ), None).unwrap();
//     path.call_method("append", ("/anaconda3/lib/python3.6/lib-dynload".to_owned(), ), None).unwrap();
//     path.call_method("append", ("/anaconda3/lib/python3.6/site-packages", ), None).unwrap();
// }

// fn create_model() {
//     let gil = Python::acquire_gil();
//     let py = gil.python();
//     let c4 = py.import("c4_model").unwrap();

//     c4.call("create_model", (), None).unwrap();
// }