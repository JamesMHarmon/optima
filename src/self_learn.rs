use super::model::Model;
use std::io::Write;
use std::io::Read;
use std::fs::{create_dir_all, OpenOptions};
use std::path::{Path,PathBuf};
use serde::{Serialize, Deserialize};

// game/run/iteration/
//                  ./games
//                  ./nets
pub struct SelfLearn<M>
    where M: Model
{
    options: SelfLearnOptions,
    game_name: String,
    run_name: String,
    model: M,
    run_directory: PathBuf
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfLearnOptions {
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

impl<M> SelfLearn<M> 
    where M: Model
{
    pub fn new(game_name: String, run_name: String, mut model: M, options: SelfLearnOptions) -> Result<Self, &'static str>
    {
        if game_name.contains("_") {
            return Err("game_name cannot contain any '_' characters");
        }

        if run_name.contains("_") {
            return Err("run_name cannot contain any '_' characters");
        }

        let run_directory = SelfLearn::<M>::initialize_directories_and_files(&game_name, &run_name, &options)?;
        let model_name = Self::get_model_name(&game_name, &run_name, 1);

        model.create(&model_name);

        Ok(Self {
            game_name,
            run_name,
            run_directory,
            model,
            options
        })
    }

    pub fn from(game_name: String, run_name: String, model: M) -> Result<Self, &'static str> {
        let run_directory = Self::get_run_directory(&game_name, &run_name);
        let options = Self::get_config(&run_directory)?;

        // TODO: get latest model name and games.

        Ok(Self {
            game_name,
            run_name,
            run_directory,
            model,
            options
        })
    }

    pub fn learn(&self) {
        // create a net
        // get number of games left to play

        loop {
            // load the net
            // play n games
            // train
        }
    }

    fn get_number_of_games_to_play() {

    }

    fn get_model_name(game_name: &str, run_name: &str, model_number: usize) -> String {
        format!("{}_{}_{:0>5}", game_name, run_name, model_number)
    }

    fn get_run_directory(game_name: &str, run_name: &str) -> PathBuf {
        PathBuf::from(format!("./{}_runs/{}", game_name, run_name))
    }

    fn get_config_path(run_directory: &Path) -> PathBuf {
        run_directory.join("config.json")
    }

    fn get_config(run_directory: &Path) -> Result<SelfLearnOptions, &'static str> {
        let config_path = Self::get_config_path(run_directory);
        let mut file = OpenOptions::new()
            .read(true)
            .open(config_path)
            .expect("Couldn't load config file.");

        let mut config_file_contents = String::new();
        file.read_to_string(&mut config_file_contents).expect("Failed to read config file");
        let options: SelfLearnOptions = serde_json::from_str(&config_file_contents).expect("Failed to parse config file");
        Ok(options)
    }

    fn initialize_directories_and_files(game_name: &str, run_name: &str, options: &SelfLearnOptions) -> Result<PathBuf, &'static str> {
        let run_directory = SelfLearn::<M>::get_run_directory(game_name, run_name);
        create_dir_all(&run_directory).expect("Run already exists or unable to create directories");

        let config_path = Self::get_config_path(&run_directory);

        if config_path.exists() {
            return Err("Run already exists");
        }

        println!("{:?}", run_directory);
        println!("{:?}", config_path);

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(config_path)
            .expect("Couldn't open or create the config file");

        let serialized_options = serde_json::to_string_pretty(options).expect("Unable to serialize options");
        file.write(serialized_options.as_bytes()).expect("Unable to write options to file");

        Ok(run_directory)
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