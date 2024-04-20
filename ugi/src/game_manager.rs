use crate::{ActionsToMoveString, InitialGameState, UGICommand, UGIOption, UGIOptions};
use engine::{GameEngine, GameState, ValidActions};
use mcts::{MCTSOptions, MCTS, PUCT};
use model::Analyzer;
use rand::seq::IteratorRandom;
use rand::thread_rng;
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{str, thread};
use tokio::sync::mpsc;

const NAME: &str = "UGI";
const AUTHOR: &str = "Author";

pub struct GameManager<S, A> {
    command_channel: mpsc::Sender<CommandInner<A>>,
    output: OutputHandle,
    options: Arc<Mutex<UGIOptions<S>>>,
    ponder_active: Arc<AtomicBool>,
}

enum CommandInner<A> {
    Ponder,
    Go,
    MakeMove(Vec<A>),
    FocusActions(Vec<A>),
    ClearFocus,
    NewGame,
}

pub enum Output {
    Command(String, String),
    Info(String),
    Debug(String),
}

impl<S, A> GameManager<S, A> {
    pub fn set_option(&self, option: UGIOption<S>) {
        self.options.lock().unwrap().set_option(option);
    }

    pub async fn command(&self, command: UGICommand<S, A>) {
        match command {
            UGICommand::UGI => {
                self.output.cmd("protocol-version", "1");
                self.output.cmd("id name", NAME);
                self.output.cmd("id author", AUTHOR);
                self.output.cmd("ugiok", "");
            }
            UGICommand::IsReady => {
                self.output.cmd("readyok", "");
            }
            UGICommand::NewGame => self.send_command(CommandInner::NewGame).await,
            UGICommand::SetPosition(state) => {
                self.set_option(UGIOption::InitialGameState(state));
            }
            UGICommand::Go => {
                self.ponder_active.store(false, Ordering::SeqCst);
                self.send_command(CommandInner::Go).await
            }
            UGICommand::GoPonder => {
                self.ponder_active.store(true, Ordering::SeqCst);
                self.send_command(CommandInner::Ponder).await
            }
            UGICommand::MakeMove(actions) => {
                self.ponder_active.store(false, Ordering::SeqCst);
                self.send_command(CommandInner::MakeMove(actions)).await
            }
            UGICommand::ClearFocus => self.send_command(CommandInner::ClearFocus).await,
            UGICommand::Focus(actions) => {
                self.send_command(CommandInner::FocusActions(actions)).await
            }
            UGICommand::Quit => panic!("Quit command is not implemented"),
            UGICommand::Stop => {
                self.ponder_active.store(false, Ordering::SeqCst);
            }
            UGICommand::SetOption(option) => self.set_option(option),
            UGICommand::Noop => {}
        }
    }

    async fn send_command(&self, command: CommandInner<A>) {
        if self.command_channel.send(command).await.is_err() {
            panic!("Failed to Send Command");
        }
    }
}

impl<S, A> GameManager<S, A>
where
    S: GameState + Clone + Send + 'static,
    A: Debug + Eq + Clone + Send + 'static,
{
    pub fn new<U, E, M>(
        ugi_mapper: Arc<U>,
        engine: E,
        model: M,
    ) -> (Self, mpsc::UnboundedReceiver<Output>)
    where
        U: InitialGameState<State = S>
            + ActionsToMoveString<State = S, Action = A>
            + Send
            + Sync
            + 'static,
        E: GameEngine<State = S, Action = A> + ValidActions<State = S, Action = A> + Send + 'static,
        M: Analyzer<State = S, Action = A, Value = E::Value> + Send + 'static,
        M::Analyzer: Send,
    {
        let (command_tx, command_rx) = mpsc::channel(1);
        let (output_tx, output_rx) = mpsc::unbounded_channel();
        let output = OutputHandle { output_tx };
        let options = Arc::new(Mutex::new(init_options()));
        let ponder_active = Arc::new(AtomicBool::new(false));

        let game_manager = Self {
            command_channel: command_tx,
            output: output.clone(),
            options: options.clone(),
            ponder_active: ponder_active.clone(),
        };

        let game_manager_inner = GameManagerInner::new(
            command_rx,
            output,
            options,
            ponder_active,
            ugi_mapper,
            engine,
            model,
        );

        let handle = tokio::runtime::Handle::current();
        thread::spawn(move || {
            let mut game_manager_inner = game_manager_inner;
            handle.block_on(async { game_manager_inner.run_game_loop().await });
        });

        (game_manager, output_rx)
    }
}

pub struct GameManagerInner<S, A, U, E, M> {
    command_rx: mpsc::Receiver<CommandInner<A>>,
    output: OutputHandle,
    options: Arc<Mutex<UGIOptions<S>>>,
    ponder_active: Arc<AtomicBool>,
    ugi_mapper: Arc<U>,
    engine: E,
    model: M,
}

impl<S, A, U, E, M> GameManagerInner<S, A, U, E, M>
where
    S: GameState + Clone,
    A: Debug + Eq + Clone,
    U: InitialGameState<State = S> + ActionsToMoveString<State = S, Action = A>,
    E: GameEngine<State = S, Action = A> + ValidActions<State = S, Action = A>,
    M: Analyzer<State = S, Action = A, Value = E::Value>,
{
    fn new(
        command_rx: mpsc::Receiver<CommandInner<A>>,
        output: OutputHandle,
        options: Arc<Mutex<UGIOptions<S>>>,
        ponder_active: Arc<AtomicBool>,
        ugi_mapper: Arc<U>,
        engine: E,
        model: M,
    ) -> Self {
        Self {
            command_rx,
            output,
            options,
            ponder_active,
            ugi_mapper,
            engine,
            model,
        }
    }

    async fn run_game_loop(&mut self) {
        let mut mcts = None;
        let mut game_state: <U as InitialGameState>::State = self.ugi_mapper.initial_game_state();
        let mut focus_game_state = game_state.clone();

        let options = self.options.clone();
        let ponder_active = self.ponder_active.clone();
        let analyzer = self.model.analyzer();

        while let Some(command) = self.command_rx.recv().await {
            if mcts.is_none() {
                let options = options.lock().unwrap();
                let cpuct_base = options.cpuct_base;
                let cpuct_init = options.cpuct_init;
                let cpuct_factor = options.cpuct_factor;
                let cpuct_root_scaling = options.cpuct_root_scaling;
                game_state = options
                    .initial_game_state
                    .as_ref()
                    .expect("Initial Game State is not defined.")
                    .clone();
                focus_game_state = game_state.clone();

                mcts = Some(MCTS::with_capacity(
                    game_state.clone(),
                    &self.engine,
                    &analyzer,
                    MCTSOptions::<S, _, _>::new(
                        None,
                        options.fpu,
                        options.fpu_root,
                        move |_, nsb, is_root| {
                            (cpuct_init
                                + cpuct_factor
                                    * ((nsb as f32 + cpuct_base + 1.0) / cpuct_base).ln())
                                * if is_root { cpuct_root_scaling } else { 1.0 }
                        },
                        |_| 0.0,
                        0.0,
                        options.moves_left_threshold,
                        options.moves_left_scale,
                        options.moves_left_factor,
                        options.parallelism,
                    ),
                    options.visits,
                ));
            }

            let mcts = mcts.as_mut().expect("MCTS should have been created");

            match command {
                CommandInner::NewGame => {
                    let option = UGIOption::InitialGameState(self.ugi_mapper.initial_game_state());
                    self.set_option(option)
                }
                CommandInner::Ponder => {
                    let options_lock = options.lock().unwrap();

                    if options_lock.visits == 0 {
                        let start_time = Instant::now();
                        let max_visits = options_lock.max_visits;

                        drop(options_lock);

                        self.output.info("ponder started");
                        mcts.search(|visits| {
                            ponder_active.load(Ordering::SeqCst) && visits < max_visits
                        })
                        .await
                        .unwrap();
                        self.output.info(&format!(
                            "ponder ended. duration: {:?}",
                            start_time.elapsed()
                        ));
                    } else {
                        self.output
                            .info("skipping ponder as options are fixed visits.");
                    }
                }
                CommandInner::MakeMove(actions) => {
                    // @TODO: Add this for arimaa
                    // let actions = self
                    //     .ugi_mapper
                    //     .convert_to_valid_composite_actions(&actions, &game_state);

                    self.output
                        .info(&format!("Updating tree with actions: {:?}", &actions));

                    for action in actions {
                        if !self.engine.valid_actions(&game_state).contains(&action) {
                            panic!("The action {:?} is not valid", action);
                        }
                        if self.engine.terminal_state(&game_state).is_some() {
                            panic!("The game is already in a terminal state");
                        }

                        game_state = self.engine.take_action(&game_state, &action);
                        mcts.advance_to_action_retain(action).await.unwrap();
                    }

                    focus_game_state = game_state.clone();
                }
                CommandInner::FocusActions(actions) => {
                    for action in actions {
                        if !self
                            .engine
                            .valid_actions(&focus_game_state)
                            .contains(&action)
                        {
                            panic!("The action {:?} is not valid", action);
                        }
                        if self.engine.terminal_state(&focus_game_state).is_some() {
                            panic!("The game is already in a terminal state");
                        }
                        focus_game_state = self.engine.take_action(&focus_game_state, &action);
                        mcts.add_focus_to_action(action);
                    }
                }
                CommandInner::ClearFocus => {
                    focus_game_state = game_state.clone();
                    mcts.clear_focus();
                }
                CommandInner::Go => {
                    let search_start = Instant::now();
                    let pre_action_game_state = focus_game_state.clone();
                    let mut focus_game_state = focus_game_state.clone();
                    let focused_actions = mcts.get_focused_actions().to_vec();
                    let current_player = self.engine.player_to_move(&focus_game_state);
                    let move_number = self.engine.move_number(&focus_game_state);
                    let options = &*options.lock().unwrap();

                    let search_duration = search_duration(options, current_player);

                    let mut actions = Vec::new();
                    let mut depths = Vec::new();
                    let mut visits = Vec::new();
                    let mut scores = Vec::new();
                    let mut moves_left = Vec::new();
                    while self.engine.player_to_move(&focus_game_state) == current_player
                        && self.engine.terminal_state(&focus_game_state).is_none()
                    {
                        let depth = if options.visits != 0 {
                            self.output
                                .info(&format!("search visits: {}", options.visits));
                            mcts.search_visits(options.visits).await.unwrap()
                        } else {
                            self.output
                                .info(&format!("search duration: {:?}", search_duration));
                            mcts.search_time_max_visits(search_duration, options.max_visits)
                                .await
                                .unwrap()
                        };

                        // Safety step to ensure there is at least one visit for the steps when in play phase and using time. An additional visit is added for PUCT to select the best action.
                        mcts.search_visits(2).await.unwrap();

                        depths.push(depth);

                        let node_details = mcts
                            .get_focus_node_details()
                            .unwrap()
                            .expect("There should have been at least one visit");

                        let (action, best_node_puct) = choose_action(
                            &node_details.children,
                            options.alternative_action_threshold,
                        );

                        scores.push(best_node_puct.Qsa);
                        moves_left.push((best_node_puct.M - move_number as f32).max(0.0));
                        visits.push(node_details.visits);
                        mcts.add_focus_to_action(action.clone());

                        focus_game_state = self.engine.take_action(&focus_game_state, action);
                        actions.push(action.clone());
                    }

                    mcts.clear_focus();
                    for action in focused_actions {
                        mcts.add_focus_to_action(action);
                    }

                    let pv = mcts.get_principal_variation().unwrap();
                    let actions = pv.iter().map(|(a, _)| a).cloned().collect::<Vec<_>>();
                    let move_string = self
                        .ugi_mapper
                        .actions_to_move_string(&pre_action_game_state, &actions);

                    self.output.info_val("pv", &move_string);
                    self.output
                        .info_val("time", &search_start.elapsed().as_secs().to_string());
                    self.output.info_val(
                        "root_score",
                        &format!("{:.3}", scores.first().unwrap_or(&0.5)),
                    );
                    self.output
                        .info_val("score", &format!("{:.3}", scores.last().unwrap_or(&0.5)));
                    self.output.info_val(
                        "moves_left",
                        &format!("{:.1}", moves_left.last().unwrap_or(&0.0)),
                    );
                    self.output
                        .info_val("visits", &visits.iter().max().unwrap_or(&0).to_string());
                    self.output
                        .info_val("depth", &depths.iter().max().unwrap_or(&0).to_string());

                    let move_string = self
                        .ugi_mapper
                        .actions_to_move_string(&pre_action_game_state, &actions);

                    self.output.cmd("bestmove", &move_string)
                }
            }
        }
    }

    fn set_option(&self, option: UGIOption<S>) {
        self.options.lock().unwrap().set_option(option);
    }
}

#[derive(Clone)]
struct OutputHandle {
    output_tx: mpsc::UnboundedSender<Output>,
}

impl OutputHandle {
    fn cmd(&self, cmd_name: &str, cmd_value: &str) {
        self.output_tx
            .send(Output::Command(cmd_name.to_string(), cmd_value.to_string()))
            .expect("Failed to send output");
    }

    fn info(&self, msg: &str) {
        self.output_tx
            .send(Output::Info(msg.to_string()))
            .expect("Failed to send output");
    }

    fn info_val(&self, msg: &str, value: &str) {
        self.output_tx
            .send(Output::Info(format!("{} {}", msg, value.to_string())))
            .expect("Failed to send output");
    }
}

fn choose_action<A>(actions: &[(A, PUCT)], alternative_action_threshold: f32) -> &(A, PUCT) {
    let max_visits = actions
        .iter()
        .map(|(_, puct)| puct.Nsa)
        .max()
        .expect("Expected at least one action");
    let visit_threshold = max_visits - (max_visits as f32 * alternative_action_threshold) as usize;
    let mut rng = thread_rng();
    actions
        .iter()
        .filter(|(_, puct)| puct.Nsa >= visit_threshold)
        .choose(&mut rng)
        .expect("Expected at least one action")
}

fn init_options<S>() -> UGIOptions<S> {
    let mut options = UGIOptions::new();

    if let Some(visits) = std::env::var("BOT_VISITS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        options.set_option(UGIOption::Visits(visits));
    }

    if let Some(bot_fixed_time) = std::env::var("BOT_FIXED_TIME")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        options.set_option(UGIOption::FixedTime(Some(bot_fixed_time)));
    }

    options
}

fn search_duration<S>(options: &UGIOptions<S>, current_player: usize) -> Duration {
    let current_g_reserve_time = options.current_g_reserve_time;
    let current_s_reserve_time = options.current_s_reserve_time;
    let reserve_time_to_use = options.reserve_time_to_use;
    let time_per_move = options.time_per_move;
    let fixed_time = options.fixed_time;
    let time_buffer = options.time_buffer;

    let reserve_time: f32 = if current_player == 1 {
        current_g_reserve_time
    } else {
        current_s_reserve_time
    };
    let reserve_time: f32 = reserve_time.min(reserve_time - time_per_move).max(0.0);
    let search_time: f32 = reserve_time * reserve_time_to_use + time_per_move;
    let search_time = search_time - time_buffer - time_per_move * 0.05;
    let search_time: f32 = fixed_time.unwrap_or(search_time);

    std::time::Duration::from_secs_f32(0f32.max(search_time))
}
