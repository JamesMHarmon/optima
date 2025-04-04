use crate::{ActionsToMoveString, InitialGameState, UGICommand, UGIOption, UGIOptions};
use common::{PropagatedGameLength, PropagatedValue};
use engine::{GameEngine, GameState, ValidActions};
use itertools::Itertools;
use mcts::{
    BackpropagationStrategy, EdgeDetails, NodeDetails, SelectionStrategy, MCTS,
};
use model::Analyzer;
use rand::seq::IteratorRandom;
use rand::thread_rng;
use std::fmt::{Debug, Display};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{str, thread};
use tokio::sync::mpsc;

const NAME: &str = "UGI";
const AUTHOR: &str = "Author";

pub struct GameManager<S, A> {
    command_channel: mpsc::Sender<CommandInner<S, A>>,
    output: OutputHandle,
    options: Arc<Mutex<UGIOptions>>,
    ponder_active: Arc<AtomicBool>,
}

enum CommandInner<S, A> {
    Ponder,
    Go,
    MakeMove(Vec<A>),
    FocusActions(Vec<A>),
    ClearFocus,
    SetPosition(S),
}

pub enum Output {
    Command(String, String),
    Info(String),
    Debug(String),
}

impl<S, A> GameManager<S, A> {
    pub fn set_option(&self, option: UGIOption) {
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
            UGICommand::SetPosition(state) => {
                self.send_command(CommandInner::SetPosition(state)).await
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

    async fn send_command(&self, command: CommandInner<S, A>) {
        if self.command_channel.send(command).await.is_err() {
            panic!("Failed to Send Command");
        }
    }
}

impl<S, A> GameManager<S, A>
where
    S: GameState + Clone + Display + Send + 'static,
    A: Display + Debug + Eq + Clone + Send + 'static,
{
    pub fn new<U, E, M, FnB, B, FnSel, Sel>(
        ugi_mapper: Arc<U>,
        engine: E,
        model: M,
        backpropagation_strategy: FnB,
        selection_strategy: FnSel,
    ) -> (Self, mpsc::UnboundedReceiver<Output>)
    where
        U: InitialGameState<State = S>
            + ActionsToMoveString<State = S, Action = A>
            + Send
            + Sync
            + 'static,
        E: GameEngine<State = S, Action = A> + ValidActions<State = S, Action = A> + Send + 'static,
        M: Analyzer<State = S, Action = A, Predictions = E::Terminal> + Send + 'static,
        B: BackpropagationStrategy<State = S, Action = A, Predictions = E::Terminal>
            + Send
            + 'static,
        FnB: Fn(&UGIOptions) -> B + Send + 'static,
        FnSel: Fn(&UGIOptions) -> Sel + Send + 'static,
        Sel: SelectionStrategy<
                State = S,
                Action = A,
                Predictions = E::Terminal,
                PropagatedValues = B::PropagatedValues,
            > + Send
            + 'static,
        M::Analyzer: Send,
        B::PropagatedValues: PropagatedValue + PropagatedGameLength + Default + Ord,
        E::Terminal: Clone,
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
            backpropagation_strategy,
            selection_strategy,
        );

        let handle = tokio::runtime::Handle::current();
        thread::spawn(move || {
            let mut game_manager_inner = game_manager_inner;
            handle.block_on(async { game_manager_inner.run_game_loop().await });
        });

        (game_manager, output_rx)
    }
}

pub struct GameManagerInner<S, A, U, E, M, FnB, FnSel> {
    command_rx: mpsc::Receiver<CommandInner<S, A>>,
    output: OutputHandle,
    options: Arc<Mutex<UGIOptions>>,
    ponder_active: Arc<AtomicBool>,
    ugi_mapper: Arc<U>,
    engine: E,
    model: M,
    backpropagation_strategy: FnB,
    selection_strategy: FnSel,
}

#[allow(clippy::too_many_arguments)]
impl<S, A, U, E, M, B, FnB, FnSel, Sel> GameManagerInner<S, A, U, E, M, FnB, FnSel>
where
    S: GameState + Clone + Display,
    A: Display + Debug + Eq + Clone,
    U: InitialGameState<State = S> + ActionsToMoveString<State = S, Action = A>,
    E: GameEngine<State = S, Action = A> + ValidActions<State = S, Action = A>,
    M: Analyzer<State = S, Action = A, Predictions = E::Terminal>,
    B: BackpropagationStrategy<State = S, Action = A, Predictions = E::Terminal>,
    FnB: Fn(&UGIOptions) -> B,
    FnSel: Fn(&UGIOptions) -> Sel,
    Sel: SelectionStrategy<
        State = S,
        Action = A,
        Predictions = E::Terminal,
        PropagatedValues = B::PropagatedValues,
    >,
    B::PropagatedValues: PropagatedValue + PropagatedGameLength + Default + Ord,
    E::Terminal: Clone,
{
    fn new(
        command_rx: mpsc::Receiver<CommandInner<S, A>>,
        output: OutputHandle,
        options: Arc<Mutex<UGIOptions>>,
        ponder_active: Arc<AtomicBool>,
        ugi_mapper: Arc<U>,
        engine: E,
        model: M,
        backpropagation_strategy: FnB,
        selection_strategy: FnSel,
    ) -> Self {
        Self {
            command_rx,
            output,
            options,
            ponder_active,
            ugi_mapper,
            engine,
            model,
            backpropagation_strategy,
            selection_strategy,
        }
    }

    async fn run_game_loop(&mut self) {
        let mut mcts_container = None;
        let mut game_state = self.ugi_mapper.initial_game_state();
        let mut focus_game_state = game_state.clone();

        let options: Arc<Mutex<UGIOptions>> = self.options.clone();
        let ponder_active = self.ponder_active.clone();
        let analyzer = self.model.analyzer();

        let mut backpropagation_strategy;
        let mut selection_strategy;

        while let Some(command) = self.command_rx.recv().await {
            if mcts_container.is_none() {
                let options = options.lock().unwrap();

                backpropagation_strategy = Some((self.backpropagation_strategy)(&options));
                selection_strategy = Some((self.selection_strategy)(&options));

                mcts_container = Some(MCTS::with_capacity(
                    game_state.clone(),
                    &self.engine,
                    &analyzer,
                    backpropagation_strategy.as_ref().unwrap(),
                    selection_strategy.as_ref().unwrap(),
                    options.visits,
                    options.parallelism,
                ));
            }

            let mcts = mcts_container
                .as_mut()
                .expect("MCTS should have been created");

            match command {
                CommandInner::SetPosition(state) => {
                    game_state = state;
                    focus_game_state = game_state.clone();
                    mcts_container = None;
                    self.display_board(&game_state);
                }
                CommandInner::Ponder => {
                    let (visits, max_visits);

                    {
                        let options_lock = options.lock().unwrap();
                        visits = options_lock.visits;
                        max_visits = options_lock.max_visits;
                    }

                    if visits == 0 {
                        let start_time = Instant::now();
                        let mut last_output = start_time;

                        self.output.info("ponder started");

                        while ponder_active.load(Ordering::SeqCst)
                            && self.command_rx.is_empty()
                            && (max_visits == 0 || mcts.num_focus_node_visits() < max_visits)
                        {
                            let depth = mcts
                                .search(|visits| {
                                    ponder_active.load(Ordering::SeqCst)
                                        && visits < max_visits
                                        && last_output.elapsed().as_secs() < 1
                                })
                                .await
                                .unwrap();

                            last_output = Instant::now();

                            let node_details = mcts
                                .get_focus_node_details()
                                .unwrap()
                                .expect("There should have been at least one visit");

                            let multi_pv = self.options.lock().unwrap().multi_pv;
                            let pv = node_details
                                .children
                                .iter()
                                .take(multi_pv)
                                .filter_map(|edge| {
                                    mcts.get_principal_variation(Some(&edge.action), 10).ok()
                                })
                                .collect_vec();

                            let best_node = choose_action(&node_details.children, 0.0);

                            let player_to_move = self.engine.player_to_move(&focus_game_state);
                            let game_length = best_node.propagated_values.game_length().max(0.0);

                            self.output_post_search_info(
                                player_to_move,
                                &pv,
                                &focus_game_state,
                                start_time,
                                &[best_node.Qsa()],
                                &[node_details.visits],
                                &[game_length],
                                &[depth],
                                &node_details,
                            );
                        }

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
                        let is_valid_action =
                            self.engine.valid_actions(&game_state).contains(&action);

                        if !is_valid_action {
                            self.output
                                .info(&format!("The action {:?} is not valid", action));

                            break;
                        }

                        let is_terminal = self.engine.terminal_state(&game_state).is_some();
                        if is_terminal {
                            self.output.info("The game is already in a terminal state");

                            break;
                        }

                        game_state = self.engine.take_action(&game_state, &action);
                        mcts.advance_to_action_retain(action).await.unwrap();
                    }

                    focus_game_state = game_state.clone();
                    self.display_board(&game_state);
                }
                CommandInner::FocusActions(actions) => {
                    for action in actions {
                        let is_valid_action = self
                            .engine
                            .valid_actions(&focus_game_state)
                            .contains(&action);

                        if !is_valid_action {
                            self.output
                                .info(&format!("The action {:?} is not valid", action));

                            break;
                        }

                        let is_terminal = self.engine.terminal_state(&focus_game_state).is_some();
                        if is_terminal {
                            self.output.info("The game is already in a terminal state");

                            break;
                        }

                        focus_game_state = self.engine.take_action(&focus_game_state, &action);
                        mcts.add_focus_to_action(action);
                    }

                    self.display_board(&focus_game_state);
                }
                CommandInner::ClearFocus => {
                    focus_game_state = game_state.clone();
                    mcts.clear_focus();

                    self.display_board(&game_state);
                }
                CommandInner::Go => {
                    let search_start = Instant::now();
                    let pre_action_game_state = focus_game_state.clone();
                    let mut focus_game_state = focus_game_state.clone();
                    let focused_actions = mcts.get_focused_actions().to_vec();
                    let current_player = self.engine.player_to_move(&focus_game_state);

                    let (
                        options_visits,
                        options_max_visits,
                        options_alternative_action_threshold,
                        search_duration,
                    );

                    {
                        let options = options.lock().unwrap();
                        options_visits = options.visits;
                        options_max_visits = options.max_visits;
                        options_alternative_action_threshold = options.alternative_action_threshold;
                        search_duration = calc_search_duration(&options, current_player);
                    }

                    let mut actions = Vec::new();
                    let mut depths = Vec::new();
                    let mut visits = Vec::new();
                    let mut game_lengths = Vec::new();
                    let mut scores = Vec::new();
                    let mut node_details_container = None;
                    while self.engine.player_to_move(&focus_game_state) == current_player
                        && self.engine.terminal_state(&focus_game_state).is_none()
                    {
                        let depth = if options_visits != 0 {
                            self.output
                                .info(&format!("search visits: {}", options_visits));
                            mcts.search_visits(options_visits).await.unwrap()
                        } else {
                            self.output
                                .info(&format!("search duration: {:?}", search_duration));
                            mcts.search_time_max_visits(search_duration, options_max_visits)
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

                        let best_node = choose_action(
                            &node_details.children,
                            options_alternative_action_threshold,
                        );

                        scores.push(best_node.Qsa());
                        game_lengths.push(best_node.propagated_values.game_length());
                        visits.push(node_details.visits);
                        mcts.add_focus_to_action(best_node.action.clone());

                        focus_game_state = self
                            .engine
                            .take_action(&focus_game_state, &best_node.action);
                        actions.push(best_node.action.clone());

                        if node_details_container.is_none() {
                            node_details_container = Some(node_details);
                        }
                    }

                    mcts.clear_focus();
                    for action in focused_actions {
                        mcts.add_focus_to_action(action);
                    }

                    let pv = mcts
                        .get_principal_variation(None, 10)
                        .into_iter()
                        .collect_vec();

                    let node_details =
                        node_details_container.expect("Expected node_details to have been set");

                    self.output_post_search_info(
                        current_player,
                        &pv,
                        &pre_action_game_state,
                        search_start,
                        &scores,
                        &visits,
                        &game_lengths,
                        &depths,
                        &node_details,
                    );

                    let move_string = self
                        .ugi_mapper
                        .actions_to_move_string(&pre_action_game_state, &actions);
                    self.output.cmd("bestmove", &move_string);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn output_post_search_info<PV: PropagatedValue + PropagatedGameLength>(
        &self,
        player_to_move: usize,
        pv: &[Vec<EdgeDetails<A, PV>>],
        pre_action_game_state: &S,
        search_start: Instant,
        scores: &[f32],
        visits: &[usize],
        game_lengths: &[f32],
        depths: &[usize],
        node_details: &NodeDetails<A, PV>,
    ) {
        self.output.info(&format!(
            "time {time} playertomove {playertomove} score {score:.3} visits {visits} game_length {game_length:.3} depth {depth}",
            time = search_start.elapsed().as_secs(),
            playertomove = player_to_move,
            score = scores.first().unwrap_or(&0.5),
            visits = visits.iter().max().unwrap_or(&0),
            game_length = game_lengths.last().unwrap_or(&0.0),
            depth = depths.iter().max().unwrap_or(&0)
        ));

        let visits_sum = (node_details.visits - 1).max(1);

        for (i, (edge, pv)) in node_details.children.iter().zip(pv).enumerate() {
            let pv_actions = pv
                .iter()
                .map(|edge| &edge.action)
                .cloned()
                .collect::<Vec<_>>();
            let pv_string = self
                .ugi_mapper
                .actions_to_move_string(pre_action_game_state, &pv_actions);

            self.output.info(&format!(
                "multipv {pv_num} score {score:.3} visits {visits} visitspct {visitspct:.3} game_length {game_length:.3} pv {pv}",
                pv_num = i + 1,
                score = edge.propagated_values.value(),
                visits = edge.Nsa,
                visitspct = edge.Nsa as f32 / visits_sum as f32,
                game_length = edge.propagated_values.game_length(),
                pv = &pv_string,
            ));
        }
    }

    fn display_board(&self, game_state: &S) {
        if self.options.lock().unwrap().display {
            game_state
                .to_string()
                .lines()
                .for_each(|line| self.output.info(line));
        }
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
}

fn choose_action<A, PV>(
    edges: &[EdgeDetails<A, PV>],
    alternative_action_threshold: f32,
) -> &EdgeDetails<A, PV> {
    let max_visits = edges
        .iter()
        .map(|details| details.Nsa)
        .max()
        .expect("Expected at least one action");
    let visit_threshold = max_visits - (max_visits as f32 * alternative_action_threshold) as usize;
    let mut rng = thread_rng();
    edges
        .iter()
        .filter(|details| details.Nsa >= visit_threshold)
        .choose(&mut rng)
        .expect("Expected at least one action")
}

fn init_options() -> UGIOptions {
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

fn calc_search_duration(options: &UGIOptions, current_player: usize) -> Duration {
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
