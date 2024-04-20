pub struct UGIOptions<S> {
    pub fpu: f32,
    pub fpu_root: f32,
    pub cpuct_base: f32,
    pub cpuct_init: f32,
    pub cpuct_factor: f32,
    pub cpuct_root_scaling: f32,
    pub eee_mode: bool,
    pub eee_reflective_symmetry: bool,
    pub moves_left_threshold: f32,
    pub moves_left_scale: f32,
    pub moves_left_factor: f32,
    pub parallelism: usize,
    pub visits: usize,
    pub max_visits: usize,
    pub time_buffer: f32,
    pub reserve_time_to_use: f32,
    pub current_g_reserve_time: f32,
    pub current_s_reserve_time: f32,
    pub time_per_move: f32,
    pub alternative_action_threshold: f32,
    pub silver_setup: String,
    pub gold_setup: String,
    pub fixed_time: Option<f32>,
    pub initial_game_state: Option<S>,
}

pub enum UGIOption<S> {
    Fpu(f32),
    FpuRoot(f32),
    CpuctBase(f32),
    CpuctInit(f32),
    CpuctFactor(f32),
    CpuctRootScaling(f32),
    EEEMode(bool),
    EEEReflectiveSymmetry(bool),
    MovesLeftThreshold(f32),
    MovesLeftScale(f32),
    MovesLeftFactor(f32),
    Parallelism(usize),
    Visits(usize),
    MaxVisits(usize),
    TimeBuffer(f32),
    CurrentGReserveTime(f32),
    CurrentSReserveTime(f32),
    TimePerMove(f32),
    AlternativeActionThreshold(f32),
    SilverSetup(String),
    GoldSetup(String),
    FixedTime(Option<f32>),
    InitialGameState(S),
}

impl<S> UGIOptions<S> {
    pub fn new() -> Self {
        UGIOptions {
            fpu: 0.0,
            fpu_root: 1.0,
            cpuct_base: 19652.0,
            cpuct_init: 1.25,
            cpuct_factor: 2.4,
            cpuct_root_scaling: 1.0,
            eee_mode: false,
            eee_reflective_symmetry: false,
            moves_left_threshold: 0.95,
            moves_left_scale: 10.0,
            moves_left_factor: 0.05,
            parallelism: 512,
            visits: 0,
            max_visits: 10_000_000,
            time_buffer: 1.0,
            reserve_time_to_use: 1.0 / 20.0,
            current_g_reserve_time: 0.0,
            current_s_reserve_time: 0.0,
            time_per_move: 0.0,
            alternative_action_threshold: 0.00,
            silver_setup: "".to_string(),
            gold_setup: "".to_string(),
            fixed_time: None,
            initial_game_state: None,
        }
    }

    pub fn set_option(&mut self, option: UGIOption<S>) {
        match option {
            UGIOption::Fpu(fpu) => self.fpu = fpu,
            UGIOption::FpuRoot(fpu_root) => self.fpu_root = fpu_root,
            UGIOption::CpuctBase(cpuct_base) => self.cpuct_base = cpuct_base,
            UGIOption::CpuctInit(cpuct_init) => self.cpuct_init = cpuct_init,
            UGIOption::CpuctFactor(cpuct_factor) => self.cpuct_factor = cpuct_factor,
            UGIOption::CpuctRootScaling(cpuct_root_scaling) => {
                self.cpuct_root_scaling = cpuct_root_scaling
            }
            UGIOption::EEEMode(eee_mode) => self.eee_mode = eee_mode,
            UGIOption::EEEReflectiveSymmetry(eee_reflective_symmetry) => {
                self.eee_reflective_symmetry = eee_reflective_symmetry
            }
            UGIOption::MovesLeftThreshold(moves_left_threshold) => {
                self.moves_left_threshold = moves_left_threshold
            }
            UGIOption::MovesLeftScale(moves_left_scale) => self.moves_left_scale = moves_left_scale,
            UGIOption::MovesLeftFactor(moves_left_factor) => {
                self.moves_left_factor = moves_left_factor
            }
            UGIOption::Parallelism(parallelism) => self.parallelism = parallelism,
            UGIOption::Visits(visits) => self.visits = visits,
            UGIOption::MaxVisits(max_visits) => self.max_visits = max_visits,
            UGIOption::TimeBuffer(time_buffer) => self.time_buffer = time_buffer,
            UGIOption::CurrentGReserveTime(current_g_reserve_time) => {
                self.current_g_reserve_time = current_g_reserve_time
            }
            UGIOption::CurrentSReserveTime(current_s_reserve_time) => {
                self.current_s_reserve_time = current_s_reserve_time
            }
            UGIOption::TimePerMove(time_per_move) => self.time_per_move = time_per_move,
            UGIOption::AlternativeActionThreshold(alternative_action_threshold) => {
                self.alternative_action_threshold = alternative_action_threshold
            }
            UGIOption::SilverSetup(silver_setup) => self.silver_setup = silver_setup,
            UGIOption::GoldSetup(gold_setup) => self.gold_setup = gold_setup,
            UGIOption::FixedTime(fixed_time) => self.fixed_time = fixed_time,
            UGIOption::InitialGameState(initial_game_state) => {
                self.initial_game_state = Some(initial_game_state)
            }
        };
    }
}
