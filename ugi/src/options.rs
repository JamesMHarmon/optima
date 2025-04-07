pub struct UGIOptions {
    pub fpu: f32,
    pub fpu_root: f32,
    pub cpuct_base: f32,
    pub cpuct_init: f32,
    pub cpuct_factor: f32,
    pub cpuct_root_scaling: f32,
    pub eee_mode: bool,
    pub eee_reflective_symmetry: bool,
    pub victory_margin_threshold: f32,
    pub victory_margin_factor: f32,
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
    pub display: bool,
    pub multi_pv: usize,
}

pub enum UGIOption {
    Fpu(f32),
    FpuRoot(f32),
    CpuctBase(f32),
    CpuctInit(f32),
    CpuctFactor(f32),
    CpuctRootScaling(f32),
    EEEMode(bool),
    EEEReflectiveSymmetry(bool),
    VictoryMarginThreshold(f32),
    VictoryMarginFactor(f32),
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
    Display(bool),
    MultiPV(usize),
}

#[allow(clippy::new_without_default)]
impl UGIOptions {
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
            victory_margin_threshold: 0.95,
            victory_margin_factor: 0.05,
            parallelism: 512,
            visits: 0,
            max_visits: 10_000_000,
            time_buffer: 1.0,
            reserve_time_to_use: 1.0 / 20.0,
            current_g_reserve_time: 0.0,
            current_s_reserve_time: 0.0,
            time_per_move: 10.0,
            alternative_action_threshold: 0.00,
            silver_setup: "".to_string(),
            gold_setup: "".to_string(),
            fixed_time: None,
            display: true,
            multi_pv: 1,
        }
    }

    pub fn set_option(&mut self, option: UGIOption) {
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
            UGIOption::VictoryMarginThreshold(victory_margin_threshold) => {
                self.victory_margin_threshold = victory_margin_threshold
            }
            UGIOption::VictoryMarginFactor(victory_margin_factor) => {
                self.victory_margin_factor = victory_margin_factor
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
            UGIOption::Display(display) => self.display = display,
            UGIOption::MultiPV(multi_pv) => self.multi_pv = multi_pv,
        };
    }
}
