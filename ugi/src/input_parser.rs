use core::panic;

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;

use crate::{InitialGameState, UGIOption};
use crate::{MoveStringToActions, ParseGameState};

pub struct InputParser<'a, M> {
    ugi_mapper: &'a M,
}

impl<'a, M> InputParser<'a, M> {
    pub fn new(ugi_mapper: &'a M) -> Self {
        Self { ugi_mapper }
    }
}

pub enum UGICommand<S, A> {
    UGI,
    IsReady,
    SetPosition(S),
    Go,
    GoPonder,
    MakeMove(Vec<A>),
    Focus(Vec<A>),
    ClearFocus,
    Quit,
    Stop,
    SetOption(UGIOption),
    Noop,
}

static MAKE_MOVE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^makemove\s+(.+)").unwrap());
static FOCUS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^focus\s+(.+)").unwrap());
static SET_POSITION_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^setposition\s+(.+)").unwrap());
static SET_OPTION_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^setoption\s+name\s+([_\-a-zA-Z0-9]+)\s+value\s+(.+)").unwrap());

impl<M, A, S> InputParser<'_, M>
where
    M: MoveStringToActions<Action = A> + ParseGameState<State = S> + InitialGameState<State = S>,
{
    pub fn parse_line(&self, line: &str) -> Result<UGICommand<S, A>> {
        match line {
            "ugi" => Ok(UGICommand::UGI),
            "isready" => Ok(UGICommand::IsReady),
            "newgame" => {
                let game_state = self.ugi_mapper.initial_game_state();
                Ok(UGICommand::SetPosition(game_state))
            }
            _ if { SET_POSITION_RE.is_match(line) } => {
                let cap = SET_POSITION_RE.captures(line).unwrap();
                let game_state = self.ugi_mapper.parse_game_state(&cap[1]);
                Ok(UGICommand::SetPosition(game_state))
            }
            "go" => Ok(UGICommand::Go),
            "go ponder" => Ok(UGICommand::GoPonder),
            _ if { MAKE_MOVE_RE.is_match(line) } => {
                let cap = MAKE_MOVE_RE.captures(line).unwrap();
                let actions = self.ugi_mapper.move_string_to_actions(&cap[1]);
                Ok(UGICommand::MakeMove(actions))
            }
            "focus clear" => Ok(UGICommand::ClearFocus),
            // focus Ee2n Ee3n Ee4n Ee5n
            _ if { FOCUS_RE.is_match(line) } => {
                let cap = FOCUS_RE.captures(line).unwrap();
                let actions = self.ugi_mapper.move_string_to_actions(&cap[1]);
                Ok(UGICommand::Focus(actions))
            }
            "quit" => Ok(UGICommand::Quit),
            "stop" => Ok(UGICommand::Stop),
            _ if { SET_OPTION_RE.is_match(line) } => {
                let cap = SET_OPTION_RE.captures(line).unwrap();
                let option_name = &cap[1];
                let option_value = &cap[2];

                let option = match option_name {
                    "fpu" => Some(UGIOption::Fpu(
                        option_value.parse().expect("Could not read option fpu"),
                    )),
                    "fpu_root" => Some(UGIOption::FpuRoot(
                        option_value
                            .parse()
                            .expect("Could not read option fpu_root"),
                    )),
                    "cpuct_base" => Some(UGIOption::CpuctBase(
                        option_value
                            .parse()
                            .expect("Could not read option cpuct_base"),
                    )),
                    "cpuct_init" => Some(UGIOption::CpuctInit(
                        option_value
                            .parse()
                            .expect("Could not read option cpuct_init"),
                    )),
                    "cpuct_factor" => Some(UGIOption::CpuctFactor(
                        option_value
                            .parse()
                            .expect("Could not read option cpuct_factor"),
                    )),
                    "cpuct_root_scaling" => Some(UGIOption::CpuctRootScaling(
                        option_value
                            .parse()
                            .expect("Could not read option cpuct_root_scaling"),
                    )),
                    "moves_left_threshold" => Some(UGIOption::MovesLeftThreshold(
                        option_value
                            .parse()
                            .expect("Could not read option moves_left_threshold"),
                    )),
                    "moves_left_scale" => Some(UGIOption::MovesLeftScale(
                        option_value
                            .parse()
                            .expect("Could not read option moves_left_scale"),
                    )),
                    "moves_left_factor" => Some(UGIOption::MovesLeftFactor(
                        option_value
                            .parse()
                            .expect("Could not read option moves_left_factor"),
                    )),
                    "fixed_time" => Some(UGIOption::FixedTime(Some(
                        option_value
                            .parse()
                            .expect("Could not read option fixed_time"),
                    ))),
                    "parallelism" => Some(UGIOption::Parallelism(
                        option_value
                            .parse()
                            .expect("Could not read option parallelism"),
                    )),
                    "visits" => Some(UGIOption::Visits(
                        option_value.parse().expect("Could not read option visits"),
                    )),
                    "max_visits" => Some(UGIOption::MaxVisits(
                        option_value
                            .parse()
                            .expect("Could not read option max_visits"),
                    )),
                    "alternative_action_threshold" => Some(UGIOption::AlternativeActionThreshold(
                        option_value
                            .parse()
                            .expect("Could not read option alternative_action_threshold"),
                    )),
                    "eee_mode" => Some(UGIOption::EEEMode(
                        option_value
                            .to_lowercase()
                            .parse()
                            .expect("Could not read option eee_mode"),
                    )),
                    "eee_reflective_symmetry" => Some(UGIOption::EEEReflectiveSymmetry(
                        option_value
                            .to_lowercase()
                            .parse()
                            .expect("Could not read option eee_reflective_symmetry"),
                    )),
                    "silver_setup" => Some(UGIOption::SilverSetup(option_value.to_owned())),
                    "gold_setup" => Some(UGIOption::GoldSetup(option_value.to_owned())),
                    "tcmove" => Some(UGIOption::TimePerMove(
                        option_value.parse().expect("Could not read option tcmove"),
                    )),
                    "time_buffer" => Some(UGIOption::TimeBuffer(
                        option_value
                            .parse()
                            .expect("Could not read option time_buffer"),
                    )),
                    "greserve" => Some(UGIOption::CurrentGReserveTime(
                        option_value
                            .parse()
                            .expect("Could not read option greserve"),
                    )),
                    "sreserve" => Some(UGIOption::CurrentSReserveTime(
                        option_value
                            .parse()
                            .expect("Could not read option sreserve"),
                    )),
                    "rated" => {
                        if std::env::var("NO_RATED").is_ok() && option_value.contains('1') {
                            panic!("NO_RATED is set and rated game was joined");
                        }
                        None
                    }
                    "display" => Some(UGIOption::Display(
                        option_value.parse().expect("Could not read option display"),
                    )),
                    "rating" => None,
                    "opponent" => None,
                    "opponent_rating" => None,
                    "moveused" => None,
                    "lastmoveused" => None,
                    "gused" => None,
                    "sused" => None,
                    "hash" => None,
                    _ => {
                        anyhow::bail!(format!(
                            "debug option {} w/ value {} is unknown",
                            option_name, option_value
                        ));
                    }
                };

                match option {
                    Some(option) => Ok(UGICommand::SetOption(option)),
                    None => Ok(UGICommand::Noop),
                }
            }
            cmd if cmd.is_empty() => Ok(UGICommand::Noop),
            _ => {
                anyhow::bail!("Command is unknown or not implemented");
            }
        }
    }
}
