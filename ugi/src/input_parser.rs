use core::panic;

use anyhow::{anyhow, Result};
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
    Details,
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
                self.ugi_mapper
                    .parse_game_state(&cap[1])
                    .map(|game_state| UGICommand::SetPosition(game_state))
            }
            "go" => Ok(UGICommand::Go),
            "go ponder" => Ok(UGICommand::GoPonder),
            _ if { MAKE_MOVE_RE.is_match(line) } => {
                let cap = MAKE_MOVE_RE.captures(line).unwrap();
                let actions = self.ugi_mapper.move_string_to_actions(&cap[1]);
                actions.map(|actions| UGICommand::MakeMove(actions))
            }
            "focus clear" => Ok(UGICommand::ClearFocus),
            // focus Ee2n Ee3n Ee4n Ee5n
            _ if { FOCUS_RE.is_match(line) } => {
                let cap = FOCUS_RE.captures(line).unwrap();
                let actions = self.ugi_mapper.move_string_to_actions(&cap[1]);
                actions.map(|actions| UGICommand::Focus(actions))
            }
            "quit" => Ok(UGICommand::Quit),
            "stop" => Ok(UGICommand::Stop),
            _ if { SET_OPTION_RE.is_match(line) } => {
                let cap = SET_OPTION_RE.captures(line).unwrap();
                let option_name = &cap[1];
                let option_value = &cap[2];

                let option = match option_name {
                    "fpu" => Some(UGIOption::Fpu(parse_option_value("fpu", option_value)?)),
                    "fpu_root" => Some(UGIOption::FpuRoot(parse_option_value(
                        "fpu_root",
                        option_value,
                    )?)),
                    "cpuct_base" => Some(UGIOption::CpuctBase(parse_option_value(
                        "cpuct_base",
                        option_value,
                    )?)),
                    "cpuct_init" => Some(UGIOption::CpuctInit(parse_option_value(
                        "cpuct_init",
                        option_value,
                    )?)),
                    "cpuct_factor" => Some(UGIOption::CpuctFactor(parse_option_value(
                        "cpuct_factor",
                        option_value,
                    )?)),
                    "cpuct_root_scaling" => Some(UGIOption::CpuctRootScaling(parse_option_value(
                        "cpuct_root_scaling",
                        option_value,
                    )?)),
                    "victory_margin_threshold" => Some(UGIOption::VictoryMarginThreshold(
                        parse_option_value("victory_margin_threshold", option_value)?,
                    )),
                    "victory_margin_factor" => Some(UGIOption::VictoryMarginFactor(
                        parse_option_value("moves_left_factor", option_value)?,
                    )),
                    "fixed_time" => Some(UGIOption::FixedTime(Some(parse_option_value(
                        "fixed_time",
                        option_value,
                    )?))),
                    "parallelism" => Some(UGIOption::Parallelism(parse_option_value(
                        "parallelism",
                        option_value,
                    )?)),
                    "visits" => Some(UGIOption::Visits(parse_option_value(
                        "visits",
                        option_value,
                    )?)),
                    "max_visits" => Some(UGIOption::MaxVisits(parse_option_value(
                        "max_visits",
                        option_value,
                    )?)),
                    "alternative_action_threshold" => Some(UGIOption::AlternativeActionThreshold(
                        parse_option_value("alternative_action_threshold", option_value)?,
                    )),
                    "eee_mode" => Some(UGIOption::EEEMode(parse_option_value(
                        "eee_mode",
                        &option_value.to_lowercase(),
                    )?)),
                    "eee_reflective_symmetry" => {
                        Some(UGIOption::EEEReflectiveSymmetry(parse_option_value(
                            "eee_reflective_symmetry",
                            &option_value.to_lowercase(),
                        )?))
                    }
                    "silver_setup" => Some(UGIOption::SilverSetup(option_value.to_owned())),
                    "gold_setup" => Some(UGIOption::GoldSetup(option_value.to_owned())),
                    "tcmove" => Some(UGIOption::TimePerMove(parse_option_value(
                        "tcmove",
                        option_value,
                    )?)),
                    "time_buffer" => Some(UGIOption::TimeBuffer(parse_option_value(
                        "time_buffer",
                        option_value,
                    )?)),
                    "greserve" => Some(UGIOption::CurrentGReserveTime(parse_option_value(
                        "greserve",
                        option_value,
                    )?)),
                    "sreserve" => Some(UGIOption::CurrentSReserveTime(parse_option_value(
                        "sreserve",
                        option_value,
                    )?)),
                    "rated" => {
                        if std::env::var("NO_RATED").is_ok() && option_value.contains('1') {
                            panic!("NO_RATED is set and rated game was joined");
                        }
                        None
                    }
                    "display" => Some(UGIOption::Display(parse_option_value(
                        "display",
                        option_value,
                    )?)),
                    "multipv" => Some(UGIOption::MultiPV(parse_option_value(
                        "multipv",
                        option_value,
                    )?)),
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
            "details" => Ok(UGICommand::Details),
            "" => Ok(UGICommand::Noop),
            _ => {
                anyhow::bail!("Command is unknown or not implemented");
            }
        }
    }
}

fn parse_option_value<T: std::str::FromStr>(option_name: &str, option_value: &str) -> Result<T> {
    option_value
        .parse::<T>()
        .map_err(|_e| anyhow!("Could not read option '{}'", option_name))
}
