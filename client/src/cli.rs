use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[clap(author, version)]
#[clap(name = "Quoridor Engine Self Learning Client")]
#[clap(about = "Does awesome things", long_about = None)]
pub struct Cli {
    #[clap(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    SelfPlay(SelfPlayCommand),
    Arena(ArenaCommand),
    Init(InitCommand),
}

#[derive(Args)]
pub struct SelfPlayCommand {
    #[clap(short, long, default_value_t = String::from("client.conf"))]
    pub config: String,
}

#[derive(Args)]
pub struct ArenaCommand {
    #[clap(short, long, default_value_t = String::from("client.conf"))]
    pub config: String,
}

#[derive(Args)]
pub struct InitCommand {}
