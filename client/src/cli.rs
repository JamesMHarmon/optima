use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[clap(author, version)]
#[clap(name = "Optima Engine Self Learning Client")]
#[clap(about = "Does awesome things", long_about = None)]
pub struct Cli {
    #[clap(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    SelfPlay(SelfPlayCommand),
    Arena(ArenaCommand),
    Ugi(UgiCommand),
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
#[clap(name = "ugi-client")]
#[clap(about = "Runs the game through cmd line interface similar to the UCI Universal Chess Interface.", long_about = None)]
pub struct UgiCommand {
    #[clap(short, long)]
    pub dir: Option<String>,

    #[clap(short, long)]
    pub model: Option<String>,
}
