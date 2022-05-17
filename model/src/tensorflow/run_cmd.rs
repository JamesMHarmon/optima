use std::process::{Command, Stdio};
use anyhow::Result;
use log::info;

pub fn run_cmd(cmd: &str) -> Result<()> {
    info!("\n");
    info!("{}", cmd);
    info!("\n");

    let mut cmd = Command::new("/bin/bash")
        .arg("-c")
        .arg(cmd)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;

    let result = cmd.wait();

    info!("OUTPUT: {:?}", result);

    Ok(())
}
