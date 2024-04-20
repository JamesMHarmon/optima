#[allow(dead_code)]
pub fn log_debug(msg: &str) {
    println!("log Debug: {}", msg);
}

#[allow(dead_code)]
pub fn log_warning(msg: &str) {
    println!("log Warning: {}", msg);
}

#[allow(dead_code)]
pub fn log_error(msg: &str) {
    println!("log Error: {}", msg);
}

pub fn output_ugi_cmd(cmd: &str, msg: &str) {
    if !msg.is_empty() {
        println!("{} {}", cmd, msg);
    } else {
        println!("{}", cmd);
    }
}

pub fn output_ugi_info(msg: &str) {
    println!("info {}", msg);
}
