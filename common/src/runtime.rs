use std::future::Future;
use tokio::runtime::{Builder, Runtime};

pub fn block_on<F: Future>(future: F) -> F::Output {
    build_basic().block_on(future)
}

pub fn build_basic() -> Runtime {
    Builder::new().basic_scheduler().build().unwrap()
}
