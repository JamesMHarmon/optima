use anyhow::{anyhow, Result};
use retry::delay::Fixed;
use retry::retry;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use log::warn;

pub fn latest(model_dir: &Path) -> Result<PathBuf> {
    let num_retries = 2;
    let seconds_between_retries = 5;

    let file = retry(
        Fixed::from(Duration::from_secs(seconds_between_retries)).take(num_retries),
        || {
            let file = fs::read_dir(model_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_ok_and(|f| f.is_file()))
                .filter_map(|f| {
                    f.metadata()
                        .ok()
                        .and_then(|m| m.created().ok())
                        .map(|m| (f, m))
                })
                .max_by_key(|(_, m)| *m);

            file.ok_or_else(|| {
                warn!("Model does not exist in {:?}", model_dir);
                anyhow!("Model does not exist")
            })
        },
    );

    file.map(|(p, _)| p.path())
        .map_err(|_| anyhow!("Failed to find model {:?}", model_dir))
}
