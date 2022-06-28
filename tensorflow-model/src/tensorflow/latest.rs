use std::{path::{PathBuf, Path}, fs, time::Duration};
use anyhow::Result;

use log::warn;

pub fn latest(model_dir: &Path) -> Result<PathBuf> {
    let mut file;

    loop {
        file = fs::read_dir(model_dir)?
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_ok_and(|f| f.is_file()))
            .filter_map(|f| {
                f.metadata()
                    .ok()
                    .and_then(|m| m.created().ok())
                    .map(|m| (f, m))
            })
            .max_by_key(|(_, m)| m.clone());

        if file.is_some() {
            break;
        }

        warn!("No models found in the directory {:?}", model_dir);

        std::thread::sleep(Duration::from_secs(10));
    }

    Ok(file.unwrap().0.path())
}