use std::path::{Path, PathBuf};

use anyhow::Result;

pub trait FsExt {
    // Converts the provided relative path to be based from the path of the currently working directory.
    // If the path is absolute, then it returns the absolute path.
    fn relative_to_cwd(&self) -> Result<PathBuf>
    where
        Self: AsRef<Path>,
    {
        let cwd_dir = std::env::current_dir()?;

        Ok(cwd_dir.join(self))
    }
}

impl FsExt for String {}

impl FsExt for &'static str {}
