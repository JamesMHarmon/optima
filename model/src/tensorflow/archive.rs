use anyhow::{anyhow, Context, Result};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use tar::Header;
use tempfile::tempdir;
use tempfile::TempDir;

use super::super::ModelInfo;
use super::TensorflowModelOptions;

pub fn archive(
    archive: impl AsRef<Path>,
    model: impl AsRef<Path>,
    model_options: &TensorflowModelOptions,
    model_info: &ModelInfo,
) -> Result<()> {
    let file = File::create(archive)?;
    let enc = GzEncoder::new(file, Compression::default());
    let mut builder = tar::Builder::new(enc);

    append_json_file(&mut builder, "model-options.json", model_options)?;
    append_json_file(&mut builder, "model-info.json", model_info)?;

    builder.append_dir_all("model", model)?;

    // Finishes writing to the archive.
    builder.into_inner()?;

    Ok(())
}

pub fn unarchive<P: AsRef<Path>>(
    archive: P,
) -> Result<(TempDir, TensorflowModelOptions, ModelInfo)> {
    let file = File::open(archive)?;
    let enc = GzDecoder::new(file);
    let mut archive = tar::Archive::new(enc);
    let mut model_options: Option<TensorflowModelOptions> = None;
    let mut model_info: Option<ModelInfo> = None;
    let temp_dir = tempdir()?;
    println!(
        "Creating a temporary path to load the model: {:?}",
        temp_dir.path()
    );

    for file in archive.entries()? {
        let mut file = file?;
        let path = file.header().path()?;
        let model_prefix = Path::new("model/");

        if path.ends_with(Path::new("model-options.json")) {
            model_options = Some(serde_json::from_reader(file)?);
        } else if path.ends_with(Path::new("model-info.json")) {
            model_info = Some(serde_json::from_reader(file)?);
        } else if dbg!(&path).starts_with(model_prefix) {
            let dest = dbg!(path.strip_prefix("model/")?);

            let dest = dbg!(temp_dir.path().join(&dest));
            file.unpack(&dest)?;
        }
    }

    Ok((
        temp_dir,
        model_options.context("Expected options to exist in model archive")?,
        model_info.context("Expected info to exist in model archive")?,
    ))
}

pub fn read_archived_info<P: AsRef<Path>>(archive: P) -> Result<ModelInfo> {
    let file = File::open(archive)?;
    let enc = GzDecoder::new(file);
    let mut archive = tar::Archive::new(enc);

    for file in archive.entries()? {
        let file = file?;
        let path = file.header().path()?;
        if path.ends_with(Path::new("model-info.json")) {
            return Ok(serde_json::from_reader(file)?);
        }
    }

    Err(anyhow!("Could not find model info within archived model."))
}

pub fn read_archived_options<P: AsRef<Path>>(archive: P) -> Result<TensorflowModelOptions> {
    let file = File::open(archive)?;
    let enc = GzDecoder::new(file);
    let mut archive = tar::Archive::new(enc);

    for file in archive.entries()? {
        let file = file?;
        let path = file.header().path()?;
        if path.ends_with(Path::new("model-options.json")) {
            return Ok(serde_json::from_reader(file)?);
        }
    }

    Err(anyhow!(
        "Could not find model options within archived model."
    ))
}

fn append_json_file(
    builder: &mut tar::Builder<impl Write>,
    path: impl AsRef<Path>,
    data: &impl Serialize,
) -> Result<()> {
    let data = serde_json::to_string(data)?;
    let data = data.as_bytes();

    let mut header = Header::new_gnu();
    header.set_path(path)?;
    header.set_size(data.len() as u64);
    header.set_cksum();

    builder.append(&header, data)?;

    Ok(())
}
