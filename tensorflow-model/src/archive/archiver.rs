use anyhow::{Context, Result, anyhow};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use log::info;
use serde::Serialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tar::Header;

use super::super::tensorflow::TensorflowModelOptions;
use ::model::{Analyzer, GameAnalyzer, GameStateAnalysis, ModelInfo, RequestId};

/// Marker file written after a model archive has been fully extracted into
/// its cache directory. Its presence means the cache directory is safe to
/// reuse as-is; its absence (including from a prior extraction that was
/// interrupted, e.g. by the process being killed) means the directory
/// should be (re-)extracted from scratch.
const EXTRACTED_MARKER: &str = ".extracted";

pub struct Archive<M> {
    inner: M,
    // Keep this reference around for the lifetime of the model. The
    // directory itself is a persistent, content-addressed cache (see
    // `unarchive`) rather than a temp dir, so there's nothing to clean up
    // here -- this just keeps the path alive alongside the model.
    _cache_dir: Arc<PathBuf>,
}

impl<M> Archive<M> {
    pub fn new(model: M, cache_dir: PathBuf) -> Self {
        Self {
            inner: model,
            _cache_dir: Arc::new(cache_dir),
        }
    }

    pub fn inner(&self) -> &M {
        &self.inner
    }
}

pub struct ArchiveAnalyzer<A> {
    inner: A,
    _cache_dir: Arc<PathBuf>,
}

impl<An> GameAnalyzer for ArchiveAnalyzer<An>
where
    An: GameAnalyzer,
{
    type Action = An::Action;
    type State = An::State;
    type Predictions = An::Predictions;

    fn send(&self, request_id: RequestId, game_state: &Self::State) {
        self.inner.send(request_id, game_state)
    }

    fn recv(
        &self,
    ) -> (
        RequestId,
        GameStateAnalysis<Self::Action, Self::Predictions>,
    ) {
        self.inner.recv()
    }

    fn analyze(
        &self,
        game_state: &Self::State,
    ) -> GameStateAnalysis<Self::Action, Self::Predictions> {
        self.inner.analyze(game_state)
    }
}

impl<M: Analyzer> Analyzer for Archive<M> {
    type State = M::State;
    type Action = M::Action;
    type Predictions = M::Predictions;
    type Analyzer = ArchiveAnalyzer<M::Analyzer>;

    fn analyzer(&self) -> ArchiveAnalyzer<M::Analyzer> {
        ArchiveAnalyzer {
            inner: self.inner.analyzer(),
            _cache_dir: self._cache_dir.clone(),
        }
    }
}

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

/// Extracts a model archive into a persistent, content-addressed cache
/// directory, reusing a prior extraction when one already exists instead of
/// re-extracting into a fresh temp directory on every call (previously this
/// used `tempfile::tempdir()`, which relies on its `Drop` impl to clean up --
/// something that never runs if the process is killed rather than exiting
/// normally, which is common for these bots. That leaked an extracted copy
/// of the model, tens of MB, into the OS temp dir on every restart).
///
/// Concurrent callers (e.g. multiple game sessions loading the same model at
/// once) extract into their own pid-namespaced staging directory and publish
/// it into the shared cache path with a single `rename`, which POSIX
/// guarantees is atomic on the same filesystem. A reader checking `marker`
/// therefore only ever sees "not extracted yet" or "fully extracted", never
/// a directory that another process is still writing into. Whichever caller
/// loses the race discards its own (redundant) staging copy and uses the
/// winner's.
pub fn unarchive<P: AsRef<Path>>(
    archive: P,
) -> Result<(PathBuf, TensorflowModelOptions, ModelInfo)> {
    let archive_path = archive.as_ref();
    let cache_dir = cache_dir_for(archive_path)?;
    let marker = cache_dir.join(EXTRACTED_MARKER);

    if marker.exists() {
        info!("Reusing cached model extraction at {:?}", cache_dir);
        let model_options = read_json_file(&cache_dir.join("model-options.json"))?;
        let model_info = read_json_file(&cache_dir.join("model-info.json"))?;
        return Ok((cache_dir, model_options, model_info));
    }

    let cache_root = cache_dir
        .parent()
        .context("Cache directory unexpectedly has no parent")?;
    fs::create_dir_all(cache_root)
        .with_context(|| format!("Failed to create cache root {:?}", cache_root))?;
    cleanup_stale_staging_dirs(cache_root, &cache_dir);

    let staging_dir = staging_dir_for(&cache_dir);
    // We own this exact path -- it's namespaced by our own pid -- so if
    // something is already there it can only be left over from an earlier
    // process that reused this pid; safe to clear before (re-)populating it.
    let _ = fs::remove_dir_all(&staging_dir);
    fs::create_dir_all(&staging_dir)
        .with_context(|| format!("Failed to create staging directory {:?}", staging_dir))?;

    info!("Extracting model into staging directory: {:?}", staging_dir);
    let (model_options, model_info) = extract_into(archive_path, &staging_dir)?;

    // Cache the parsed options/info alongside the extracted model files so a
    // cache hit can be served with a couple of small local file reads,
    // rather than re-decoding the (gzipped) archive from scratch just to
    // pull two small JSON blobs back out of it.
    fs::write(
        staging_dir.join("model-options.json"),
        serde_json::to_string(&model_options)?,
    )?;
    fs::write(
        staging_dir.join("model-info.json"),
        serde_json::to_string(&model_info)?,
    )?;

    // Write the marker only after everything above succeeded, so a staging
    // directory from an interrupted extraction (e.g. the process is killed
    // mid-way) is never published as if it were complete.
    fs::write(staging_dir.join(EXTRACTED_MARKER), "")?;

    match fs::rename(&staging_dir, &cache_dir) {
        Ok(()) => {}
        // `rename` onto an existing non-empty directory fails: another
        // process's staging dir already won the race and got published
        // first. Its extraction is from the same source archive, so its
        // result is equally valid -- just drop our redundant copy.
        Err(_) if marker.exists() => {
            let _ = fs::remove_dir_all(&staging_dir);
        }
        Err(err) => {
            return Err(err)
                .with_context(|| format!("Failed to publish cache directory {:?}", cache_dir));
        }
    }

    Ok((cache_dir, model_options, model_info))
}

fn extract_into(
    archive_path: &Path,
    dest_dir: &Path,
) -> Result<(TensorflowModelOptions, ModelInfo)> {
    let file =
        File::open(archive_path).with_context(|| format!("Failed to open {:?}", archive_path))?;
    let enc = GzDecoder::new(file);
    let mut archive = tar::Archive::new(enc);
    let mut model_options: Option<TensorflowModelOptions> = None;
    let mut model_info: Option<ModelInfo> = None;

    for file in archive.entries()? {
        let mut file = file?;
        let path = file.header().path()?;
        let model_prefix = Path::new("model/");

        if path.ends_with(Path::new("model-options.json")) {
            model_options = Some(serde_json::from_reader(file)?);
        } else if path.ends_with(Path::new("model-info.json")) {
            model_info = Some(serde_json::from_reader(file)?);
        } else if path.starts_with(model_prefix) {
            let dest = path.strip_prefix("model/")?;

            let dest = dest_dir.join(dest);
            file.unpack(&dest)?;
        }
    }

    let model_options = model_options.context("Expected options to exist in model archive")?;
    let model_info = model_info.context("Expected info to exist in model archive")?;

    Ok((model_options, model_info))
}

fn read_json_file<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path).with_context(|| format!("Failed to open {:?}", path))?;
    Ok(serde_json::from_reader(file)?)
}

/// Derives the pid-namespaced staging directory a given cache directory is
/// extracted into before being atomically published (see `unarchive`).
fn staging_dir_for(cache_dir: &Path) -> PathBuf {
    let key = cache_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model");
    cache_dir.with_file_name(format!("{key}{STAGING_SUFFIX}{}", std::process::id()))
}

/// Removes staging directories left behind by processes that were killed
/// mid-extraction (and so never reached the publishing `rename`) and are no
/// longer running, so they don't accumulate indefinitely. Best-effort: any
/// failure here just means cleanup happens on a later call instead.
fn cleanup_stale_staging_dirs(cache_root: &Path, cache_dir: &Path) {
    let prefix = format!(
        "{}{STAGING_SUFFIX}",
        cache_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model")
    );

    let Ok(entries) = fs::read_dir(cache_root) else {
        return;
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(pid) = name.strip_prefix(&prefix).and_then(|p| p.parse::<u32>().ok()) else {
            continue;
        };

        if !is_pid_alive(pid) {
            info!(
                "Removing stale staging directory from dead pid {pid}: {:?}",
                entry.path()
            );
            let _ = fs::remove_dir_all(entry.path());
        }
    }
}

const STAGING_SUFFIX: &str = ".staging-";

fn is_pid_alive(pid: u32) -> bool {
    Path::new("/proc").join(pid.to_string()).exists()
}

/// Derives a stable cache directory for a model archive, keyed on the
/// archive's file name plus its size and modified time. This avoids needing
/// to hash the (potentially large) archive contents while still being safe
/// against a file name being reused for different content.
fn cache_dir_for(archive_path: &Path) -> Result<PathBuf> {
    let metadata = fs::metadata(archive_path)
        .with_context(|| format!("Failed to read metadata for {:?}", archive_path))?;
    let modified_secs = metadata
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let file_name = archive_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("Model archive path has no file name")?;

    let key = format!("{}-{}-{}", file_name, metadata.len(), modified_secs);

    Ok(std::env::temp_dir().join("optima-model-cache").join(key))
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
