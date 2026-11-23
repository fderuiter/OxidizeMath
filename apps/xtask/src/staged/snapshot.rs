use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::time::UNIX_EPOCH;

const BATCH_CHUNK_SIZE: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GitSnapshotEnv {
    pub git_dir: PathBuf,
    pub git_index_file: Option<PathBuf>,
}

impl GitSnapshotEnv {
    pub(crate) fn resolve(repo_root: &Path) -> Result<Self, String> {
        let out = Command::new("git")
            .current_dir(repo_root)
            .args(["--no-optional-locks", "rev-parse", "--git-dir"])
            .output()
            .map_err(|e| format!("Failed to invoke git rev-parse --git-dir: {e}"))?;
        if !out.status.success() {
            return Err("git rev-parse --git-dir exited with failure".to_string());
        }
        let raw_dir = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
        let git_dir = if raw_dir.is_absolute() {
            raw_dir
        } else {
            repo_root.join(raw_dir)
        };
        let git_index_file = std::env::var_os("GIT_INDEX_FILE").map(|raw| {
            let p = PathBuf::from(raw);
            if p.is_absolute() {
                p
            } else {
                repo_root.join(p)
            }
        });
        Ok(Self {
            git_dir,
            git_index_file,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct IndexEntry {
    mode: String,
    oid: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ManifestRecord {
    mode: String,
    oid: String,
    len: u64,
    mtime_secs: u64,
    mtime_nanos: u32,
}

fn validate_rel_path(path: &str) -> Result<(), String> {
    if path.is_empty() {
        return Err("Empty path in git index".to_string());
    }
    for comp in Path::new(path).components() {
        if !matches!(comp, Component::Normal(_)) {
            return Err(format!("Unsafe relative path in git index: {path}"));
        }
    }
    Ok(())
}

fn read_index_entries(repo_root: &Path) -> Result<BTreeMap<String, IndexEntry>, String> {
    let out = Command::new("git")
        .current_dir(repo_root)
        .args(["--no-optional-locks", "ls-files", "--stage", "-z"])
        .output()
        .map_err(|e| format!("Failed to read git index via ls-files: {e}"))?;
    if !out.status.success() {
        return Err("git ls-files --stage failed".to_string());
    }

    let mut entries = BTreeMap::new();
    for raw_record in out.stdout.split(|&b| b == 0) {
        if raw_record.is_empty() {
            continue;
        }
        let record = std::str::from_utf8(raw_record)
            .map_err(|e| format!("Non-UTF-8 path in git index: {e}"))?;
        let Some((meta_part, raw_path)) = record.split_once('\t') else {
            continue;
        };
        let mut fields = meta_part.split_whitespace();
        let (Some(mode), Some(oid), Some(stage)) = (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        if stage != "0" {
            return Err(format!(
                "Unmerged git index conflict detected at '{raw_path}' (stage {stage})"
            ));
        }
        let norm_path = raw_path.replace('\\', "/");
        validate_rel_path(&norm_path)?;
        if mode == "160000" {
            continue;
        }
        entries.insert(
            norm_path,
            IndexEntry {
                mode: mode.to_string(),
                oid: oid.to_string(),
            },
        );
    }
    Ok(entries)
}

fn load_manifest(manifest_path: &Path) -> BTreeMap<String, ManifestRecord> {
    let Ok(content) = fs::read_to_string(manifest_path) else {
        return BTreeMap::new();
    };
    let mut records = BTreeMap::new();
    for line in content.lines() {
        let mut parts = line.splitn(6, '\t');
        let (Some(mode), Some(oid), Some(len_s), Some(sec_s), Some(nano_s), Some(path)) = (
            parts.next(),
            parts.next(),
            parts.next(),
            parts.next(),
            parts.next(),
            parts.next(),
        ) else {
            continue;
        };
        if let (Ok(len), Ok(mtime_secs), Ok(mtime_nanos)) =
            (len_s.parse(), sec_s.parse(), nano_s.parse())
        {
            records.insert(
                path.to_string(),
                ManifestRecord {
                    mode: mode.to_string(),
                    oid: oid.to_string(),
                    len,
                    mtime_secs,
                    mtime_nanos,
                },
            );
        }
    }
    records
}

fn save_manifest(
    manifest_path: &Path,
    records: &BTreeMap<String, ManifestRecord>,
) -> Result<(), String> {
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create {}: {e}", parent.display()))?;
    }
    let mut buf = String::new();
    for (path, rec) in records {
        buf.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            rec.mode, rec.oid, rec.len, rec.mtime_secs, rec.mtime_nanos, path
        ));
    }
    fs::write(manifest_path, buf)
        .map_err(|e| format!("Failed to write {}: {e}", manifest_path.display()))
}

fn metadata_mtime(meta: &fs::Metadata) -> (u64, u32) {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| (d.as_secs(), d.subsec_nanos()))
        .unwrap_or((0, 0))
}

fn mode_matches_metadata(mode: &str, meta: &fs::Metadata) -> bool {
    if mode == "120000" {
        return meta.file_type().is_symlink();
    }
    if !meta.file_type().is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let is_exec = (meta.permissions().mode() & 0o111) != 0;
        let want_exec = mode == "100755";
        is_exec == want_exec
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn is_entry_fresh(
    snapshot_dir: &Path,
    path: &str,
    entry: &IndexEntry,
    prev: &BTreeMap<String, ManifestRecord>,
) -> Option<ManifestRecord> {
    let rec = prev.get(path)?;
    if rec.mode != entry.mode || rec.oid != entry.oid {
        return None;
    }
    let meta = fs::symlink_metadata(snapshot_dir.join(path)).ok()?;
    if !mode_matches_metadata(&entry.mode, &meta) {
        return None;
    }
    if entry.mode != "120000" {
        let (secs, nanos) = metadata_mtime(&meta);
        if meta.len() != rec.len || secs != rec.mtime_secs || nanos != rec.mtime_nanos {
            return None;
        }
    }
    Some(rec.clone())
}

fn collect_expected_dirs(index_entries: &BTreeMap<String, IndexEntry>) -> HashSet<PathBuf> {
    let mut dirs = HashSet::new();
    for path in index_entries.keys() {
        let mut cur = Path::new(path).parent();
        while let Some(parent) = cur {
            if parent.as_os_str().is_empty() {
                break;
            }
            if !dirs.insert(parent.to_path_buf()) {
                break;
            }
            cur = parent.parent();
        }
    }
    dirs
}

fn prune_snapshot_dir(
    snapshot_dir: &Path,
    index_entries: &BTreeMap<String, IndexEntry>,
) -> Result<(), String> {
    if !snapshot_dir.exists() {
        return Ok(());
    }
    let expected_dirs = collect_expected_dirs(index_entries);
    let mut stack = vec![snapshot_dir.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|e| format!("Failed reading snapshot dir {}: {e}", dir.display()))?;
        for item in entries {
            let item = item.map_err(|e| format!("Snapshot dir entry error: {e}"))?;
            let full_path = item.path();
            let file_type = item
                .file_type()
                .map_err(|e| format!("Failed reading file type {}: {e}", full_path.display()))?;
            let rel_path = full_path
                .strip_prefix(snapshot_dir)
                .map_err(|e| format!("Snapshot prefix error: {e}"))?;
            let rel_key = rel_path.to_string_lossy().replace('\\', "/");

            if file_type.is_dir() {
                if expected_dirs.contains(rel_path) {
                    stack.push(full_path);
                } else {
                    fs::remove_dir_all(&full_path)
                        .map_err(|e| format!("Failed removing {}: {e}", full_path.display()))?;
                }
            } else if !index_entries.contains_key(&rel_key) {
                fs::remove_file(&full_path)
                    .map_err(|e| format!("Failed removing {}: {e}", full_path.display()))?;
            }
        }
    }
    Ok(())
}

fn prepare_parent_dirs(snapshot_dir: &Path, rel_path: &str) -> Result<PathBuf, String> {
    let dest = snapshot_dir.join(rel_path);
    if let Ok(meta) = fs::symlink_metadata(&dest)
        && meta.file_type().is_dir()
    {
        fs::remove_dir_all(&dest)
            .map_err(|e| format!("Failed removing conflicting dir {}: {e}", dest.display()))?;
    }
    let mut cur = PathBuf::from(snapshot_dir);
    if let Some(parent_rel) = Path::new(rel_path).parent() {
        for comp in parent_rel.components() {
            cur.push(comp);
            if let Ok(meta) = fs::symlink_metadata(&cur)
                && !meta.file_type().is_dir()
            {
                fs::remove_file(&cur).map_err(|e| {
                    format!("Failed removing conflicting file {}: {e}", cur.display())
                })?;
            }
        }
        fs::create_dir_all(&cur)
            .map_err(|e| format!("Failed creating snapshot parent {}: {e}", cur.display()))?;
    }
    Ok(dest)
}

fn write_blob_to_snapshot(dest: &Path, mode: &str, bytes: &[u8]) -> Result<(), String> {
    if mode == "120000" {
        if fs::symlink_metadata(dest).is_ok() {
            let _ = fs::remove_file(dest);
        }
        #[cfg(unix)]
        {
            let target = std::str::from_utf8(bytes)
                .map_err(|e| format!("Invalid UTF-8 symlink target at {}: {e}", dest.display()))?;
            return std::os::unix::fs::symlink(target, dest)
                .map_err(|e| format!("Failed creating symlink {}: {e}", dest.display()));
        }
    }
    fs::write(dest, bytes).map_err(|e| format!("Failed writing {}: {e}", dest.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perm_mode = if mode == "100755" { 0o755 } else { 0o644 };
        fs::set_permissions(dest, fs::Permissions::from_mode(perm_mode))
            .map_err(|e| format!("Failed setting permissions on {}: {e}", dest.display()))?;
    }
    Ok(())
}

fn process_cat_file_chunk(
    snapshot_dir: &Path,
    chunk: &[(&String, &IndexEntry)],
    stdin: &mut ChildStdin,
    reader: &mut BufReader<ChildStdout>,
) -> Result<(), String> {
    for (_, entry) in chunk {
        writeln!(stdin, "{}", entry.oid)
            .map_err(|e| format!("Failed writing to git cat-file stdin: {e}"))?;
    }
    stdin
        .flush()
        .map_err(|e| format!("Failed flushing git cat-file stdin: {e}"))?;

    let mut header = String::new();
    for (path, entry) in chunk {
        header.clear();
        reader
            .read_line(&mut header)
            .map_err(|e| format!("Failed reading git cat-file header for {path}: {e}"))?;
        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() < 3 || parts[1] != "blob" {
            return Err(format!(
                "Unexpected git cat-file header for {path}: {}",
                header.trim()
            ));
        }
        let size: usize = parts[2]
            .parse()
            .map_err(|e| format!("Invalid blob size in git cat-file for {path}: {e}"))?;
        let mut buf = vec![0u8; size];
        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("Failed reading blob content for {path}: {e}"))?;
        let mut nl = [0u8; 1];
        reader
            .read_exact(&mut nl)
            .map_err(|e| format!("Failed reading trailing newline for {path}: {e}"))?;

        let dest = prepare_parent_dirs(snapshot_dir, path)?;
        write_blob_to_snapshot(&dest, &entry.mode, &buf)?;
    }
    Ok(())
}

fn materialize_dirty_entries(
    repo_root: &Path,
    snapshot_dir: &Path,
    dirty: &[(&String, &IndexEntry)],
) -> Result<(), String> {
    if dirty.is_empty() {
        return Ok(());
    }
    let mut child = Command::new("git")
        .current_dir(repo_root)
        .args(["--no-optional-locks", "cat-file", "--batch"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed spawning git cat-file --batch: {e}"))?;

    let result = {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| "Missing git cat-file stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Missing git cat-file stdout".to_string())?;
        let mut reader = BufReader::new(stdout);
        let mut step_res = Ok(());
        for chunk in dirty.chunks(BATCH_CHUNK_SIZE) {
            if let Err(err) = process_cat_file_chunk(snapshot_dir, chunk, stdin, &mut reader) {
                step_res = Err(err);
                break;
            }
        }
        step_res
    };
    let _ = child.stdin.take();
    let status = child
        .wait()
        .map_err(|e| format!("Failed waiting on git cat-file --batch: {e}"))?;
    result?;
    if !status.success() {
        return Err("git cat-file --batch exited with failure".to_string());
    }
    Ok(())
}

pub(crate) fn sync_index_snapshot(
    repo_root: &Path,
    snapshot_dir: &Path,
    manifest_path: &Path,
) -> Result<(), String> {
    fs::create_dir_all(snapshot_dir).map_err(|e| {
        format!(
            "Failed creating snapshot directory {}: {e}",
            snapshot_dir.display()
        )
    })?;
    let index_entries = read_index_entries(repo_root)?;
    let prev_manifest = load_manifest(manifest_path);

    prune_snapshot_dir(snapshot_dir, &index_entries)?;

    let mut next_manifest = BTreeMap::new();
    let mut dirty = Vec::new();
    for (path, entry) in &index_entries {
        if let Some(fresh_rec) = is_entry_fresh(snapshot_dir, path, entry, &prev_manifest) {
            next_manifest.insert(path.clone(), fresh_rec);
        } else {
            dirty.push((path, entry));
        }
    }

    materialize_dirty_entries(repo_root, snapshot_dir, &dirty)?;

    for (path, entry) in dirty {
        let meta = fs::symlink_metadata(snapshot_dir.join(path))
            .map_err(|e| format!("Missing materialized snapshot file {path}: {e}"))?;
        let (mtime_secs, mtime_nanos) = metadata_mtime(&meta);
        next_manifest.insert(
            path.clone(),
            ManifestRecord {
                mode: entry.mode.clone(),
                oid: entry.oid.clone(),
                len: meta.len(),
                mtime_secs,
                mtime_nanos,
            },
        );
    }

    save_manifest(manifest_path, &next_manifest)
}
