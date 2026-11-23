use super::DiffSource;
use std::fs;
use std::path::Path;
use std::process::Command;

pub(crate) fn sanitize_sha(raw: Option<&str>) -> Option<String> {
    raw.map(|s| s.trim().trim_matches('"').trim())
        .filter(|s| !s.is_empty() && !s.chars().all(|c| c == '0'))
        .map(ToString::to_string)
}

pub(crate) fn git_stdout(repo_dir: &Path, args: &[&str]) -> Option<String> {
    let mut full_args = Vec::with_capacity(2 + args.len());
    full_args.push("-c");
    full_args.push("core.quotepath=false");
    full_args.extend_from_slice(args);
    Command::new("git")
        .current_dir(repo_dir)
        .args(&full_args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
}

pub(crate) fn commit_oid(repo_dir: &Path, rev: &str) -> Option<String> {
    let spec = format!("{rev}^{{commit}}");
    git_stdout(repo_dir, &["rev-parse", "--verify", "--quiet", &spec])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub(crate) fn resolve_merge_base(repo_dir: &Path, target_ref: &str) -> Option<String> {
    for branch in ["origin/main", "main"] {
        if commit_oid(repo_dir, branch).is_some()
            && let Some(out) = git_stdout(repo_dir, &["merge-base", target_ref, branch])
        {
            let trimmed = out.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn inspect_head_commit(repo_dir: &Path) -> String {
    git_stdout(
        repo_dir,
        &[
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--name-only",
            "-r",
            "HEAD",
        ],
    )
    .unwrap_or_default()
}

pub(crate) fn inspect_staged_index(repo_dir: &Path) -> String {
    git_stdout(repo_dir, &["diff", "--cached", "--name-only"]).unwrap_or_default()
}

fn inspect_cached_from(repo_dir: &Path, base: &str) -> Option<String> {
    git_stdout(repo_dir, &["diff", "--cached", "--name-only", base])
}

pub(crate) fn inspect_commit_range(repo_dir: &Path, base: &str, head: &str) -> Option<String> {
    if commit_oid(repo_dir, base).is_none() || commit_oid(repo_dir, head).is_none() {
        eprintln!("Error: Invalid or unreachable commit range {base}...{head}; failing closed.");
        return None;
    }
    let Some(diffs) = git_stdout(
        repo_dir,
        &["diff", "--name-only", &format!("{base}...{head}")],
    ) else {
        eprintln!("Error: git diff failed for range {base}...{head}; failing closed.");
        return None;
    };
    Some(diffs)
}

fn select_without_explicit_base(
    repo_dir: &Path,
    head_sha: Option<&str>,
) -> Option<(String, DiffSource)> {
    if let Some(explicit_head) = head_sha
        && commit_oid(repo_dir, explicit_head).is_none()
    {
        eprintln!("Error: Invalid or unreachable HEAD_SHA {explicit_head}; failing closed.");
        return None;
    }

    let target_ref = head_sha.unwrap_or("HEAD");
    let head_oid = commit_oid(repo_dir, "HEAD");
    let target_oid = commit_oid(repo_dir, target_ref);

    if let Some(mb) = resolve_merge_base(repo_dir, target_ref) {
        if target_oid.as_deref() == head_oid.as_deref() {
            let cached = inspect_cached_from(repo_dir, &mb).unwrap_or_default();
            if target_oid.as_deref() == Some(mb.as_str()) && cached.trim().is_empty() {
                return Some((inspect_head_commit(repo_dir), DiffSource::Head));
            }
            return Some((cached, DiffSource::CachedBase(mb)));
        }
        let diffs = inspect_commit_range(repo_dir, &mb, target_ref)?;
        if diffs.trim().is_empty() {
            return Some((inspect_head_commit(repo_dir), DiffSource::Head));
        }
        return Some((
            diffs,
            DiffSource::Range {
                base: mb,
                head: target_ref.to_string(),
            },
        ));
    }

    let staged = inspect_staged_index(repo_dir);
    if !staged.trim().is_empty() {
        return Some((staged, DiffSource::Staged));
    }
    Some((inspect_head_commit(repo_dir), DiffSource::Head))
}

pub(crate) fn select_changed_files(
    repo_dir: &Path,
    base_sha: Option<&str>,
    head_sha: Option<&str>,
    force_staged: bool,
) -> Option<(String, DiffSource)> {
    if force_staged {
        return Some((inspect_staged_index(repo_dir), DiffSource::Staged));
    }

    if let Some(base) = base_sha {
        let head = head_sha.unwrap_or("HEAD");
        let diffs = inspect_commit_range(repo_dir, base, head)?;
        if diffs.trim().is_empty() {
            return Some((inspect_head_commit(repo_dir), DiffSource::Head));
        }
        return Some((
            diffs,
            DiffSource::Range {
                base: base.to_string(),
                head: head.to_string(),
            },
        ));
    }

    select_without_explicit_base(repo_dir, head_sha)
}

fn split_normalized_lines(raw: String) -> Vec<String> {
    raw.replace('\\', "/")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(ToString::to_string)
        .collect()
}

pub(crate) fn active_changed_files(repo_dir: &Path, source: &DiffSource) -> Vec<String> {
    let out = match source {
        DiffSource::Staged => git_stdout(
            repo_dir,
            &["diff", "--cached", "--diff-filter=ACMR", "--name-only"],
        ),
        DiffSource::CachedBase(base) => git_stdout(
            repo_dir,
            &[
                "diff",
                "--cached",
                "--diff-filter=ACMR",
                "--name-only",
                base,
            ],
        ),
        DiffSource::Range { base, head } => git_stdout(
            repo_dir,
            &[
                "diff",
                "--diff-filter=ACMR",
                "--name-only",
                &format!("{base}...{head}"),
            ],
        ),
        DiffSource::Head => git_stdout(
            repo_dir,
            &[
                "diff-tree",
                "--root",
                "--no-commit-id",
                "--diff-filter=ACMR",
                "--name-only",
                "-r",
                "HEAD",
            ],
        ),
    };
    split_normalized_lines(out.unwrap_or_default())
}

const ZERO_CTX_FLAGS: &[&str] = &["-U0", "--no-renames", "--src-prefix=a/", "--dst-prefix=b/"];

fn git_zero_ctx_diff(repo_dir: &Path, prefix: &[&str], suffix: &[&str]) -> String {
    let mut args = Vec::with_capacity(prefix.len() + ZERO_CTX_FLAGS.len() + suffix.len());
    args.extend_from_slice(prefix);
    args.extend_from_slice(ZERO_CTX_FLAGS);
    args.extend_from_slice(suffix);
    git_stdout(repo_dir, &args).unwrap_or_default()
}

fn head_zero_context_diff(repo_dir: &Path) -> String {
    git_zero_ctx_diff(repo_dir, &["show", "--root", "--format="], &["HEAD"])
}

pub(crate) fn unified_zero_context_diff(repo_dir: &Path, source: &DiffSource) -> String {
    match source {
        DiffSource::Staged => git_zero_ctx_diff(repo_dir, &["diff", "--cached"], &[]),
        DiffSource::CachedBase(base) => git_zero_ctx_diff(repo_dir, &["diff", "--cached"], &[base]),
        DiffSource::Range { base, head } => {
            let range_spec = format!("{base}...{head}");
            let range_diff = git_zero_ctx_diff(repo_dir, &["diff"], &[&range_spec]);
            if range_diff.trim().is_empty() {
                head_zero_context_diff(repo_dir)
            } else {
                range_diff
            }
        }
        DiffSource::Head => head_zero_context_diff(repo_dir),
    }
}

pub(crate) fn read_target_file(
    repo_dir: &Path,
    source: &DiffSource,
    rel_path: &str,
) -> Option<String> {
    match source {
        DiffSource::Staged | DiffSource::CachedBase(_) => {
            git_stdout(repo_dir, &["show", &format!(":{rel_path}")])
                .or_else(|| fs::read_to_string(repo_dir.join(rel_path)).ok())
        }
        DiffSource::Range { head, .. } => {
            git_stdout(repo_dir, &["show", &format!("{head}:{rel_path}")])
                .or_else(|| fs::read_to_string(repo_dir.join(rel_path)).ok())
        }
        DiffSource::Head => git_stdout(repo_dir, &["show", &format!("HEAD:{rel_path}")])
            .or_else(|| fs::read_to_string(repo_dir.join(rel_path)).ok()),
    }
}
