use std::env;
use std::path::Path;
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DiffSource<'a> {
    Staged,
    Range(&'a str, &'a str),
    Head,
}

fn sanitize_sha(raw: Option<&str>) -> Option<String> {
    raw.map(|s| s.trim().trim_matches('"').trim())
        .filter(|s| !s.is_empty() && !s.chars().all(|c| c == '0'))
        .map(ToString::to_string)
}

fn git_stdout(repo_dir: &Path, args: &[&str]) -> Option<String> {
    Command::new("git")
        .current_dir(repo_dir)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
}

fn inspect_head_commit(repo_dir: &Path) -> String {
    git_stdout(
        repo_dir,
        &["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
    )
    .unwrap_or_default()
}

fn inspect_staged_index(repo_dir: &Path) -> String {
    git_stdout(repo_dir, &["diff", "--cached", "--name-only"]).unwrap_or_default()
}

fn inspect_commit_range(repo_dir: &Path, base: &str, head: &str) -> Option<String> {
    let Some(diffs) = git_stdout(
        repo_dir,
        &["diff", "--name-only", &format!("{base}...{head}")],
    ) else {
        eprintln!(
            "Warning: git diff failed for range {base}...{head}; falling back to HEAD inspection."
        );
        return None;
    };
    Some(diffs)
}

fn select_changed_files<'a>(
    repo_dir: &Path,
    base_sha: Option<&'a str>,
    head_sha: Option<&'a str>,
    force_staged: bool,
) -> (String, DiffSource<'a>) {
    if force_staged {
        return (inspect_staged_index(repo_dir), DiffSource::Staged);
    }

    if let (Some(base), Some(head)) = (base_sha, head_sha)
        && let Some(diffs) = inspect_commit_range(repo_dir, base, head)
    {
        if diffs.trim().is_empty() {
            return (inspect_head_commit(repo_dir), DiffSource::Head);
        }
        return (diffs, DiffSource::Range(base, head));
    }

    let staged = inspect_staged_index(repo_dir);
    if !staged.trim().is_empty() {
        return (staged, DiffSource::Staged);
    }

    (inspect_head_commit(repo_dir), DiffSource::Head)
}

pub fn verify_records(args: &[String]) -> bool {
    let force_staged = args.iter().any(|a| a == "--staged");
    let raw_base = env::var("BASE_SHA").ok();
    let raw_head = env::var("HEAD_SHA").ok();
    verify_records_in_dir_opts(
        Path::new("."),
        raw_base.as_deref(),
        raw_head.as_deref(),
        force_staged,
    )
}

#[cfg(test)]
pub fn verify_records_in_dir(
    repo_dir: &Path,
    raw_base: Option<&str>,
    raw_head: Option<&str>,
) -> bool {
    verify_records_in_dir_opts(repo_dir, raw_base, raw_head, false)
}

pub fn verify_records_in_dir_opts(
    repo_dir: &Path,
    raw_base: Option<&str>,
    raw_head: Option<&str>,
    force_staged: bool,
) -> bool {
    let base_sha = sanitize_sha(raw_base);
    let head_sha = sanitize_sha(raw_head);

    let (changed_files, source) = select_changed_files(
        repo_dir,
        base_sha.as_deref(),
        head_sha.as_deref(),
        force_staged,
    );

    let changed_files = changed_files.replace('\\', "/");
    println!("Changed files:\n{}", changed_files);

    let changed_lines: Vec<&str> = changed_files.lines().filter(|l| !l.is_empty()).collect();
    let core_modified = changed_lines.iter().any(|l| {
        l.starts_with("math_explorer/")
            || (l.starts_with("crates/") && !l.starts_with("crates/unified_verification/"))
    });

    if !core_modified {
        println!("No core logic areas modified. Skipping ADR verification.");
        return true;
    }

    if is_pure_deletion_for_source(repo_dir, source) {
        println!("Pure deletion detected in core logic. Skipping ADR verification.");
        return true;
    }

    println!("Core logic areas (math_explorer/ or crates/) were modified.");
    let adr_modified = changed_lines
        .iter()
        .any(|l| l.starts_with("docs/adr/") && l.ends_with(".md"));
    if !adr_modified {
        println!(
            "Verification failed! A Markdown ADR in docs/adr/ must be created or updated when modifying core logic."
        );
        return false;
    }
    println!("ADR verification passed.");
    true
}

fn is_pure_deletion_for_source(repo_dir: &Path, source: DiffSource<'_>) -> bool {
    let diff = match source {
        DiffSource::Staged => git_stdout(
            repo_dir,
            &["diff", "--cached", "--src-prefix=a/", "--dst-prefix=b/"],
        )
        .unwrap_or_default(),
        DiffSource::Range(base, head) => {
            let range_diff =
                git_stdout(repo_dir, &["diff", &format!("{base}...{head}")]).unwrap_or_default();
            if range_diff.trim().is_empty() {
                git_stdout(repo_dir, &["show", "--format=", "HEAD"]).unwrap_or_default()
            } else {
                range_diff
            }
        }
        DiffSource::Head => {
            git_stdout(repo_dir, &["show", "--format=", "HEAD"]).unwrap_or_default()
        }
    };

    let mut in_core_file = false;
    let mut added_code_lines = 0;

    for line in diff.lines() {
        if line.starts_with("diff --git") {
            in_core_file = line.contains(" b/math_explorer/")
                || line.contains(" a/math_explorer/")
                || ((line.contains(" b/crates/") || line.contains(" a/crates/"))
                    && !line.contains(" b/crates/unified_verification/")
                    && !line.contains(" a/crates/unified_verification/"));
        } else if in_core_file && line.starts_with('+') && !line.starts_with("+++") {
            let content = line[1..].trim();
            if !content.is_empty()
                && !content.starts_with("//")
                && !content.starts_with("/*")
                && !content.starts_with('*')
            {
                added_code_lines += 1;
            }
        }
    }
    added_code_lines == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static REPO_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .current_dir(dir)
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {:?} failed", args);
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn init_temp_repo() -> PathBuf {
        let id = REPO_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            env::temp_dir().join(format!("oxidize_verify_test_{}_{}", std::process::id(), id));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        git(&dir, &["init"]);
        git(&dir, &["config", "user.email", "test@example.com"]);
        git(&dir, &["config", "user.name", "Test"]);
        fs::write(dir.join("README.md"), "init\n").unwrap();
        git(&dir, &["add", "."]);
        git(&dir, &["commit", "-m", "initial commit"]);
        dir
    }

    fn stage_file(dir: &Path, rel_path: &str, content: &str) {
        let full = dir.join(rel_path);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(full, content).unwrap();
        git(dir, &["add", rel_path]);
    }

    fn commit_file(dir: &Path, rel_path: &str, content: &str, msg: &str) -> String {
        stage_file(dir, rel_path, content);
        git(dir, &["commit", "-m", msg]);
        git(dir, &["rev-parse", "HEAD"])
    }

    #[test]
    fn test_sanitize_sha() {
        assert_eq!(sanitize_sha(None), None);
        assert_eq!(sanitize_sha(Some("")), None);
        assert_eq!(sanitize_sha(Some("   ")), None);
        assert_eq!(sanitize_sha(Some("\"\"")), None);
        let zero40 = "0".repeat(40);
        let zero64 = "0".repeat(64);
        assert_eq!(sanitize_sha(Some(&zero40)), None);
        assert_eq!(sanitize_sha(Some(&zero64)), None);
        assert_eq!(sanitize_sha(Some(&format!("\"{}\"", zero40))), None);
        assert_eq!(
            sanitize_sha(Some("  \"2fc42a4b\"  ")),
            Some("2fc42a4b".to_string())
        );
    }

    #[test]
    fn test_multi_commit_range_and_fallbacks() {
        let dir = init_temp_repo();
        let base = git(&dir, &["rev-parse", "HEAD"]);

        commit_file(&dir, "crates/pure_math/src/lib.rs", "pub fn x() {}\n", "c1");
        let head = commit_file(&dir, "README.md", "updated\n", "c2");

        assert!(verify_records_in_dir(&dir, None, None));
        assert!(!verify_records_in_dir(&dir, Some(&base), Some(&head)));

        let zero_sha = "0".repeat(40);
        assert!(verify_records_in_dir(&dir, Some(&zero_sha), Some(&head)));

        let missing_sha = "deadbeef".repeat(5);
        assert!(verify_records_in_dir(&dir, Some(&missing_sha), Some(&head)));

        let head_with_adr = commit_file(&dir, "docs/adr/0001-test.md", "# ADR\n", "c3");
        assert!(verify_records_in_dir(
            &dir,
            Some(&base),
            Some(&head_with_adr)
        ));

        let head_bad = commit_file(&dir, "math_explorer/src/lib.rs", "pub fn y() {}\n", "c4");
        assert!(!verify_records_in_dir(
            &dir,
            Some(&zero_sha),
            Some(&head_bad)
        ));
        assert!(!verify_records_in_dir(
            &dir,
            Some(&missing_sha),
            Some(&head_bad)
        ));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_staged_index_validation() {
        let dir = init_temp_repo();

        // Empty staged index passes in --staged mode
        assert!(verify_records_in_dir_opts(&dir, None, None, true));

        // Staged core edit without ADR fails both --staged and default fallback
        stage_file(&dir, "crates/pure_math/src/lib.rs", "pub fn z() {}\n");
        assert!(!verify_records_in_dir_opts(&dir, None, None, true));
        assert!(!verify_records_in_dir(&dir, None, None));

        // Staging an ADR alongside the core edit passes
        stage_file(&dir, "docs/adr/0002-staged.md", "# Staged ADR\n");
        assert!(verify_records_in_dir_opts(&dir, None, None, true));
        assert!(verify_records_in_dir(&dir, None, None));

        let _ = fs::remove_dir_all(&dir);
    }
}
