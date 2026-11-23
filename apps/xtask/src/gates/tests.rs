use super::*;
use std::fs;
use std::sync::atomic::AtomicUsize;

static GATE_REPO_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap();
    assert!(out.status.success(), "git {:?} failed", args);
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn init_gate_repo() -> PathBuf {
    let id = GATE_REPO_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("xtask_gates_test_{}_{}", std::process::id(), id));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    git(&dir, &["init"]);
    git(&dir, &["branch", "-M", "main"]);
    git(&dir, &["config", "user.email", "test@example.com"]);
    git(&dir, &["config", "user.name", "Test"]);
    fs::write(dir.join("README.md"), "init\n").unwrap();
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-m", "init"]);
    dir
}

#[test]
fn test_branch_records_base_sha_handles_staged_files_and_origin_main() {
    let dir = init_gate_repo();
    let base = git(&dir, &["rev-parse", "HEAD"]);
    assert_eq!(branch_records_base_sha_in_dir(&dir, None), None);

    // Set refs/remotes/origin/main to base, then commit directly on local main
    git(&dir, &["update-ref", "refs/remotes/origin/main", &base]);
    fs::write(dir.join("crates_touch.txt"), "branch edit\n").unwrap();
    git(&dir, &["add", "crates_touch.txt"]);
    git(
        &dir,
        &["commit", "-m", "commit on local main ahead of origin/main"],
    );

    // Prefer origin/main over local main when local main == HEAD
    assert_eq!(
        branch_records_base_sha_in_dir(&dir, None),
        Some(base.clone())
    );

    // Staged files must NOT clear branch_records_base_sha
    fs::write(dir.join("staged_note.txt"), "staged\n").unwrap();
    git(&dir, &["add", "staged_note.txt"]);
    assert_eq!(
        branch_records_base_sha_in_dir(&dir, None),
        Some(base.clone())
    );

    // Zero BASE_SHA still computes merge-base; explicit non-zero BASE_SHA defers to caller
    let zero40 = "0".repeat(40);
    assert_eq!(
        branch_records_base_sha_in_dir(&dir, Some(&zero40)),
        Some(base)
    );
    assert_eq!(branch_records_base_sha_in_dir(&dir, Some("deadbeef")), None);

    let _ = fs::remove_dir_all(&dir);
}
