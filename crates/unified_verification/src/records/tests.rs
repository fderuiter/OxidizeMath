use super::adr_validation::{check_markdown_headings, is_valid_yyyymmdd};
use super::comment_scanner::{RustScanState, count_added_core_code_lines, scan_rust_line};
use super::git_range::sanitize_sha;
use super::{verify_records_in_dir, verify_records_in_dir_opts};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static REPO_COUNTER: AtomicUsize = AtomicUsize::new(0);

const VALID_ADR_1: &str = "# ADR 20261123: Initial Decision\n\n## Status\n\nAccepted\n";
const VALID_ADR_2: &str = "# ADR 20261123: Staged Decision\n\n## Status\n\nAccepted\n";

pub(crate) fn git_in_test(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

pub(crate) fn init_temp_repo() -> PathBuf {
    let id = REPO_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = env::temp_dir().join(format!("oxidize_verify_test_{}_{}", std::process::id(), id));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    git_in_test(&dir, &["init"]);
    git_in_test(&dir, &["branch", "-M", "main"]);
    git_in_test(&dir, &["config", "user.email", "test@example.com"]);
    git_in_test(&dir, &["config", "user.name", "Test"]);
    fs::write(dir.join("README.md"), "init\n").unwrap();
    git_in_test(&dir, &["add", "."]);
    git_in_test(&dir, &["commit", "-m", "initial commit"]);
    dir
}

pub(crate) fn stage_file(dir: &Path, rel_path: &str, content: &str) {
    let full = dir.join(rel_path);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(full, content).unwrap();
    git_in_test(dir, &["add", rel_path]);
}

pub(crate) fn commit_file(dir: &Path, rel_path: &str, content: &str, msg: &str) -> String {
    stage_file(dir, rel_path, content);
    git_in_test(dir, &["commit", "-m", msg]);
    git_in_test(dir, &["rev-parse", "HEAD"])
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
    let base = git_in_test(&dir, &["rev-parse", "HEAD"]);
    git_in_test(&dir, &["checkout", "-b", "feature/test-branch"]);

    commit_file(&dir, "crates/pure_math/src/lib.rs", "pub fn x() {}\n", "c1");
    let head = commit_file(&dir, "README.md", "updated\n", "c2");

    // Branch merge-base resolution and bad BASE_SHA fail closed instead of inspecting HEAD (c2)
    assert!(!verify_records_in_dir(&dir, None, None));
    assert!(!verify_records_in_dir(&dir, Some(&base), Some(&head)));
    let zero_sha = "0".repeat(40);
    assert!(!verify_records_in_dir(&dir, Some(&zero_sha), Some(&head)));
    let missing_sha = "deadbeef".repeat(5);
    assert!(!verify_records_in_dir(
        &dir,
        Some(&missing_sha),
        Some(&head)
    ));

    // Staging an unrelated file must not hide unrecorded branch commits
    stage_file(&dir, "NOTES.md", "unrelated staged note\n");
    assert!(!verify_records_in_dir(&dir, None, None));
    git_in_test(&dir, &["reset", "--hard", "HEAD"]);

    // Missing ## Status fails, then valid ADR with ## Status passes
    let head_no_status = commit_file(&dir, "docs/adr/20261123-test.md", "# ADR 20261123\n", "c3a");
    assert!(!verify_records_in_dir(
        &dir,
        Some(&base),
        Some(&head_no_status)
    ));
    let head_with_adr = commit_file(&dir, "docs/adr/20261123-test.md", VALID_ADR_1, "c3b");
    assert!(verify_records_in_dir(
        &dir,
        Some(&base),
        Some(&head_with_adr)
    ));
    assert!(verify_records_in_dir(
        &dir,
        Some(&zero_sha),
        Some(&head_with_adr)
    ));
    assert!(!verify_records_in_dir(
        &dir,
        Some(&missing_sha),
        Some(&head_with_adr)
    ));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_git_rm_adr_and_duplicate_id_enforcement() {
    let dir = init_temp_repo();
    commit_file(&dir, "docs/adr/0001-existing.md", VALID_ADR_1, "seed adr");
    let base = git_in_test(&dir, &["rev-parse", "HEAD"]);
    git_in_test(&dir, &["checkout", "-b", "feature/rm-bypass"]);

    // Deleting an existing ADR via git rm must never satisfy core crate modifications
    stage_file(&dir, "crates/pure_math/src/lib.rs", "pub fn added() {}\n");
    git_in_test(&dir, &["rm", "docs/adr/0001-existing.md"]);
    assert!(!verify_records_in_dir_opts(&dir, None, None, true));
    let head_rm = git_in_test(&dir, &["commit", "-m", "core edit + rm adr"]);
    let _ = head_rm;
    let head_sha = git_in_test(&dir, &["rev-parse", "HEAD"]);
    assert!(!verify_records_in_dir(&dir, Some(&base), Some(&head_sha)));
    assert!(!verify_records_in_dir(&dir, None, None));

    // Pre-existing legacy colliding 0001 files fail without allowlist, pass when allowlisted, and fail with a third unallowlisted file
    git_in_test(&dir, &["checkout", "main"]);
    commit_file(
        &dir,
        "docs/adr/0001-collision.md",
        VALID_ADR_2,
        "seed collision",
    );
    git_in_test(&dir, &["checkout", "-b", "feature/legacy-allowlist"]);
    stage_file(&dir, "NOTES.md", "doc note\n");
    assert!(!verify_records_in_dir_opts(&dir, None, None, true));

    stage_file(
        &dir,
        "verification_whitelist.toml",
        "[adr_verification]\nallowlist = [\"docs/adr/0001-existing.md\", \"docs/adr/0001-collision.md\"]\n",
    );
    assert!(verify_records_in_dir_opts(&dir, None, None, true));

    git_in_test(&dir, &["reset", "--hard", "main"]);
    commit_file(&dir, "docs/adr/0001-third.md", VALID_ADR_2, "seed third");
    git_in_test(&dir, &["checkout", "-b", "feature/legacy-third"]);
    stage_file(
        &dir,
        "verification_whitelist.toml",
        "[adr_verification]\nallowlist = [\"docs/adr/0001-existing.md\", \"docs/adr/0001-collision.md\"]\n",
    );
    assert!(!verify_records_in_dir_opts(&dir, None, None, true));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_staged_index_reads_staged_blobs_and_fences() {
    let dir = init_temp_repo();
    assert!(verify_records_in_dir_opts(&dir, None, None, true));

    // Staged core edit without ADR fails
    stage_file(&dir, "crates/pure_math/src/lib.rs", "pub fn z() {}\n");
    assert!(!verify_records_in_dir_opts(&dir, None, None, true));

    // Stage ADR missing ## Status, then add ## Status ONLY in working tree (unstaged)
    stage_file(
        &dir,
        "docs/adr/20261123-staged.md",
        "# Staged ADR\n\n```markdown\n## Status\n```\n",
    );
    fs::write(dir.join("docs/adr/20261123-staged.md"), VALID_ADR_2).unwrap();
    assert!(
        !verify_records_in_dir_opts(&dir, None, None, true),
        "Unstaged working-tree edits must not mask missing ## Status in staged blob"
    );

    // Staging the fixed ADR passes
    git_in_test(&dir, &["add", "docs/adr/20261123-staged.md"]);
    assert!(verify_records_in_dir_opts(&dir, None, None, true));

    let (title, status) =
        check_markdown_headings("```text\n# Fake\n## Status\n```\n# Real Title\n");
    assert!(title);
    assert!(!status);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_deref_and_stateful_comment_scanner() {
    let dir = init_temp_repo();
    commit_file(
        &dir,
        "crates/pure_math/src/lib.rs",
        "/*\n * Header line 1\n */\npub fn write_out(out: &mut i32) {\n    let _ = out;\n}\n",
        "seed core",
    );
    let base = git_in_test(&dir, &["rev-parse", "HEAD"]);
    git_in_test(&dir, &["checkout", "-b", "feature/deref"]);

    // Adding a comment line inside an existing block comment across a -U0 hunk is exempt
    let head_comment = commit_file(
        &dir,
        "crates/pure_math/src/lib.rs",
        "/*\n * Header line 1\n * Added comment inside block\n */\npub fn write_out(out: &mut i32) {\n    let _ = out;\n}\n",
        "comment only",
    );
    assert!(verify_records_in_dir(
        &dir,
        Some(&base),
        Some(&head_comment)
    ));

    // Adding pointer dereference `*out = 42;` or `/* comment */ *out = 42;` requires an ADR
    let head_deref = commit_file(
        &dir,
        "crates/pure_math/src/lib.rs",
        "/*\n * Header line 1\n */\npub fn write_out(out: &mut i32) {\n    *out = 42;\n}\n",
        "deref edit",
    );
    assert!(!verify_records_in_dir(&dir, Some(&base), Some(&head_deref)));

    let mut st = RustScanState::default();
    assert!(scan_rust_line(&mut st, "/* comment */ *out = 42;"));
    assert!(!scan_rust_line(&mut st, "// line comment"));

    let sample_diff = concat!(
        "diff --git a/crates/pure_math/src/lib.rs b/crates/pure_math/src/lib.rs\n",
        "--- a/crates/pure_math/src/lib.rs\n",
        "+++ b/crates/pure_math/src/lib.rs\n",
        "@@ -1,0 +2,1 @@\n",
        "+/* comment */ *out = 42;\n",
    );
    assert_eq!(count_added_core_code_lines(sample_diff, |_| None), 1);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_tag_or_main_head_inspection_when_merge_base_matches_head() {
    let dir = init_temp_repo();
    // Commit directly on main so merge-base(HEAD, main) == HEAD (simulating tag push / workflow_dispatch)
    let clean_head = commit_file(&dir, "README.md", "release notes\n", "docs commit on main");
    let zero_sha = "0".repeat(40);
    assert!(verify_records_in_dir(
        &dir,
        Some(&zero_sha),
        Some(&clean_head)
    ));

    let unrecorded_main = commit_file(
        &dir,
        "math_explorer/src/lib.rs",
        "pub fn unrecorded() {}\n",
        "unrecorded core commit on main",
    );
    assert!(!verify_records_in_dir(
        &dir,
        Some(&zero_sha),
        Some(&unrecorded_main)
    ));
    assert!(!verify_records_in_dir(&dir, None, None));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_origin_main_preferred_and_workspace_adrs_valid() {
    let dir = init_temp_repo();
    let base = git_in_test(&dir, &["rev-parse", "HEAD"]);
    git_in_test(&dir, &["update-ref", "refs/remotes/origin/main", &base]);

    // Commit directly on local main ahead of origin/main without ADR: must fail
    commit_file(
        &dir,
        "crates/pure_math/src/lib.rs",
        "pub fn direct_on_main() {}\n",
        "c1 on local main",
    );
    commit_file(&dir, "README.md", "touch readme\n", "c2 on local main");
    assert!(!verify_records_in_dir(&dir, None, None));

    commit_file(&dir, "docs/adr/20261123-fix.md", VALID_ADR_1, "c3 add adr");
    assert!(verify_records_in_dir(&dir, None, None));
    let _ = fs::remove_dir_all(&dir);

    let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert!(
        super::adr_validation::validate_repository_adrs(&ws_root, &super::DiffSource::Head),
        "All workspace docs/adr/*.md files must pass structural and prefix checks"
    );
}

#[test]
fn test_dated_adr_calendar_bounds_and_headings() {
    for valid in ["19700101", "20000229", "20240229", "20261123", "99991231"] {
        assert!(is_valid_yyyymmdd(valid), "expected {valid} to be valid");
    }
    for invalid in [
        "19691231",
        "00420101",
        "20260010",
        "20261301",
        "20261100",
        "20260431",
        "20260229",
        "21000229",
        "2026112",
        "202611230",
        "202611ab",
    ] {
        assert!(
            !is_valid_yyyymmdd(invalid),
            "expected {invalid} to be invalid"
        );
    }

    let dir = init_temp_repo();
    stage_file(&dir, "crates/pure_math/src/lib.rs", "pub fn f() {}\n");
    stage_file(&dir, "docs/adr/20260229-bad-leap.md", VALID_ADR_1);
    assert!(!verify_records_in_dir_opts(&dir, None, None, true));
    git_in_test(&dir, &["rm", "-f", "docs/adr/20260229-bad-leap.md"]);

    stage_file(
        &dir,
        "docs/adr/20240229-no-title.md",
        "## Status\n\nAccepted\n",
    );
    assert!(!verify_records_in_dir_opts(&dir, None, None, true));
    git_in_test(&dir, &["rm", "-f", "docs/adr/20240229-no-title.md"]);

    stage_file(&dir, "docs/adr/20240229-leap-day.md", VALID_ADR_1);
    assert!(verify_records_in_dir_opts(&dir, None, None, true));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_duplicate_dated_slug_rejected() {
    let dir = init_temp_repo();
    commit_file(
        &dir,
        "docs/adr/20261122-shared-slug.md",
        VALID_ADR_1,
        "seed dated adr",
    );
    let base = git_in_test(&dir, &["rev-parse", "HEAD"]);
    git_in_test(&dir, &["checkout", "-b", "feature/dup-slug"]);

    stage_file(&dir, "crates/pure_math/src/lib.rs", "pub fn dup() {}\n");
    stage_file(&dir, "docs/adr/20261123-shared-slug.md", VALID_ADR_2);
    assert!(!verify_records_in_dir_opts(&dir, None, None, true));
    let bad_head = commit_file(
        &dir,
        "docs/adr/20261123-shared-slug.md",
        VALID_ADR_2,
        "dup slug",
    );
    assert!(!verify_records_in_dir(&dir, Some(&base), Some(&bad_head)));

    git_in_test(
        &dir,
        &[
            "mv",
            "docs/adr/20261123-shared-slug.md",
            "docs/adr/20261123-unique-slug.md",
        ],
    );
    assert!(verify_records_in_dir_opts(&dir, None, None, true));
    git_in_test(&dir, &["commit", "-m", "rename to unique slug"]);
    let ok_head = git_in_test(&dir, &["rev-parse", "HEAD"]);
    assert!(verify_records_in_dir(&dir, Some(&base), Some(&ok_head)));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_newly_added_four_digit_adr_rejected_while_legacy_edits_pass() {
    let dir = init_temp_repo();
    commit_file(
        &dir,
        "docs/adr/0042-legacy-kept.md",
        "# ADR 0042: Legacy Kept\n\n## Status\n\nAccepted\n",
        "seed 0042",
    );
    let base = git_in_test(&dir, &["rev-parse", "HEAD"]);
    git_in_test(&dir, &["checkout", "-b", "feature/freeze-legacy"]);

    // Adding a new 4-digit ADR (even unused 0041 or unfrozen 0047) fails staged, range, and HEAD
    stage_file(&dir, "docs/adr/0041-new-legacy.md", VALID_ADR_1);
    assert!(!verify_records_in_dir_opts(&dir, None, None, true));
    let bad_41 = commit_file(&dir, "docs/adr/0041-new-legacy.md", VALID_ADR_1, "add 0041");
    assert!(!verify_records_in_dir(&dir, Some(&base), Some(&bad_41)));
    assert!(!verify_records_in_dir(&dir, None, None));

    git_in_test(&dir, &["reset", "--hard", &base]);
    stage_file(&dir, "docs/adr/0047-unfrozen.md", VALID_ADR_1);
    assert!(!verify_records_in_dir_opts(&dir, None, None, true));

    // Modifying an existing frozen legacy ADR remains valid
    git_in_test(&dir, &["reset", "--hard", &base]);
    let edited_legacy = commit_file(
        &dir,
        "docs/adr/0042-legacy-kept.md",
        "# ADR 0042: Legacy Kept\n\n## Status\n\nAccepted (updated link)\n",
        "update legacy adr",
    );
    assert!(verify_records_in_dir(
        &dir,
        Some(&base),
        Some(&edited_legacy)
    ));

    let _ = fs::remove_dir_all(&dir);
}
