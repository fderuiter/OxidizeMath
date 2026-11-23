use super::snapshot::sync_index_snapshot;
use super::*;
use crate::gates::{HOST_CLIPPY_ARGS, WASM_CLIPPY_ARGS};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

static SNAPSHOT_TEST_NONCE: AtomicUsize = AtomicUsize::new(0);

#[allow(clippy::too_many_arguments)]
fn make_crate(
    dir: &str,
    pkg: &str,
    edition: &str,
    has_lib: bool,
    has_bins: bool,
    is_web: bool,
    normal_deps: &[&str],
    dev_deps: &[&str],
) -> WorkspaceCrate {
    WorkspaceCrate {
        dir: dir.to_string(),
        package_name: pkg.to_string(),
        edition: edition.to_string(),
        has_lib,
        has_bins,
        is_web,
        normal_deps: normal_deps.iter().map(|s| (*s).to_string()).collect(),
        dev_deps: dev_deps.iter().map(|s| (*s).to_string()).collect(),
    }
}

fn sample_crates() -> Vec<WorkspaceCrate> {
    vec![
        make_crate(
            "crates/scientific_metadata",
            "scientific_metadata",
            "2024",
            true,
            false,
            true,
            &[],
            &[],
        ),
        make_crate(
            "crates/domain_physics",
            "domain_physics",
            "2024",
            true,
            false,
            true,
            &["scientific_metadata"],
            &[],
        ),
        make_crate(
            "egui_plot",
            "egui_plot",
            "2024",
            true,
            false,
            true,
            &[],
            &[],
        ),
        make_crate(
            "math_explorer_gui",
            "math_explorer_gui",
            "2021",
            false,
            true,
            true,
            &["domain_physics", "egui_plot"],
            &[],
        ),
        make_crate(
            "crates/markdown_tests",
            "markdown_tests",
            "2021",
            true,
            false,
            false,
            &[],
            &["domain_physics"],
        ),
        make_crate("apps/xtask", "xtask", "2024", false, true, false, &[], &[]),
    ]
}

#[test]
fn test_host_only_leaf_crate_skips_wasm_workflows_and_markdown() {
    let crates = sample_crates();
    let staged = vec!["apps/xtask/src/main.rs".to_string()];
    let plan = build_staged_plan(&staged, &staged, &crates);

    assert!(!plan.full_workspace);
    assert_eq!(
        plan.rustfmt_by_edition.get("2024"),
        Some(&vec!["apps/xtask/src/main.rs".to_string()])
    );
    assert!(!plan.run_workflows);
    assert!(!plan.scan_all_markdown);
    assert!(plan.markdown_files.is_empty());
    assert_eq!(plan.host_clippy_packages, vec!["xtask"]);
    assert!(plan.wasm_clippy_args.is_none());
}

#[test]
fn test_reverse_dependents_expand_host_and_wasm_clippy() {
    let crates = sample_crates();
    let staged = vec!["crates/scientific_metadata/src/lib.rs".to_string()];
    let plan = build_staged_plan(&staged, &staged, &crates);

    assert_eq!(
        plan.host_clippy_packages,
        vec![
            "scientific_metadata",
            "domain_physics",
            "math_explorer_gui",
            "markdown_tests",
        ]
    );
    let wasm = plan
        .wasm_clippy_args
        .expect("shared web crate edit must trigger wasm32 clippy");
    assert!(wasm.contains(&"scientific_metadata".to_string()));
    assert!(wasm.contains(&"domain_physics".to_string()));
    assert!(wasm.contains(&"math_explorer_gui".to_string()));
    assert!(!wasm.contains(&"markdown_tests".to_string()));
    assert!(wasm.contains(&"--lib".to_string()));
    assert!(wasm.contains(&"--bins".to_string()));
    assert!(wasm.contains(&"--tests".to_string()));
}

#[test]
fn test_web_leaf_and_library_target_flags() {
    let crates = sample_crates();

    let gui_only = vec!["math_explorer_gui/src/main.rs".to_string()];
    let gui_plan = build_staged_plan(&gui_only, &gui_only, &crates);
    assert_eq!(
        gui_plan.rustfmt_by_edition.get("2021"),
        Some(&vec!["math_explorer_gui/src/main.rs".to_string()])
    );
    assert_eq!(gui_plan.host_clippy_packages, vec!["math_explorer_gui"]);
    let gui_wasm = gui_plan
        .wasm_clippy_args
        .expect("gui should run wasm clippy");
    assert!(!gui_wasm.iter().any(|a| a == "--lib"));
    assert!(gui_wasm.iter().any(|a| a == "--bins"));
    assert!(gui_wasm.iter().any(|a| a == "--tests"));

    let plot_only = vec!["egui_plot/Cargo.toml".to_string()];
    let plot_plan = build_staged_plan(&plot_only, &plot_only, &crates);
    assert_eq!(
        plot_plan.host_clippy_packages,
        vec!["egui_plot", "math_explorer_gui"]
    );
    let plot_wasm = plot_plan
        .wasm_clippy_args
        .expect("plot and gui dependent should run wasm clippy");
    assert!(plot_wasm.iter().any(|a| a == "--lib"));
    assert!(plot_wasm.iter().any(|a| a == "--bins"));
}

#[test]
fn test_workflow_triggers_and_deleted_file_markdown_escalation() {
    let crates = sample_crates();
    let staged = vec![
        ".github/workflows/ci.yml".to_string(),
        "README.md".to_string(),
        "crates/domain_physics/README.md".to_string(),
    ];
    let plan = build_staged_plan(&staged, &staged, &crates);

    assert!(plan.rustfmt_by_edition.is_empty());
    assert!(plan.run_workflows);
    assert!(!plan.scan_all_markdown);
    assert_eq!(
        plan.markdown_files,
        vec![
            PathBuf::from("README.md"),
            PathBuf::from("crates/domain_physics/README.md"),
        ]
    );
    assert!(plan.host_clippy_packages.is_empty());
    assert!(plan.host_clippy_args.is_none());
    assert!(plan.wasm_clippy_args.is_none());

    let deleted_staged = vec!["docs/removed_guide.md".to_string(), "README.md".to_string()];
    let existing = vec!["README.md".to_string()];
    let del_plan = build_staged_plan(&deleted_staged, &existing, &crates);
    assert!(
        del_plan.scan_all_markdown,
        "Deleted or renamed files must escalate to full-repository markdown link scanning"
    );
}

#[test]
fn test_root_config_edits_escalate_to_full_workspace_checks() {
    let crates = sample_crates();
    for root_cfg in [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        ".clippy.toml",
        ".rustfmt.toml",
        ".cargo/config.toml",
        "verification_whitelist.toml",
    ] {
        let staged = vec![root_cfg.to_string()];
        let plan = build_staged_plan(&staged, &staged, &crates);
        assert!(plan.full_workspace, "{root_cfg} must escalate workspace");
        assert!(plan.run_workflows, "{root_cfg} must run workflow checks");
        assert!(plan.scan_all_markdown, "{root_cfg} must scan all markdown");
        assert_eq!(plan.host_clippy_packages.len(), crates.len());
        assert_eq!(
            plan.host_clippy_args.as_deref(),
            Some(
                HOST_CLIPPY_ARGS
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect::<Vec<_>>()
                    .as_slice()
            )
        );
        assert_eq!(
            plan.wasm_clippy_args.as_deref(),
            Some(
                WASM_CLIPPY_ARGS
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect::<Vec<_>>()
                    .as_slice()
            )
        );
    }
}

#[test]
fn test_load_real_workspace_discovers_all_17_web_crates_and_dependents() {
    let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let crates = load_workspace_crates(&ws_root);
    assert_eq!(crates.len(), 20);

    let mut web_names: Vec<&str> = crates
        .iter()
        .filter(|c| c.is_web)
        .map(|c| c.package_name.as_str())
        .collect();
    web_names.sort_unstable();
    assert_eq!(
        web_names,
        vec![
            "diagnostics",
            "domain_ai",
            "domain_applied",
            "domain_biology",
            "domain_climate",
            "domain_epidemiology",
            "domain_physics",
            "egui_plot",
            "federated_registry",
            "math_commons",
            "math_explorer",
            "math_explorer_gui",
            "oxidize_core",
            "pure_math",
            "scientific_metadata",
            "verified_engine",
            "verified_engine_macros",
        ]
    );

    for host_only in ["markdown_tests", "unified_verification", "xtask"] {
        let krate = crates.iter().find(|c| c.package_name == host_only).unwrap();
        assert!(!krate.is_web, "{host_only} should not be marked is_web");
    }

    let meta_staged = vec!["crates/scientific_metadata/src/lib.rs".to_string()];
    let meta_plan = build_staged_plan(&meta_staged, &meta_staged, &crates);
    for expected_dep in ["verified_engine", "math_explorer", "math_explorer_gui"] {
        assert!(
            meta_plan
                .host_clippy_packages
                .contains(&expected_dep.to_string()),
            "scientific_metadata edit must lint reverse dependent {expected_dep}"
        );
    }
}

fn run_git(repo: &Path, args: &[&str]) {
    let status = Command::new("git")
        .current_dir(repo)
        .args(args)
        .status()
        .expect("git command failed to spawn");
    assert!(status.success(), "git {args:?} failed");
}

fn init_temp_git_repo() -> PathBuf {
    let id = SNAPSHOT_TEST_NONCE.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "xtask_snapshot_sync_test_{}_{}",
        std::process::id(),
        id
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("src")).unwrap();
    run_git(&dir, &["init", "-q"]);
    run_git(&dir, &["config", "user.email", "test@example.com"]);
    run_git(&dir, &["config", "user.name", "Test"]);
    fs::write(dir.join("src/keep.rs"), "pub fn keep() -> i32 { 1 }\n").unwrap();
    fs::write(dir.join("src/edit.rs"), "pub fn clean() -> i32 { 2 }\n").unwrap();
    fs::write(dir.join("obsolete.md"), "# old\n").unwrap();
    run_git(&dir, &["add", "."]);
    run_git(&dir, &["commit", "-q", "-m", "initial"]);
    dir
}

#[test]
fn test_sync_index_snapshot_isolates_staged_blobs_and_preserves_warm_mtime() {
    let repo = init_temp_git_repo();
    let snap = repo.join("target/lint-staged-snapshot");
    let manifest = repo.join("target/lint-staged-snapshot.manifest");

    sync_index_snapshot(&repo, &snap, &manifest).expect("initial sync should succeed");
    let keep_mtime_before = fs::metadata(snap.join("src/keep.rs"))
        .unwrap()
        .modified()
        .unwrap();

    // Stage a clean hunk, then write invalid unstaged WIP to working tree + untracked file in snapshot
    fs::write(
        repo.join("src/edit.rs"),
        "pub fn staged_ok() -> i32 { 3 }\n",
    )
    .unwrap();
    run_git(&repo, &["add", "src/edit.rs"]);
    run_git(&repo, &["rm", "-q", "obsolete.md"]);
    fs::write(repo.join("src/edit.rs"), "BROKEN UNSTAGED SYNTAX {{{{\n").unwrap();
    fs::create_dir_all(snap.join("untracked_dir")).unwrap();
    fs::write(snap.join("untracked_dir/junk.txt"), "stale").unwrap();

    thread::sleep(Duration::from_millis(20));
    sync_index_snapshot(&repo, &snap, &manifest).expect("warm sync should succeed");

    assert_eq!(
        fs::read_to_string(snap.join("src/edit.rs")).unwrap(),
        "pub fn staged_ok() -> i32 { 3 }\n"
    );
    assert_eq!(
        fs::read_to_string(repo.join("src/edit.rs")).unwrap(),
        "BROKEN UNSTAGED SYNTAX {{{{\n"
    );
    assert!(!snap.join("obsolete.md").exists());
    assert!(!snap.join("untracked_dir").exists());
    assert!(!repo.join(".git/index.lock").exists());
    assert_eq!(
        fs::metadata(snap.join("src/keep.rs"))
            .unwrap()
            .modified()
            .unwrap(),
        keep_mtime_before,
        "Unchanged files must preserve mtime across snapshot syncs"
    );

    let _ = fs::remove_dir_all(repo);
}
