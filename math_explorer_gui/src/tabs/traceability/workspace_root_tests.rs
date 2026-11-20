#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use std::path::{Path, PathBuf};

pub(super) fn strip_verbatim_prefix(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if let Some(rest) = raw.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{rest}"))
    } else if let Some(rest) = raw.strip_prefix(r"\\?\") {
        PathBuf::from(rest)
    } else {
        path.to_path_buf()
    }
}

pub(super) struct CwdRestoreGuard {
    raw_cwd: PathBuf,
    canonical_cwd: PathBuf,
}

impl CwdRestoreGuard {
    pub(super) fn capture() -> Self {
        let canonical_cwd = std::env::current_dir()
            .expect("current_dir")
            .canonicalize()
            .expect("canonicalize current_dir");
        let raw_cwd = strip_verbatim_prefix(&canonical_cwd);
        // Ensure DEFAULT_SCAN_CACHE is initialized while still inside the workspace.
        let mut warm_tab = TraceabilityTab::default();
        warm_tab.refresh();
        Self {
            raw_cwd,
            canonical_cwd,
        }
    }

    pub(super) fn original(&self) -> &Path {
        &self.canonical_cwd
    }

    pub(super) fn filesystem_root(&self) -> PathBuf {
        let mut root = PathBuf::new();
        for comp in self.raw_cwd.components() {
            match comp {
                std::path::Component::Prefix(_) | std::path::Component::RootDir => {
                    root.push(comp.as_os_str());
                }
                _ => break,
            }
        }
        if root.as_os_str().is_empty() {
            PathBuf::from("/")
        } else {
            root
        }
    }
}

impl Drop for CwdRestoreGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let _ = std::env::set_current_dir(&self.raw_cwd);
        } else {
            std::env::set_current_dir(&self.raw_cwd).expect("restore original current_dir");
        }
    }
}

pub(super) fn create_non_workspace_canary_dirs() -> (PathBuf, PathBuf, PathBuf) {
    let pid = std::process::id();
    let external_canary_dir =
        std::env::temp_dir().join(format!("oxidize_canary_root_escape_{pid}"));
    std::fs::create_dir_all(external_canary_dir.join("papers")).unwrap();
    std::fs::write(
        external_canary_dir
            .join("papers")
            .join("canary_root_escape.tex"),
        "\\documentclass{article}\\begin{document}RootCanary\\end{document}",
    )
    .unwrap();

    let fake_home = std::env::temp_dir().join(format!("oxidize_fake_home_{pid}"));
    let home_project_dir = fake_home.join("projects").join("untrusted_repo");
    std::fs::create_dir_all(home_project_dir.join("papers")).unwrap();
    std::fs::write(
        home_project_dir
            .join("papers")
            .join("canary_home_escape.tex"),
        "\\documentclass{article}\\begin{document}HomeCanary\\end{document}",
    )
    .unwrap();

    (external_canary_dir, fake_home, home_project_dir)
}

pub(super) fn build_mws_envelope(repo_path: &str) -> String {
    let mut tab_states = std::collections::HashMap::new();
    tab_states.insert(
        "Traceability Portal".to_string(),
        serde_json::json!({ "repo_path": repo_path }).to_string(),
    );
    let state = crate::app::AppState {
        selected_tab_name: Some("Traceability Portal".to_string()),
        tab_states,
        ..Default::default()
    };
    crate::app::WorkspaceSessionManager::default().create_envelope(state)
}

fn shape_has_text(shape: &egui::epaint::Shape, needle: &str) -> bool {
    match shape {
        egui::epaint::Shape::Text(t) => t.galley.text().contains(needle),
        egui::epaint::Shape::Vec(items) => items.iter().any(|s| shape_has_text(s, needle)),
        _ => false,
    }
}

pub(super) fn assert_mws_session_rejected(
    app: &mut crate::app::MathExplorerApp,
    canary_dir: &Path,
    canary_tex: &str,
) {
    use eframe::App;

    assert_eq!(TraceabilityTab::canonical_workspace_root(), None);
    let portal_json = app
        .get_app_state()
        .tab_states
        .get("Traceability Portal")
        .cloned()
        .expect("Traceability Portal state");
    let parsed: TraceabilityState = serde_json::from_str(&portal_json).unwrap();
    assert_eq!(
        parsed.repo_path, None,
        "Non-workspace cwd must reject repo_path in .mws session"
    );

    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let out = ctx.run(raw, |ctx| app.update(ctx, &mut frame));
    assert!(out
        .shapes
        .iter()
        .any(|cs| shape_has_text(&cs.shape, "Scanning: default workspace")));
    assert!(!out
        .shapes
        .iter()
        .any(|cs| shape_has_text(&cs.shape, canary_tex)));

    let mut tab = TraceabilityTab {
        repo_path: Some(canary_dir.to_string_lossy().into_owned()),
        ..TraceabilityTab::default()
    };
    tab.refresh();
    assert_eq!(tab.repo_path, None);
    super::wait_for_scan_completion(&mut tab);
    assert_eq!(tab.online_document_url(canary_tex), None);
}

#[test]
fn test_has_workspace_header_strips_comments_and_matches_tables() {
    assert!(TraceabilityTab::has_workspace_header(
        "[workspace]\nmembers = []\n"
    ));
    assert!(TraceabilityTab::has_workspace_header(
        "  [ workspace ]  # workspace root\n"
    ));
    assert!(TraceabilityTab::has_workspace_header(
        "[workspace.package]\nversion = \"0.1.0\"\n"
    ));
    assert!(TraceabilityTab::has_workspace_header(
        "[workspace.dependencies] # shared deps\n"
    ));

    assert!(!TraceabilityTab::has_workspace_header(
        "# [workspace]\n[package]\nname = \"demo\"\n"
    ));
    assert!(!TraceabilityTab::has_workspace_header(
        "[package]\nname = \"demo\" # [workspace]\ndescription = \"[workspace]\"\n"
    ));
    assert!(!TraceabilityTab::has_workspace_header(
        "[workspace_hack]\n[[workspace]]\n"
    ));
}

#[test]
fn test_spoofed_cargo_toml_comments_outside_workspace_fail_closed() {
    let _mutex = crate::reflective_ui::tests::lock_test_mutex();
    let _cwd_guard = CwdRestoreGuard::capture();

    let pid = std::process::id();
    let spoof_root = std::env::temp_dir().join(format!("oxidize_spoof_ws_{pid}"));
    let sub_dir = spoof_root.join("nested");
    std::fs::create_dir_all(sub_dir.join("papers")).unwrap();
    std::fs::write(
        spoof_root.join("Cargo.toml"),
        "# [workspace]\n[package]\nname = \"spoof\" # [workspace.package]\ndescription = \"[workspace]\"\n",
    )
    .unwrap();
    std::fs::write(sub_dir.join("papers").join("spoof_canary.tex"), "x").unwrap();

    std::env::set_current_dir(&spoof_root).expect("set_current_dir to spoof_root");
    assert_eq!(TraceabilityTab::canonical_workspace_root(), None);

    let mut app = crate::app::MathExplorerApp::default();
    let mws = build_mws_envelope(&sub_dir.to_string_lossy());
    app.load_state_from_json(&mws);
    assert_mws_session_rejected(&mut app, &sub_dir, "spoof_canary.tex");

    drop(_cwd_guard);
    let _ = std::fs::remove_dir_all(&spoof_root);
}

#[test]
fn test_join_relative_segments_normalizes_separators_on_canonical_root() {
    let _mutex = crate::reflective_ui::tests::lock_test_mutex();
    let ws_root = TraceabilityTab::canonical_workspace_root().expect("workspace root");
    let joined = TraceabilityTab::join_relative_segments(&ws_root, "./.github/workflows\\ci.yml");
    assert_eq!(
        joined,
        ws_root.join(".github").join("workflows").join("ci.yml")
    );
    assert!(joined.is_file());
    assert_eq!(TraceabilityTab::join_relative_segments(&ws_root, "./"), ws_root);

    let mut tab = TraceabilityTab::default();
    tab.load_state(r#"{"repo_path":"crates/pure_math"}"#);
    assert!(
        tab.repo_path
            .as_deref()
            .is_some_and(|p| p.ends_with("crates/pure_math"))
    );
}

#[test]
fn test_cwd_restore_guard_strips_verbatim_prefixes_and_survives_repeated_root_roundtrips() {
    assert_eq!(
        strip_verbatim_prefix(Path::new(r"\\?\D:\a\OxidizeMath\OxidizeMath")),
        PathBuf::from(r"D:\a\OxidizeMath\OxidizeMath")
    );
    assert_eq!(
        strip_verbatim_prefix(Path::new(r"\\?\UNC\server\share\OxidizeMath")),
        PathBuf::from(r"\\server\share\OxidizeMath")
    );

    let _mutex = crate::reflective_ui::tests::lock_test_mutex();
    for _ in 0..3 {
        let guard = CwdRestoreGuard::capture();
        assert!(!guard.raw_cwd.to_string_lossy().starts_with(r"\\?\"));
        let root = guard.filesystem_root();
        assert!(!root.to_string_lossy().starts_with(r"\\?\"));
        std::env::set_current_dir(&root).expect("chdir to filesystem_root");
        assert_eq!(TraceabilityTab::canonical_workspace_root(), None);
        drop(guard);
        assert!(TraceabilityTab::canonical_workspace_root().is_some());
    }
}
