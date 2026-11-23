#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use std::path::{Path, PathBuf};

pub(super) struct CwdRestoreGuard {
    raw_cwd: PathBuf,
    canonical_cwd: PathBuf,
}

impl CwdRestoreGuard {
    pub(super) fn capture() -> Self {
        let raw_cwd = std::env::current_dir().expect("current_dir");
        let canonical_cwd = raw_cwd.canonicalize().expect("canonicalize current_dir");
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
}

impl Drop for CwdRestoreGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.raw_cwd);
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
        serde_json::json!({ "repo_path": repo_path })
            .to_string()
            .into(),
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
    let ci_path = TraceabilityTab::join_relative_segments(&ws_root, ".github/workflows/ci.yml");
    assert!(ci_path.canonicalize().is_ok());
    let mixed_path =
        TraceabilityTab::join_relative_segments(&ws_root, "./crates\\oxidize_core/src/lib.rs");
    assert!(mixed_path.canonicalize().is_ok());

    #[cfg(windows)]
    {
        let ws_norm = oxidize_core::path_utils::normalize_path(&ws_root);
        let verbatim_root = PathBuf::from(format!(r"\\?\{}", ws_norm.replace('/', r"\")));
        let verbatim_ci =
            TraceabilityTab::join_relative_segments(&verbatim_root, ".github/workflows/ci.yml");
        assert!(!verbatim_ci.to_string_lossy().contains('/'));
        assert!(std::fs::metadata(&verbatim_ci).is_ok());
        let verbatim_mixed = TraceabilityTab::join_relative_segments(
            &verbatim_root,
            "./crates/oxidize_core\\src/lib.rs",
        );
        assert!(!verbatim_mixed.to_string_lossy().contains('/'));
        assert!(std::fs::read_to_string(&verbatim_mixed).is_ok());
    }
}
