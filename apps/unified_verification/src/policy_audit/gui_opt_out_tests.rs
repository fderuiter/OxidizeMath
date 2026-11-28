use super::gui_opt_out::audit_headless_gui_opt_out;
use super::run_policy_audit_with_whitelist;
use crate::utils::test_utils::TempTestDir;
use std::fs;

#[test]
fn rejects_headless_gui_tool_opt_out_without_ui_imports_and_allows_ui_files() {
    let headless_ast = syn::parse_file(
        r#"
        #![cfg_attr(any(), verified(opt_out = "gui_tool"))]
        use math_explorer::pure_math::statistics::markov::parrondo::ParrondoParams;

        pub fn step_solver(p: &ParrondoParams) -> f64 {
            p.p_a()
        }
        "#,
    )
    .expect("valid syntax");
    assert!(!audit_headless_gui_opt_out(
        &headless_ast,
        "apps/math_explorer_gui/src/tabs/paradox_lab/headless.rs",
        Some("math_explorer_gui"),
    ));

    let egui_ast = syn::parse_file(
        r#"
        #![cfg_attr(any(), verified(opt_out = "gui_tool"))]
        use eframe::egui;

        pub fn render(ui: &mut egui::Ui) {
            let _ = ui;
        }
        "#,
    )
    .expect("valid syntax");
    assert!(audit_headless_gui_opt_out(
        &egui_ast,
        "apps/math_explorer_gui/src/tabs/paradox_lab/ui.rs",
        Some("math_explorer_gui"),
    ));

    let framework_ast = syn::parse_file(
        r#"
        #![cfg_attr(any(), verified(opt_out = "gui_tool"))]
        use crate::framework::InteractiveTool;

        pub struct ToolStub;
        "#,
    )
    .expect("valid syntax");
    assert!(audit_headless_gui_opt_out(
        &framework_ast,
        "apps/math_explorer_gui/src/tabs/paradox_lab/tool.rs",
        Some("math_explorer_gui"),
    ));

    // Non-GUI crate paths are unaffected by the math_explorer_gui/src/ check.
    assert!(audit_headless_gui_opt_out(
        &headless_ast,
        "apps/math_explorer_gui/build_snapshot.rs",
        Some("math_explorer_gui"),
    ));
}

#[test]
fn policy_audit_fails_closed_on_headless_gui_tool_member_file() {
    let dir = TempTestDir::new("policy_audit_headless_gui_tool");
    let gui_dir = dir.join("apps/math_explorer_gui");
    let tab_dir = gui_dir.join("src/tabs/demo");
    fs::create_dir_all(&tab_dir).unwrap();
    fs::write(
        gui_dir.join("Cargo.toml"),
        "[package]\nname = \"math_explorer_gui\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    fs::write(
        gui_dir.join("src/main.rs"),
        "//! Demo GUI binary.\nfn main() {}\n",
    )
    .unwrap();
    fs::write(
        tab_dir.join("solver.rs"),
        "//! Headless solver.\n#![cfg_attr(any(), verified(opt_out = \"gui_tool\"))]\npub fn solve(x: f64) -> f64 { x * 2.0 }\n",
    )
    .unwrap();

    let whitelist_file = dir.join("verification_whitelist.toml");
    fs::write(
        &whitelist_file,
        "[doc_verification]\nallowlist = []\n\n[entropy_guard]\nscan_paths = [\"apps\"]\n",
    )
    .unwrap();

    let member = crate::whitelist::lossy_slash_path(&gui_dir);
    assert!(!run_policy_audit_with_whitelist(
        &[member.as_str()],
        &whitelist_file,
    ));
}
