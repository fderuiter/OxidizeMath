use super::*;
use std::path::Path;

fn sample_crates() -> Vec<WorkspaceCrate> {
    vec![
        WorkspaceCrate {
            dir: "crates/domain_physics".to_string(),
            package_name: "domain_physics".to_string(),
            edition: "2024".to_string(),
            has_lib: true,
            has_bins: false,
            is_web: false,
        },
        WorkspaceCrate {
            dir: "crates/oxidize_core".to_string(),
            package_name: "oxidize_core".to_string(),
            edition: "2024".to_string(),
            has_lib: true,
            has_bins: true,
            is_web: true,
        },
        WorkspaceCrate {
            dir: "egui_plot".to_string(),
            package_name: "egui_plot".to_string(),
            edition: "2024".to_string(),
            has_lib: true,
            has_bins: false,
            is_web: true,
        },
        WorkspaceCrate {
            dir: "math_explorer_gui".to_string(),
            package_name: "math_explorer_gui".to_string(),
            edition: "2021".to_string(),
            has_lib: false,
            has_bins: true,
            is_web: true,
        },
    ]
}

#[test]
fn test_domain_crate_edit_skips_wasm_workflows_and_markdown() {
    let crates = sample_crates();
    let staged = vec!["crates/domain_physics/src/quantum.rs".to_string()];
    let plan = build_staged_plan(&staged, &staged, &crates);

    assert_eq!(
        plan.rustfmt_by_edition.get("2024"),
        Some(&vec!["crates/domain_physics/src/quantum.rs".to_string()])
    );
    assert!(!plan.run_workflows);
    assert!(plan.markdown_files.is_empty());
    assert_eq!(plan.host_clippy_packages, vec!["domain_physics"]);
    assert!(plan.wasm_clippy_args.is_none());
}

#[test]
fn test_web_crates_select_matching_wasm_targets_and_editions() {
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
    let plot_wasm = plot_plan
        .wasm_clippy_args
        .expect("plot should run wasm clippy");
    assert!(plot_wasm.iter().any(|a| a == "--lib"));
    assert!(!plot_wasm.iter().any(|a| a == "--bins"));

    let both = vec![
        "crates/oxidize_core/src/lib.rs".to_string(),
        "math_explorer_gui/src/main.rs".to_string(),
    ];
    let both_plan = build_staged_plan(&both, &both, &crates);
    let both_wasm = both_plan
        .wasm_clippy_args
        .expect("web crates should run wasm clippy");
    assert!(both_wasm.iter().any(|a| a == "--lib"));
    assert!(both_wasm.iter().any(|a| a == "--bins"));
}

#[test]
fn test_docs_and_workflows_only_skip_clippy() {
    let crates = sample_crates();
    let staged = vec![
        ".github/workflows/ci.yml".to_string(),
        "README.md".to_string(),
        "crates/domain_physics/README.md".to_string(),
    ];
    let plan = build_staged_plan(&staged, &staged, &crates);

    assert!(plan.rustfmt_by_edition.is_empty());
    assert!(plan.run_workflows);
    assert_eq!(
        plan.markdown_files,
        vec![
            PathBuf::from("README.md"),
            PathBuf::from("crates/domain_physics/README.md"),
        ]
    );
    assert!(plan.host_clippy_packages.is_empty());
    assert!(plan.wasm_clippy_args.is_none());
}

#[test]
fn test_load_real_workspace_crates_classifies_web_and_editions() {
    let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let crates = load_workspace_crates(&ws_root);
    assert!(!crates.is_empty());

    let web_names: Vec<&str> = crates
        .iter()
        .filter(|c| c.is_web)
        .map(|c| c.package_name.as_str())
        .collect();
    assert_eq!(web_names.len(), 3);
    assert!(web_names.contains(&"oxidize_core"));
    assert!(web_names.contains(&"egui_plot"));
    assert!(web_names.contains(&"math_explorer_gui"));

    let gui = crates
        .iter()
        .find(|c| c.package_name == "math_explorer_gui")
        .expect("missing math_explorer_gui");
    assert_eq!(gui.edition, "2021");
    assert!(!gui.has_lib);
    assert!(gui.has_bins);
}
