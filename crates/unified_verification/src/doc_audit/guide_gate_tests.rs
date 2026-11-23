use super::cargo_cli_checks::{check_markdown_cargo_commands, check_rustdoc_cargo_commands};
use super::catalog_and_guide_checks::{audit_examples_catalog, audit_paper_and_gate_instructions};
use super::facade_imports::audit_workspace_guides_and_manifests;
use super::manifest_model::WorkspaceManifestModel;
use super::toml_dep_checks::check_markdown_toml_dependencies;
use super::trait_contracts::{TraitContractRegistry, check_markdown_trait_and_gui_contracts};
use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn real_workspace_passes_all_static_guide_and_manifest_gates() {
    let root = workspace_root();
    let violations = audit_workspace_guides_and_manifests(&root);
    assert!(
        violations.is_empty(),
        "Expected zero workspace guide/manifest violations, got:\n{}",
        violations.join("\n")
    );
}

#[test]
fn cargo_cli_check_flags_missing_example_features_and_virtual_manifest_test() {
    let root = workspace_root();
    let model = WorkspaceManifestModel::load(&root).expect("load manifests");
    let md_path = root.join("README.md");
    let bad_md = "\
```bash
cargo run --package math_explorer --example hello_world
cargo run --package math_explorer --example <example_name>
cargo test --lib pure_math::statistics::markov
cargo doc --workspace --no-deps
```
";
    let violations = check_markdown_cargo_commands(&md_path, &root, bad_md, &model);
    assert_eq!(violations.len(), 4, "Violations: {violations:?}");
    assert!(violations[0].contains("requires feature(s) [\"physics\"]"));
    assert!(violations[1].contains("Template `--example <example_name>`"));
    assert!(violations[2].contains("omits `-p <package>`"));
    assert!(violations[3].contains("omits `--all-features`"));
}

#[test]
fn cargo_cli_check_accepts_valid_feature_gated_commands_and_rustdoc_headers() {
    let root = workspace_root();
    let model = WorkspaceManifestModel::load(&root).expect("load manifests");
    let md_path = root.join("README.md");
    let good_md = "\
```bash
cargo run --package math_explorer --features physics --example hello_world
cargo run --package math_explorer --features applied --example grpo_demo
cargo run --package math_explorer --features <domain_feature> --example <example_name>
cargo test -p pure_math --lib pure_math::statistics::markov
RUSTDOCFLAGS=\"-D warnings\" cargo doc --workspace --no-deps --all-features
```
";
    assert!(check_markdown_cargo_commands(&md_path, &root, good_md, &model).is_empty());

    let bad_ex =
        "//! Run with:\n//!   cargo run --package math_explorer --example transformer_simple\n";
    let ex_violations =
        check_rustdoc_cargo_commands(Path::new("transformer_simple.rs"), bad_ex, &model);
    assert_eq!(ex_violations.len(), 1);
    assert!(ex_violations[0].contains("requires feature(s) [\"ai\"]"));
}

#[test]
fn toml_dependency_check_flags_missing_or_mismatched_domain_features() {
    let root = workspace_root();
    let model = WorkspaceManifestModel::load(&root).expect("load manifests");

    let root_guide = root.join("math_explorer/README.md");
    let bare_dep = "```toml\n[dependencies]\nmath_explorer = \"0.4\"\n```\n";
    let v1 = check_markdown_toml_dependencies(&root_guide, &root, bare_dep, &model);
    assert_eq!(v1.len(), 1);
    assert!(v1[0].contains("omits explicit `features = [...]`"));

    let grpo_guide = root.join("crates/domain_applied/src/applied/grpo/README.md");
    let wrong_dom = "```toml\n[dependencies]\nmath_explorer = { version = \"0.4\", features = [\"physics\"] }\n```\n";
    let v2 = check_markdown_toml_dependencies(&grpo_guide, &root, wrong_dom, &model);
    assert_eq!(v2.len(), 1);
    assert!(v2[0].contains("must enable feature `\"applied\"`"));

    let right_dom = "```toml\n[dependencies]\nmath_explorer = { version = \"0.4\", features = [\"applied\"] }\n```\n";
    assert!(check_markdown_toml_dependencies(&grpo_guide, &root, right_dom, &model).is_empty());
}

#[test]
fn catalog_check_flags_unregistered_file_missing_entry_and_wrong_domain_heading() {
    let dir = crate::utils::test_utils::create_temp_test_dir("catalog_parity_gate");
    let ex_dir = dir.join("math_explorer/examples");
    let src_dir = dir.join("math_explorer/src");
    fs::create_dir_all(&ex_dir).unwrap();
    fs::create_dir_all(&src_dir).unwrap();

    fs::write(
        dir.join("Cargo.toml"),
        "[workspace]\nmembers = [\"math_explorer\"]\n",
    )
    .unwrap();
    fs::write(src_dir.join("lib.rs"), "").unwrap();
    fs::write(
        dir.join("math_explorer/Cargo.toml"),
        "[package]\nname = \"math_explorer\"\nversion = \"0.4.0\"\n[features]\ndefault = []\nphysics = []\napplied = []\n[[example]]\nname = \"mmwave_setup\"\nrequired-features = [\"physics\"]\n[[example]]\nname = \"grpo_demo\"\nrequired-features = [\"applied\"]\n",
    )
    .unwrap();
    fs::write(ex_dir.join("mmwave_setup.rs"), "fn main() {}\n").unwrap();
    fs::write(ex_dir.join("grpo_demo.rs"), "fn main() {}\n").unwrap();
    fs::write(ex_dir.join("check_statrs.rs"), "fn main() {}\n").unwrap();
    fs::write(
        ex_dir.join("README.md"),
        "## Examples\n### Applied\n- `mmwave_setup`: Wrong section.\n",
    )
    .unwrap();

    let model = WorkspaceManifestModel::load(&dir).unwrap();
    let violations = audit_examples_catalog(&dir, &model);
    let _ = fs::remove_dir_all(&dir);

    assert!(violations.iter().any(|v| v.contains("check_statrs.rs")));
    assert!(violations.iter().any(|v| v.contains("requires `physics`")));
    assert!(violations.iter().any(|v| v.contains("grpo_demo")));
}

#[test]
fn paper_and_gate_check_rejects_tectonic_and_missing_all_features() {
    let dir = crate::utils::test_utils::create_temp_test_dir("paper_and_gate_check");
    fs::create_dir_all(dir.join("apps/xtask/src")).unwrap();
    fs::create_dir_all(dir.join(".github/workflows")).unwrap();

    fs::write(
        dir.join("apps/xtask/src/papers.rs"),
        "fn run() { let _ = \"pdflatex\"; }\n",
    )
    .unwrap();
    fs::write(
        dir.join("apps/xtask/src/main.rs"),
        "fn help() { println!(\"Requires Tectonic\"); }\n",
    )
    .unwrap();
    fs::write(
        dir.join("apps/xtask/src/gates.rs"),
        "const ARGS: &[&str] = &[\"doc\", \"--workspace\", \"--no-deps\"];\n",
    )
    .unwrap();
    fs::write(
        dir.join(".github/workflows/ci.yml"),
        "run: cargo doc --workspace --no-deps\n",
    )
    .unwrap();
    let readme = dir.join("README.md");
    fs::write(&readme, "Install Tectonic 0.15.0 to compile papers.\n").unwrap();

    let violations = audit_paper_and_gate_instructions(&dir, &[readme]);
    let _ = fs::remove_dir_all(&dir);

    assert_eq!(violations.len(), 4, "Violations: {violations:?}");
}

#[test]
fn trait_and_gui_contract_check_flags_stale_theory_methods_and_manual_tab_wiring() {
    let root = workspace_root();
    let registry = TraitContractRegistry::load(&root);
    let gui_walkthrough = root.join("docs/architecture/interactive_tool_walkthrough.md");

    let stale_guide = "\
In `src/tabs/mod.rs`, add your module and update `src/app.rs`.

```rust,ignore
impl TheoryDescribable for DemoTool {
    fn theory_title(&self) -> &'static str { \"Demo\" }
    fn theory_description(&self, _level: AudienceLevel) -> &'static str { \"Text\" }
    fn theory_equations(&self) -> &'static [&'static str] { &[] }
}

impl InteractiveTool for DemoTool {
    // ...
    fn step_simulation(&mut self, dt: f64) {}
}
```

```rust,ignore
impl ExplorerTab for DemoTab {
    fn name(&self) -> &'static str { \"Demo\" }
    fn show(&mut self, _ctx: &egui::Context, _frame: &mut eframe::Frame) {}
}
```
Running at 60 FPS (`dt = 0.016`).
";
    let violations =
        check_markdown_trait_and_gui_contracts(&gui_walkthrough, stale_guide, &registry);
    assert!(violations.iter().any(|v| v.contains("theory_title")));
    assert!(violations.iter().any(|v| v.contains("theory_equations")));
    assert!(
        violations
            .iter()
            .any(|v| v.contains("theory_description` signature mismatch"))
    );
    assert!(
        violations
            .iter()
            .any(|v| v.contains("`impl ExplorerTab` defines non-existent trait method `show`"))
    );
    assert!(
        violations
            .iter()
            .any(|v| v.contains("`impl ExplorerTab` omits required trait method `show_inside`"))
    );
    assert!(violations.iter().any(|v| v.contains("src/app.rs")));
    assert!(violations.iter().any(|v| v.contains("src/tabs/mod.rs")));
    assert!(violations.iter().any(|v| v.contains("step_simulation")));
}
