use super::facade_imports::audit_workspace_guides_and_manifests;
use super::manifest_model::WorkspaceManifestModel;
use super::static_gates::{
    audit_system_overview_crate_catalog, check_markdown_binary_gui_imports,
    check_markdown_crate_counts, check_markdown_ignore_fences, check_rs_binary_gui_imports,
    collect_binary_only_packages,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

#[test]
fn ignore_fence_gate_rejects_unapproved_files_cap_overflow_and_non_gui_blocks() {
    let root = Path::new("/ws");
    let mut caps = BTreeMap::new();
    caps.insert(
        "docs/architecture/interactive_tool_walkthrough.md".to_string(),
        1,
    );

    let readme = root.join("README.md");
    let bad_readme = "```rust,ignore\nlet x = 1 + 1;\n```\n";
    let v_readme = check_markdown_ignore_fences(&readme, root, bad_readme, &caps);
    assert_eq!(v_readme.len(), 1);
    assert!(v_readme[0].contains("Unapproved `rust,ignore`"));

    let walkthrough = root.join("docs/architecture/interactive_tool_walkthrough.md");
    let valid_gui = "```rust,ignore\nuse crate::framework::InteractiveTool;\n```\n";
    assert!(check_markdown_ignore_fences(&walkthrough, root, valid_gui, &caps).is_empty());
    let valid_out_of_line_mod =
        "```rust,ignore\n#[cfg(feature = \"physics\")]\npub mod fluid_dynamics;\n```\n";
    assert!(
        check_markdown_ignore_fences(&walkthrough, root, valid_out_of_line_mod, &caps).is_empty()
    );

    let overflow_and_theory = "\
```rust,ignore
use crate::framework::InteractiveTool;
```
```rust,ignore
impl TheoryDescribable for DemoTool {
    fn theory_description(&self) -> TheoryDescription { todo!() }
}
```
";
    let v_walk = check_markdown_ignore_fences(&walkthrough, root, overflow_and_theory, &caps);
    assert!(
        v_walk
            .iter()
            .any(|v| v.contains("exceeds approved `rust,ignore` fence cap"))
    );
    assert!(
        v_walk
            .iter()
            .any(|v| v.contains("must be a compiled `rust` doctest"))
    );
}

#[test]
fn binary_gui_import_gate_rejects_markdown_rustdoc_and_source_imports() {
    let mut binary_pkgs = BTreeSet::new();
    binary_pkgs.insert("math_explorer_gui".to_string());

    let bad_md = "\
```rust,ignore
use math_explorer_gui::framework::SimulationFramework;
```
";
    let v_md = check_markdown_binary_gui_imports(Path::new("guide.md"), bad_md, &binary_pkgs);
    assert_eq!(v_md.len(), 1);
    assert!(v_md[0].contains("Invalid `math_explorer_gui::` import"));

    let good_md = "```rust,ignore\nuse crate::framework::SimulationFramework;\n```\n";
    assert!(
        check_markdown_binary_gui_imports(Path::new("guide.md"), good_md, &binary_pkgs).is_empty()
    );

    let bad_rs = "\
//! ```rust,no_run
//! use math_explorer_gui::tabs::ExplorerTab;
//! ```
pub fn bad() {
    let _ = math_explorer_gui::tabs::all();
}
";
    let v_rs = check_rs_binary_gui_imports(Path::new("tabs/mod.rs"), bad_rs, &binary_pkgs);
    assert_eq!(v_rs.len(), 2, "Violations: {v_rs:?}");
    assert!(v_rs[0].contains("Rustdoc snippet imports binary-only package"));
    assert!(v_rs[1].contains("Source file imports binary-only package"));
}

#[test]
fn crate_count_gate_flags_stale_counts_and_allows_web_compiled_subset() {
    let md_path = Path::new("docs/README.md");
    let stale_md = "\
- [System Overview](architecture/system_overview.md): Catalog of all 18 workspace crates.
Run tests across 18 workspace crates.
";
    let v_stale = check_markdown_crate_counts(md_path, stale_md, 20);
    assert_eq!(v_stale.len(), 2, "Violations: {v_stale:?}");
    assert!(v_stale[0].contains("Stale workspace crate count `18`"));

    let clean_md = "\
- [System Overview](architecture/system_overview.md): Catalog of all 20 workspace crates.
- Zero WebAssembly compilation warnings across all 17 web-compiled crates.
";
    assert!(check_markdown_crate_counts(md_path, clean_md, 20).is_empty());
}

#[test]
fn system_overview_catalog_and_adr_historical_exclusion_work_end_to_end() {
    let dir = crate::utils::test_utils::create_temp_test_dir("static_doc_gates_e2e");
    fs::create_dir_all(dir.join("crates/alpha/src")).unwrap();
    fs::create_dir_all(dir.join("math_explorer_gui/src")).unwrap();
    fs::create_dir_all(dir.join("docs/architecture")).unwrap();
    fs::create_dir_all(dir.join("docs/adr")).unwrap();

    fs::write(
        dir.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/alpha\", \"math_explorer_gui\"]\n",
    )
    .unwrap();
    fs::write(
        dir.join("crates/alpha/Cargo.toml"),
        "[package]\nname = \"alpha\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(dir.join("crates/alpha/src/lib.rs"), "").unwrap();
    fs::write(
        dir.join("math_explorer_gui/Cargo.toml"),
        "[package]\nname = \"math_explorer_gui\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(dir.join("math_explorer_gui/src/main.rs"), "fn main() {}\n").unwrap();

    // Historical ADR containing legacy rust,ignore, Tectonic, math_explorer_gui::, and 18 crates must be exempt
    fs::write(
        dir.join("docs/adr/0009-legacy.md"),
        "All 18 workspace crates and Tectonic.\n```rust,ignore\nuse math_explorer_gui::tabs;\n```\n",
    )
    .unwrap();

    // Active system_overview missing math_explorer_gui in table and mermaid
    fs::write(
        dir.join("docs/architecture/system_overview.md"),
        "```mermaid\ngraph LR\nA --> B\n```\n```mermaid\ngraph TD\nalpha\n```\n| `crates/alpha` | Core |\n",
    )
    .unwrap();

    let model = WorkspaceManifestModel::load(&dir).unwrap();
    let bins = collect_binary_only_packages(&model);
    assert!(bins.contains("math_explorer_gui"));
    assert!(!bins.contains("alpha"));

    let cat_v = audit_system_overview_crate_catalog(&dir, &model);
    assert_eq!(cat_v.len(), 2, "Catalog violations: {cat_v:?}");

    // Fix system_overview so end-to-end audit verifies ADR exclusion cleanly
    fs::write(
        dir.join("docs/architecture/system_overview.md"),
        "All 2 workspace crates.\n```mermaid\ngraph LR\nA --> B\n```\n```mermaid\ngraph TD\nalpha\nmath_explorer_gui\n```\n| `crates/alpha` | Core |\n| `math_explorer_gui` | GUI |\n",
    )
    .unwrap();

    let e2e_v = audit_workspace_guides_and_manifests(&dir);
    let _ = fs::remove_dir_all(&dir);
    assert!(
        e2e_v.is_empty(),
        "Expected ADR historical exclusions to pass cleanly, got: {e2e_v:?}"
    );
}
