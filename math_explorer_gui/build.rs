//! Build script for dynamic tab discovery in math_explorer_gui.
mod build_snapshot;
mod build_stage;

use oxidize_core::path_utils;
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);
    let dest_path = out_path.join("generated_tabs.rs");

    // Tell cargo to rerun if the build script, manifest, or tabs directory changes
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/tabs");
    println!("cargo:rerun-if-changed=src/tabs/mod.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");

    build_stage::stage_gui_inputs_to_out_dir(out_path);

    let mut discovered_tabs = Vec::new();
    let mut local_tab_gates = Vec::new();

    // 1. Scan local tabs in src/tabs and verify static declarations in src/tabs/mod.rs
    let tabs_dir = Path::new("src/tabs");
    if tabs_dir.exists() {
        scan_local_tabs(tabs_dir, &mut discovered_tabs, &mut local_tab_gates);
        validate_static_tab_modules(&tabs_dir.join("mod.rs"), &local_tab_gates);
    }

    // 2. Scan external path dependencies for tabs
    scan_cargo_toml_deps(&mut discovered_tabs);

    // Sort by order, then by instantiation string
    discovered_tabs.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));

    let generated_code = render_generated_tabs_code(&discovered_tabs);
    build_stage::write_if_changed(&dest_path, generated_code.as_bytes());

    // 3. Generate deterministic precomputed traceability snapshot for static web builds
    build_snapshot::generate_traceability_snapshot(out_path);
}

fn render_generated_tabs_code(discovered_tabs: &[(String, Option<String>, i32)]) -> String {
    let mut generated_code = String::from("#[allow(clippy::vec_init_then_push)]\n");
    generated_code
        .push_str("fn tab_constructors() -> Vec<fn() -> Box<dyn crate::tabs::ExplorerTab>> {\n");
    generated_code.push_str(
        "    let mut builders: Vec<fn() -> Box<dyn crate::tabs::ExplorerTab>> = vec![];\n",
    );

    for (instantiation, feature, _) in discovered_tabs {
        if let Some(feat) = feature {
            generated_code.push_str(&format!("    #[cfg(feature = \"{}\")]\n", feat));
        }
        generated_code.push_str(&format!(
            "    builders.push(|| Box::new({}));\n",
            instantiation
        ));
    }

    generated_code.push_str("    builders\n");
    generated_code.push_str("}\n\n");
    generated_code
        .push_str("pub fn instantiate_tabs() -> Vec<Box<dyn crate::tabs::ExplorerTab>> {\n");
    generated_code.push_str("    tab_constructors().into_iter().map(|build| build()).collect()\n");
    generated_code.push_str("}\n\n");
    generated_code.push_str(
        "pub fn instantiate_tab(index: usize) -> Option<Box<dyn crate::tabs::ExplorerTab>> {\n",
    );
    generated_code.push_str("    tab_constructors().get(index).map(|build| build())\n");
    generated_code.push_str("}\n");
    generated_code
}

fn scan_local_tabs(
    tabs_dir: &Path,
    discovered_tabs: &mut Vec<(String, Option<String>, i32)>,
    local_tab_gates: &mut Vec<(String, Option<String>)>,
) {
    let mut entries: Vec<_> = fs::read_dir(tabs_dir)
        .unwrap()
        .map(|e| e.unwrap())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        let file_name = entry.file_name().into_string().unwrap();

        if file_name == "mod.rs" || file_name == "generated_tabs.rs" || file_name == "generated.rs"
        {
            continue;
        }

        let is_dir = path.is_dir();
        let is_rs_file = file_name.ends_with(".rs") && path.is_file();

        if !is_dir && !is_rs_file {
            continue;
        }

        if is_dir {
            println!(
                "cargo:rerun-if-changed={}",
                path_utils::normalize_path(&path)
            );
        }

        register_local_tab(&path, &file_name, is_dir, discovered_tabs, local_tab_gates);
    }
}

fn register_local_tab(
    path: &Path,
    file_name: &str,
    is_dir: bool,
    discovered_tabs: &mut Vec<(String, Option<String>, i32)>,
    local_tab_gates: &mut Vec<(String, Option<String>)>,
) {
    let mod_name = if is_dir {
        file_name.to_string()
    } else {
        file_name
            .strip_suffix(".rs")
            .unwrap_or(file_name)
            .to_string()
    };

    let target_file = if is_dir {
        path.join("mod.rs")
    } else {
        path.to_path_buf()
    };

    if !target_file.exists() {
        return;
    }

    let rel_path_str = path_utils::normalize_path(&target_file);
    println!("cargo:rerun-if-changed={}", rel_path_str);

    let content = fs::read_to_string(&target_file).unwrap();
    if let Some(s_name) = extract_struct_name(&content) {
        let (feature, order) = extract_metadata(&content);
        local_tab_gates.push((mod_name.clone(), feature.clone()));
        let instantiation = format!("{}::{}::default()", mod_name, s_name);
        discovered_tabs.push((instantiation, feature, order));
    }
}

fn parse_static_tab_declarations(
    tabs_mod_content: &str,
) -> std::collections::HashMap<String, Option<String>> {
    let mut declared = std::collections::HashMap::new();
    let mut pending_feature: Option<String> = None;

    for raw_line in tabs_mod_content.lines() {
        let trimmed = raw_line.split("//").next().unwrap_or("").trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("#[cfg(feature = \"") {
            if let Some(feat) = rest.strip_suffix("\")]") {
                pending_feature = Some(feat.to_string());
                continue;
            }
        }
        if trimmed.starts_with("#[") {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("pub mod ") {
            if let Some(mod_name) = rest.strip_suffix(';') {
                declared.insert(mod_name.trim().to_string(), pending_feature.take());
                continue;
            }
        }
        pending_feature = None;
    }

    declared
}

fn validate_static_tab_modules(tabs_mod_path: &Path, local_tab_gates: &[(String, Option<String>)]) {
    let content = fs::read_to_string(tabs_mod_path).unwrap_or_else(|err| {
        panic!("Failed to read {}: {err}", tabs_mod_path.display());
    });
    let declared = parse_static_tab_declarations(&content);

    for (mod_name, expected_feature) in local_tab_gates {
        let Some(actual_feature) = declared.get(mod_name) else {
            let gate_hint = expected_feature
                .as_deref()
                .map(|f| format!("#[cfg(feature = \"{f}\")]\n"))
                .unwrap_or_default();
            panic!(
                "Local ExplorerTab module `{mod_name}` is missing a static declaration in `{}`.\nAdd:\n{gate_hint}pub mod {mod_name};",
                tabs_mod_path.display()
            );
        };
        assert_eq!(
            actual_feature,
            expected_feature,
            "Feature gate mismatch for `pub mod {mod_name};` in `{}`: expected {:?}, found {:?}",
            tabs_mod_path.display(),
            expected_feature,
            actual_feature
        );
    }
}

fn scan_cargo_toml_deps(discovered_tabs: &mut Vec<(String, Option<String>, i32)>) {
    let cargo_toml = fs::read_to_string("Cargo.toml").unwrap_or_default();
    let mut in_dependencies = false;
    for raw_line in cargo_toml.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_dependencies = line[1..line.len() - 1].trim() == "dependencies";
            continue;
        }
        if !in_dependencies || !line.contains("path = ") {
            continue;
        }
        let parts: Vec<&str> = line.split('=').collect();
        if parts.len() < 2 {
            continue;
        }
        let crate_name = parts[0].trim().to_string();

        if let Some(start) = line.find("path = \"") {
            let rest = &line[start + 8..];
            if let Some(end) = rest.find('"') {
                let dep_path = &rest[..end];
                let dep_src_str = path_utils::join_and_normalize(dep_path, "src");
                let dep_src = Path::new(&dep_src_str);

                if dep_src.is_dir() {
                    println!("cargo:rerun-if-changed={}", dep_src_str);
                    scan_external_crate(dep_src, &crate_name, discovered_tabs, dep_src);
                }
            }
        }
    }
}

fn scan_external_crate(
    dir: &Path,
    crate_name: &str,
    tabs: &mut Vec<(String, Option<String>, i32)>,
    root_src: &Path,
) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current_dir) = stack.pop() {
        let Ok(read_dir) = fs::read_dir(&current_dir) else {
            continue;
        };
        let mut entries: Vec<_> = read_dir.flatten().map(|e| e.path()).collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                println!(
                    "cargo:rerun-if-changed={}",
                    path_utils::normalize_path(&path)
                );
                stack.push(path);
            } else if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("rs") {
                println!(
                    "cargo:rerun-if-changed={}",
                    path_utils::normalize_path(&path)
                );
                inspect_external_rs_file(&path, crate_name, tabs, root_src);
            }
        }
    }
}

fn inspect_external_rs_file(
    path: &Path,
    crate_name: &str,
    tabs: &mut Vec<(String, Option<String>, i32)>,
    root_src: &Path,
) {
    let Ok(content) = fs::read_to_string(path) else {
        return;
    };
    let Some(s_name) = extract_struct_name(&content) else {
        return;
    };
    let (feature, order) = extract_metadata(&content);

    let mut mod_path = String::new();
    if let Ok(rel_path) = path.strip_prefix(root_src) {
        let mut comps = Vec::new();
        for comp in rel_path.components() {
            let c = comp.as_os_str().to_string_lossy();
            if c == "mod.rs" || c == "lib.rs" || c == "main.rs" {
                continue;
            }
            if let Some(stripped) = c.strip_suffix(".rs") {
                comps.push(stripped.to_string());
            } else {
                comps.push(c.to_string());
            }
        }
        if !comps.is_empty() {
            mod_path = format!("::{}", comps.join("::"));
        }
    }

    let clean_crate_name = crate_name.replace('-', "_");
    tabs.push((
        format!("{}{}::{}", clean_crate_name, mod_path, s_name),
        feature,
        order,
    ));
}

fn extract_struct_name(content: &str) -> Option<String> {
    for line in content.lines() {
        if let Some(idx) = line.find("impl ExplorerTab for ") {
            let rest = &line[idx + "impl ExplorerTab for ".len()..];
            let end = rest
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(rest.len());
            return Some(rest[..end].to_string());
        }
        if let Some(idx) = line.find("impl crate::tabs::ExplorerTab for ") {
            let rest = &line[idx + "impl crate::tabs::ExplorerTab for ".len()..];
            let end = rest
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(rest.len());
            return Some(rest[..end].to_string());
        }
    }
    None
}

fn extract_metadata(content: &str) -> (Option<String>, i32) {
    let mut feature = None;
    let mut order = 100;

    for line in content.lines() {
        if let Some(idx) = line.find("// @explorer_feature = \"") {
            let rest = &line[idx + "// @explorer_feature = \"".len()..];
            if let Some(end) = rest.find('"') {
                feature = Some(rest[..end].to_string());
            }
        }
        if let Some(idx) = line.find("// @explorer_order = ") {
            let rest = &line[idx + "// @explorer_order = ".len()..];
            let end = rest.find(|c: char| !c.is_numeric()).unwrap_or(rest.len());
            if let Ok(val) = rest[..end].parse::<i32>() {
                order = val;
            }
        }
    }

    (feature, order)
}
