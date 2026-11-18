//! Build script for dynamic tab discovery in math_explorer_gui.
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("generated_tabs.rs");

    // Tell cargo to rerun if the tabs directory changes
    println!("cargo:rerun-if-changed=src/tabs");
    println!("cargo:rerun-if-changed=Cargo.toml");

    let mut discovered_tabs = Vec::new();
    let mut generated_mods = String::new();

    // 1. Scan local tabs in src/tabs
    let tabs_dir = Path::new("src/tabs");
    if tabs_dir.exists() {
        scan_local_tabs(tabs_dir, &mut discovered_tabs, &mut generated_mods);
    }

    // 2. Scan external path dependencies for tabs
    scan_cargo_toml_deps(&mut discovered_tabs);

    // Sort by order, then by instantiation string
    discovered_tabs.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));

    let mut generated_code = String::new();
    generated_code.push_str(&generated_mods);

    generated_code.push_str("\n#[allow(clippy::vec_init_then_push)]\n");
    generated_code
        .push_str("fn tab_constructors() -> Vec<fn() -> Box<dyn crate::tabs::ExplorerTab>> {\n");
    generated_code.push_str(
        "    let mut builders: Vec<fn() -> Box<dyn crate::tabs::ExplorerTab>> = vec![];\n",
    );

    for (instantiation, feature, _) in &discovered_tabs {
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

    fs::write(&dest_path, generated_code).unwrap();
}

fn scan_local_tabs(
    tabs_dir: &Path,
    discovered_tabs: &mut Vec<(String, Option<String>, i32)>,
    generated_mods: &mut String,
) {
    for entry in fs::read_dir(tabs_dir).unwrap() {
        let entry = entry.unwrap();
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

        let mod_name = if is_dir {
            file_name.clone()
        } else if let Some(stripped) = file_name.strip_suffix(".rs") {
            stripped.to_string()
        } else {
            file_name.clone()
        };

        let target_file = if is_dir {
            path.join("mod.rs")
        } else {
            path.clone()
        };

        if !target_file.exists() {
            continue;
        }

        let content = fs::read_to_string(&target_file).unwrap();

        if let Some(s_name) = extract_struct_name(&content) {
            let (feature, order) = extract_metadata(&content);

            // Add to generated mods
            if let Some(feat) = &feature {
                generated_mods.push_str(&format!("#[cfg(feature = \"{}\")]\n", feat));
            }
            let rel_path_str = target_file.to_string_lossy().replace('\\', "/");
            generated_mods.push_str(&format!(
                "pub mod {} {{\n    include!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/\", \"{}\"));\n}}\n",
                mod_name, rel_path_str
            ));

            let instantiation = format!("{}::{}::default()", mod_name, s_name);
            discovered_tabs.push((instantiation, feature, order));
        }
    }
}

fn scan_cargo_toml_deps(discovered_tabs: &mut Vec<(String, Option<String>, i32)>) {
    let cargo_toml = fs::read_to_string("Cargo.toml").unwrap_or_default();
    for line in cargo_toml.lines() {
        if line.contains("path = ") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() < 2 {
                continue;
            }
            let crate_name = parts[0].trim().to_string();

            if let Some(start) = line.find("path = \"") {
                let rest = &line[start + 8..];
                if let Some(end) = rest.find('"') {
                    let dep_path = &rest[..end];
                    let abs_dep_path = env::current_dir().unwrap().join(dep_path).join("src");

                    if abs_dep_path.exists() {
                        scan_external_crate(
                            &abs_dep_path,
                            &crate_name,
                            discovered_tabs,
                            &abs_dep_path,
                        );
                    }
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
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                scan_external_crate(&path, crate_name, tabs, root_src);
            } else if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("rs") {
                if let Ok(content) = fs::read_to_string(&path) {
                    if let Some(s_name) = extract_struct_name(&content) {
                        let (feature, order) = extract_metadata(&content);

                        // Construct module path
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

                        let clean_crate_name = crate_name.replace("-", "_");
                        let full_path = format!("{}{}{}", clean_crate_name, mod_path, "::");

                        tabs.push((format!("{}{}", full_path, s_name), feature, order));
                    }
                }
            }
        }
    }
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
