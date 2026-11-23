//! Build script for local dynamic tab discovery and traceability snapshot generation in `math_explorer_gui`.
mod build_snapshot;

use oxidize_core::path_utils;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::Path;

struct TabMetadata {
    feature: Option<String>,
    order: i32,
    domain: Option<String>,
    slug: Option<String>,
}

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);
    let dest_path = out_path.join("generated_tabs.rs");

    // Rerun only when build scripts, manifest, or local src/tabs modules change.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/tabs");
    println!("cargo:rerun-if-changed=Cargo.toml");

    let mut discovered_tabs = Vec::new();
    let mut seen_slugs = BTreeSet::new();
    let mut generated_mods = String::new();

    let tabs_dir = Path::new("src/tabs");
    if tabs_dir.exists() {
        scan_local_tabs(
            tabs_dir,
            &mut discovered_tabs,
            &mut seen_slugs,
            &mut generated_mods,
        );
    }

    // Sort by order, then by instantiation string.
    discovered_tabs.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));

    let generated_code = render_generated_tabs_code(&discovered_tabs, &generated_mods);
    build_snapshot::write_if_changed(&dest_path, generated_code.as_bytes())
        .expect("Failed to write OUT_DIR/generated_tabs.rs");

    // Generate deterministic precomputed traceability snapshot for static web builds.
    build_snapshot::generate_traceability_snapshot(out_path);
}

fn render_generated_tabs_code(
    discovered_tabs: &[(String, Option<String>, i32)],
    generated_mods: &str,
) -> String {
    let mut generated_code = String::from(generated_mods);

    generated_code.push_str("\n#[allow(clippy::vec_init_then_push)]\n");
    generated_code
        .push_str("fn tab_constructors() -> Vec<fn() -> Box<dyn crate::tabs::ExplorerTab>> {\n");
    generated_code.push_str(
        "    let mut builders: Vec<fn() -> Box<dyn crate::tabs::ExplorerTab>> = vec![];\n",
    );

    for (instantiation, feature, _) in discovered_tabs {
        if let Some(feat) = feature {
            generated_code.push_str(&format!("    #[cfg(feature = \"{feat}\")]\n"));
        }
        generated_code.push_str(&format!(
            "    builders.push(|| Box::new({instantiation}));\n"
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
    seen_slugs: &mut BTreeSet<String>,
    generated_mods: &mut String,
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

        register_local_tab(
            &path,
            &file_name,
            is_dir,
            discovered_tabs,
            seen_slugs,
            generated_mods,
        );
    }
}

fn register_local_tab(
    path: &Path,
    file_name: &str,
    is_dir: bool,
    discovered_tabs: &mut Vec<(String, Option<String>, i32)>,
    seen_slugs: &mut BTreeSet<String>,
    generated_mods: &mut String,
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
    println!("cargo:rerun-if-changed={rel_path_str}");

    let content = fs::read_to_string(&target_file).unwrap();
    let Some(s_name) = extract_struct_name(&content) else {
        return;
    };
    let meta = extract_metadata(&content);
    let domain = meta.domain.unwrap_or_else(|| {
        panic!("Tab module '{rel_path_str}' must declare // @explorer_domain = \"...\"")
    });
    let slug = meta.slug.unwrap_or_else(|| {
        panic!("Tab module '{rel_path_str}' must declare // @explorer_slug = \"...\"")
    });
    assert!(
        !slug.trim().is_empty() && slug != "simulation",
        "Tab module '{rel_path_str}' declared invalid @explorer_slug = \"{slug}\""
    );
    assert!(
        seen_slugs.insert(slug.clone()),
        "Duplicate @explorer_slug = \"{slug}\" in '{rel_path_str}'"
    );

    if let Some(feat) = &meta.feature {
        generated_mods.push_str(&format!("#[cfg(feature = \"{feat}\")]\n"));
    }
    generated_mods.push_str(&format!(
        "pub mod {mod_name} {{\n    const EXPLORER_DOMAIN: &str = {domain:?};\n    const EXPLORER_SLUG: &str = {slug:?};\n    include!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/\", {rel_path_str:?}));\n}}\n"
    ));

    let instantiation = format!("{mod_name}::{s_name}::default()");
    discovered_tabs.push((instantiation, meta.feature, meta.order));
}

fn extract_ident_prefix(s: &str) -> Option<String> {
    let trimmed = s.trim_start();
    let end = trimmed
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(trimmed.len());
    (end > 0).then(|| trimmed[..end].to_string())
}

fn extract_struct_name(content: &str) -> Option<String> {
    for line in content.lines() {
        let code = line.split("//").next().unwrap_or("");
        for marker in [
            "impl ExplorerTab for ",
            "impl crate::tabs::ExplorerTab for ",
        ] {
            if let Some(idx) = code.find(marker)
                && let Some(ident) = extract_ident_prefix(&code[idx + marker.len()..])
            {
                return Some(ident);
            }
        }
    }

    if let Some(macro_idx) = content.find("declare_simulation_tab!") {
        let after_macro = &content[macro_idx + "declare_simulation_tab!".len()..];
        for line in after_macro.lines() {
            let code = line.split("//").next().unwrap_or("");
            if let Some(struct_idx) = code.find("struct ")
                && let Some(ident) = extract_ident_prefix(&code[struct_idx + "struct ".len()..])
            {
                return Some(ident);
            }
        }
    }
    None
}

fn parse_quoted_directive(line: &str, prefix: &str) -> Option<String> {
    let idx = line.find(prefix)?;
    let rest = &line[idx + prefix.len()..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn extract_metadata(content: &str) -> TabMetadata {
    let mut meta = TabMetadata {
        feature: None,
        order: 100,
        domain: None,
        slug: None,
    };

    for line in content.lines() {
        if let Some(feat) = parse_quoted_directive(line, "// @explorer_feature = \"") {
            meta.feature = Some(feat);
        }
        if let Some(dom) = parse_quoted_directive(line, "// @explorer_domain = \"") {
            meta.domain = Some(dom);
        }
        if let Some(slug) = parse_quoted_directive(line, "// @explorer_slug = \"") {
            meta.slug = Some(slug);
        }
        if let Some(idx) = line.find("// @explorer_order = ") {
            let rest = &line[idx + "// @explorer_order = ".len()..];
            let end = rest.find(|c: char| !c.is_numeric()).unwrap_or(rest.len());
            if let Ok(val) = rest[..end].parse::<i32>() {
                meta.order = val;
            }
        }
    }

    meta
}
