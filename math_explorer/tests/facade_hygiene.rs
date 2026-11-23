//! Integration tests verifying `math_explorer` facade guide and error hygiene.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf()
    }

    fn collect_files(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name == "target" || name == ".git" {
                continue;
            }
            if path.is_dir() {
                collect_files(&path, ext, out);
            } else if path.extension().and_then(|s| s.to_str()) == Some(ext) {
                out.push(path);
            }
        }
    }

    fn is_rust_fence(tag: &str) -> bool {
        let t = tag.trim();
        t.is_empty()
            || matches!(
                t,
                "rust" | "no_run" | "should_panic" | "compile_fail" | "ignore"
            )
            || t.starts_with("rust,")
            || t.starts_with("rust ")
    }

    fn contains_direct_internal_crate_path(line: &str) -> Option<&'static str> {
        const FORBIDDEN: &[&str] = &[
            "pure_math::",
            "domain_ai::",
            "domain_applied::",
            "domain_biology::",
            "domain_climate::",
            "domain_epidemiology::",
            "domain_physics::",
        ];
        let code = line.split("//").next().unwrap_or(line);
        for needle in FORBIDDEN {
            let mut search_from = 0;
            while let Some(rel_idx) = code[search_from..].find(needle) {
                let idx = search_from + rel_idx;
                let prev = code[..idx].chars().next_back();
                let boundary_ok =
                    prev.is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_' && c != ':');
                if boundary_ok {
                    return Some(needle);
                }
                search_from = idx + needle.len();
            }
        }
        None
    }

    #[test]
    fn active_guides_have_zero_direct_internal_crate_imports() {
        let root = workspace_root();
        let mut guides = vec![root.join("README.md")];
        collect_files(&root.join("math_explorer"), "md", &mut guides);
        for krate in [
            "pure_math",
            "domain_ai",
            "domain_applied",
            "domain_biology",
            "domain_climate",
            "domain_epidemiology",
            "domain_physics",
        ] {
            collect_files(&root.join("crates").join(krate), "md", &mut guides);
        }
        let mut docs = Vec::new();
        collect_files(&root.join("docs"), "md", &mut docs);
        let adr_dir = root.join("docs/adr");
        guides.extend(docs.into_iter().filter(|p| !p.starts_with(&adr_dir)));

        let mut violations = Vec::new();
        for guide in guides {
            let content = fs::read_to_string(&guide).expect("readable markdown guide");
            let mut in_fence = false;
            let mut rust_fence = false;
            for (idx, line) in content.lines().enumerate() {
                let trimmed = line.trim();
                if let Some(after) = trimmed.strip_prefix("```") {
                    if !in_fence {
                        in_fence = true;
                        rust_fence = is_rust_fence(after);
                    } else {
                        in_fence = false;
                        rust_fence = false;
                    }
                    continue;
                }
                if !in_fence
                    || !rust_fence
                    || trimmed.starts_with("# ")
                    || trimmed == "#"
                    || trimmed.starts_with("//")
                {
                    continue;
                }
                if let Some(bad) = contains_direct_internal_crate_path(trimmed) {
                    violations.push(format!("{}:{}: {bad}", guide.display(), idx + 1));
                }
            }
        }
        assert!(
            violations.is_empty(),
            "Active guide hygiene violations: {violations:?}"
        );
    }

    #[test]
    fn math_explorer_error_module_reexports_all_twenty_eight_crate_error_enums() {
        let root = workspace_root();
        let facade_error_src =
            fs::read_to_string(root.join("math_explorer/src/error.rs")).expect("error.rs");
        let mut crate_enums = BTreeSet::new();
        for rel in [
            "crates/math_commons/src/error.rs",
            "crates/pure_math/src/error.rs",
            "crates/domain_ai/src/error.rs",
            "crates/domain_applied/src/error.rs",
            "crates/domain_biology/src/error.rs",
            "crates/domain_climate/src/error.rs",
            "crates/domain_epidemiology/src/error.rs",
            "crates/domain_physics/src/error.rs",
        ] {
            let src = fs::read_to_string(root.join(rel)).expect("crate error.rs");
            for line in src.lines() {
                if let Some(rest) = line.trim().strip_prefix("pub enum ") {
                    let name: String = rest
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    if !name.is_empty() {
                        crate_enums.insert(name);
                    }
                }
            }
        }
        assert_eq!(crate_enums.len(), 28);
        for enum_name in &crate_enums {
            assert!(
                facade_error_src.contains(enum_name),
                "math_explorer/src/error.rs is missing re-export for `{enum_name}`"
            );
        }
    }
}
