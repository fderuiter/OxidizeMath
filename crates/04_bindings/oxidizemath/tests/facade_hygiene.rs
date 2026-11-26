//! Integration tests verifying facade guide and error re-export hygiene via ancestor root discovery.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn has_workspace_section(manifest_path: &Path) -> bool {
        let Ok(content) = fs::read_to_string(manifest_path) else {
            return false;
        };
        content
            .lines()
            .any(|raw| raw.split('#').next().unwrap_or("").trim() == "[workspace]")
    }

    fn workspace_root() -> PathBuf {
        let start = Path::new(env!("CARGO_MANIFEST_DIR"));
        for ancestor in start.ancestors() {
            if has_workspace_section(&ancestor.join("Cargo.toml")) {
                return ancestor.to_path_buf();
            }
        }
        panic!(
            "Unable to locate workspace Cargo.toml from {}",
            start.display()
        );
    }

    fn parse_workspace_members(root_toml: &str) -> Vec<String> {
        let mut in_ws = false;
        let mut body = String::new();
        for raw in root_toml.lines() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.starts_with('[') && line.ends_with(']') {
                if in_ws {
                    break;
                }
                in_ws = line == "[workspace]";
                continue;
            }
            if in_ws {
                body.push_str(line);
                body.push('\n');
            }
        }
        let Some((_, rest)) = body.split_once("members") else {
            return Vec::new();
        };
        let Some(open) = rest.find('[') else {
            return Vec::new();
        };
        let Some(close) = rest[open + 1..].find(']') else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut chars = rest[open + 1..open + 1 + close].chars();
        while let Some(ch) = chars.next() {
            if ch == '"' || ch == '\'' {
                let q = ch;
                let mut item = String::new();
                for next in chars.by_ref() {
                    if next == q {
                        out.push(item);
                        break;
                    }
                    item.push(next);
                }
            }
        }
        out
    }

    fn parse_manifest_key(manifest: &str, section: &str, key: &str) -> Option<String> {
        let target = format!("[{section}]");
        let mut in_sec = false;
        for raw in manifest.lines() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.starts_with('[') && line.ends_with(']') {
                if in_sec {
                    break;
                }
                in_sec = line == target;
                continue;
            }
            if in_sec
                && let Some((k, v)) = line.split_once('=')
                && k.trim() == key
            {
                return Some(v.trim().trim_matches('"').trim_matches('\'').to_string());
            }
        }
        None
    }

    fn discover_role_members(root: &Path) -> Vec<(PathBuf, String, String)> {
        let root_toml = fs::read_to_string(root.join("Cargo.toml")).expect("root Cargo.toml");
        parse_workspace_members(&root_toml)
            .into_iter()
            .filter_map(|m| {
                let dir = root.join(&m);
                let manifest = fs::read_to_string(dir.join("Cargo.toml")).ok()?;
                let pkg = parse_manifest_key(&manifest, "package", "name")?;
                let role = parse_manifest_key(&manifest, "package.metadata.oxidize", "facade_role")
                    .unwrap_or_else(|| "none".to_string());
                Some((dir, pkg, role))
            })
            .collect()
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

    fn contains_direct_internal_crate_path<'a>(
        line: &str,
        forbidden: &'a [String],
    ) -> Option<&'a str> {
        let code = line.split("//").next().unwrap_or(line);
        for needle in forbidden {
            let mut search_from = 0;
            while let Some(rel_idx) = code[search_from..].find(needle.as_str()) {
                let idx = search_from + rel_idx;
                let prev = code[..idx].chars().next_back();
                let boundary_ok =
                    prev.is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_' && c != ':');
                if boundary_ok {
                    return Some(needle.as_str());
                }
                search_from = idx + needle.len();
            }
        }
        None
    }

    #[test]
    fn active_guides_have_zero_direct_internal_crate_imports() {
        let root = workspace_root();
        let members = discover_role_members(&root);
        let forbidden: Vec<String> = members
            .iter()
            .filter(|(_, _, role)| role == "domain")
            .map(|(_, pkg, _)| format!("{pkg}::"))
            .collect();
        assert!(!forbidden.is_empty(), "expected domain crates in workspace");

        let mut guides = vec![root.join("README.md")];
        for (dir, pkg, role) in &members {
            if matches!(role.as_str(), "facade" | "compat_facade" | "domain")
                || pkg == "oxidize-cli"
            {
                collect_files(dir, "md", &mut guides);
            }
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
                if let Some(bad) = contains_direct_internal_crate_path(trimmed, &forbidden) {
                    violations.push(format!("{}:{}: {bad}", guide.display(), idx + 1));
                }
            }
        }
        assert!(
            violations.is_empty(),
            "Active guide hygiene violations: {violations:?}"
        );
    }

    fn collect_error_enums_in_crate(crate_dir: &Path, out: &mut BTreeSet<String>) {
        let mut candidates = vec![
            crate_dir.join("src/error.rs"),
            crate_dir.join("src/error/stochastic.rs"),
        ];
        if !candidates[0].is_file() {
            let mut rs_files = Vec::new();
            collect_files(&crate_dir.join("src"), "rs", &mut rs_files);
            candidates = rs_files;
        }
        for file in candidates {
            let Ok(src) = fs::read_to_string(&file) else {
                continue;
            };
            for line in src.lines() {
                if let Some(rest) = line.trim().strip_prefix("pub enum ") {
                    let name: String = rest
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    if name.ends_with("Error") {
                        out.insert(name);
                    }
                }
            }
        }
    }

    #[test]
    fn facade_error_module_reexports_all_engine_and_domain_error_enums() {
        let root = workspace_root();
        let members = discover_role_members(&root);
        let facade_dir = members
            .iter()
            .find(|(_, _, role)| role == "facade")
            .map(|(dir, _, _)| dir)
            .expect("primary facade crate");
        let facade_error_src =
            fs::read_to_string(facade_dir.join("src/error.rs")).expect("facade error.rs");

        let mut crate_enums = BTreeSet::new();
        for (dir, _, role) in &members {
            if matches!(role.as_str(), "engine" | "domain") {
                collect_error_enums_in_crate(dir, &mut crate_enums);
            }
        }
        assert!(
            crate_enums.len() >= 32,
            "expected engine and domain error enums, found {}",
            crate_enums.len()
        );
        for enum_name in &crate_enums {
            assert!(
                facade_error_src.contains(enum_name),
                "{}/src/error.rs is missing re-export for `{enum_name}`",
                facade_dir.display()
            );
        }
    }
}
