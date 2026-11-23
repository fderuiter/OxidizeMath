use super::WorkspaceCrate;
use std::collections::{BTreeSet, VecDeque};
use std::fs;
use std::path::Path;

const WEB_ROOT_PACKAGE: &str = "math_explorer_gui";

#[derive(Clone, Copy, PartialEq, Eq)]
enum DepSection {
    None,
    Table(DepKind),
    Dotted(DepKind),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DepKind {
    Normal,
    Dev,
}

pub(crate) fn is_root_config_escalation(path: &str) -> bool {
    matches!(
        path,
        "Cargo.toml"
            | "Cargo.lock"
            | "rust-toolchain.toml"
            | "rust-toolchain"
            | ".clippy.toml"
            | "clippy.toml"
            | ".rustfmt.toml"
            | "rustfmt.toml"
            | "verification_whitelist.toml"
    ) || path.starts_with(".cargo/")
}

pub(crate) fn collect_quoted_strings(line: &str, out: &mut Vec<String>) {
    let mut rest = line;
    while let Some(start) = rest.find('"') {
        rest = &rest[start + 1..];
        if let Some(end) = rest.find('"') {
            let val = rest[..end].trim();
            if !val.is_empty() {
                out.push(val.to_string());
            }
            rest = &rest[end + 1..];
        } else {
            break;
        }
    }
}

pub(crate) fn parse_workspace_members(root_manifest: &str) -> Vec<String> {
    let mut in_workspace = false;
    let mut in_members = false;
    let mut members = Vec::new();

    for raw_line in root_manifest.lines() {
        let line = strip_toml_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && !in_members {
            in_workspace = line == "[workspace]";
            continue;
        }
        if in_workspace && !in_members && line.starts_with("members") {
            in_members = true;
            collect_quoted_strings(line, &mut members);
            if line.contains(']') {
                break;
            }
            continue;
        }
        if in_members {
            collect_quoted_strings(line, &mut members);
            if line.contains(']') {
                break;
            }
        }
    }
    members
}

pub(crate) fn parse_package_fields(manifest: &str, fallback_name: &str) -> (String, String) {
    let mut in_package = false;
    let mut name = fallback_name.to_string();
    let mut edition = "2024".to_string();

    for raw_line in manifest.lines() {
        let line = strip_toml_comment(raw_line).trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
            continue;
        }
        if !in_package {
            continue;
        }
        if let Some(rest) = line.strip_prefix("name") {
            let mut vals = Vec::new();
            collect_quoted_strings(rest, &mut vals);
            if let Some(first) = vals.into_iter().next() {
                name = first;
            }
        } else if let Some(rest) = line.strip_prefix("edition") {
            let mut vals = Vec::new();
            collect_quoted_strings(rest, &mut vals);
            if let Some(first) = vals.into_iter().next() {
                edition = first;
            }
        }
    }
    (name, edition)
}

fn strip_toml_comment(line: &str) -> &str {
    let mut in_quotes = false;
    let mut in_single = false;
    let mut escaped = false;
    for (idx, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_quotes {
            escaped = true;
        } else if ch == '"' && !in_single {
            in_quotes = !in_quotes;
        } else if ch == '\'' && !in_quotes {
            in_single = !in_single;
        } else if ch == '#' && !in_quotes && !in_single {
            return &line[..idx];
        }
    }
    line
}

fn classify_section_header(inner: &str) -> (DepSection, Option<String>) {
    let trimmed = inner.trim();
    if trimmed == "dependencies" || trimmed == "build-dependencies" {
        return (DepSection::Table(DepKind::Normal), None);
    }
    if trimmed == "dev-dependencies" {
        return (DepSection::Table(DepKind::Dev), None);
    }
    if trimmed.ends_with(".dependencies") || trimmed.ends_with(".build-dependencies") {
        return (DepSection::Table(DepKind::Normal), None);
    }
    if trimmed.ends_with(".dev-dependencies") {
        return (DepSection::Table(DepKind::Dev), None);
    }
    for (marker, kind) in [
        ("dev-dependencies.", DepKind::Dev),
        ("build-dependencies.", DepKind::Normal),
        ("dependencies.", DepKind::Normal),
    ] {
        if let Some(rest) = trimmed
            .strip_prefix(marker)
            .or_else(|| trimmed.rsplit_once(&format!(".{marker}")).map(|(_, r)| r))
        {
            let pkg = rest.trim().trim_matches('"').trim_matches('\'');
            if !pkg.is_empty() && !pkg.contains('.') {
                return (DepSection::Dotted(kind), Some(pkg.to_string()));
            }
        }
    }
    (DepSection::None, None)
}

fn extract_inline_package_override(rhs: &str) -> Option<String> {
    for part in rhs.split([',', '{', '}']) {
        let trimmed = part.trim();
        if let Some(rest) = trimmed.strip_prefix("package")
            && rest.trim_start().starts_with('=')
        {
            let mut vals = Vec::new();
            collect_quoted_strings(rest, &mut vals);
            if let Some(pkg) = vals.into_iter().next() {
                return Some(pkg);
            }
        }
    }
    None
}

fn update_nesting_depth(line: &str, depth: &mut i32) {
    let mut in_quotes = false;
    let mut escaped = false;
    for ch in line.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_quotes {
            escaped = true;
        } else if ch == '"' {
            in_quotes = !in_quotes;
        } else if !in_quotes {
            if ch == '{' || ch == '[' {
                *depth += 1;
            } else if ch == '}' || ch == ']' {
                *depth = (*depth - 1).max(0);
            }
        }
    }
}

fn record_dep(
    pkg: String,
    kind: DepKind,
    self_pkg: &str,
    known_packages: &BTreeSet<String>,
    normal: &mut BTreeSet<String>,
    dev: &mut BTreeSet<String>,
) {
    if pkg != self_pkg && known_packages.contains(&pkg) {
        match kind {
            DepKind::Normal => {
                normal.insert(pkg);
            }
            DepKind::Dev => {
                dev.insert(pkg);
            }
        }
    }
}

fn parse_workspace_deps(
    manifest: &str,
    self_pkg: &str,
    known_packages: &BTreeSet<String>,
) -> (Vec<String>, Vec<String>) {
    let mut section = DepSection::None;
    let mut dotted_candidate: Option<(String, DepKind)> = None;
    let mut nesting = 0i32;
    let mut normal = BTreeSet::new();
    let mut dev = BTreeSet::new();

    for raw_line in manifest.lines() {
        let line = strip_toml_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        if nesting == 0 && line.starts_with('[') && line.ends_with(']') && !line.starts_with("[[") {
            if let Some((cand, kind)) = dotted_candidate.take() {
                record_dep(cand, kind, self_pkg, known_packages, &mut normal, &mut dev);
            }
            let (next_sec, dotted_pkg) = classify_section_header(&line[1..line.len() - 1]);
            section = next_sec;
            if let (DepSection::Dotted(kind), Some(pkg)) = (next_sec, dotted_pkg) {
                dotted_candidate = Some((pkg, kind));
            }
            continue;
        }
        if nesting == 0
            && line.starts_with("[[")
            && let Some((cand, kind)) = dotted_candidate.take()
        {
            record_dep(cand, kind, self_pkg, known_packages, &mut normal, &mut dev);
            section = DepSection::None;
        }
        if nesting == 0
            && let Some((lhs, rhs)) = line.split_once('=')
        {
            let key = lhs.trim().trim_matches('"').trim_matches('\'');
            match section {
                DepSection::Table(kind) => {
                    let pkg =
                        extract_inline_package_override(rhs).unwrap_or_else(|| key.to_string());
                    record_dep(pkg, kind, self_pkg, known_packages, &mut normal, &mut dev);
                }
                DepSection::Dotted(_) if key == "package" => {
                    let mut vals = Vec::new();
                    collect_quoted_strings(rhs, &mut vals);
                    if let (Some(override_name), Some((cand, _))) =
                        (vals.into_iter().next(), dotted_candidate.as_mut())
                    {
                        *cand = override_name;
                    }
                }
                _ => {}
            }
        }
        update_nesting_depth(line, &mut nesting);
    }
    if let Some((cand, kind)) = dotted_candidate.take() {
        record_dep(cand, kind, self_pkg, known_packages, &mut normal, &mut dev);
    }
    (normal.into_iter().collect(), dev.into_iter().collect())
}

fn compute_web_package_set(crates: &[WorkspaceCrate]) -> BTreeSet<String> {
    let mut web = BTreeSet::new();
    let Some(gui) = crates.iter().find(|c| c.package_name == WEB_ROOT_PACKAGE) else {
        return web;
    };
    let mut queue = VecDeque::new();
    web.insert(gui.package_name.clone());
    for dep in gui.normal_deps.iter().chain(gui.dev_deps.iter()) {
        if web.insert(dep.clone()) {
            queue.push_back(dep.clone());
        }
    }
    while let Some(pkg) = queue.pop_front() {
        if let Some(krate) = crates.iter().find(|c| c.package_name == pkg) {
            for dep in &krate.normal_deps {
                if web.insert(dep.clone()) {
                    queue.push_back(dep.clone());
                }
            }
        }
    }
    web
}

pub(crate) fn load_workspace_crates(root: &Path) -> Vec<WorkspaceCrate> {
    let Ok(root_toml) = fs::read_to_string(root.join("Cargo.toml")) else {
        return Vec::new();
    };
    let dirs = parse_workspace_members(&root_toml);
    let mut manifests = Vec::with_capacity(dirs.len());
    let mut known_packages = BTreeSet::new();

    for dir in dirs {
        let member_root = root.join(&dir);
        let fallback = dir.rsplit('/').next().unwrap_or(&dir);
        let manifest = fs::read_to_string(member_root.join("Cargo.toml")).unwrap_or_default();
        let (package_name, edition) = parse_package_fields(&manifest, fallback);
        known_packages.insert(package_name.clone());
        manifests.push((dir, member_root, manifest, package_name, edition));
    }

    let mut crates = Vec::with_capacity(manifests.len());
    for (dir, member_root, manifest, package_name, edition) in manifests {
        let has_lib = member_root.join("src/lib.rs").is_file()
            || manifest.lines().any(|l| l.trim() == "[lib]");
        let has_bins = member_root.join("src/main.rs").is_file()
            || member_root.join("src/bin").is_dir()
            || manifest.lines().any(|l| l.trim() == "[[bin]]");
        let (normal_deps, dev_deps) =
            parse_workspace_deps(&manifest, &package_name, &known_packages);
        crates.push(WorkspaceCrate {
            dir,
            package_name,
            edition,
            has_lib,
            has_bins,
            is_web: false,
            normal_deps,
            dev_deps,
        });
    }

    let web_set = compute_web_package_set(&crates);
    for krate in &mut crates {
        krate.is_web = web_set.contains(&krate.package_name);
    }
    crates
}

pub(crate) fn expand_reverse_dependents<'a>(
    direct: &[&'a WorkspaceCrate],
    crates: &'a [WorkspaceCrate],
) -> Vec<&'a WorkspaceCrate> {
    let mut normal_closure: BTreeSet<&str> = BTreeSet::new();
    let mut queue: VecDeque<&str> = VecDeque::new();
    for krate in direct {
        if normal_closure.insert(krate.package_name.as_str()) {
            queue.push_back(krate.package_name.as_str());
        }
    }
    while let Some(target_pkg) = queue.pop_front() {
        for candidate in crates {
            if candidate.normal_deps.iter().any(|d| d == target_pkg)
                && normal_closure.insert(candidate.package_name.as_str())
            {
                queue.push_back(candidate.package_name.as_str());
            }
        }
    }

    let mut affected = normal_closure.clone();
    for candidate in crates {
        if candidate
            .dev_deps
            .iter()
            .any(|d| normal_closure.contains(d.as_str()))
        {
            affected.insert(candidate.package_name.as_str());
        }
    }

    crates
        .iter()
        .filter(|c| affected.contains(c.package_name.as_str()))
        .collect()
}
