//! Deterministic build-time generator for `traceability_manifest.json`.
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "build_snapshot_scan.rs"]
mod build_snapshot_scan;

use build_snapshot_scan::{
    discover_workspace_active_files, emit_rerun, extract_citations, extract_macro_modules,
};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const MANIFEST_FILENAME: &str = "traceability_manifest.json";

/// Aggregated build-time traceability manifest written for WebAssembly static hosting.
#[derive(Default)]
struct ManifestBuilder {
    scanned_files: usize,
    paper_coverage: BTreeMap<String, Vec<String>>,
    orphaned_papers: Vec<String>,
    unlinked_code: Vec<String>,
    invalid_links: Vec<(String, String)>,
    semantic_integrity_status: BTreeMap<String, String>,
    certificate_raw: Option<String>,
}

/// Generates `traceability_manifest.json` into `out_dir` and Cargo's target directory.
pub fn generate_traceability_snapshot(out_dir: &Path) {
    emit_rerun(Path::new("build_snapshot.rs"));
    emit_rerun(Path::new("build_snapshot_scan.rs"));
    let ws_root = Path::new("..");
    let registry_path = ws_root.join("traceability.toml");
    let cert_path = ws_root.join("Verification_Certificate.json");
    let papers_dir = ws_root.join("papers");

    emit_if_exists(&registry_path);
    emit_if_exists(&cert_path);
    emit_if_exists(&papers_dir);

    let valid_papers = collect_valid_papers(&papers_dir);
    let registry_links = parse_registry_links(&registry_path);
    let registered_modules: BTreeSet<String> = registry_links.keys().cloned().collect();

    let mut builder = ManifestBuilder {
        certificate_raw: read_optional_certificate_object(&cert_path),
        ..ManifestBuilder::default()
    };
    for paper in &valid_papers {
        builder.paper_coverage.insert(paper.clone(), Vec::new());
    }
    link_registry_entries(&valid_papers, &registry_links, &mut builder);

    let active_files = discover_workspace_active_files(ws_root);
    for (disk_path, repo_rel_path) in active_files {
        process_source_file(
            &disk_path,
            &repo_rel_path,
            &registered_modules,
            &mut builder,
        );
    }

    finalize_manifest(&valid_papers, &registry_links, &mut builder);
    let json = serialize_manifest_json(&builder);
    write_manifest_outputs(out_dir, &json);
}

fn emit_if_exists(path: &Path) {
    if path.exists() {
        emit_rerun(path);
    }
}

fn read_optional_certificate_object(cert_path: &Path) -> Option<String> {
    let raw = fs::read_to_string(cert_path).ok()?;
    let trimmed = raw.trim();
    (trimmed.starts_with('{') && trimmed.ends_with('}') && trimmed.contains("\"passed\""))
        .then(|| trimmed.to_string())
}

fn collect_valid_papers(papers_dir: &Path) -> BTreeSet<String> {
    let mut papers = BTreeSet::new();
    let Ok(entries) = fs::read_dir(papers_dir) else {
        return papers;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("tex") {
            emit_rerun(&path);
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                papers.insert(name.to_string());
            }
        }
    }
    papers
}

fn parse_registry_links(registry_path: &Path) -> BTreeMap<String, String> {
    let mut links = BTreeMap::new();
    let Ok(content) = fs::read_to_string(registry_path) else {
        return links;
    };
    let mut in_links = false;
    for raw_line in content.lines() {
        let line = strip_toml_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_links = line[1..line.len() - 1].trim() == "links";
            continue;
        }
        if !in_links {
            continue;
        }
        if let Some((raw_key, raw_val)) = line.split_once('=') {
            let key = raw_key.trim().trim_matches('"');
            let val = raw_val.trim();
            if let Some(unquoted) = val.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                links.insert(key.to_string(), unquoted.to_string());
            } else if !key.is_empty() {
                links.insert(key.to_string(), "INVALID_FORMAT".to_string());
            }
        }
    }
    links
}

fn strip_toml_comment(line: &str) -> &str {
    let mut in_quotes = false;
    let mut escaped = false;
    for (idx, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_quotes {
            escaped = true;
        } else if ch == '"' {
            in_quotes = !in_quotes;
        } else if ch == '#' && !in_quotes {
            return &line[..idx];
        }
    }
    line
}

fn link_registry_entries(
    valid_papers: &BTreeSet<String>,
    registry_links: &BTreeMap<String, String>,
    builder: &mut ManifestBuilder,
) {
    for (module_name, paper_name) in registry_links {
        if paper_name == "INVALID_FORMAT" {
            builder
                .invalid_links
                .push((module_name.clone(), paper_name.clone()));
        } else if valid_papers.contains(paper_name)
            || paper_name.starts_with("spec:")
            || paper_name.starts_with("registry:")
        {
            builder
                .paper_coverage
                .entry(paper_name.clone())
                .or_default()
                .push(module_name.clone());
        } else {
            builder
                .invalid_links
                .push((module_name.clone(), paper_name.clone()));
        }
    }
}

fn process_source_file(
    disk_path: &Path,
    repo_rel_path: &str,
    registered_modules: &BTreeSet<String>,
    builder: &mut ManifestBuilder,
) {
    builder.scanned_files += 1;
    let Ok(content) = fs::read_to_string(disk_path) else {
        return;
    };

    let is_module = repo_rel_path.ends_with("mod.rs")
        || repo_rel_path.contains("/tabs/")
        || repo_rel_path.ends_with("lib.rs");

    let mut verified_modules = Vec::new();
    for cite in extract_citations(&content) {
        if registered_modules.contains(&cite) {
            verified_modules.push(cite);
        } else {
            builder
                .invalid_links
                .push((repo_rel_path.to_string(), cite));
        }
    }
    verified_modules.extend(extract_macro_modules(&content));

    for module in &verified_modules {
        builder
            .semantic_integrity_status
            .insert(module.clone(), "Verified".to_string());
        if is_module && !registered_modules.contains(module) {
            builder.unlinked_code.push(repo_rel_path.to_string());
        }
    }
}

fn finalize_manifest(
    valid_papers: &BTreeSet<String>,
    registry_links: &BTreeMap<String, String>,
    builder: &mut ManifestBuilder,
) {
    for paper in valid_papers {
        let stem = paper.strip_suffix(".tex").unwrap_or(paper);
        if builder.semantic_integrity_status.contains_key(stem) {
            continue;
        }
        let verified = registry_links.iter().any(|(module, target)| {
            let clean = target.strip_suffix(".tex").unwrap_or(target);
            (target == paper || clean == stem)
                && builder
                    .semantic_integrity_status
                    .get(module)
                    .is_some_and(|s| s == "Verified")
        });
        builder.semantic_integrity_status.insert(
            stem.to_string(),
            if verified { "Verified" } else { "Unverified" }.to_string(),
        );
    }

    for (paper, linked) in &mut builder.paper_coverage {
        linked.sort();
        linked.dedup();
        if linked.is_empty() {
            builder.orphaned_papers.push(paper.clone());
        }
    }
    builder.orphaned_papers.sort();
    builder.orphaned_papers.dedup();
    builder.unlinked_code.sort();
    builder.unlinked_code.dedup();
    builder.invalid_links.sort();
    builder.invalid_links.dedup();
}

fn json_escape(raw: &str) -> String {
    let mut escaped = String::with_capacity(raw.len() + 2);
    escaped.push('"');
    for ch in raw.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if c.is_control() => escaped.push_str(&format!("\\u{:04x}", c as u32)),
            c => escaped.push(c),
        }
    }
    escaped.push('"');
    escaped
}

fn serialize_string_array(items: &[String]) -> String {
    let inner: Vec<String> = items.iter().map(|s| json_escape(s)).collect();
    format!("[{}]", inner.join(","))
}

fn serialize_manifest_json(builder: &ManifestBuilder) -> String {
    let coverage_entries: Vec<String> = builder
        .paper_coverage
        .iter()
        .map(|(paper, modules)| {
            format!("{}:{}", json_escape(paper), serialize_string_array(modules))
        })
        .collect();
    let invalid_entries: Vec<String> = builder
        .invalid_links
        .iter()
        .map(|(f, p)| format!("[{},{}]", json_escape(f), json_escape(p)))
        .collect();
    let semantic_entries: Vec<String> = builder
        .semantic_integrity_status
        .iter()
        .map(|(k, v)| format!("{}:{}", json_escape(k), json_escape(v)))
        .collect();
    let cert_suffix = builder
        .certificate_raw
        .as_deref()
        .map_or_else(String::new, |raw| format!(",\"certificate\":{raw}"));

    format!(
        "{{\"scanned_files\":{},\"invalid_links\":[{}],\"orphaned_papers\":{},\"unlinked_code\":{},\"unverified_modules\":[],\"paper_coverage\":{{{}}},\"total_funcs\":0,\"total_asserts\":0,\"verified_funcs\":0,\"verified_asserts\":0,\"invalid_tiers\":[],\"vacuous_bypasses\":[],\"semantic_integrity_status\":{{{}}}{}}}\n",
        builder.scanned_files,
        invalid_entries.join(","),
        serialize_string_array(&builder.orphaned_papers),
        serialize_string_array(&builder.unlinked_code),
        coverage_entries.join(","),
        semantic_entries.join(","),
        cert_suffix
    )
}

fn write_if_changed(path: &Path, content: &str) -> std::io::Result<()> {
    if fs::read(path).is_ok_and(|existing| existing == content.as_bytes()) {
        return Ok(());
    }
    fs::write(path, content)
}

fn write_manifest_outputs(out_dir: &Path, json: &str) {
    write_if_changed(&out_dir.join(MANIFEST_FILENAME), json)
        .expect("Failed to write OUT_DIR/traceability_manifest.json");

    if let Some(profile_dir) = out_dir
        .ancestors()
        .find(|p| p.file_name().and_then(|n| n.to_str()) == Some("build"))
        .and_then(Path::parent)
    {
        let _ = write_if_changed(&profile_dir.join(MANIFEST_FILENAME), json);
        if let Some(parent_target) = profile_dir.parent() {
            let _ = write_if_changed(&parent_target.join(MANIFEST_FILENAME), json);
        }
    }

    if let Some(custom_target) = env::var_os("CARGO_TARGET_DIR") {
        let target_path = PathBuf::from(custom_target);
        if target_path.is_dir() {
            let _ = write_if_changed(&target_path.join(MANIFEST_FILENAME), json);
        }
    } else if Path::new("../target").is_dir() {
        let _ = write_if_changed(&Path::new("../target").join(MANIFEST_FILENAME), json);
    }
}
