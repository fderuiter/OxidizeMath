//! Compile-time traceability manifest generator for `math_explorer_gui`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

struct CertificateMetrics {
    native_execution_coverage_pct: f64,
    wasm_path_coverage_pct: f64,
    total_assertion_density: f64,
    verified_modules_density: f64,
    unverified_modules_density: f64,
    passed: bool,
}

type ManifestResolution = (
    Vec<(String, Vec<String>)>,
    Vec<String>,
    Vec<(String, String)>,
);

/// Generates `$OUT_DIR/traceability_manifest.json` from `../traceability.toml`,
/// `../papers/*.tex`, and `../Verification_Certificate.json`.
pub fn generate_traceability_manifest(out_dir: &Path) {
    assert!(!out_dir.as_os_str().is_empty(), "OUT_DIR must not be empty");
    println!("cargo:rerun-if-changed=build_manifest.rs");
    println!("cargo:rerun-if-changed=../traceability.toml");
    println!("cargo:rerun-if-changed=../papers");

    let cert_path = Path::new("../Verification_Certificate.json");
    if cert_path.exists() {
        println!("cargo:rerun-if-changed=../Verification_Certificate.json");
    }

    let tex_papers = collect_tex_papers(Path::new("../papers"));
    assert!(
        !tex_papers.is_empty(),
        "Expected academic .tex manuscripts in ../papers"
    );

    let registry_raw = fs::read_to_string("../traceability.toml")
        .expect("Failed to read ../traceability.toml in build.rs");
    let links = parse_traceability_links(&registry_raw);
    let (papers, orphaned_papers, invalid_links) = resolve_manifest_entries(&tex_papers, &links);

    let cert = fs::read_to_string(cert_path)
        .ok()
        .and_then(|raw| parse_verification_certificate(&raw));

    let json = render_manifest_json(&papers, &orphaned_papers, &invalid_links, cert.as_ref());
    assert!(
        !json.is_empty(),
        "Generated manifest JSON must not be empty"
    );

    let manifest_dest = out_dir.join("traceability_manifest.json");
    fs::write(&manifest_dest, json).expect("Failed to write traceability_manifest.json");
}

fn collect_tex_papers(papers_dir: &Path) -> Vec<String> {
    let entries = fs::read_dir(papers_dir).expect("Failed to read ../papers directory");
    let mut papers = BTreeSet::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("tex") {
            println!(
                "cargo:rerun-if-changed={}",
                super::path_utils::normalize_path(&path)
            );
            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                papers.insert(name.to_string());
            }
        }
    }
    papers.into_iter().collect()
}

fn strip_toml_comment(line: &str) -> &str {
    let mut in_quotes = false;
    let mut escaped = false;
    for (idx, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if in_quotes && ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            in_quotes = !in_quotes;
        } else if !in_quotes && ch == '#' {
            return &line[..idx];
        }
    }
    line
}

fn parse_traceability_links(content: &str) -> Vec<(String, String)> {
    let mut in_links = false;
    let mut pairs = Vec::new();

    for raw_line in content.lines() {
        let line = strip_toml_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        if let Some(header) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            in_links = header.trim() == "links";
            continue;
        }
        if !in_links {
            continue;
        }
        if let Some((raw_key, raw_val)) = line.split_once('=') {
            let key = raw_key.trim().trim_matches('"').trim();
            let val = raw_val
                .trim()
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'));
            if !key.is_empty() {
                if let Some(target) = val {
                    pairs.push((key.to_string(), target.to_string()));
                }
            }
        }
    }
    pairs
}

fn resolve_manifest_entries(
    tex_papers: &[String],
    links: &[(String, String)],
) -> ManifestResolution {
    let mut paper_map: BTreeMap<String, Vec<String>> = tex_papers
        .iter()
        .map(|name| (name.clone(), Vec::new()))
        .collect();
    let mut invalid_links = Vec::new();

    for (module_key, target) in links {
        if let Some(modules) = paper_map.get_mut(target) {
            modules.push(module_key.clone());
        } else if !target.starts_with("spec:") && !target.starts_with("registry:") {
            invalid_links.push((module_key.clone(), target.clone()));
        }
    }

    let mut papers = Vec::with_capacity(paper_map.len());
    let mut orphaned_papers = Vec::new();
    for (name, mut linked_code) in paper_map {
        linked_code.sort();
        linked_code.dedup();
        if linked_code.is_empty() {
            orphaned_papers.push(name);
        } else {
            papers.push((name, linked_code));
        }
    }

    invalid_links.sort();
    invalid_links.dedup();
    (papers, orphaned_papers, invalid_links)
}

fn extract_json_token<'a>(raw: &'a str, key: &str) -> Option<&'a str> {
    let pattern = format!("\"{key}\"");
    let idx = raw.find(&pattern)?;
    let after_key = &raw[idx + pattern.len()..];
    let colon = after_key.find(':')?;
    let rest = after_key[colon + 1..].trim_start();
    let end = rest
        .find([',', '}', '\n', '\r', ' ', '\t'])
        .unwrap_or(rest.len());
    Some(&rest[..end])
}

fn parse_verification_certificate(raw: &str) -> Option<CertificateMetrics> {
    Some(CertificateMetrics {
        native_execution_coverage_pct: extract_json_token(raw, "native_execution_coverage_pct")?
            .parse()
            .ok()?,
        wasm_path_coverage_pct: extract_json_token(raw, "wasm_path_coverage_pct")?
            .parse()
            .ok()?,
        total_assertion_density: extract_json_token(raw, "total_assertion_density")?
            .parse()
            .ok()?,
        verified_modules_density: extract_json_token(raw, "verified_modules_density")?
            .parse()
            .ok()?,
        unverified_modules_density: extract_json_token(raw, "unverified_modules_density")?
            .parse()
            .ok()?,
        passed: extract_json_token(raw, "passed")?.parse().ok()?,
    })
}

fn escape_json_str(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 2);
    out.push('"');
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn render_string_array(items: &[String]) -> String {
    let joined: Vec<String> = items.iter().map(|s| escape_json_str(s)).collect();
    format!("[{}]", joined.join(","))
}

fn render_manifest_json(
    papers: &[(String, Vec<String>)],
    orphaned_papers: &[String],
    invalid_links: &[(String, String)],
    cert: Option<&CertificateMetrics>,
) -> String {
    let paper_objs: Vec<String> = papers
        .iter()
        .map(|(name, linked)| {
            format!(
                "{{\"name\":{},\"linked_code\":{}}}",
                escape_json_str(name),
                render_string_array(linked)
            )
        })
        .collect();
    let invalid_objs: Vec<String> = invalid_links
        .iter()
        .map(|(code, paper)| format!("[{},{}]", escape_json_str(code), escape_json_str(paper)))
        .collect();
    let cert_json = match cert {
        Some(c) => format!(
            "{{\"native_execution_coverage_pct\":{:.6},\"wasm_path_coverage_pct\":{:.6},\"total_assertion_density\":{:.6},\"verified_modules_density\":{:.6},\"unverified_modules_density\":{:.6},\"passed\":{}}}",
            c.native_execution_coverage_pct,
            c.wasm_path_coverage_pct,
            c.total_assertion_density,
            c.verified_modules_density,
            c.unverified_modules_density,
            c.passed
        ),
        None => "null".to_string(),
    };

    format!(
        "{{\"papers\":[{}],\"orphaned_papers\":{},\"unlinked_code\":[],\"invalid_links\":[{}],\"certificate\":{}}}",
        paper_objs.join(","),
        render_string_array(orphaned_papers),
        invalid_objs.join(","),
        cert_json
    )
}
