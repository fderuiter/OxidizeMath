//! Native `Cargo.lock` supply-chain, SPDX license, duplicate-crate ceiling, and Tier 5
//! security harness gates invoked by `unified_verification check-security`.

use crate::whitelist::{
    SupplyChainConfig, WORKSPACE_WHITELIST_CANDIDATES, find_whitelist_path, load_whitelist_strict,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::process::Command;

/// Canonical first-party math crates required to maintain `VerifiedAllocator` and `tests/security_*.rs`.
pub const MATH_CRATE_DIRS: &[&str] = &[
    "crates/math_commons",
    "crates/pure_math",
    "crates/domain_ai",
    "crates/domain_applied",
    "crates/domain_biology",
    "crates/domain_climate",
    "crates/domain_epidemiology",
    "crates/domain_physics",
];

const MAX_SPDX_PARSE_DEPTH: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
enum SpdxToken {
    LParen,
    RParen,
    And,
    Or,
    Ident(String),
}

fn tokenize_spdx(expr: &str) -> Result<Vec<SpdxToken>, String> {
    let mut raw_tokens = Vec::new();
    let mut cur = String::new();
    for ch in expr.chars() {
        match ch {
            '(' | ')' | '/' => {
                if !cur.is_empty() {
                    raw_tokens.push(std::mem::take(&mut cur));
                }
                raw_tokens.push(ch.to_string());
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    raw_tokens.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        raw_tokens.push(cur);
    }

    let mut tokens = Vec::new();
    let mut idx = 0;
    while idx < raw_tokens.len() {
        match raw_tokens[idx].as_str() {
            "(" => tokens.push(SpdxToken::LParen),
            ")" => tokens.push(SpdxToken::RParen),
            "/" => tokens.push(SpdxToken::Or),
            w if w.eq_ignore_ascii_case("OR") => tokens.push(SpdxToken::Or),
            w if w.eq_ignore_ascii_case("AND") => tokens.push(SpdxToken::And),
            w if w.eq_ignore_ascii_case("WITH") => {
                let Some(SpdxToken::Ident(base)) = tokens.pop() else {
                    return Err(format!("Malformed SPDX WITH clause in `{expr}`"));
                };
                let Some(exception) = raw_tokens.get(idx + 1) else {
                    return Err(format!("Missing SPDX exception after WITH in `{expr}`"));
                };
                tokens.push(SpdxToken::Ident(format!("{base} WITH {exception}")));
                idx += 1;
            }
            ident => tokens.push(SpdxToken::Ident(ident.to_string())),
        }
        idx += 1;
    }
    if tokens.is_empty() {
        return Err("Empty SPDX license expression".to_string());
    }
    Ok(tokens)
}

fn eval_spdx_or(
    tokens: &[SpdxToken],
    pos: &mut usize,
    allowed: &BTreeSet<&str>,
    depth: usize,
) -> Result<bool, String> {
    if depth > MAX_SPDX_PARSE_DEPTH {
        return Err("SPDX expression exceeds maximum nesting depth".to_string());
    }
    let mut acc = eval_spdx_and(tokens, pos, allowed, depth)?;
    while matches!(tokens.get(*pos), Some(SpdxToken::Or)) {
        *pos += 1;
        let rhs = eval_spdx_and(tokens, pos, allowed, depth)?;
        acc = acc || rhs;
    }
    Ok(acc)
}

fn eval_spdx_and(
    tokens: &[SpdxToken],
    pos: &mut usize,
    allowed: &BTreeSet<&str>,
    depth: usize,
) -> Result<bool, String> {
    let mut acc = eval_spdx_atom(tokens, pos, allowed, depth)?;
    while matches!(tokens.get(*pos), Some(SpdxToken::And)) {
        *pos += 1;
        let rhs = eval_spdx_atom(tokens, pos, allowed, depth)?;
        acc = acc && rhs;
    }
    Ok(acc)
}

fn eval_spdx_atom(
    tokens: &[SpdxToken],
    pos: &mut usize,
    allowed: &BTreeSet<&str>,
    depth: usize,
) -> Result<bool, String> {
    match tokens.get(*pos) {
        Some(SpdxToken::LParen) => {
            *pos += 1;
            let inner = eval_spdx_or(tokens, pos, allowed, depth + 1)?;
            if !matches!(tokens.get(*pos), Some(SpdxToken::RParen)) {
                return Err("Unclosed '(' in SPDX expression".to_string());
            }
            *pos += 1;
            Ok(inner)
        }
        Some(SpdxToken::Ident(id)) => {
            *pos += 1;
            Ok(allowed.contains(id.as_str()))
        }
        other => Err(format!("Unexpected SPDX token: {other:?}")),
    }
}

/// Evaluates whether a compound SPDX license expression is satisfied by `allowed_licenses`.
pub fn evaluate_spdx_expression(expr: &str, allowed_licenses: &[String]) -> Result<bool, String> {
    let tokens = tokenize_spdx(expr)?;
    let allowed: BTreeSet<&str> = allowed_licenses.iter().map(String::as_str).collect();
    let mut pos = 0;
    let ok = eval_spdx_or(&tokens, &mut pos, &allowed, 0)?;
    if pos != tokens.len() {
        return Err(format!("Trailing tokens in SPDX expression `{expr}`"));
    }
    Ok(ok)
}

/// Audits `Cargo.lock` for allowed package registries and the duplicate crate family ceiling.
pub fn audit_cargo_lock(lock_toml: &str, cfg: &SupplyChainConfig) -> Vec<String> {
    let mut violations = Vec::new();
    let parsed: toml::Value = match toml::from_str(lock_toml) {
        Ok(v) => v,
        Err(err) => return vec![format!("[!] Failed to parse Cargo.lock: {err}")],
    };
    let Some(packages) = parsed.get("package").and_then(toml::Value::as_array) else {
        return vec!["[!] Cargo.lock missing [[package]] array".to_string()];
    };
    let allowed_regs: BTreeSet<&str> = cfg.allowed_registries.iter().map(String::as_str).collect();
    let mut versions_by_name: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for pkg in packages {
        let name = pkg
            .get("name")
            .and_then(toml::Value::as_str)
            .unwrap_or("<unknown>");
        let version = pkg
            .get("version")
            .and_then(toml::Value::as_str)
            .unwrap_or("<unknown>");
        versions_by_name
            .entry(name.to_string())
            .or_default()
            .insert(version.to_string());

        if let Some(src) = pkg.get("source").and_then(toml::Value::as_str)
            && !allowed_regs.contains(src)
        {
            violations.push(format!(
                "[!] Unauthorized registry/git source for `{name}` {version}: `{src}`"
            ));
        }
    }

    let duplicate_families = versions_by_name.values().filter(|v| v.len() > 1).count();
    let Some(max_dups) = cfg.max_duplicate_families else {
        violations.push(
            "[!] [supply_chain] in verification_whitelist.toml must set max_duplicate_families"
                .to_string(),
        );
        return violations;
    };
    if duplicate_families > max_dups {
        violations.push(format!(
            "[!] Cargo.lock duplicate crate families ({duplicate_families}) exceed ceiling ({max_dups})"
        ));
    }
    violations
}

/// Audits `cargo metadata` JSON for third-party SPDX license compliance.
pub fn audit_metadata_licenses(
    metadata_json: &serde_json::Value,
    cfg: &SupplyChainConfig,
) -> Vec<String> {
    let mut violations = Vec::new();
    let ws_members: BTreeSet<&str> = metadata_json
        .get("workspace_members")
        .and_then(serde_json::Value::as_array)
        .map(|arr| arr.iter().filter_map(serde_json::Value::as_str).collect())
        .unwrap_or_default();

    let Some(packages) = metadata_json
        .get("packages")
        .and_then(serde_json::Value::as_array)
    else {
        return vec!["[!] cargo metadata JSON missing `packages` array".to_string()];
    };

    for pkg in packages {
        let id = pkg
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if ws_members.contains(id) {
            continue;
        }
        let name = pkg
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        let version = pkg
            .get("version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        let Some(license) = pkg.get("license").and_then(serde_json::Value::as_str) else {
            violations.push(format!(
                "[!] Third-party crate `{name}` {version} is missing an SPDX `license` field"
            ));
            continue;
        };
        match evaluate_spdx_expression(license, &cfg.allowed_licenses) {
            Ok(true) => {}
            Ok(false) => violations.push(format!(
                "[!] Disallowed SPDX license on `{name}` {version}: `{license}`"
            )),
            Err(err) => violations.push(format!(
                "[!] Invalid SPDX license on `{name}` {version} (`{license}`): {err}"
            )),
        }
    }
    violations
}

fn file_declares_verified_allocator_and_test(path: &Path, require_test_fn: bool) -> (bool, bool) {
    let Ok(src) = fs::read_to_string(path) else {
        return (false, false);
    };
    let Ok(ast) = syn::parse_file(&src) else {
        return (false, false);
    };
    let has_alloc = ast.items.iter().any(|item| match item {
        syn::Item::Static(st) => {
            let is_global = st
                .attrs
                .iter()
                .any(|a| a.path().is_ident("global_allocator"));
            let ty_str = quote::ToTokens::to_token_stream(&st.ty).to_string();
            is_global && ty_str.contains("VerifiedAllocator")
        }
        _ => false,
    });
    if !require_test_fn {
        return (has_alloc, true);
    }
    let has_test = ast.items.iter().any(|item| match item {
        syn::Item::Fn(func) => func.attrs.iter().any(|a| {
            a.path()
                .segments
                .last()
                .is_some_and(|seg| seg.ident == "test")
        }),
        _ => false,
    });
    (has_alloc, has_test)
}

/// Audits all 8 math crates for `#[global_allocator]` in `src/lib.rs` and `tests/security_*.rs` suites.
pub fn audit_tier5_security_suites(ws_root: &Path, crate_dirs: &[&str]) -> Vec<String> {
    let mut violations = Vec::new();
    for &rel_dir in crate_dirs {
        let crate_dir = ws_root.join(rel_dir);
        let lib_rs = crate_dir.join("src/lib.rs");
        let (lib_alloc, _) = file_declares_verified_allocator_and_test(&lib_rs, false);
        if !lib_alloc {
            violations.push(format!(
                "[!] `{rel_dir}/src/lib.rs` must register `#[global_allocator]` with `VerifiedAllocator`"
            ));
        }

        let tests_dir = crate_dir.join("tests");
        let mut security_files = Vec::new();
        if let Ok(entries) = fs::read_dir(&tests_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("rs")
                    && path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .is_some_and(|s| s.starts_with("security_"))
                {
                    security_files.push(path);
                }
            }
        }
        security_files.sort();
        if security_files.is_empty() {
            violations.push(format!(
                "[!] Math crate `{rel_dir}` is missing a Tier 5 `tests/security_*.rs` suite"
            ));
            continue;
        }
        for sec_file in security_files {
            let (has_alloc, has_test) = file_declares_verified_allocator_and_test(&sec_file, true);
            let disp = sec_file
                .strip_prefix(ws_root)
                .unwrap_or(&sec_file)
                .display();
            if !has_alloc {
                violations.push(format!(
                    "[!] Security suite `{disp}` must declare `#[global_allocator]` with `VerifiedAllocator`"
                ));
            }
            if !has_test {
                violations.push(format!(
                    "[!] Security suite `{disp}` must contain at least one `#[test]`"
                ));
            }
        }
    }
    violations
}

/// Executes the `unified_verification check-security` gate (`--offline` skips remote OSV query).
pub fn run_check_security(args: &[String]) -> bool {
    let mut offline = false;
    for arg in args {
        match arg.as_str() {
            "--offline" => offline = true,
            other => {
                eprintln!("[!] Unknown check-security argument: {other}");
                return false;
            }
        }
    }

    let ws_root = crate::manifest_cache::workspace_root_dir();
    let whitelist_path = find_whitelist_path();
    let whitelist =
        match load_whitelist_strict(whitelist_path.as_deref(), WORKSPACE_WHITELIST_CANDIDATES) {
            Ok(w) => w,
            Err(err) => {
                eprintln!("[!] {err}");
                return false;
            }
        };
    let Some(cfg) = whitelist.supply_chain else {
        eprintln!("[!] Missing [supply_chain] section in verification_whitelist.toml");
        return false;
    };

    let mut violations = Vec::new();
    match fs::read_to_string(ws_root.join("Cargo.lock")) {
        Ok(lock_str) => violations.extend(audit_cargo_lock(&lock_str, &cfg)),
        Err(err) => violations.push(format!("[!] Failed to read Cargo.lock: {err}")),
    }

    let meta_out = Command::new("cargo")
        .current_dir(&ws_root)
        .args(["metadata", "--format-version", "1", "--locked"])
        .output();
    match meta_out {
        Ok(out) if out.status.success() => match serde_json::from_slice(&out.stdout) {
            Ok(json) => violations.extend(audit_metadata_licenses(&json, &cfg)),
            Err(err) => violations.push(format!("[!] Failed to parse cargo metadata JSON: {err}")),
        },
        Ok(out) => violations.push(format!(
            "[!] cargo metadata --locked failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )),
        Err(err) => violations.push(format!("[!] Failed to spawn cargo metadata: {err}")),
    }

    violations.extend(audit_tier5_security_suites(&ws_root, MATH_CRATE_DIRS));

    for v in &violations {
        eprintln!("{v}");
    }
    let mut passed = violations.is_empty();
    if !offline && !crate::vulnerabilities::check_osv_vulnerabilities() {
        passed = false;
    }
    if passed {
        println!(
            "Supply-chain & Tier 5 security checks passed (registries, SPDX licenses, duplicate ceiling <= {:?}, 8 math crates).",
            cfg.max_duplicate_families
        );
    }
    passed
}

#[cfg(test)]
#[path = "supply_chain_tests.rs"]
mod tests;
