#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use toml::Value;

#[cfg(not(target_arch = "wasm32"))]
pub const BATCH_SIZE: usize = 128;
#[cfg(not(target_arch = "wasm32"))]
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

#[cfg(not(target_arch = "wasm32"))]
pub fn check_osv_vulnerabilities() -> bool {
    check_osv_vulnerabilities_with_path_and_url(
        Path::new("Cargo.lock"),
        "https://api.osv.dev/v1/querybatch",
    )
}

#[cfg(not(target_arch = "wasm32"))]
pub fn check_osv_vulnerabilities_with_path_and_url(lock_path: &Path, osv_url: &str) -> bool {
    check_osv_vulnerabilities_with_path_url_and_timeout(lock_path, osv_url, REQUEST_TIMEOUT)
}

#[cfg(not(target_arch = "wasm32"))]
fn extract_crates_io_packages(parsed: &Value) -> (Vec<serde_json::Value>, Vec<(String, String)>) {
    let mut queries = Vec::new();
    let mut pkg_names = Vec::new();

    if let Some(packages) = parsed.get("package").and_then(|p| p.as_array()) {
        for pkg in packages {
            if let (Some(name), Some(version)) = (
                pkg.get("name").and_then(|n| n.as_str()),
                pkg.get("version").and_then(|v| v.as_str()),
            ) {
                let source = pkg.get("source").and_then(|s| s.as_str()).unwrap_or("");
                if source.contains("crates.io") {
                    queries.push(ureq::json!({
                        "package": {
                            "name": name,
                            "ecosystem": "crates.io"
                        },
                        "version": version
                    }));
                    pkg_names.push((name.to_string(), version.to_string()));
                }
            }
        }
    }

    (queries, pkg_names)
}

#[cfg(not(target_arch = "wasm32"))]
fn query_osv_batch(
    agent: &ureq::Agent,
    osv_url: &str,
    queries: &[serde_json::Value],
) -> Option<serde_json::Value> {
    let req_body = ureq::json!({ "queries": queries });
    let res = match agent
        .post(osv_url)
        .set("Content-Type", "application/json")
        .send_json(req_body)
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to query remote vulnerability database: {}", e);
            return None;
        }
    };

    if res.status() < 200 || res.status() >= 300 {
        eprintln!(
            "Remote vulnerability database returned HTTP status {}",
            res.status()
        );
        return None;
    }

    match res.into_json() {
        Ok(j) => Some(j),
        Err(e) => {
            eprintln!("Failed to deserialize OSV response JSON: {}", e);
            None
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn check_osv_vulnerabilities_with_path_url_and_timeout(
    lock_path: &Path,
    osv_url: &str,
    timeout: Duration,
) -> bool {
    let content = match fs::read_to_string(lock_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read {}: {}", lock_path.display(), e);
            return false;
        }
    };

    let parsed: Value = match toml::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to parse Cargo.lock: {}", e);
            return false;
        }
    };

    let (queries, pkg_names) = extract_crates_io_packages(&parsed);
    if queries.is_empty() {
        return true;
    }

    let agent: ureq::Agent = ureq::AgentBuilder::new().timeout(timeout).build();
    let mut all_passed = true;

    for (query_chunk, pkg_chunk) in queries.chunks(BATCH_SIZE).zip(pkg_names.chunks(BATCH_SIZE)) {
        let Some(res_json) = query_osv_batch(&agent, osv_url, query_chunk) else {
            return false;
        };

        if !parse_advisories(&res_json, pkg_chunk) {
            all_passed = false;
        }
    }

    all_passed
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_ignored_advisories(
    whitelist_path: Option<&Path>,
) -> Result<Vec<crate::whitelist::IgnoredAdvisoryEntry>, String> {
    let resolved = whitelist_path
        .map(Path::to_path_buf)
        .or_else(crate::whitelist::find_whitelist_path);
    let parsed = crate::whitelist::load_whitelist_strict(
        resolved.as_deref(),
        crate::whitelist::WORKSPACE_WHITELIST_CANDIDATES,
    )?;
    Ok(parsed
        .supply_chain
        .map(|sc| sc.ignored_advisories)
        .unwrap_or_default())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn parse_advisories_with_ignored(
    res_json: &serde_json::Value,
    pkg_names: &[(String, String)],
    ignored: &[crate::whitelist::IgnoredAdvisoryEntry],
) -> bool {
    let mut passed = true;

    if let Some(results) = res_json.get("results").and_then(|r| r.as_array()) {
        for (i, result) in results.iter().enumerate() {
            if let Some(vulns) = result.get("vulns").and_then(|v| v.as_array()) {
                for vuln in vulns {
                    let vuln_id = vuln.get("id").and_then(|id| id.as_str()).unwrap_or("");
                    let (pkg_name, pkg_version) = pkg_names
                        .get(i)
                        .map(|(n, v)| (n.as_str(), v.as_str()))
                        .unwrap_or(("unknown", "unknown"));
                    if let Some(entry) = ignored.iter().find(|entry| entry.id == vuln_id) {
                        println!(
                            "[-] Ignoring audited advisory {} ({} {}): {}",
                            vuln_id, pkg_name, pkg_version, entry.reason
                        );
                        continue;
                    }
                    println!(
                        "[!] Vulnerability found in {} {}: {}",
                        pkg_name, pkg_version, vuln_id
                    );
                    passed = false;
                }
            }
        }
    }
    passed
}

#[cfg(not(target_arch = "wasm32"))]
pub fn parse_advisories(res_json: &serde_json::Value, pkg_names: &[(String, String)]) -> bool {
    let Ok(ignored) = load_ignored_advisories(None) else {
        eprintln!("Failed to load audited ignored_advisories from verification_whitelist.toml");
        return false;
    };
    parse_advisories_with_ignored(res_json, pkg_names, &ignored)
}

pub use crate::supply_chain::run_check_security;

#[cfg(target_arch = "wasm32")]
pub fn check_osv_vulnerabilities() -> bool {
    true
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "vulnerabilities_tests.rs"]
mod tests;
