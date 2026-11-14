#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use toml::Value;

#[cfg(not(target_arch = "wasm32"))]
pub fn check_osv_vulnerabilities() -> bool {
    check_osv_vulnerabilities_with_path_and_url(
        Path::new("Cargo.lock"),
        "https://api.osv.dev/v1/querybatch",
    )
}

#[cfg(not(target_arch = "wasm32"))]
pub fn check_osv_vulnerabilities_with_path_and_url(lock_path: &Path, osv_url: &str) -> bool {
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

    if queries.is_empty() {
        return true;
    }

    let req_body = ureq::json!({ "queries": queries });
    let res = match ureq::post(osv_url)
        .set("Content-Type", "application/json")
        .send_json(req_body)
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to query remote vulnerability database: {}", e);
            return false;
        }
    };

    if res.status() < 200 || res.status() >= 300 {
        eprintln!(
            "Remote vulnerability database returned HTTP status {}",
            res.status()
        );
        return false;
    }

    let res_json: serde_json::Value = match res.into_json() {
        Ok(j) => j,
        Err(e) => {
            eprintln!("Failed to deserialize OSV response JSON: {}", e);
            return false;
        }
    };

    parse_advisories(&res_json, &pkg_names)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn parse_advisories(res_json: &serde_json::Value, pkg_names: &[(String, String)]) -> bool {
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

#[cfg(target_arch = "wasm32")]
pub fn check_osv_vulnerabilities() -> bool {
    true
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TempLockFile {
        path: std::path::PathBuf,
    }

    impl TempLockFile {
        fn new(content: &str) -> Self {
            let id = COUNTER.fetch_add(1, Ordering::SeqCst);
            let filename = format!(
                "test_cargo_{}_{}_{}.lock",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                id
            );
            let path = std::env::temp_dir().join(filename);
            fs::write(&path, content).expect("Failed to write temporary test lock file");
            Self { path }
        }
    }

    impl Drop for TempLockFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    fn spawn_mock_server(
        status_line: &'static str,
        body: &'static str,
    ) -> (String, thread::JoinHandle<()>) {
        let listener =
            TcpListener::bind("127.0.0.1:0").expect("Failed to bind mock server TCP listener");
        let addr = listener
            .local_addr()
            .expect("Failed to retrieve local address");
        let url = format!("http://{}", addr);

        let handle = thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let response = format!(
                    "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status_line,
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });

        (url, handle)
    }

    const SAMPLE_LOCK_WITH_CRATES: &str = r#"
[[package]]
name = "vulnerable-crate"
version = "1.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
"#;

    #[test]
    fn test_missing_cargo_lock_fails() {
        let missing = Path::new("/tmp/nonexistent_cargo_lock_path_123456.lock");
        assert!(!check_osv_vulnerabilities_with_path_and_url(
            missing,
            "http://127.0.0.1:9"
        ));
    }

    #[test]
    fn test_invalid_toml_cargo_lock_fails() {
        let lock = TempLockFile::new("invalid toml syntax [[[[");
        assert!(!check_osv_vulnerabilities_with_path_and_url(
            &lock.path,
            "http://127.0.0.1:9"
        ));
    }

    #[test]
    fn test_empty_crates_io_packages_passes() {
        let lock = TempLockFile::new(
            r#"
[[package]]
name = "local-only"
version = "0.1.0"
"#,
        );
        // Should not query network and should return true
        assert!(check_osv_vulnerabilities_with_path_and_url(
            &lock.path,
            "http://127.0.0.1:9"
        ));
    }

    #[test]
    fn test_network_connection_failure_fails() {
        let lock = TempLockFile::new(SAMPLE_LOCK_WITH_CRATES);
        // Find an unused port and immediately close the listener
        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind unused port");
        let addr = listener.local_addr().expect("Failed to get addr");
        drop(listener);

        let url = format!("http://{}", addr);
        assert!(!check_osv_vulnerabilities_with_path_and_url(
            &lock.path, &url
        ));
    }

    #[test]
    fn test_server_error_non_200_fails() {
        let lock = TempLockFile::new(SAMPLE_LOCK_WITH_CRATES);
        let (url, handle) = spawn_mock_server("500 Internal Server Error", r#"{"error":"boom"}"#);
        assert!(!check_osv_vulnerabilities_with_path_and_url(
            &lock.path, &url
        ));
        let _ = handle.join();
    }

    #[test]
    fn test_server_not_found_fails() {
        let lock = TempLockFile::new(SAMPLE_LOCK_WITH_CRATES);
        let (url, handle) = spawn_mock_server("404 Not Found", r#"{"error":"not found"}"#);
        assert!(!check_osv_vulnerabilities_with_path_and_url(
            &lock.path, &url
        ));
        let _ = handle.join();
    }

    #[test]
    fn test_deserialization_failure_fails() {
        let lock = TempLockFile::new(SAMPLE_LOCK_WITH_CRATES);
        let (url, handle) = spawn_mock_server("200 OK", "not a valid json string");
        assert!(!check_osv_vulnerabilities_with_path_and_url(
            &lock.path, &url
        ));
        let _ = handle.join();
    }

    #[test]
    fn test_empty_advisory_response_passes() {
        let lock = TempLockFile::new(SAMPLE_LOCK_WITH_CRATES);
        let (url, handle) = spawn_mock_server("200 OK", r#"{"results":[{"vulns":[]}]}"#);
        assert!(check_osv_vulnerabilities_with_path_and_url(
            &lock.path, &url
        ));
        let _ = handle.join();

        // Also test direct parse_advisories helper
        let res_json = serde_json::json!({
            "results": [
                { "vulns": [] }
            ]
        });
        let pkg_names = vec![("vulnerable-crate".to_string(), "1.0.0".to_string())];
        assert!(parse_advisories(&res_json, &pkg_names));
    }

    #[test]
    fn test_rustsec_advisory_detected() {
        let lock = TempLockFile::new(SAMPLE_LOCK_WITH_CRATES);
        let mock_body = r#"{"results":[{"vulns":[{"id":"RUSTSEC-2024-0429"}]}]}"#;
        let (url, handle) = spawn_mock_server("200 OK", mock_body);
        assert!(!check_osv_vulnerabilities_with_path_and_url(
            &lock.path, &url
        ));
        let _ = handle.join();

        let res_json = serde_json::json!({
            "results": [
                {
                    "vulns": [
                        { "id": "RUSTSEC-2024-0429" }
                    ]
                }
            ]
        });
        let pkg_names = vec![("glib".to_string(), "0.18.5".to_string())];
        assert!(!parse_advisories(&res_json, &pkg_names));
    }

    #[test]
    fn test_ghsa_advisory_detected() {
        let lock = TempLockFile::new(SAMPLE_LOCK_WITH_CRATES);
        let mock_body = r#"{"results":[{"vulns":[{"id":"GHSA-wrw7-89jp-8q8g"}]}]}"#;
        let (url, handle) = spawn_mock_server("200 OK", mock_body);
        assert!(!check_osv_vulnerabilities_with_path_and_url(
            &lock.path, &url
        ));
        let _ = handle.join();

        let res_json = serde_json::json!({
            "results": [
                {
                    "vulns": [
                        { "id": "GHSA-wrw7-89jp-8q8g" }
                    ]
                }
            ]
        });
        let pkg_names = vec![("glib".to_string(), "0.18.5".to_string())];
        assert!(!parse_advisories(&res_json, &pkg_names));
    }
}
