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
        let filename = format!("test_cargo_{}_{}.lock", std::process::id(), id);
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

fn spawn_mock_server_batches(
    status_line: &'static str,
    body: &'static str,
    expected_requests: usize,
) -> (String, std::sync::Arc<AtomicUsize>, thread::JoinHandle<()>) {
    let listener =
        TcpListener::bind("127.0.0.1:0").expect("Failed to bind mock server TCP listener");
    let addr = listener
        .local_addr()
        .expect("Failed to retrieve local address");
    let url = format!("http://{}", addr);
    let count = std::sync::Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        for _ in 0..expected_requests {
            if let Ok((mut stream, _)) = listener.accept() {
                count_clone.fetch_add(1, Ordering::SeqCst);
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
                let _ = stream.shutdown(std::net::Shutdown::Write);
                let mut drain = [0u8; 512];
                while let Ok(n) = stream.read(&mut drain) {
                    if n == 0 {
                        break;
                    }
                }
            }
        }
    });

    (url, count, handle)
}

fn spawn_mock_server(
    status_line: &'static str,
    body: &'static str,
) -> (String, thread::JoinHandle<()>) {
    let (url, _, handle) = spawn_mock_server_batches(status_line, body, 1);
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
    assert!(check_osv_vulnerabilities_with_path_and_url(
        &lock.path,
        "http://127.0.0.1:9"
    ));
}

#[test]
fn test_network_connection_failure_fails() {
    let lock = TempLockFile::new(SAMPLE_LOCK_WITH_CRATES);
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

#[test]
fn test_ignored_unmaintained_advisory_passes() {
    let res_json = serde_json::json!({
        "results": [
            {
                "vulns": [
                    { "id": "RUSTSEC-2024-0436" },
                    { "id": "RUSTSEC-2026-0543" }
                ]
            }
        ]
    });
    let pkg_names = vec![("paste".to_string(), "1.0.15".to_string())];
    assert!(parse_advisories(&res_json, &pkg_names));
}

#[test]
fn test_timeout_fails_on_unresponsive_server() {
    let lock = TempLockFile::new(SAMPLE_LOCK_WITH_CRATES);
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind mock server");
    let addr = listener.local_addr().expect("Failed to get local address");
    let url = format!("http://{}", addr);

    let handle = thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            thread::sleep(Duration::from_millis(300));
        }
    });

    let passed = check_osv_vulnerabilities_with_path_url_and_timeout(
        &lock.path,
        &url,
        Duration::from_millis(50),
    );
    assert!(!passed);
    let _ = handle.join();
}

#[test]
fn test_batch_chunking_with_multiple_batches() {
    assert_eq!(BATCH_SIZE, 128);
    assert_eq!(REQUEST_TIMEOUT, Duration::from_secs(30));

    let mut lock_content = String::new();
    for i in 0..130 {
        lock_content.push_str(&format!(
            "[[package]]\nname = \"crate-{}\"\nversion = \"1.0.0\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n\n",
            i
        ));
    }
    let lock = TempLockFile::new(&lock_content);
    let (url, count, handle) = spawn_mock_server_batches("200 OK", r#"{"results":[]}"#, 2);

    assert!(check_osv_vulnerabilities_with_path_and_url(
        &lock.path, &url
    ));
    let _ = handle.join();
    assert_eq!(count.load(Ordering::SeqCst), 2);
}
