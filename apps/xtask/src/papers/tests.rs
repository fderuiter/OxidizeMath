use super::*;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};

static TEST_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct TempOutputDir {
    path: PathBuf,
}

impl TempOutputDir {
    fn new(label: &str) -> Self {
        let id = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir_name = format!(
            "oxidizemath_papers_test_{}_{}_{}",
            std::process::id(),
            label,
            id
        );
        let path = env::temp_dir().join(dir_name);
        if path.exists() {
            let _ = fs::remove_dir_all(&path);
        }
        fs::create_dir_all(&path).expect("failed to create temp test dir");
        Self { path }
    }
}

impl Drop for TempOutputDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[test]
fn test_is_safe_bundle_filename_rejects_leading_dashes() {
    assert!(is_safe_bundle_filename("paper1.pdf"));
    assert!(is_safe_bundle_filename("index.html"));
    assert!(is_safe_bundle_filename("Verification_Certificate.json"));

    assert!(!is_safe_bundle_filename("-"));
    assert!(!is_safe_bundle_filename("-v.pdf"));
    assert!(!is_safe_bundle_filename("-index.html"));
    assert!(!is_safe_bundle_filename("--checkpoint=1.pdf"));
    assert!(!is_safe_bundle_filename(
        "--checkpoint-action=exec=touch CANARY_PWNED.pdf"
    ));
    assert!(!is_safe_bundle_filename("SHA256SUMS"));
    assert!(!is_safe_bundle_filename("SHA256SUMS-papers"));
    assert!(!is_safe_bundle_filename("notes.txt"));
}

#[test]
fn test_collect_bundle_artifacts_filters_dash_prefixed_files() {
    let temp = TempOutputDir::new("collect");
    fs::write(temp.path.join("algebra.pdf"), b"%PDF-1.4\n").unwrap();
    fs::write(temp.path.join("index.html"), b"<html></html>").unwrap();
    fs::write(temp.path.join("Verification_Certificate.json"), b"{}").unwrap();
    fs::write(temp.path.join("--checkpoint=1.pdf"), b"malicious").unwrap();
    fs::write(
        temp.path
            .join("--checkpoint-action=exec=touch CANARY_PWNED.pdf"),
        b"malicious",
    )
    .unwrap();
    fs::write(temp.path.join("-rf.pdf"), b"malicious").unwrap();

    let collected = collect_bundle_artifacts(&temp.path);
    assert_eq!(
        collected,
        vec![
            "Verification_Certificate.json".to_string(),
            "algebra.pdf".to_string(),
            "index.html".to_string(),
        ]
    );
}

#[test]
fn test_tar_and_checksum_args_include_end_of_options_delimiter() {
    let files = vec!["algebra.pdf".to_string(), "index.html".to_string()];
    let tar_args = build_tar_args(&files);
    assert_eq!(
        tar_args,
        vec![
            "-czf",
            "papers-bundle.tar.gz",
            "--",
            "algebra.pdf",
            "index.html"
        ]
    );

    let sha256_args = build_checksum_args("sha256sum", &files);
    assert_eq!(sha256_args, vec!["--", "algebra.pdf", "index.html"]);

    let shasum_args = build_checksum_args("shasum", &files);
    assert_eq!(
        shasum_args,
        vec!["-a", "256", "--", "algebra.pdf", "index.html"]
    );
}

#[test]
#[cfg(unix)]
fn test_package_release_bundle_prevents_option_injection() {
    let temp = TempOutputDir::new("bundle_exec");
    fs::write(temp.path.join("paper_alpha.pdf"), b"%PDF-1.4 sample\n").unwrap();
    fs::write(temp.path.join("index.html"), b"<!DOCTYPE html></html>").unwrap();
    fs::write(temp.path.join("Verification_Certificate.json"), b"{}").unwrap();

    // Plant GNU tar option-injection payloads in output_dir
    fs::write(temp.path.join("--checkpoint=1.pdf"), b"payload").unwrap();
    fs::write(
        temp.path
            .join("--checkpoint-action=exec=touch CANARY_PWNED.pdf"),
        b"payload",
    )
    .unwrap();

    package_release_bundle(&temp.path);

    // Verify injected command was NOT executed
    assert!(
        !temp.path.join("CANARY_PWNED.pdf").exists(),
        "Option injection payload must not execute during bundle creation"
    );

    // Verify archive exists and contains SHA256SUMS plus legitimate inner artifacts
    let bundle_path = temp.path.join("papers-bundle.tar.gz");
    assert!(bundle_path.exists(), "Release bundle should be created");

    let tar_list = Command::new("tar")
        .current_dir(&temp.path)
        .args(["-tzf", "papers-bundle.tar.gz"])
        .output()
        .expect("failed to list tar contents");
    assert!(tar_list.status.success());
    let mut entries: Vec<String> = String::from_utf8_lossy(&tar_list.stdout)
        .lines()
        .map(|s| s.trim_start_matches("./").to_string())
        .collect();
    entries.sort();
    assert_eq!(
        entries,
        vec![
            "SHA256SUMS".to_string(),
            "Verification_Certificate.json".to_string(),
            "index.html".to_string(),
            "paper_alpha.pdf".to_string(),
        ]
    );

    // Verify inner SHA256SUMS hashes inner artifacts only and excludes dash-prefixed entries
    let sums = fs::read_to_string(temp.path.join("SHA256SUMS")).expect("SHA256SUMS missing");
    assert!(!sums.contains("papers-bundle.tar.gz"));
    assert!(sums.contains("paper_alpha.pdf"));
    assert!(sums.contains("index.html"));
    assert!(sums.contains("Verification_Certificate.json"));
    assert!(!sums.contains("--checkpoint"));
    assert!(!sums.contains("CANARY_PWNED.pdf"));

    // Verify outer SHA256SUMS-papers hashes only papers-bundle.tar.gz
    let outer_sums =
        fs::read_to_string(temp.path.join("SHA256SUMS-papers")).expect("SHA256SUMS-papers missing");
    assert!(outer_sums.contains("papers-bundle.tar.gz"));
    assert!(!outer_sums.contains("paper_alpha.pdf"));
    assert!(!outer_sums.contains("index.html"));
    assert!(!outer_sums.contains("Verification_Certificate.json"));
    assert_eq!(outer_sums.lines().count(), 1);
}

#[test]
#[cfg(unix)]
fn test_split_manifests_verify_in_isolated_release_and_extracted_dirs() {
    let build_dir = TempOutputDir::new("split_build");
    fs::write(build_dir.path.join("paper_one.pdf"), b"%PDF-1.4 one\n").unwrap();
    fs::write(build_dir.path.join("paper_two.pdf"), b"%PDF-1.4 two\n").unwrap();
    fs::write(build_dir.path.join("index.html"), b"<html>index</html>").unwrap();
    fs::write(
        build_dir.path.join("Verification_Certificate.json"),
        b"{\"passed\":true}",
    )
    .unwrap();

    package_release_bundle(&build_dir.path);

    let cmd = detect_sha256_cmd();

    // Simulate release-assets/ containing only papers-bundle.tar.gz and SHA256SUMS-papers
    let release_dir = TempOutputDir::new("split_release");
    fs::copy(
        build_dir.path.join("papers-bundle.tar.gz"),
        release_dir.path.join("papers-bundle.tar.gz"),
    )
    .unwrap();
    fs::copy(
        build_dir.path.join("SHA256SUMS-papers"),
        release_dir.path.join("SHA256SUMS-papers"),
    )
    .unwrap();

    let outer_check = Command::new(cmd)
        .current_dir(&release_dir.path)
        .args(build_check_args(cmd, "SHA256SUMS-papers"))
        .status()
        .expect("failed to run checksum check on SHA256SUMS-papers");
    assert!(
        outer_check.success(),
        "SHA256SUMS-papers must verify cleanly"
    );

    // Extract papers-bundle.tar.gz into an empty directory and verify internal SHA256SUMS
    let extract_dir = TempOutputDir::new("split_extract");
    let extract_status = Command::new("tar")
        .arg("-xzf")
        .arg(release_dir.path.join("papers-bundle.tar.gz"))
        .arg("-C")
        .arg(&extract_dir.path)
        .status()
        .expect("failed to extract papers-bundle.tar.gz");
    assert!(extract_status.success());

    let inner_check = Command::new(cmd)
        .current_dir(&extract_dir.path)
        .args(build_check_args(cmd, "SHA256SUMS"))
        .status()
        .expect("failed to run checksum check on extracted SHA256SUMS");
    assert!(
        inner_check.success(),
        "Extracted SHA256SUMS must verify cleanly"
    );
}
