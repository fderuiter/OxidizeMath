use std::fs;
use std::path::Path;
use std::process::{Command, exit};

pub(super) fn is_safe_bundle_filename(name: &str) -> bool {
    !name.starts_with('-')
        && (name.ends_with(".pdf")
            || name == "Verification_Certificate.json"
            || name == "index.html")
        && name != "SHA256SUMS"
}

pub(super) fn collect_bundle_artifacts(output_dir: &Path) -> Vec<String> {
    let entries = fs::read_dir(output_dir).unwrap();
    let mut artifacts = Vec::new();
    for entry in entries.filter_map(|e| e.ok()) {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('-') {
            eprintln!("Warning: ignoring filename with leading dash: {}", name);
            continue;
        }
        if is_safe_bundle_filename(&name) {
            artifacts.push(name);
        }
    }
    artifacts.sort();
    artifacts.dedup();
    artifacts
}

pub(super) fn build_tar_args(files_to_pack: &[String]) -> Vec<&str> {
    let mut tar_args = vec!["-czf", "papers-bundle.tar.gz", "--"];
    let file_refs: Vec<&str> = files_to_pack.iter().map(String::as_str).collect();
    tar_args.extend(file_refs);
    tar_args
}

pub(super) fn build_checksum_args<'a>(cmd: &str, checksum_targets: &'a [String]) -> Vec<&'a str> {
    let mut cmd_args: Vec<&'a str> = Vec::new();
    if cmd == "shasum" {
        cmd_args.push("-a");
        cmd_args.push("256");
    }
    cmd_args.push("--");
    let target_refs: Vec<&'a str> = checksum_targets.iter().map(String::as_str).collect();
    cmd_args.extend(target_refs);
    cmd_args
}

/// Package compiled PDFs, index HTML, and verification certificate into a release tarball with SHA-256 checksums.
pub fn package_release_bundle(output_dir: &Path) {
    let cert_path = Path::new("Verification_Certificate.json");
    if cert_path.exists() {
        let _ = fs::copy(cert_path, output_dir.join("Verification_Certificate.json"));
    }

    let bundle_file = output_dir.join("papers-bundle.tar.gz");
    let files_to_pack = collect_bundle_artifacts(output_dir);
    let tar_args = build_tar_args(&files_to_pack);

    let tar_status = Command::new("tar")
        .current_dir(output_dir)
        .args(&tar_args)
        .status();

    if !tar_status.map(|s| s.success()).unwrap_or(false) {
        eprintln!("Error creating papers bundle archive.");
        exit(1);
    }
    println!("Created release bundle at {:?}", bundle_file);

    // Generate SHA256 checksums
    generate_sha256sums(output_dir);
}

fn generate_sha256sums(output_dir: &Path) {
    let (cmd, flag) = if Command::new("sha256sum").arg("--version").output().is_ok() {
        ("sha256sum", "-c")
    } else {
        ("shasum", "-c")
    };

    let mut checksum_targets = vec!["papers-bundle.tar.gz".to_string()];
    checksum_targets.extend(collect_bundle_artifacts(output_dir));
    checksum_targets.sort();
    checksum_targets.dedup();

    let cmd_args = build_checksum_args(cmd, &checksum_targets);

    let output = Command::new(cmd)
        .current_dir(output_dir)
        .args(&cmd_args)
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let sums_path = output_dir.join("SHA256SUMS");
            fs::write(&sums_path, &out.stdout).unwrap();
            println!("Generated SHA256SUMS at {:?}", sums_path);

            let check_status = Command::new(cmd)
                .current_dir(output_dir)
                .arg(flag)
                .arg("--")
                .arg("SHA256SUMS")
                .status();
            if !check_status.map(|s| s.success()).unwrap_or(false) {
                eprintln!("Checksum verification failed!");
                exit(1);
            }
            println!("SHA256 checksums successfully verified.");
        }
        _ => {
            eprintln!("Error generating SHA-256 sums.");
            exit(1);
        }
    }
}
