#![doc = include_str!("../README.md")]
use std::env;
use std::fs;
use std::process::{Command, exit};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args[1] == "-h" || args[1] == "--help" {
        print_help();
        if args.len() < 2 {
            exit(1);
        } else {
            exit(0);
        }
    }
    match args[1].as_str() {
        "setup" => setup(),
        "test-features" => test_features(&args[2..]),
        "verify-suite" => verify_suite(&args[2..]),
        "verify-records" => verify_records(),
        "compile-papers" => compile_papers(),
        "traceability" => traceability(),
        "check-file-lengths" => check_file_lengths(),
        "check-staged-duplicates" => check_staged_duplicates(),
        "regenerate-baseline" => regenerate_baseline(),
        _ => {
            println!("Unknown command");
            exit(1);
        }
    }
}

fn print_help() {
    println!("Usage: xtask <command> [args...]\n");
    println!("Active Quality Gates:");
    println!(
        "  check-file-lengths       - Run centralized verification suite for file-length constraints"
    );
    println!("  check-staged-duplicates  - Run staged duplicates check");
    println!("  test-features            - Run tests across feature combinations");
    println!("  traceability             - Generate traceability report");
    println!("  verify-records           - Verify verification records");
    println!("  verify-suite             - Run high-integrity verified suite\n");
    println!("Legacy Utility Tasks:");
    println!("  compile-papers           - Compile academic papers with Tectonic");
    println!("  regenerate-baseline      - Regenerate public API baseline");
    println!("  setup                    - Run setup script and install pre-commit hook");
}

fn run_cmd(cmd: &str, args: &[&str]) {
    println!("Running: {} {}", cmd, args.join(" "));
    let status = Command::new(cmd).args(args).status();
    match status {
        Ok(s) if s.success() => (),
        _ => {
            eprintln!("Command failed!");
            exit(1);
        }
    }
}

fn setup_pre_commit_hook(hook_path: &str, hook_content: &str) {
    if std::path::Path::new(hook_path).exists() {
        let existing = fs::read_to_string(hook_path).unwrap_or_default();
        if !existing.contains("auto-generated pre-commit hook") {
            println!(
                "Warning: A custom pre-commit hook exists at {}. Please merge the verification check manually or remove it to allow auto-installation.",
                hook_path
            );
        } else {
            fs::write(hook_path, hook_content).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(hook_path).unwrap().permissions();
                perms.set_mode(0o755);
                fs::set_permissions(hook_path, perms).unwrap();
            }
            println!("Pre-commit hook updated.");
        }
    } else {
        if let Some(parent) = std::path::Path::new(hook_path).parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(hook_path, hook_content).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(hook_path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(hook_path, perms).unwrap();
        }
        println!("Pre-commit hook installed.");
    }
}

fn setup() {
    println!("=== Math Explorer Setup Script ===");
    run_cmd("cargo", &["build"]);
    run_cmd("cargo", &["test"]);

    let hook_path = ".git/hooks/pre-commit";
    let hook_content = r#"#!/bin/sh
# auto-generated pre-commit hook

for profile in "$HOME/.bash_profile" "$HOME/.zprofile" "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$profile" ]; then
        . "$profile" >/dev/null 2>&1 || true
    fi
done

if ! command -v cargo >/dev/null 2>&1; then
    if [ -d "$HOME/.cargo/bin" ]; then
        export PATH="$PATH:$HOME/.cargo/bin"
    fi
fi

echo "Running staged duplicates check..."
cargo run -p xtask -- check-staged-duplicates

echo "Running centralized verification suite..."
OUTPUT=$(cargo run -p xtask -- check-file-lengths 2>&1)
EXIT_CODE=$?

echo "$OUTPUT"

if [ $EXIT_CODE -ne 0 ]; then
    echo "Verification failed! Commit blocked due to file-length constraints."
    exit 1
fi

case "$OUTPUT" in
    *"File length violation"*)
        echo "Verification failed! Commit blocked due to file-length constraints."
        exit 1
        ;;
esac

exit 0
"#;

    setup_pre_commit_hook(hook_path, hook_content);
    println!("=== Setup Complete ===");
}

fn test_features(args: &[String]) {
    if args.is_empty() {
        let features = [
            "pure_math",
            "applied",
            "ai",
            "biology",
            "climate",
            "epidemiology",
            "physics",
        ];
        println!("=== Running core-only (no features) ===");
        run_cmd(
            "cargo",
            &["test", "-p", "math_explorer", "--no-default-features"],
        );
        for f in features.iter() {
            println!("=== Running tests for feature: {} ===", f);
            run_cmd(
                "cargo",
                &[
                    "test",
                    "-p",
                    "math_explorer",
                    "--no-default-features",
                    "--features",
                    f,
                ],
            );
        }
        println!("=== Running all features ===");
        run_cmd("cargo", &["test", "-p", "math_explorer", "--all-features"]);
        println!("All feature combinations passed successfully!");
    } else {
        let feature = &args[0];
        if feature == "core-only" || feature.is_empty() {
            run_cmd(
                "cargo",
                &["test", "-p", "math_explorer", "--no-default-features"],
            );
        } else {
            run_cmd(
                "cargo",
                &[
                    "test",
                    "-p",
                    "math_explorer",
                    "--no-default-features",
                    "--features",
                    feature,
                ],
            );
        }
    }
}

fn verify_suite(args: &[String]) {
    println!("=== High-Integrity Verified Suite ===");
    let mut cmd_args = vec![
        "run",
        "-p",
        "unified_verification",
        "--release",
        "--",
        "verify-suite",
    ];
    let args_str: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    cmd_args.extend(args_str);
    run_cmd("cargo", &cmd_args);
}

fn verify_records() {
    println!("=== Verify Records ===");
    run_cmd(
        "cargo",
        &[
            "run",
            "-p",
            "unified_verification",
            "--release",
            "--",
            "verify-records",
        ],
    );
}

fn compile_papers() {
    println!("=== Compiling Papers with pdflatex ===");

    let check = Command::new("pdflatex").arg("--version").output();
    match check {
        Ok(output) if output.status.success() => {}
        _ => {
            eprintln!("Error: pdflatex is not installed or not in PATH.");
            eprintln!("Please install TeXLive to compile academic papers:");
            eprintln!(
                "sudo apt-get install -y texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra texlive-science"
            );
            exit(1);
        }
    }

    let papers_dir = std::path::Path::new("papers");
    let output_dir = std::path::Path::new("papers/output");

    if !output_dir.exists() {
        fs::create_dir_all(output_dir).unwrap();
    }

    for entry in fs::read_dir(papers_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "tex") {
            println!("Compiling {:?}", path);
            let status = Command::new("pdflatex")
                .arg("-interaction=nonstopmode")
                .arg("-halt-on-error")
                .arg("-output-directory")
                .arg(output_dir)
                .arg(&path)
                .status();

            match status {
                Ok(s) if s.success() => println!("Successfully compiled {:?}", path),
                _ => {
                    eprintln!("Failed to compile {:?}", path);
                    exit(1);
                }
            }
        }
    }
    println!("=== All papers compiled successfully ===");
}

fn traceability() {
    println!("=== Traceability Report ===");
    run_cmd(
        "cargo",
        &[
            "run",
            "-p",
            "unified_verification",
            "--release",
            "--",
            "traceability",
        ],
    );
}

fn check_file_lengths() {
    println!("=== Check File Lengths ===");
    run_cmd(
        "cargo",
        &[
            "run",
            "-p",
            "unified_verification",
            "--release",
            "--",
            "check-file-lengths",
        ],
    );
}
fn check_staged_duplicates() {
    run_cmd(
        "cargo",
        &[
            "run",
            "-p",
            "unified_verification",
            "--release",
            "--",
            "check-staged-duplicates",
        ],
    );
}

fn regenerate_baseline() {
    println!("=== Regenerate Public API Baseline ===");
    run_cmd(
        "cargo",
        &[
            "run",
            "-p",
            "unified_verification",
            "--release",
            "--",
            "regenerate-baseline",
        ],
    );
}
