#![doc = include_str!("../README.md")]
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
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
        "lint-all" | "lint" => lint_all(),
        "test-features" => test_features(&args[2..]),
        "verify-suite" => verify_suite(&args[2..]),
        "verify-records" => verify_records(),
        "compile-papers" => compile_papers(),
        "traceability" => traceability(),
        "check-file-lengths" => check_file_lengths(),
        "check-staged-duplicates" => check_staged_duplicates(),
        "check-markdown-links" => check_markdown_links(),
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
    println!(
        "  check-markdown-links     - Check all internal relative markdown documentation links"
    );
    println!("  check-staged-duplicates  - Run staged duplicates check");
    println!(
        "  lint-all                 - Run format, clippy, file-length, duplicate, doc, and doctest checks"
    );
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
    run_cmd_with_env(cmd, args, &[]);
}

fn run_cmd_with_env(cmd: &str, args: &[&str], envs: &[(&str, &str)]) {
    println!("Running: {} {}", cmd, args.join(" "));
    let mut command = Command::new(cmd);
    command.args(args);
    for (k, v) in envs {
        command.env(k, v);
    }
    let status = command.status();
    match status {
        Ok(s) if s.success() => (),
        _ => {
            eprintln!("Command failed!");
            exit(1);
        }
    }
}

fn lint_all() {
    println!("=== Running Workspace Lint Suite ===");
    println!("\n--- Step 1: Format Check ---");
    run_cmd("cargo", &["fmt", "--all", "--", "--check"]);

    println!("\n--- Step 2: Clippy Lints ---");
    run_cmd(
        "cargo",
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ],
    );

    println!("\n--- Step 3: Staged Duplicates Check ---");
    check_staged_duplicates();

    println!("\n--- Step 4: File Length Checks ---");
    check_file_lengths();

    println!("\n--- Step 5: Cargo Doc Verification ---");
    run_cmd_with_env(
        "cargo",
        &["doc", "--workspace", "--no-deps", "--all-features"],
        &[("RUSTDOCFLAGS", "-D warnings")],
    );

    println!("\n--- Step 6: Doc Tests ---");
    run_cmd("cargo", &["test", "--doc", "--workspace", "--all-features"]);

    println!("\n--- Step 7: Markdown Documentation Links ---");
    check_markdown_links();

    println!("\n=== All Lint Checks Passed Successfully ===");
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

echo "Running pre-commit lint check..."
cargo run -p xtask -- lint-all
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "Linting failed! Commit blocked."
    exit 1
fi

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
                .env("TEXINPUTS", "papers//:")
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

fn check_markdown_links() {
    println!("=== Checking Markdown Documentation Links ===");
    let mut md_files = Vec::new();
    collect_md_files(Path::new("."), &mut md_files);

    let mut broken = Vec::new();
    for file in &md_files {
        if let Ok(content) = fs::read_to_string(file) {
            let parent = file.parent().unwrap_or(Path::new("."));
            let links = extract_md_links(&content);
            for link in links {
                let target = link.split_whitespace().next().unwrap_or(&link);
                if target.starts_with("http://")
                    || target.starts_with("https://")
                    || target.starts_with("mailto:")
                    || target.starts_with('#')
                    || target.contains("::")
                {
                    continue;
                }
                let path_part = target.split('#').next().unwrap_or("");
                if path_part.is_empty() {
                    continue;
                }
                let rel_dest = parent.join(path_part);
                let root_dest = Path::new(path_part);
                if !rel_dest.exists() && !root_dest.exists() {
                    broken.push((file.display().to_string(), target.to_string()));
                }
            }
        }
    }

    if !broken.is_empty() {
        eprintln!("Found {} broken markdown link(s):", broken.len());
        for (f, l) in &broken {
            eprintln!("  In {}: broken link target -> {}", f, l);
        }
        exit(1);
    }
    println!(
        "All markdown documentation links in {} files resolved successfully.",
        md_files.len()
    );
}

fn collect_md_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name == "target" || name == ".git" || name == "node_modules" {
                    continue;
                }
                collect_md_files(&path, out);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("md") {
                out.push(path);
            }
        }
    }
}

fn extract_md_links(content: &str) -> Vec<String> {
    let mut links = Vec::new();
    let mut rest = content;
    while let Some(start_bracket) = rest.find('[') {
        rest = &rest[start_bracket + 1..];
        if let Some(end_bracket) = rest.find(']') {
            let after_bracket = &rest[end_bracket + 1..];
            if after_bracket.starts_with('(') {
                if let Some(end_paren) = after_bracket.find(')') {
                    let link = &after_bracket[1..end_paren];
                    links.push(link.trim().to_string());
                    rest = &after_bracket[end_paren + 1..];
                } else {
                    rest = after_bracket;
                }
            } else {
                rest = after_bracket;
            }
        }
    }
    links
}
