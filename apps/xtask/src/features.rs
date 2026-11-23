use crate::gates::run_cmd;
use std::process::{Command, exit};

pub(crate) fn assert_no_serde_in_normal_deps(features: Option<&str>) {
    for banned in ["serde", "serde_derive"] {
        let mut args = vec![
            "tree",
            "-p",
            "math_explorer",
            "-e",
            "normal",
            "--no-default-features",
        ];
        if let Some(f) = features.filter(|s| !s.is_empty()) {
            args.extend_from_slice(&["--features", f]);
        }
        args.extend_from_slice(&["-i", banned]);
        let out = Command::new("cargo")
            .args(&args)
            .output()
            .expect("failed to execute cargo tree");
        let stdout = String::from_utf8_lossy(&out.stdout);
        if !out.status.success() || !stdout.trim().is_empty() {
            eprintln!(
                "Unexpected `{banned}` normal dependency in `cargo {}`:\n{stdout}",
                args.join(" ")
            );
            exit(1);
        }
    }
}

fn test_single_feature(feature: &str) {
    if feature == "core-only" || feature.is_empty() {
        assert_no_serde_in_normal_deps(None);
        run_cmd(
            "cargo",
            &["test", "-p", "math_explorer", "--no-default-features"],
        );
        run_cmd(
            "cargo",
            &[
                "test",
                "-p",
                "math_explorer",
                "--no-default-features",
                "--features",
                "serde",
            ],
        );
        return;
    }
    if feature.split(',').any(|f| f.trim() == "serde") {
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
        return;
    }
    assert_no_serde_in_normal_deps(Some(feature));
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
    let with_serde = format!("{feature},serde");
    run_cmd(
        "cargo",
        &[
            "test",
            "-p",
            "math_explorer",
            "--no-default-features",
            "--features",
            &with_serde,
        ],
    );
}

pub fn test_features(args: &[String]) {
    if let Some(feature) = args.first() {
        test_single_feature(feature);
        return;
    }
    let domains = [
        "pure_math",
        "applied",
        "ai",
        "biology",
        "climate",
        "epidemiology",
        "physics",
    ];
    println!("=== Running core-only (without and with serde) ===");
    test_single_feature("core-only");
    assert_no_serde_in_normal_deps(Some(&domains.join(",")));
    for f in &domains {
        println!("=== Running tests for feature: {f} (without and with serde) ===");
        test_single_feature(f);
    }
    println!("=== Running all features ===");
    run_cmd("cargo", &["test", "-p", "math_explorer", "--all-features"]);
    println!("All feature combinations passed successfully!");
}
