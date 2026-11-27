use crate::staged::parse_workspace_members;
use std::fs;
use std::path::Path;

const NON_INSTALLER_MEMBERS: &[&str] = &[
    "apps/xtask",
    "crates/04_bindings/markdown_tests",
    "apps/unified_verification",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CaseArm {
    pub patterns: Vec<String>,
    pub assignment: Option<(String, String)>,
}

pub(super) fn glob_matches(pattern: &str, text: &str) -> bool {
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = text.chars().collect();
    let (mut p_idx, mut t_idx) = (0usize, 0usize);
    let mut star_idx: Option<usize> = None;
    let mut match_idx = 0usize;
    let max_steps = (pat.len() + 1) * (txt.len() + 1) + 1;
    let mut steps = 0usize;

    while t_idx < txt.len() && steps < max_steps {
        steps += 1;
        if p_idx < pat.len() && (pat[p_idx] == '?' || pat[p_idx] == txt[t_idx]) {
            p_idx += 1;
            t_idx += 1;
        } else if p_idx < pat.len() && pat[p_idx] == '*' {
            star_idx = Some(p_idx);
            match_idx = t_idx;
            p_idx += 1;
        } else if let Some(s_idx) = star_idx {
            p_idx = s_idx + 1;
            match_idx += 1;
            t_idx = match_idx;
        } else {
            return false;
        }
    }
    while p_idx < pat.len() && pat[p_idx] == '*' {
        p_idx += 1;
    }
    assert!(steps <= max_steps);
    p_idx == pat.len()
}

fn parse_assignment(line: &str) -> Option<(String, String)> {
    let (lhs, rhs) = line.split_once('=')?;
    let key = lhs.trim();
    let val = rhs.trim().trim_matches('"').trim_matches('\'');
    (!key.is_empty()).then(|| (key.to_string(), val.to_string()))
}

fn parse_case_arms(case_body: &str) -> Vec<CaseArm> {
    let mut arms = Vec::new();
    let mut header_buf = String::new();
    let mut current_patterns: Option<Vec<String>> = None;
    let mut current_assign: Option<(String, String)> = None;

    for raw_line in case_body.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(patterns) = &current_patterns {
            if let Some(before_terminator) = line.strip_suffix(";;") {
                let stmt = before_terminator.trim();
                if !stmt.is_empty() {
                    current_assign = parse_assignment(stmt);
                }
                arms.push(CaseArm {
                    patterns: patterns.clone(),
                    assignment: current_assign.take(),
                });
                current_patterns = None;
            } else if let Some(parsed) = parse_assignment(line) {
                current_assign = Some(parsed);
            }
        } else if let Some(continued) = line.strip_suffix('\\') {
            header_buf.push_str(continued.trim());
        } else {
            header_buf.push_str(line);
            let pat_list = header_buf
                .strip_suffix(')')
                .expect("case pattern arm must end with ')'");
            let parsed_pats: Vec<String> = pat_list
                .trim_start_matches('(')
                .split('|')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect();
            assert!(!parsed_pats.is_empty(), "case arm must declare patterns");
            header_buf.clear();
            current_patterns = Some(parsed_pats);
        }
    }
    assert!(current_patterns.is_none(), "unterminated case arm");
    arms
}

pub(super) fn extract_detect_changes_step(ci_yaml: &str) -> (String, Vec<Vec<CaseArm>>) {
    let job_start = ci_yaml
        .find("\n  detect-changes:\n")
        .expect("ci.yml missing top-level detect-changes job");
    let preflight_start = ci_yaml[job_start..]
        .find("\n  pre-flight:\n")
        .expect("ci.yml missing pre-flight job after detect-changes")
        + job_start;
    let detect_job = &ci_yaml[job_start..preflight_start];
    assert!(
        detect_job.contains("timeout-minutes: 5"),
        "detect-changes job must set timeout-minutes: 5"
    );
    assert!(
        !detect_job.contains("Set up Rust") && !detect_job.contains("cargo "),
        "detect-changes job must remain bash-only without Rust or Cargo steps"
    );

    let step_offset = detect_job
        .find("- name: Detect Scoped Change Paths")
        .expect("detect-changes job missing Detect Scoped Change Paths step");
    let step = &detect_job[step_offset..];

    let mut blocks = Vec::new();
    let marker = "case \"$file_path\" in\n";
    let mut cursor = step;
    while let Some(case_pos) = cursor.find(marker) {
        let after_case = &cursor[case_pos + marker.len()..];
        let esac_pos = after_case
            .find("\n                esac")
            .or_else(|| after_case.find("esac"))
            .expect("missing esac in Detect Scoped Change Paths");
        blocks.push(parse_case_arms(&after_case[..esac_pos]));
        cursor = &after_case[esac_pos + 4..];
    }
    assert_eq!(blocks.len(), 2, "expected papers and installer case blocks");
    (step.to_string(), blocks)
}

pub(super) fn evaluate_pull_request_diff(
    blocks: &[Vec<CaseArm>],
    changed_files: &[&str],
) -> (bool, bool) {
    let mut papers_changed = false;
    let mut installer_changed = false;

    for file_path in changed_files.iter().copied().filter(|p| !p.is_empty()) {
        for arms in blocks {
            let matched = arms
                .iter()
                .find(|arm| arm.patterns.iter().any(|pat| glob_matches(pat, file_path)));
            if let Some(Some((key, val))) = matched.map(|arm| &arm.assignment) {
                match key.as_str() {
                    "papers_changed" => papers_changed = val == "true",
                    "installer_changed" => installer_changed = val == "true",
                    other => panic!("unexpected case assignment variable: {other}"),
                }
            }
        }
    }
    (papers_changed, installer_changed)
}

pub(super) fn assert_ci_static_and_dedup_parity(ci_yaml: &str) {
    for required_cmd in [
        "cargo build -p unified_verification",
        "cargo run -p unified_verification -- batch-static",
        "cargo run -p unified_verification -- check-release-tag",
        "cargo run -p unified_verification -- check-release-tag --require-tag --github-output \"$GITHUB_OUTPUT\"",
        "cargo run -p unified_verification -- verify-suite --coverage-only",
        "- name: Build (Native)\n        if: runner.os != 'Linux' && matrix.target != 'wasm32-unknown-unknown'",
    ] {
        assert!(
            ci_yaml.contains(required_cmd),
            "ci.yml missing expected CI command: {required_cmd}"
        );
    }
    for forbidden in [
        "unified_verification --release",
        "name: Check Workspace (WASM)",
        "name: Check Workspace Tests (WASM)",
    ] {
        assert!(
            !ci_yaml.contains(forbidden),
            "ci.yml must not retain redundant step or release verification flag: {forbidden}"
        );
    }
}

fn slice_job_block<'a>(ci_yaml: &'a str, job_id: &str, next_job_id: Option<&str>) -> &'a str {
    let header = format!("\n  {job_id}:\n");
    let start = ci_yaml
        .find(&header)
        .unwrap_or_else(|| panic!("ci.yml missing job '{job_id}'"));
    match next_job_id {
        Some(next) => {
            let next_header = format!("\n  {next}:\n");
            let rel_end = ci_yaml[start..]
                .find(&next_header)
                .unwrap_or_else(|| panic!("ci.yml missing job '{next}' after '{job_id}'"));
            &ci_yaml[start..start + rel_end]
        }
        None => &ci_yaml[start..],
    }
}

fn assert_parallel_dag_invariants(ci_yaml: &str) {
    let preflight = slice_job_block(ci_yaml, "pre-flight", Some("build-and-test"));
    let build_and_test = slice_job_block(ci_yaml, "build-and-test", Some("compile-papers"));
    let compile_papers = slice_job_block(ci_yaml, "compile-papers", Some("feature-matrix"));
    let feature_matrix = slice_job_block(ci_yaml, "feature-matrix", Some("package"));
    let package = slice_job_block(ci_yaml, "package", Some("deploy-pages"));
    let deploy_pages = slice_job_block(ci_yaml, "deploy-pages", Some("publish-release"));
    let publish_release = slice_job_block(ci_yaml, "publish-release", None);

    for (job_name, block) in [
        ("pre-flight", preflight),
        ("feature-matrix", feature_matrix),
    ] {
        assert!(
            !block.contains("\n    needs:"),
            "{job_name} must start immediately without job-level needs"
        );
    }
    assert!(!preflight.contains("\n    outputs:"));
    assert!(build_and_test.contains("\n    needs: detect-changes\n"));
    assert!(
        build_and_test
            .contains("include: ${{ fromJSON(needs.detect-changes.outputs.native_matrix) }}")
    );
    assert!(build_and_test.contains(
        "continue-on-error: ${{ matrix.desktop && !startsWith(github.ref, 'refs/tags/v') }}"
    ));
    assert!(compile_papers.contains("\n    needs: detect-changes\n"));
    assert!(compile_papers.contains("needs.detect-changes.outputs.papers_changed == 'true'"));
    assert!(package.contains("\n    needs: detect-changes\n"));
    assert!(package.contains("needs.detect-changes.outputs.installer_changed == 'true'"));
    assert!(package.contains("IS_PRERELEASE: ${{ needs.detect-changes.outputs.is_prerelease }}"));
    assert!(package.contains("WIX_VERSION: ${{ needs.detect-changes.outputs.wix_version }}"));
    assert!(!ci_yaml.contains("needs.pre-flight.outputs."));
    assert!(
        deploy_pages.contains(
            "\n    needs: [pre-flight, build-and-test, compile-papers, feature-matrix]\n"
        )
    );
    assert!(publish_release.contains(
        "\n    needs: [pre-flight, build-and-test, package, compile-papers, feature-matrix]\n"
    ));
    assert!(publish_release.contains("scope: preflight\n          profile: dev"));
}

pub(super) fn assert_detect_changes_invariants(ci_yaml: &str, step: &str) {
    for env_binding in ["EVENT_NAME:", "BASE_SHA:", "HEAD_SHA:"] {
        assert!(
            step.contains(env_binding),
            "Detect Scoped Change Paths must bind {env_binding} in env:"
        );
    }
    let default_papers = step
        .find("papers_changed=\"true\"")
        .expect("missing default papers_changed=true");
    let default_installer = step
        .find("installer_changed=\"true\"")
        .expect("missing default installer_changed=true");
    let pr_guard = step
        .find("if [ \"${EVENT_NAME:-}\" = \"pull_request\" ]; then")
        .expect("missing pull_request guard");
    let reset_papers = step
        .find("papers_changed=\"false\"")
        .expect("missing pull_request papers_changed=false reset");
    let reset_installer = step
        .find("installer_changed=\"false\"")
        .expect("missing pull_request installer_changed=false reset");
    assert!(default_papers < pr_guard && default_installer < pr_guard);
    assert!(reset_papers > pr_guard && reset_installer > pr_guard);
    assert!(
        !ci_yaml.contains("No GUI or workspace manifest changes detected"),
        "stale package skip message must be updated"
    );
    assert!(ci_yaml.contains("MATRIX_TARGET: ${{ matrix.target }}"));
    assert!(ci_yaml.contains("skipping ${MATRIX_TARGET} installer build on pull_request."));
    assert_parallel_dag_invariants(ci_yaml);
}

pub(super) fn assert_workspace_members_parity(blocks: &[Vec<CaseArm>], root_toml: &str) {
    let members = parse_workspace_members(root_toml);
    assert!(
        members.len() >= 20,
        "expected full root workspace member list, got {}",
        members.len()
    );
    let mut linked_count = 0usize;
    for member in &members {
        let probe = format!("{member}/Cargo.toml");
        let (_, installer) = evaluate_pull_request_diff(blocks, &[&probe]);
        if NON_INSTALLER_MEMBERS.contains(&member.as_str()) {
            assert!(
                !installer,
                "non-linked workspace member {member} should not set installer_changed"
            );
        } else {
            linked_count += 1;
            assert!(
                installer,
                "linked workspace member {member} must set installer_changed"
            );
        }
    }
    assert!(
        linked_count >= 17,
        "expected math_explorer_gui plus 16 linked crates, got {linked_count}"
    );
    assert_eq!(
        evaluate_pull_request_diff(blocks, &["crates/future_new_crate/src/lib.rs"]),
        (false, true),
        "new crates under crates/* must automatically trigger installer_changed"
    );
}

pub(super) fn assert_scoped_triggers_and_skips(blocks: &[Vec<CaseArm>]) {
    let expected_cases: &[(&str, bool, bool)] = &[
        ("LICENSE", false, true),
        ("apps/math_explorer_gui/src/main.rs", false, true),
        ("apps/oxidize-cli/src/main.rs", false, true),
        ("crates/04_bindings/oxidizemath/src/lib.rs", false, true),
        ("crates/04_bindings/math_explorer/src/lib.rs", false, true),
        ("egui_plot/src/lib.rs", false, true),
        ("crates/01_core/oxidize_core/build.rs", false, true),
        (
            "crates/00_foundation/verified_engine_macros/src/embed_theory.rs",
            false,
            true,
        ),
        ("papers/cera_framework.tex", true, true),
        ("apps/xtask/Cargo.toml", true, false),
        ("apps/xtask/README.md", true, false),
        ("apps/xtask/src/main.rs", true, false),
        ("apps/xtask/src/papers/bundle.rs", true, false),
        ("traceability.toml", true, false),
        ("Verification_Certificate.json", true, false),
        ("Cargo.toml", true, true),
        ("Cargo.lock", true, true),
        ("rust-toolchain.toml", true, true),
        (".cargo/config.toml", true, true),
        (".github/actions/cargo-cache/action.yml", true, true),
        (".github/workflows/ci.yml", true, true),
        ("docs/README.md", false, false),
        ("docs/adr/0001-dynamic-workspace-discovery.md", false, false),
        ("AGENTS.md", false, false),
        ("CONTRIBUTING.md", false, false),
        ("apps/unified_verification/src/main.rs", false, false),
        ("crates/04_bindings/markdown_tests/build.rs", false, false),
    ];
    for &(path, exp_papers, exp_installer) in expected_cases {
        assert_eq!(
            evaluate_pull_request_diff(blocks, &[path]),
            (exp_papers, exp_installer),
            "unexpected (papers_changed, installer_changed) for {path}"
        );
    }
    assert_eq!(
        evaluate_pull_request_diff(
            blocks,
            &[
                "crates/03_domains/domain_physics/src/lib.rs",
                "apps/unified_verification/src/main.rs",
                "docs/README.md",
            ],
        ),
        (false, true),
        "ignored trailing paths must not reset installer_changed once set"
    );
}

#[test]
fn test_glob_matcher_and_agents_md_parity() {
    assert!(glob_matches("crates/*", "crates/oxidize_core/src/lib.rs"));
    assert!(glob_matches(
        "apps/xtask/*",
        "apps/xtask/src/papers/bundle.rs"
    ));
    assert!(!glob_matches("papers/*", "docs/papers/notes.md"));

    let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let agents_md = fs::read_to_string(ws_root.join("AGENTS.md"))
        .expect("failed to read AGENTS.md")
        .replace("\r\n", "\n");
    for expected_doc_token in [
        "crates/04_bindings/markdown_tests/*",
        "apps/unified_verification/*",
        "crates/*",
        "apps/*",
        "egui_plot/*",
        "LICENSE",
        "papers/*",
        "apps/xtask/*",
        "Verification_Certificate.json",
        "rust-toolchain.toml",
        ".cargo/*",
        ".github/actions/cargo-cache/*",
        "direct merges to `main`",
    ] {
        assert!(
            agents_md.contains(expected_doc_token),
            "AGENTS.md missing scoped trigger note: {expected_doc_token}"
        );
    }
}

#[test]
#[cfg(unix)]
fn test_detect_changes_bash_case_equivalence() {
    use std::process::Command;

    let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let ci_yaml = fs::read_to_string(ws_root.join(".github/workflows/ci.yml"))
        .expect("failed to read ci.yml")
        .replace("\r\n", "\n");
    let (step, blocks) = extract_detect_changes_step(&ci_yaml);
    let loop_start = step
        .find("papers_changed=\"false\"")
        .expect("missing papers_changed reset");
    let loop_end = step
        .find("done <<< \"$changed_files\"")
        .expect("missing done <<< changed_files")
        + "done <<< \"$changed_files\"".len();
    let loop_body: String = step[loop_start..loop_end]
        .lines()
        .map(|l| l.strip_prefix("              ").unwrap_or(l))
        .collect::<Vec<_>>()
        .join("\n");
    let script = format!(
        "set -euo pipefail\n{loop_body}\nprintf '%s,%s' \"$papers_changed\" \"$installer_changed\"\n"
    );

    for sample in [
        "LICENSE",
        "crates/03_domains/domain_ai/src/lib.rs",
        "apps/unified_verification/src/main.rs",
        "crates/04_bindings/markdown_tests/build.rs",
        "apps/xtask/src/main.rs",
        "Verification_Certificate.json",
        ".cargo/config.toml",
        ".github/actions/cargo-cache/action.yml",
        "docs/README.md",
        "crates/03_domains/pure_math/src/lib.rs\napps/unified_verification/src/main.rs",
    ] {
        let out = Command::new("bash")
            .arg("-c")
            .arg(&script)
            .env("changed_files", sample)
            .output()
            .expect("failed to execute bash case parity check");
        assert!(out.status.success(), "bash snippet failed for {sample}");
        let lines: Vec<&str> = sample.lines().collect();
        let (exp_p, exp_i) = evaluate_pull_request_diff(&blocks, &lines);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("{exp_p},{exp_i}"),
            "bash vs Rust mismatch for {sample}"
        );
    }
}
