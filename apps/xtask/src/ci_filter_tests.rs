use crate::staged::parse_workspace_members;
use std::fs;
use std::path::Path;

const NON_INSTALLER_MEMBERS: &[&str] = &[
    "apps/xtask",
    "crates/markdown_tests",
    "crates/unified_verification",
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
    let start = ci_yaml
        .find("- name: Detect Scoped Change Paths")
        .expect("ci.yml missing Detect Scoped Change Paths step");
    let rest = &ci_yaml[start..];
    let end = rest
        .find("- name: Set up Rust")
        .expect("ci.yml missing Set up Rust step after Detect Scoped Change Paths");
    let step = &rest[..end];

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
    assert!(ci_yaml.contains("skipping ${{ matrix.target }} installer build on pull_request."));
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
        ("math_explorer_gui/src/main.rs", false, true),
        ("math_explorer/src/lib.rs", false, true),
        ("egui_plot/src/lib.rs", false, true),
        ("crates/oxidize_core/build.rs", false, true),
        (
            "crates/verified_engine_macros/src/embed_theory.rs",
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
        ("crates/unified_verification/src/main.rs", false, false),
        ("crates/markdown_tests/build.rs", false, false),
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
                "crates/domain_physics/src/lib.rs",
                "crates/unified_verification/src/main.rs",
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
        "crates/markdown_tests/*",
        "crates/unified_verification/*",
        "crates/*",
        "math_explorer_gui/*",
        "math_explorer/*",
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
        "crates/domain_ai/src/lib.rs",
        "crates/unified_verification/src/main.rs",
        "crates/markdown_tests/build.rs",
        "apps/xtask/src/main.rs",
        "Verification_Certificate.json",
        ".cargo/config.toml",
        ".github/actions/cargo-cache/action.yml",
        "docs/README.md",
        "crates/pure_math/src/lib.rs\ncrates/unified_verification/src/main.rs",
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
