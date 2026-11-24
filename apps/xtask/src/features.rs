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

pub(crate) const DOMAIN_FEATURES: &[&str] = &[
    "pure_math",
    "applied",
    "ai",
    "biology",
    "climate",
    "epidemiology",
    "physics",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FeatureSelector {
    CoreOnly,
    AllFeatures,
    Features(Vec<String>),
}

impl FeatureSelector {
    fn banner(&self) -> String {
        match self {
            Self::CoreOnly => "=== Running core-only (no features) ===".to_string(),
            Self::AllFeatures => "=== Running all features ===".to_string(),
            Self::Features(list) => {
                format!("=== Running tests for feature(s): {} ===", list.join(","))
            }
        }
    }

    fn cargo_feature_flags(&self) -> Vec<String> {
        match self {
            Self::CoreOnly => vec!["--no-default-features".to_string()],
            Self::AllFeatures => vec!["--all-features".to_string()],
            Self::Features(list) => vec![
                "--no-default-features".to_string(),
                "--features".to_string(),
                list.join(","),
            ],
        }
    }

    fn gui_feature_flags(&self) -> Vec<String> {
        match self {
            Self::Features(list) if list.iter().any(|f| f == "serde") => {
                let gui_features: Vec<&str> = list
                    .iter()
                    .map(String::as_str)
                    .filter(|&f| f != "serde")
                    .collect();
                if gui_features.is_empty() {
                    vec!["--no-default-features".to_string()]
                } else {
                    vec![
                        "--no-default-features".to_string(),
                        "--features".to_string(),
                        gui_features.join(","),
                    ]
                }
            }
            _ => self.cargo_feature_flags(),
        }
    }
}

pub(crate) fn parse_selectors(args: &[String]) -> Vec<FeatureSelector> {
    if args.is_empty() {
        let mut selectors = Vec::with_capacity(DOMAIN_FEATURES.len() + 2);
        selectors.push(FeatureSelector::CoreOnly);
        for feature in DOMAIN_FEATURES {
            selectors.push(FeatureSelector::Features(vec![(*feature).to_string()]));
        }
        selectors.push(FeatureSelector::AllFeatures);
        return selectors;
    }

    let mut parsed_features: Vec<String> = Vec::new();
    for raw in args {
        for piece in raw.split(|c: char| c == ',' || c.is_whitespace()) {
            let token = piece.trim();
            if token.is_empty() || token == "--features" {
                continue;
            }
            if token == "all-features" || token == "--all-features" {
                return vec![FeatureSelector::AllFeatures];
            }
            if token == "core-only" || token == "--no-default-features" {
                continue;
            }
            if !parsed_features.iter().any(|existing| existing == token) {
                parsed_features.push(token.to_string());
            }
        }
    }

    if parsed_features.is_empty() {
        vec![FeatureSelector::CoreOnly]
    } else {
        vec![FeatureSelector::Features(parsed_features)]
    }
}

pub(crate) fn cargo_invocations_for_selector(selector: &FeatureSelector) -> Vec<Vec<String>> {
    let flags = selector.cargo_feature_flags();
    let gui_flags = selector.gui_feature_flags();

    let mut lib_test = vec![
        "test".to_string(),
        "-p".to_string(),
        "math_explorer".to_string(),
    ];
    lib_test.extend(flags.iter().cloned());

    let mut lib_benches = vec![
        "test".to_string(),
        "-p".to_string(),
        "math_explorer".to_string(),
        "--benches".to_string(),
    ];
    lib_benches.extend(flags);

    let mut gui_check = vec![
        "check".to_string(),
        "-p".to_string(),
        "math_explorer_gui".to_string(),
        "--bins".to_string(),
        "--tests".to_string(),
    ];
    gui_check.extend(gui_flags.iter().cloned());

    let mut gui_clippy = vec![
        "clippy".to_string(),
        "--no-deps".to_string(),
        "-p".to_string(),
        "math_explorer_gui".to_string(),
        "--bins".to_string(),
        "--tests".to_string(),
    ];
    gui_clippy.extend(gui_flags);
    gui_clippy.extend(["--".to_string(), "-D".to_string(), "warnings".to_string()]);

    vec![lib_test, lib_benches, gui_check, gui_clippy]
}

pub(crate) fn serde_lib_invocations_for_selector(selector: &FeatureSelector) -> Vec<Vec<String>> {
    let serde_feature = match selector {
        FeatureSelector::CoreOnly => "serde".to_string(),
        FeatureSelector::AllFeatures => return Vec::new(),
        FeatureSelector::Features(list) if list.iter().any(|f| f == "serde") => return Vec::new(),
        FeatureSelector::Features(list) => format!("{},serde", list.join(",")),
    };
    vec![
        vec![
            "test".to_string(),
            "-p".to_string(),
            "math_explorer".to_string(),
            "--no-default-features".to_string(),
            "--features".to_string(),
            serde_feature.clone(),
        ],
        vec![
            "test".to_string(),
            "-p".to_string(),
            "math_explorer".to_string(),
            "--benches".to_string(),
            "--no-default-features".to_string(),
            "--features".to_string(),
            serde_feature,
        ],
    ]
}

fn check_selector_serde_isolation(selector: &FeatureSelector) {
    match selector {
        FeatureSelector::CoreOnly => assert_no_serde_in_normal_deps(None),
        FeatureSelector::Features(list) if !list.iter().any(|f| f == "serde") => {
            assert_no_serde_in_normal_deps(Some(&list.join(",")));
        }
        FeatureSelector::AllFeatures | FeatureSelector::Features(_) => {}
    }
}

pub(crate) fn test_features(args: &[String]) {
    let selectors = parse_selectors(args);
    let is_full_matrix = args.is_empty();
    if is_full_matrix {
        assert_no_serde_in_normal_deps(Some(&DOMAIN_FEATURES.join(",")));
    }
    for selector in &selectors {
        println!("{}", selector.banner());
        check_selector_serde_isolation(selector);
        for invocation in cargo_invocations_for_selector(selector)
            .into_iter()
            .chain(serde_lib_invocations_for_selector(selector))
        {
            let arg_refs: Vec<&str> = invocation.iter().map(String::as_str).collect();
            run_cmd("cargo", &arg_refs);
        }
    }
    if is_full_matrix {
        println!("All feature combinations passed successfully!");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    fn strs(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn test_core_only_command_generation() {
        for raw in [
            strs(&["core-only"]),
            strs(&[""]),
            strs(&["--no-default-features"]),
        ] {
            let selectors = parse_selectors(&raw);
            assert_eq!(selectors, vec![FeatureSelector::CoreOnly]);
            let cmds = cargo_invocations_for_selector(&selectors[0]);
            assert_eq!(
                cmds,
                vec![
                    strs(&["test", "-p", "math_explorer", "--no-default-features"]),
                    strs(&[
                        "test",
                        "-p",
                        "math_explorer",
                        "--benches",
                        "--no-default-features",
                    ]),
                    strs(&[
                        "check",
                        "-p",
                        "math_explorer_gui",
                        "--bins",
                        "--tests",
                        "--no-default-features",
                    ]),
                    strs(&[
                        "clippy",
                        "--no-deps",
                        "-p",
                        "math_explorer_gui",
                        "--bins",
                        "--tests",
                        "--no-default-features",
                        "--",
                        "-D",
                        "warnings",
                    ]),
                ]
            );
        }
    }

    #[test]
    fn test_single_feature_command_generation() {
        let selectors = parse_selectors(&strs(&["physics"]));
        assert_eq!(
            selectors,
            vec![FeatureSelector::Features(strs(&["physics"]))]
        );
        let cmds = cargo_invocations_for_selector(&selectors[0]);
        assert_eq!(
            cmds,
            vec![
                strs(&[
                    "test",
                    "-p",
                    "math_explorer",
                    "--no-default-features",
                    "--features",
                    "physics",
                ]),
                strs(&[
                    "test",
                    "-p",
                    "math_explorer",
                    "--benches",
                    "--no-default-features",
                    "--features",
                    "physics",
                ]),
                strs(&[
                    "check",
                    "-p",
                    "math_explorer_gui",
                    "--bins",
                    "--tests",
                    "--no-default-features",
                    "--features",
                    "physics",
                ]),
                strs(&[
                    "clippy",
                    "--no-deps",
                    "-p",
                    "math_explorer_gui",
                    "--bins",
                    "--tests",
                    "--no-default-features",
                    "--features",
                    "physics",
                    "--",
                    "-D",
                    "warnings",
                ]),
            ]
        );
    }

    fn words(line: &str) -> Vec<String> {
        line.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn test_multi_feature_tokens_and_csv_command_generation() {
        let from_tokens = parse_selectors(&strs(&["biology", "epidemiology", "biology"]));
        let from_csv = parse_selectors(&strs(&["--features", "biology,epidemiology"]));
        let expected = vec![FeatureSelector::Features(strs(&[
            "biology",
            "epidemiology",
        ]))];
        assert_eq!(from_tokens, expected);
        assert_eq!(from_csv, expected);

        let cmds = cargo_invocations_for_selector(&from_tokens[0]);
        assert_eq!(
            cmds,
            vec![
                words(
                    "test -p math_explorer --no-default-features --features biology,epidemiology"
                ),
                words(
                    "test -p math_explorer --benches --no-default-features --features biology,epidemiology"
                ),
                words(
                    "check -p math_explorer_gui --bins --tests --no-default-features --features biology,epidemiology"
                ),
                words(
                    "clippy --no-deps -p math_explorer_gui --bins --tests --no-default-features --features biology,epidemiology -- -D warnings"
                ),
            ]
        );
    }

    #[test]
    fn test_all_features_and_default_sweep_generation() {
        for raw in [strs(&["all-features"]), strs(&["--all-features"])] {
            let selectors = parse_selectors(&raw);
            assert_eq!(selectors, vec![FeatureSelector::AllFeatures]);
            let cmds = cargo_invocations_for_selector(&selectors[0]);
            assert_eq!(
                cmds,
                vec![
                    strs(&["test", "-p", "math_explorer", "--all-features"]),
                    strs(&["test", "-p", "math_explorer", "--benches", "--all-features"]),
                    strs(&[
                        "check",
                        "-p",
                        "math_explorer_gui",
                        "--bins",
                        "--tests",
                        "--all-features",
                    ]),
                    strs(&[
                        "clippy",
                        "--no-deps",
                        "-p",
                        "math_explorer_gui",
                        "--bins",
                        "--tests",
                        "--all-features",
                        "--",
                        "-D",
                        "warnings",
                    ]),
                ]
            );
        }

        let default_sweep = parse_selectors(&[]);
        assert_eq!(default_sweep.len(), DOMAIN_FEATURES.len() + 2);
        assert_eq!(default_sweep.first(), Some(&FeatureSelector::CoreOnly));
        assert_eq!(default_sweep.last(), Some(&FeatureSelector::AllFeatures));
    }

    #[test]
    fn test_ci_feature_matrix_installs_clippy_component() {
        let ci_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows/ci.yml");
        let ci_yaml = fs::read_to_string(&ci_path)
            .expect("failed to read ci.yml")
            .replace("\r\n", "\n");
        let start = ci_yaml
            .find("  feature-matrix:\n")
            .expect("ci.yml missing feature-matrix job");
        let section = &ci_yaml[start..];
        let end = section.find("\n  package:\n").unwrap_or(section.len());
        let job_yaml = &section[..end];
        assert!(
            job_yaml.contains("components: clippy"),
            "feature-matrix must install the clippy rustup component"
        );
        assert!(job_yaml.contains("scope: features-shared"));
    }

    #[test]
    fn test_serde_companion_and_gui_flag_filtering() {
        assert_eq!(
            serde_lib_invocations_for_selector(&FeatureSelector::CoreOnly),
            vec![
                words("test -p math_explorer --no-default-features --features serde"),
                words("test -p math_explorer --benches --no-default-features --features serde"),
            ]
        );
        let phys = FeatureSelector::Features(strs(&["physics"]));
        assert_eq!(
            serde_lib_invocations_for_selector(&phys),
            vec![
                words("test -p math_explorer --no-default-features --features physics,serde"),
                words(
                    "test -p math_explorer --benches --no-default-features --features physics,serde"
                ),
            ]
        );
        assert!(serde_lib_invocations_for_selector(&FeatureSelector::AllFeatures).is_empty());

        let explicit_serde = FeatureSelector::Features(strs(&["serde"]));
        assert!(serde_lib_invocations_for_selector(&explicit_serde).is_empty());
        let cmds = cargo_invocations_for_selector(&explicit_serde);
        assert_eq!(
            cmds[2],
            words("check -p math_explorer_gui --bins --tests --no-default-features")
        );
    }
}
