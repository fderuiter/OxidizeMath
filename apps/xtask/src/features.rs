/// Optional domain feature flags shared across `math_explorer` and `math_explorer_gui`.
pub const DOMAIN_FEATURES: &[&str] = &[
    "pure_math",
    "applied",
    "ai",
    "biology",
    "climate",
    "epidemiology",
    "physics",
];

/// Workspace crates validated across the domain feature matrix.
pub const FEATURE_PACKAGES: &[&str] = &["math_explorer", "math_explorer_gui"];

/// Builds `cargo test` argument vectors for a single feature selector (`core-only` or `<feature>`).
pub fn single_feature_commands(feature: &str) -> Vec<Vec<String>> {
    let is_core_only = feature == "core-only" || feature.is_empty();
    FEATURE_PACKAGES
        .iter()
        .map(|pkg| {
            if is_core_only {
                vec![
                    "test".to_string(),
                    "-p".to_string(),
                    (*pkg).to_string(),
                    "--no-default-features".to_string(),
                ]
            } else {
                vec![
                    "test".to_string(),
                    "-p".to_string(),
                    (*pkg).to_string(),
                    "--no-default-features".to_string(),
                    "--features".to_string(),
                    feature.to_string(),
                ]
            }
        })
        .collect()
}

/// Builds `cargo test` argument vectors for `--all-features` across feature-gated packages.
pub fn all_features_commands() -> Vec<Vec<String>> {
    FEATURE_PACKAGES
        .iter()
        .map(|pkg| {
            vec![
                "test".to_string(),
                "-p".to_string(),
                (*pkg).to_string(),
                "--all-features".to_string(),
            ]
        })
        .collect()
}

/// Resolves the ordered list of `cargo` argument vectors for `xtask test-features`.
pub fn build_test_feature_commands(args: &[String]) -> Vec<Vec<String>> {
    if let Some(feature) = args.first() {
        return single_feature_commands(feature);
    }

    let mut commands = Vec::with_capacity((DOMAIN_FEATURES.len() + 2) * FEATURE_PACKAGES.len());
    commands.extend(single_feature_commands("core-only"));
    for feature in DOMAIN_FEATURES {
        commands.extend(single_feature_commands(feature));
    }
    commands.extend(all_features_commands());
    commands
}

/// Executes `cargo test` across `math_explorer` and `math_explorer_gui` for the requested feature mode.
pub fn test_features(args: &[String]) {
    let commands = build_test_feature_commands(args);
    if args.is_empty() {
        println!("=== Running core-only, domain features, and all-features matrix ===");
    }
    for cmd_args in &commands {
        println!(">>> cargo {}", cmd_args.join(" "));
        let arg_refs: Vec<&str> = cmd_args.iter().map(String::as_str).collect();
        super::run_cmd("cargo", &arg_refs);
    }
    if args.is_empty() {
        println!("All feature combinations passed successfully!");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_strings(slice: &[&str]) -> Vec<String> {
        slice.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn test_core_only_and_empty_selector_arguments() {
        let expected = vec![
            to_strings(&["test", "-p", "math_explorer", "--no-default-features"]),
            to_strings(&["test", "-p", "math_explorer_gui", "--no-default-features"]),
        ];

        assert_eq!(
            build_test_feature_commands(&["core-only".to_string()]),
            expected
        );
        assert_eq!(build_test_feature_commands(&[String::new()]), expected);
    }

    #[test]
    fn test_single_domain_feature_arguments() {
        for &feature in DOMAIN_FEATURES {
            let cmds = build_test_feature_commands(&[feature.to_string()]);
            assert_eq!(cmds.len(), 2);
            assert_eq!(
                cmds[0],
                to_strings(&[
                    "test",
                    "-p",
                    "math_explorer",
                    "--no-default-features",
                    "--features",
                    feature
                ])
            );
            assert_eq!(
                cmds[1],
                to_strings(&[
                    "test",
                    "-p",
                    "math_explorer_gui",
                    "--no-default-features",
                    "--features",
                    feature
                ])
            );
        }
    }

    #[test]
    fn test_full_suite_no_args_includes_core_domains_and_all_features() {
        let cmds = build_test_feature_commands(&[]);
        // (1 core-only + 7 domain features + 1 all-features) * 2 crates = 18 commands
        assert_eq!(cmds.len(), 18);

        assert_eq!(
            cmds[0],
            to_strings(&["test", "-p", "math_explorer", "--no-default-features"])
        );
        assert_eq!(
            cmds[1],
            to_strings(&["test", "-p", "math_explorer_gui", "--no-default-features"])
        );

        for (idx, &feature) in DOMAIN_FEATURES.iter().enumerate() {
            let base = 2 + idx * 2;
            assert_eq!(
                cmds[base],
                to_strings(&[
                    "test",
                    "-p",
                    "math_explorer",
                    "--no-default-features",
                    "--features",
                    feature
                ])
            );
            assert_eq!(
                cmds[base + 1],
                to_strings(&[
                    "test",
                    "-p",
                    "math_explorer_gui",
                    "--no-default-features",
                    "--features",
                    feature
                ])
            );
        }

        assert_eq!(
            cmds[16],
            to_strings(&["test", "-p", "math_explorer", "--all-features"])
        );
        assert_eq!(
            cmds[17],
            to_strings(&["test", "-p", "math_explorer_gui", "--all-features"])
        );
    }
}
