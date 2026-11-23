use serde::Deserialize;

fn default_min_verified_density() -> f64 {
    0.40
}

fn default_coverage_tolerance_pct() -> f64 {
    1.5
}

/// Configuration for `verify-suite` and `check-certificate` loaded from `[verify_suite]`.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct VerifySuiteConfig {
    /// Minimum required combined assertion density across verified functions (default `0.40`).
    #[serde(default = "default_min_verified_density")]
    pub min_verified_assertion_density: f64,
    /// Ratcheted minimum production assertion density across all non-test functions.
    #[serde(default)]
    pub min_total_production_density: f64,
    /// Ratcheted minimum production assertion density across `#[verified]` non-test functions.
    #[serde(default)]
    pub min_verified_production_density: f64,
    /// Allowed percentage-point tolerance for `native_execution_coverage_pct` in `--check` mode.
    #[serde(default = "default_coverage_tolerance_pct")]
    pub coverage_tolerance_pct: f64,
    /// Non-domain workspace crates permitted to omit `theory_verification!` / `stochastic_signature_verification!`.
    #[serde(default)]
    pub unverified_modules: Vec<String>,
    /// Authorized opt-out attributes (`path` + `reason`) permitted without failing `verify-suite`.
    #[serde(default)]
    pub opt_outs: Vec<OptOutAllowEntry>,
}

impl Default for VerifySuiteConfig {
    fn default() -> Self {
        Self {
            min_verified_assertion_density: default_min_verified_density(),
            min_total_production_density: 0.0,
            min_verified_production_density: 0.0,
            coverage_tolerance_pct: default_coverage_tolerance_pct(),
            unverified_modules: Vec::new(),
            opt_outs: Vec::new(),
        }
    }
}

impl VerifySuiteConfig {
    /// Returns `true` when `member` is explicitly listed as an allowed non-domain unverified crate.
    pub fn is_unverified_module_allowed(&self, member: &str) -> bool {
        let norm_member = member.replace('\\', "/");
        let clean_member = norm_member.trim_start_matches("./").trim_end_matches('/');
        self.unverified_modules.iter().any(|allowed| {
            let norm_allowed = allowed.replace('\\', "/");
            norm_allowed.trim_start_matches("./").trim_end_matches('/') == clean_member
        })
    }

    /// Returns `true` when `(file, reason)` matches an authorized `[[verify_suite.opt_outs]]` rule.
    pub fn is_opt_out_allowed(&self, file: &str, reason: &str) -> bool {
        let norm_file = file.replace('\\', "/");
        let clean_file = norm_file.trim_start_matches("./");
        self.opt_outs.iter().any(|entry| {
            if entry.reason != reason {
                return false;
            }
            let norm_pat = entry.path.replace('\\', "/");
            let clean_pat = norm_pat.trim_start_matches("./");
            if clean_pat.is_empty() {
                return false;
            }
            if clean_pat.ends_with('/') {
                clean_file.starts_with(clean_pat)
            } else {
                clean_file == clean_pat
            }
        })
    }
}

/// Authorized opt-out rule matching an exact file path or directory prefix and opt-out reason.
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OptOutAllowEntry {
    /// Exact workspace-relative file path or directory prefix ending in `/`.
    pub path: String,
    /// Exact `opt_out` reason string permitted at `path`.
    pub reason: String,
}
