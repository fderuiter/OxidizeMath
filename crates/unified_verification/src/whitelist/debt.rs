use super::{PathMatchOptions, path_matches_lossy};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

const DEBT_PATH_OPTS: PathMatchOptions = PathMatchOptions {
    trim_dot_slash: true,
    require_non_empty: true,
    match_file_name: false,
};

/// Ratcheted verification debt caps and assertion density floors loaded from `[verification_debt]`.
#[derive(Deserialize, Debug, Clone, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VerificationDebtConfig {
    /// Workspace member directories explicitly exempt from requiring `theory_verification!`.
    #[serde(default)]
    pub exempt_crates: Vec<String>,
    /// Default per-file maximum `opt_out` count when not overridden in `file_opt_out_caps`.
    #[serde(default)]
    pub max_opt_outs_per_file: Option<usize>,
    /// Minimum required assertion density (`verified_asserts / verified_funcs`) for verified functions.
    pub min_verified_density: f64,
    /// Minimum required overall assertion density (`total_asserts / total_funcs`) for production code.
    #[serde(default)]
    pub min_total_density: f64,
    /// Optional minimum required assertion density (`unverified_asserts / unverified_funcs`).
    #[serde(default)]
    pub min_unverified_density: Option<f64>,
    /// Authorized `opt_out` reason strings mapped to workspace-wide maximum counts (`>= 1`).
    #[serde(default, alias = "reason_caps")]
    pub opt_out_reasons: BTreeMap<String, usize>,
    /// Per-crate maximum `opt_out` counts (`>= 1`; unlisted crates have an implicit cap of `0`).
    #[serde(default, alias = "crate_caps")]
    pub crate_opt_out_caps: BTreeMap<String, usize>,
    /// Per-file `opt_out` cap overrides (`>= 1`) for files allowed above `max_opt_outs_per_file`.
    #[serde(default, alias = "file_caps")]
    pub file_opt_out_caps: BTreeMap<String, usize>,
}

fn validate_positive_cap_map(
    map: &BTreeMap<String, usize>,
    section: &str,
    path: &Path,
) -> Result<(), String> {
    for (key, &cap) in map {
        if key.trim().is_empty() || cap == 0 {
            return Err(format!(
                "Invalid [{section}] entry `{key}` in {}: key must be non-empty and cap must be >= 1",
                path.display()
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_verification_debt_config(
    cfg: &VerificationDebtConfig,
    path: &Path,
) -> Result<(), String> {
    if cfg.exempt_crates.iter().any(|c| c.trim().is_empty()) {
        return Err(format!(
            "Invalid [verification_debt] exempt_crates in {}: entries must be non-empty",
            path.display()
        ));
    }
    if cfg.max_opt_outs_per_file == Some(0) {
        return Err(format!(
            "Invalid [verification_debt] max_opt_outs_per_file in {}: must be >= 1",
            path.display()
        ));
    }
    if !cfg.min_verified_density.is_finite()
        || cfg.min_verified_density < 0.0
        || !cfg.min_total_density.is_finite()
        || cfg.min_total_density < 0.0
        || cfg
            .min_unverified_density
            .is_some_and(|v| !v.is_finite() || v < 0.0)
    {
        return Err(format!(
            "Invalid [verification_debt] density floors in {}: values must be finite and >= 0.0",
            path.display()
        ));
    }
    validate_positive_cap_map(
        &cfg.opt_out_reasons,
        "verification_debt.opt_out_reasons",
        path,
    )?;
    validate_positive_cap_map(
        &cfg.crate_opt_out_caps,
        "verification_debt.crate_opt_out_caps",
        path,
    )?;
    validate_positive_cap_map(
        &cfg.file_opt_out_caps,
        "verification_debt.file_opt_out_caps",
        path,
    )?;
    Ok(())
}

/// Maps a normalized workspace-relative file path to its owning workspace crate key.
pub(crate) fn crate_prefix_for_file(file_path: &str) -> String {
    let norm = file_path.trim().replace('\\', "/");
    let trimmed = norm.trim_start_matches("./");
    let segs: Vec<&str> = trimmed.split('/').filter(|s| !s.is_empty()).collect();
    if segs.len() >= 3 && (segs[0] == "crates" || segs[0] == "apps") {
        format!("{}/{}", segs[0], segs[1])
    } else if !segs.is_empty() {
        segs[0].to_string()
    } else {
        trimmed.to_string()
    }
}

fn lookup_crate_cap(cfg: &VerificationDebtConfig, krate: &str) -> usize {
    let actual = Path::new(krate);
    cfg.crate_opt_out_caps
        .iter()
        .find(|(key, _)| path_matches_lossy(actual, key, DEBT_PATH_OPTS))
        .map_or(0, |(_, &cap)| cap)
}

fn lookup_file_cap(cfg: &VerificationDebtConfig, file: &str) -> Option<usize> {
    let actual = Path::new(file);
    cfg.file_opt_out_caps
        .iter()
        .find(|(key, _)| path_matches_lossy(actual, key, DEBT_PATH_OPTS))
        .map(|(_, &cap)| cap)
        .or(cfg.max_opt_outs_per_file)
}

/// Evaluates `m.opt_outs` against per-reason, per-crate, and per-file caps in `cfg`.
pub(crate) fn evaluate_opt_out_caps(
    cfg: &VerificationDebtConfig,
    opt_outs: &[(String, String)],
) -> Vec<String> {
    let mut violations = Vec::new();
    let mut by_reason: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_crate: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_file: BTreeMap<String, usize> = BTreeMap::new();

    for (raw_file, reason) in opt_outs {
        let norm_file = raw_file
            .replace('\\', "/")
            .trim_start_matches("./")
            .to_string();
        if reason.starts_with("Legacy function call:") {
            violations.push(format!(
                "Unauthorized legacy function in {norm_file}: {reason}"
            ));
            continue;
        }
        *by_reason.entry(reason.clone()).or_insert(0) += 1;
        *by_crate
            .entry(crate_prefix_for_file(&norm_file))
            .or_insert(0) += 1;
        *by_file.entry(norm_file).or_insert(0) += 1;
    }

    for (reason, count) in by_reason {
        match cfg.opt_out_reasons.get(&reason) {
            None => violations.push(format!(
                "Unauthorized opt_out reason '{reason}' ({count} occurrence(s); not in [verification_debt.opt_out_reasons])"
            )),
            Some(&cap) if count > cap => violations.push(format!(
                "Exceeded opt_out reason cap for '{reason}': found {count}, cap is {cap}"
            )),
            Some(_) => {}
        }
    }

    for (krate, count) in by_crate {
        let cap = lookup_crate_cap(cfg, &krate);
        if count > cap {
            violations.push(format!(
                "Exceeded crate opt_out cap for '{krate}': found {count}, cap is {cap}"
            ));
        }
    }

    for (file, count) in by_file {
        if let Some(cap) = lookup_file_cap(cfg, &file)
            && count > cap
        {
            violations.push(format!(
                "Exceeded file opt_out cap for '{file}': found {count}, cap is {cap}"
            ));
        }
    }

    violations
}

/// Returns `true` when workspace member `member` is listed in `exempt_crates`.
pub(crate) fn is_exempt_crate(member: &str, exempt_crates: &[String]) -> bool {
    let member_path = Path::new(member.trim_end_matches('/'));
    exempt_crates
        .iter()
        .any(|exempt| path_matches_lossy(member_path, exempt, DEBT_PATH_OPTS))
}
