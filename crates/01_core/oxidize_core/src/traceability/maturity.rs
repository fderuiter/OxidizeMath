//! Explicit maturity tier registry parsed from `traceability.toml` for host and WebAssembly builds.

use scientific_metadata::theory::MaturityTier;
use std::collections::BTreeMap;
use std::path::Path;

const EMBEDDED_TRACEABILITY_TOML: &str =
    include_str!(concat!(env!("OUT_DIR"), "/traceability.toml"));
const SKIP_PATH_SEGMENTS: &[&str] = &[
    "crates",
    "apps",
    "00_foundation",
    "01_core",
    "02_engines",
    "03_domains",
    "04_bindings",
    "src",
    "tests",
    "tabs",
    "applied",
    "physics",
];

/// Expected library and mathematical crates that must declare an explicit tier under `[crates]`.
pub const EXPECTED_WORKSPACE_CRATES: &[&str] = &[
    "diagnostics",
    "domain_ai",
    "domain_applied",
    "domain_biology",
    "domain_climate",
    "domain_epidemiology",
    "domain_physics",
    "engine_diffeq",
    "engine_linalg",
    "engine_pde",
    "federated_registry",
    "math_commons",
    "math_diagnostics",
    "math_explorer",
    "math_rng",
    "oxidize_core",
    "oxidizemath",
    "pure_math",
    "scientific_metadata",
    "verified_engine",
    "verified_engine_macros",
];

/// Parsed representation of `[semver_lock]`, `[links]`, `[crates]`, and `[modules]` in `traceability.toml`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MaturityRegistry {
    /// Workspace SemVer version recorded when `[modules]` / `[crates]` `Certified` signatures were locked.
    pub certified_baseline_version: Option<String>,
    /// Explicit crate-level maturity tiers from `[crates]`.
    pub crates: BTreeMap<String, MaturityTier>,
    /// Explicit module-level maturity tiers from `[modules]`.
    pub modules: BTreeMap<String, MaturityTier>,
    /// Registered module-to-paper links from `[links]`.
    pub links: BTreeMap<String, String>,
    /// Diagnostic messages for malformed tier strings in `[crates]` or `[modules]`.
    pub invalid_entries: Vec<String>,
}

fn strip_comment(line: &str) -> &str {
    let mut in_double = false;
    let mut in_single = false;
    let mut escaped = false;
    for (idx, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_double {
            escaped = true;
            continue;
        }
        if ch == '"' && !in_single {
            in_double = !in_double;
        } else if ch == '\'' && !in_double {
            in_single = !in_single;
        } else if ch == '#' && !in_double && !in_single {
            return &line[..idx];
        }
    }
    line
}

fn unquote(raw: &str) -> Option<&str> {
    let trimmed = raw.trim();
    if trimmed.len() >= 2
        && ((trimmed.starts_with('"') && trimmed.ends_with('"'))
            || (trimmed.starts_with('\'') && trimmed.ends_with('\'')))
    {
        Some(&trimmed[1..trimmed.len() - 1])
    } else {
        None
    }
}

fn clean_key(raw: &str) -> &str {
    unquote(raw).unwrap_or_else(|| raw.trim())
}

impl MaturityRegistry {
    /// Parses `traceability.toml` text into a [`MaturityRegistry`] without requiring `toml` on `wasm32`.
    pub fn parse_toml_str(content: &str) -> Self {
        let mut reg = Self::default();
        let mut section = "";
        for (idx, raw_line) in content.lines().enumerate() {
            let line = strip_comment(raw_line).trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                section = &line[1..line.len() - 1];
                continue;
            }
            let Some((raw_k, raw_v)) = line.split_once('=') else {
                continue;
            };
            let key = clean_key(raw_k).to_string();
            let Some(val) = unquote(raw_v) else {
                if matches!(section, "crates" | "modules") {
                    reg.invalid_entries.push(format!(
                        "line {}: `[{}].{}` must be a quoted tier string",
                        idx + 1,
                        section,
                        key
                    ));
                }
                continue;
            };
            match section {
                "links" => {
                    reg.links.insert(key, val.to_string());
                }
                "semver_lock" if key == "certified_baseline_version" => {
                    reg.certified_baseline_version = Some(val.to_string());
                }
                "crates" => match MaturityTier::parse_label(val) {
                    Some(tier) => {
                        reg.crates.insert(key, tier);
                    }
                    None => reg.invalid_entries.push(format!(
                        "line {}: invalid `[crates].{} = \"{}\"` (expected Experimental, Validated, or Certified)",
                        idx + 1,
                        key,
                        val
                    )),
                },
                "modules" => match MaturityTier::parse_label(val) {
                    Some(tier) => {
                        reg.modules.insert(key, tier);
                    }
                    None => reg.invalid_entries.push(format!(
                        "line {}: invalid `[modules].{} = \"{}\"` (expected Experimental, Validated, or Certified)",
                        idx + 1,
                        key,
                        val
                    )),
                },
                _ => {}
            }
        }
        reg
    }

    /// Returns the compile-time embedded workspace [`MaturityRegistry`].
    pub fn embedded() -> Self {
        Self::parse_toml_str(EMBEDDED_TRACEABILITY_TOML)
    }

    /// Loads `traceability.toml` from `workspace_root` on native targets, or falls back to embedded TOML.
    pub fn load_or_embedded(workspace_root: Option<&Path>) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(root) = workspace_root {
                let candidate = root.join("traceability.toml");
                if let Ok(text) = std::fs::read_to_string(candidate) {
                    return Self::parse_toml_str(&text);
                }
            }
            if let Ok(text) = std::fs::read_to_string("traceability.toml") {
                return Self::parse_toml_str(&text);
            }
            if let Some(root) = crate::workspace::find_workspace_root()
                && let Ok(text) = std::fs::read_to_string(root.join("traceability.toml"))
            {
                return Self::parse_toml_str(&text);
            }
        }
        #[cfg(target_arch = "wasm32")]
        let _ = workspace_root;
        Self::embedded()
    }

    /// Looks up the explicit [`MaturityTier`] for `crate_name`.
    pub fn crate_tier(&self, crate_name: &str) -> Option<MaturityTier> {
        self.crates.get(crate_name).copied()
    }

    /// Looks up the explicit [`MaturityTier`] for `module_name`.
    pub fn module_tier(&self, module_name: &str) -> Option<MaturityTier> {
        self.modules.get(module_name).copied()
    }

    /// Returns the crate name owning a normalized repository-relative or package-keyed path, if any.
    pub fn crate_for_repo_path(rel_path: &str) -> Option<&str> {
        let norm = rel_path.trim_start_matches("./");
        if let Some((pkg, _)) = norm.split_once(':')
            && !pkg.is_empty()
            && !pkg.contains('/')
        {
            return Some(pkg);
        }
        if let Some(rest) = norm.strip_prefix("crates/") {
            let mut parts = rest.split('/').filter(|s| !s.is_empty());
            let first = parts.next()?;
            if matches!(
                first,
                "00_foundation" | "01_core" | "02_engines" | "03_domains" | "04_bindings"
            ) {
                return parts.next();
            }
            return Some(first);
        }
        if let Some(rest) = norm.strip_prefix("apps/") {
            return rest.split('/').find(|s| !s.is_empty());
        }
        if let Some((first, _)) = norm.split_once('/')
            && matches!(first, "math_explorer" | "oxidizemath" | "math_explorer_gui")
        {
            return Some(first);
        }
        None
    }

    /// Resolves path-owned module keys in `[modules]` for `rel_path`, ordered from most specific to broadest.
    pub fn path_owned_modules(&self, rel_path: &str) -> Vec<String> {
        let norm = crate::path_utils::normalize_path(rel_path);
        let trimmed = norm.trim_start_matches("./");
        let (_, body_after_pkg) = trimmed
            .split_once(':')
            .filter(|(p, _)| !p.contains('/'))
            .unwrap_or(("", trimmed));
        let mut out = Vec::new();
        let segments: Vec<&str> = body_after_pkg.split('/').collect();
        let stem = segments
            .last()
            .and_then(|f| f.strip_suffix(".rs"))
            .unwrap_or_default();

        if stem == "cera" && self.modules.contains_key("cera_framework") {
            out.push("cera_framework".to_string());
        }
        for seg in segments.iter().rev().skip(1) {
            if SKIP_PATH_SEGMENTS.contains(seg) {
                continue;
            }
            if self.modules.contains_key(*seg) && !out.iter().any(|m| m == seg) {
                out.push((*seg).to_string());
            }
        }
        if body_after_pkg.contains("/tests/") || body_after_pkg.starts_with("tests/") {
            self.append_test_stem_modules(trimmed, stem, &mut out);
        }
        if let Some(m) = Self::crate_for_repo_path(trimmed).and_then(|c| match c {
            "domain_biology" => Some("biology"),
            "domain_epidemiology" => Some("epidemiology"),
            "domain_climate" => Some("climate"),
            "pure_math" => Some("pure_math"),
            "math_commons" => Some("math_commons"),
            _ => None,
        }) && self.modules.contains_key(m)
            && !out.iter().any(|x| x == m)
        {
            out.push(m.to_string());
        }
        let gui_tabs_rest = trimmed
            .strip_prefix("math_explorer_gui/src/tabs/")
            .or_else(|| trimmed.strip_prefix("apps/math_explorer_gui/src/tabs/"))
            .or_else(|| trimmed.strip_prefix("math_explorer_gui:src/tabs/"));
        if out.is_empty()
            && let Some(rest) = gui_tabs_rest
            && let Some(tab_domain) = rest.split('/').next()
        {
            let gui_mod = match tab_domain {
                "analysis" | "geometry_topology" | "number_theory" => Some("pure_math"),
                "morphogenesis" | "neuroscience" => Some("biology"),
                "optics" | "optics_waves" => Some("optics"),
                "algorithmic_information" => Some("algorithmic_information_rust"),
                _ => None,
            };
            if let Some(m) = gui_mod
                && self.modules.contains_key(m)
            {
                out.push(m.to_string());
            }
        }
        out
    }

    fn append_test_stem_modules(&self, norm_path: &str, stem: &str, out: &mut Vec<String>) {
        let alias = match stem {
            "verified_clinical_alloc" => Some("clinical_trials"),
            "bench_lbm_custom" => Some("fluid_dynamics"),
            "test_sandpile" => Some("stat_mech"),
            "test_optical_motion" | "test_radar_gating" => Some("medical"),
            "test_hydrogenic_3d" => Some("quantum"),
            _ => None,
        };
        if let Some(mapped) = alias
            && self.modules.contains_key(mapped)
            && !out.iter().any(|m| m == mapped)
        {
            out.push(mapped.to_string());
        }
        for mod_key in self.modules.keys() {
            if mod_key.ends_with("_mod") {
                continue;
            }
            let exact_or_affix = stem == mod_key
                || stem.starts_with(&format!("{mod_key}_"))
                || stem.ends_with(&format!("_{mod_key}"))
                || stem.contains(&format!("_{mod_key}_"));
            if exact_or_affix && !out.iter().any(|m| m == mod_key) {
                out.push(mod_key.clone());
            }
        }
        let _ = norm_path;
    }

    /// Resolves the governing [`MaturityTier`] for a repository-relative Rust file path.
    pub fn tier_for_repo_path(&self, rel_path: &str) -> MaturityTier {
        if let Some(first_mod) = self.path_owned_modules(rel_path).first()
            && let Some(tier) = self.module_tier(first_mod)
        {
            return tier;
        }
        if let Some(crate_name) = Self::crate_for_repo_path(rel_path)
            && let Some(tier) = self.crate_tier(crate_name)
        {
            return tier;
        }
        MaturityTier::Experimental
    }
}
