use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

pub const WHITELIST_FILE_NAME: &str = "verification_whitelist.toml";

pub const WORKSPACE_WHITELIST_CANDIDATES: &[&str] = &[
    "verification_whitelist.toml",
    "../../verification_whitelist.toml",
];

pub const ROOT_ONLY_WHITELIST_CANDIDATES: &[&str] = &["verification_whitelist.toml"];

/// Unified top-level schema for `verification_whitelist.toml` shared by all verification gates.
#[derive(Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct VerificationWhitelist {
    #[serde(default)]
    pub doc_verification: Option<DocVerificationConfig>,
    #[serde(default)]
    pub entropy_guard: Option<EntropyGuardConfig>,
    #[serde(default)]
    pub workflow_linter: Option<WorkflowLinterConfig>,
    #[serde(default)]
    pub dead_code: Vec<DeadCodeEntry>,
}

/// Configuration for `doc_audit` loaded from `[doc_verification]`.
#[derive(Deserialize, Debug, Clone, Default)]
pub(crate) struct DocVerificationConfig {
    #[serde(default)]
    pub allowlist: Option<Vec<String>>,
}

/// Configuration for `entropy_guard` loaded from `[entropy_guard]`.
#[derive(Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct EntropyGuardConfig {
    #[serde(default, alias = "targets", alias = "paths", alias = "scan_targets")]
    pub scan_paths: Option<Vec<String>>,
    #[serde(default, alias = "extensions", alias = "file_extensions")]
    pub target_extensions: Option<Vec<String>>,
    #[serde(default, alias = "exemptions", alias = "path_exemptions")]
    pub allowlist: Option<Vec<String>>,
}

/// Configuration for `workflow_linter` loaded from `verification_whitelist.toml`.
#[derive(Deserialize, Debug, Clone, Default)]
pub struct WorkflowLinterConfig {
    /// Explicit job-level write permission exceptions for deployment workflows.
    #[serde(default, alias = "allowed_job_permissions")]
    pub exceptions: Vec<JobPermissionException>,
}

/// Scoped exception allowing specific write permissions on a named job in a specific workflow.
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct JobPermissionException {
    /// Relative path to the workflow file (e.g., `.github/workflows/pages.yml`).
    pub workflow: String,
    /// Job identifier within the `jobs` mapping (e.g., `deploy`).
    pub job: String,
    /// Permission scopes explicitly permitted to use `write` on this job (e.g., `["pages", "id-token"]`).
    #[serde(default, alias = "permissions", alias = "write_scopes")]
    pub allow_write: Vec<String>,
    /// Optional human-readable justification for the exception.
    #[serde(default)]
    pub reason: Option<String>,
}

/// Authorized dead-code suppression entry loaded from `[[dead_code]]`.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct DeadCodeEntry {
    pub(crate) file: String,
    pub(crate) identifier: Option<String>,
}

/// Resolves the first existing whitelist path from an explicit slice of candidate paths.
pub(crate) fn resolve_whitelist_path<P: AsRef<Path>>(candidates: &[P]) -> Option<PathBuf> {
    candidates
        .iter()
        .map(AsRef::as_ref)
        .find(|candidate| candidate.exists())
        .map(Path::to_path_buf)
}

/// Locates `verification_whitelist.toml` relative to the current working directory or workspace root.
pub fn find_whitelist_path() -> Option<PathBuf> {
    resolve_whitelist_path(WORKSPACE_WHITELIST_CANDIDATES).or_else(|| {
        oxidize_core::workspace::find_workspace_root()
            .map(|root| root.join(WHITELIST_FILE_NAME))
            .filter(|candidate| candidate.exists())
    })
}

fn read_and_parse_whitelist(path: &Path) -> Result<VerificationWhitelist, String> {
    let content = fs::read_to_string(path)
        .map_err(|err| format!("Failed to read {}: {}", path.display(), err))?;
    toml::from_str::<VerificationWhitelist>(&content)
        .map_err(|err| format!("Failed to parse {}: {}", path.display(), err))
}

/// Loads `verification_whitelist.toml` strictly, failing closed on read errors or invalid TOML.
///
/// When `explicit_path` is `None` and no path in `candidates` exists, returns default configuration.
pub(crate) fn load_whitelist_strict<P: AsRef<Path>>(
    explicit_path: Option<&Path>,
    candidates: &[P],
) -> Result<VerificationWhitelist, String> {
    if let Some(path) = explicit_path {
        return read_and_parse_whitelist(path);
    }
    let Some(resolved) = resolve_whitelist_path(candidates) else {
        return Ok(VerificationWhitelist::default());
    };
    read_and_parse_whitelist(&resolved)
}

/// Loads `verification_whitelist.toml` with fallback semantics, returning defaults when missing or invalid.
pub(crate) fn load_whitelist_or_default<P: AsRef<Path>>(
    explicit_path: Option<&Path>,
    candidates: &[P],
) -> VerificationWhitelist {
    load_whitelist_strict(explicit_path, candidates).unwrap_or_default()
}

/// Converts a path lossily to UTF-8 and replaces backslashes with forward slashes.
pub(crate) fn lossy_slash_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Configurable options for lossy path matching across verification gates.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PathMatchOptions {
    pub trim_dot_slash: bool,
    pub require_non_empty: bool,
    pub match_file_name: bool,
}

/// Matches `path` against `pattern` using lossy slash normalization and per-gate matching flags.
pub(crate) fn path_matches_lossy(path: &Path, pattern: &str, opts: PathMatchOptions) -> bool {
    let raw_actual = lossy_slash_path(path);
    let norm_actual = if opts.trim_dot_slash {
        raw_actual.trim_start_matches("./")
    } else {
        raw_actual.as_str()
    };

    let raw_expected = pattern.replace('\\', "/");
    let norm_expected = if opts.trim_dot_slash {
        raw_expected.trim_start_matches("./")
    } else {
        raw_expected.as_str()
    };

    if opts.require_non_empty && norm_expected.is_empty() {
        return false;
    }

    norm_actual == norm_expected
        || norm_actual.ends_with(&format!("/{norm_expected}"))
        || (opts.match_file_name
            && path.file_name().and_then(|f| f.to_str()) == Some(norm_expected))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test_utils::TempTestDir;

    #[test]
    fn test_resolve_whitelist_path_respects_candidate_order() {
        let dir = TempTestDir::new("whitelist_candidates");
        let missing = dir.join("missing.toml");
        let first = dir.join("first.toml");
        let second = dir.join("second.toml");
        fs::write(&second, "").unwrap();
        assert_eq!(
            resolve_whitelist_path(&[&missing, &first, &second]),
            Some(second.clone())
        );

        fs::write(&first, "").unwrap();
        assert_eq!(
            resolve_whitelist_path(&[&missing, &first, &second]),
            Some(first)
        );
        assert_eq!(resolve_whitelist_path(&[&missing]), None);
    }

    #[test]
    fn test_strict_and_fallback_loaders_preserve_aliases_and_errors() {
        let dir = TempTestDir::new("whitelist_loaders");
        let valid = dir.join("valid.toml");
        fs::write(
            &valid,
            concat!(
                "[doc_verification]\nallowlist = [\"crates/demo\"]\n",
                "[entropy_guard]\nscan_targets = [\"crates\"]\n",
                "file_extensions = [\"rs\"]\npath_exemptions = [\"allow.rs\"]\n",
                "[[workflow_linter.allowed_job_permissions]]\n",
                "workflow = \".github/workflows/ci.yml\"\njob = \"deploy\"\nwrite_scopes = [\"pages\"]\n",
                "[[dead_code]]\nfile = \"crates/demo/src/lib.rs\"\nidentifier = \"Item\"\n",
            ),
        )
        .unwrap();

        let parsed = load_whitelist_strict(Some(&valid), &[] as &[&Path]).unwrap();
        assert_eq!(
            parsed.doc_verification.unwrap().allowlist,
            Some(vec!["crates/demo".to_string()])
        );
        assert_eq!(
            parsed.entropy_guard.unwrap().allowlist,
            Some(vec!["allow.rs".to_string()])
        );
        assert_eq!(
            parsed.workflow_linter.unwrap().exceptions[0].allow_write,
            vec!["pages".to_string()]
        );
        assert_eq!(parsed.dead_code.len(), 1);

        let invalid = dir.join("invalid.toml");
        fs::write(&invalid, "[broken").unwrap();
        assert!(load_whitelist_strict(Some(&invalid), &[] as &[&Path]).is_err());
        let fallback = load_whitelist_or_default(Some(&invalid), &[] as &[&Path]);
        assert!(fallback.doc_verification.is_none());
        assert!(fallback.dead_code.is_empty());
    }

    #[test]
    fn test_path_matches_lossy_per_gate_flags() {
        let entropy_opts = PathMatchOptions {
            trim_dot_slash: true,
            require_non_empty: false,
            match_file_name: true,
        };
        assert!(path_matches_lossy(
            Path::new("./crates/foo/bar.rs/"),
            "bar.rs",
            entropy_opts
        ));

        let workflow_opts = PathMatchOptions {
            trim_dot_slash: true,
            require_non_empty: true,
            match_file_name: false,
        };
        assert!(!path_matches_lossy(
            Path::new(".github/workflows/ci.yml"),
            "",
            workflow_opts
        ));
        assert!(!path_matches_lossy(
            Path::new(".github/workflows/ci.yml"),
            "./",
            workflow_opts
        ));
        assert!(!path_matches_lossy(
            Path::new(".github/workflows/ci.yml/"),
            "ci.yml",
            workflow_opts
        ));
        assert!(path_matches_lossy(
            Path::new("./.github/workflows/ci.yml"),
            ".github\\workflows\\ci.yml",
            workflow_opts
        ));

        let policy_opts = PathMatchOptions {
            trim_dot_slash: false,
            require_non_empty: false,
            match_file_name: false,
        };
        assert!(!path_matches_lossy(
            Path::new("crates/foo/lib.rs"),
            "./crates/foo/lib.rs",
            policy_opts
        ));
        assert!(path_matches_lossy(
            Path::new("root/crates/foo/lib.rs"),
            "crates\\foo\\lib.rs",
            policy_opts
        ));
    }

    #[test]
    #[cfg(unix)]
    fn test_path_matches_lossy_preserves_non_utf8_prefix_segments() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let bad_dir = Path::new(OsStr::from_bytes(b"/tmp/non_utf8_\xff_dir"));
        let full_path = bad_dir.join("crates/demo/src/lib.rs");
        let opts = PathMatchOptions {
            trim_dot_slash: true,
            require_non_empty: true,
            match_file_name: false,
        };
        assert!(path_matches_lossy(
            &full_path,
            "crates/demo/src/lib.rs",
            opts
        ));
    }
}
