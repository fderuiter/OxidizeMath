use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const WIX_ALPHA_BASE: u64 = 1_000;
const WIX_BETA_BASE: u64 = 10_000;
const WIX_RC_BASE: u64 = 20_000;
const WIX_MAX_MAJOR_MINOR: u64 = 255;
const WIX_MAX_PATCH_REVISION: u64 = 65_535;

/// Validated SemVer release version within Windows WiX `ProductVersion` bounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReleaseVersion {
    pub(crate) major: u8,
    pub(crate) minor: u8,
    pub(crate) patch: u16,
    pub(crate) prerelease: Option<String>,
    pub(crate) wix_revision: u16,
}

impl ReleaseVersion {
    /// Returns `true` when the version carries a SemVer pre-release suffix.
    pub(crate) fn is_prerelease(&self) -> bool {
        self.prerelease.is_some()
    }

    /// Formats the SemVer version passed to `cargo-packager` on Windows `wix` builds.
    ///
    /// Stable releases keep `MAJOR.MINOR.PATCH`; pre-releases map to `MAJOR.MINOR.PATCH-<revision>`.
    pub(crate) fn wix_packager_version(&self) -> String {
        if self.is_prerelease() {
            format!(
                "{}.{}.{}-{}",
                self.major, self.minor, self.patch, self.wix_revision
            )
        } else {
            format!("{}.{}.{}", self.major, self.minor, self.patch)
        }
    }
}

/// Result of running `check-release-tag` against a workspace root manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReleaseTagOutcome {
    pub(crate) workspace_version: String,
    pub(crate) release_tag: Option<String>,
    pub(crate) is_prerelease: bool,
    pub(crate) wix_version: String,
    pub(crate) wix_product_version: String,
}

#[derive(Default)]
struct CliOptions {
    tag: Option<String>,
    require_tag: bool,
    github_output: Option<PathBuf>,
    apply_wix_override: Option<PathBuf>,
}

fn parse_semver_uint(seg: &str, label: &str) -> Result<u64, String> {
    if seg.is_empty() || !seg.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!(
            "Invalid {label} '{seg}': expected non-empty ASCII digits"
        ));
    }
    if seg.len() > 1 && seg.starts_with('0') {
        return Err(format!(
            "Invalid {label} '{seg}': SemVer numeric identifiers must not have leading zeroes"
        ));
    }
    seg.parse::<u64>()
        .map_err(|e| format!("Invalid {label} '{seg}': {e}"))
}

fn parse_core_triplet(core: &str) -> Result<(u8, u8, u16), String> {
    let parts: Vec<&str> = core.split('.').collect();
    if parts.len() != 3 {
        return Err(format!(
            "Invalid SemVer core '{core}': expected MAJOR.MINOR.PATCH"
        ));
    }
    let major = parse_semver_uint(parts[0], "major version")?;
    let minor = parse_semver_uint(parts[1], "minor version")?;
    let patch = parse_semver_uint(parts[2], "patch version")?;
    if major > WIX_MAX_MAJOR_MINOR {
        return Err(format!(
            "Major version {major} in '{core}' exceeds WiX bound ({WIX_MAX_MAJOR_MINOR})"
        ));
    }
    if minor > WIX_MAX_MAJOR_MINOR {
        return Err(format!(
            "Minor version {minor} in '{core}' exceeds WiX bound ({WIX_MAX_MAJOR_MINOR})"
        ));
    }
    if patch > WIX_MAX_PATCH_REVISION {
        return Err(format!(
            "Patch version {patch} in '{core}' exceeds WiX bound ({WIX_MAX_PATCH_REVISION})"
        ));
    }
    Ok((major as u8, minor as u8, patch as u16))
}

fn validate_prerelease_identifiers(pre: &str) -> Result<(), String> {
    if pre.is_empty() {
        return Err("SemVer pre-release suffix must not be empty".to_string());
    }
    for ident in pre.split('.') {
        if ident.is_empty()
            || !ident
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        {
            return Err(format!("Invalid SemVer pre-release identifier '{ident}'"));
        }
        if ident.bytes().all(|b| b.is_ascii_digit()) && ident.len() > 1 && ident.starts_with('0') {
            return Err(format!(
                "SemVer numeric pre-release identifier '{ident}' must not have leading zeroes"
            ));
        }
    }
    Ok(())
}

fn map_prerelease_to_wix_revision(pre: &str) -> Result<u16, String> {
    validate_prerelease_identifiers(pre)?;
    let mapped = if let Some(rest) = pre.strip_prefix("alpha.") {
        let n = parse_semver_uint(rest, "alpha pre-release number")?;
        WIX_ALPHA_BASE.checked_add(n)
    } else if let Some(rest) = pre.strip_prefix("beta.") {
        let n = parse_semver_uint(rest, "beta pre-release number")?;
        WIX_BETA_BASE.checked_add(n)
    } else if let Some(rest) = pre.strip_prefix("rc.") {
        let n = parse_semver_uint(rest, "rc pre-release number")?;
        WIX_RC_BASE.checked_add(n)
    } else if !pre.contains('.') && pre.bytes().all(|b| b.is_ascii_digit()) {
        Some(parse_semver_uint(pre, "numeric pre-release revision")?)
    } else {
        return Err(format!(
            "Unsupported pre-release '{pre}' for WiX mapping: expected alpha.N, beta.N, rc.N, or numeric revision <= {WIX_MAX_PATCH_REVISION}"
        ));
    };

    match mapped {
        Some(rev) if rev <= WIX_MAX_PATCH_REVISION => Ok(rev as u16),
        _ => Err(format!(
            "Mapped WiX revision for pre-release '{pre}' exceeds {WIX_MAX_PATCH_REVISION}"
        )),
    }
}

/// Parses and validates a workspace or tag version against SemVer and WiX bounds, rejecting `+build`.
pub(crate) fn parse_release_version(raw: &str) -> Result<ReleaseVersion, String> {
    if raw.is_empty() || raw.trim() != raw {
        return Err(format!(
            "Version '{raw}' must not be empty or contain surrounding whitespace"
        ));
    }
    if raw.contains('+') {
        return Err(format!(
            "SemVer +build metadata is prohibited in release version '{raw}' (breaks WiX numeric revision mapping)"
        ));
    }
    let (core, pre_opt) = match raw.split_once('-') {
        Some((c, p)) => (c, Some(p)),
        None => (raw, None),
    };
    let (major, minor, patch) = parse_core_triplet(core)?;
    let (prerelease, wix_revision) = match pre_opt {
        Some(pre) => (Some(pre.to_string()), map_prerelease_to_wix_revision(pre)?),
        None => (None, 0),
    };
    Ok(ReleaseVersion {
        major,
        minor,
        patch,
        prerelease,
        wix_revision,
    })
}

/// Mirrors `cargo-packager`'s `wix::convert_version` 4-field `ProductVersion` conversion.
pub(crate) fn convert_version(packager_version: &str) -> Result<String, String> {
    let (without_build, build_opt) = match packager_version.split_once('+') {
        Some((lhs, b)) => (lhs, Some(b)),
        None => (packager_version, None),
    };
    let (core, pre_opt) = match without_build.split_once('-') {
        Some((c, p)) => (c, Some(p)),
        None => (without_build, None),
    };
    let (major, minor, patch) = parse_core_triplet(core)?;
    let fourth = if let Some(build) = build_opt {
        let b = parse_semver_uint(build, "WiX build metadata")?;
        u16::try_from(b)
            .map_err(|_| format!("WiX build metadata {b} exceeds {WIX_MAX_PATCH_REVISION}"))?
    } else if let Some(pre) = pre_opt {
        let p = parse_semver_uint(pre, "WiX numeric pre-release")?;
        u16::try_from(p)
            .map_err(|_| format!("WiX pre-release {p} exceeds {WIX_MAX_PATCH_REVISION}"))?
    } else {
        0
    };
    Ok(format!("{major}.{minor}.{patch}.{fourth}"))
}

fn is_non_tag_ref_name(git_ref: &str, ref_name: &str) -> bool {
    git_ref
        .strip_prefix("refs/heads/")
        .is_some_and(|b| b == ref_name)
        || git_ref
            .strip_prefix("refs/pull/")
            .is_some_and(|p| p == ref_name)
}

/// Resolves the active `v*` release tag from CLI flags or CI `GIT_REF` / `RELEASE_TAG` bindings.
pub(crate) fn resolve_effective_tag(
    cli_tag: Option<&str>,
    release_tag_env: Option<&str>,
    git_ref_env: Option<&str>,
    require_tag: bool,
) -> Result<Option<String>, String> {
    if let Some(explicit) = cli_tag {
        let trimmed = explicit.trim();
        if trimmed.is_empty() {
            return Err("--tag value must not be empty".to_string());
        }
        return Ok(Some(trimmed.to_string()));
    }

    let git_ref = git_ref_env.map(str::trim).filter(|s| !s.is_empty());
    let rel_tag = release_tag_env.map(str::trim).filter(|s| !s.is_empty());

    if let Some(gr) = git_ref {
        if let Some(tag_from_ref) = gr.strip_prefix("refs/tags/") {
            if !tag_from_ref.starts_with('v') {
                if require_tag {
                    return Err(format!("Expected a refs/tags/v* release tag, got '{gr}'"));
                }
                return Ok(None);
            }
            if let Some(rt) = rel_tag
                && rt != tag_from_ref
            {
                return Err(format!(
                    "GIT_REF ('{gr}') and RELEASE_TAG ('{rt}') refer to different tags"
                ));
            }
            return Ok(Some(tag_from_ref.to_string()));
        }
        if let Some(rt) = rel_tag
            && !is_non_tag_ref_name(gr, rt)
        {
            return Ok(Some(rt.to_string()));
        }
        if require_tag {
            return Err(format!(
                "Release tag required, but GIT_REF is '{gr}' (expected refs/tags/v*)"
            ));
        }
        return Ok(None);
    }

    if let Some(rt) = rel_tag {
        return Ok(Some(rt.to_string()));
    }
    if require_tag {
        return Err("RELEASE_TAG or refs/tags/v* GIT_REF must be set".to_string());
    }
    Ok(None)
}

pub(crate) fn read_workspace_package_version(root_manifest_path: &Path) -> Result<String, String> {
    let content = fs::read_to_string(root_manifest_path)
        .map_err(|e| format!("Failed to read {}: {e}", root_manifest_path.display()))?;
    let parsed: toml::Value = toml::from_str(&content)
        .map_err(|e| format!("Failed to parse {}: {e}", root_manifest_path.display()))?;
    parsed
        .get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("version"))
        .and_then(toml::Value::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| {
            format!(
                "Missing [workspace.package].version string in {}",
                root_manifest_path.display()
            )
        })
}

/// Validates workspace version invariants and optional `v*` release tag parity.
pub(crate) fn evaluate_release_tag(
    workspace_version: &str,
    effective_tag: Option<&str>,
) -> Result<ReleaseTagOutcome, String> {
    let parsed_ws = parse_release_version(workspace_version)?;
    if let Some(tag) = effective_tag {
        if tag.contains('+') {
            return Err(format!(
                "SemVer +build metadata is prohibited in RELEASE_TAG '{tag}'"
            ));
        }
        let Some(tag_ver_str) = tag.strip_prefix('v') else {
            return Err(format!(
                "Invalid release tag '{tag}': release tags must start with 'v' (expected 'v{workspace_version}')"
            ));
        };
        let _parsed_tag = parse_release_version(tag_ver_str)?;
        if tag_ver_str != workspace_version {
            return Err(format!(
                "Release tag parity mismatch: tag '{tag}' resolves to '{tag_ver_str}', but [workspace.package].version is '{workspace_version}'"
            ));
        }
    }

    let wix_version = parsed_ws.wix_packager_version();
    let wix_product_version = convert_version(&wix_version)?;
    Ok(ReleaseTagOutcome {
        workspace_version: workspace_version.to_string(),
        release_tag: effective_tag.map(ToString::to_string),
        is_prerelease: parsed_ws.is_prerelease(),
        wix_version,
        wix_product_version,
    })
}

/// Injects a transient `version = "<wix_version>"` override into `[package.metadata.packager]`.
pub(crate) fn inject_transient_wix_packager_version(
    manifest_content: &str,
    wix_version: &str,
) -> Result<String, String> {
    let _ = convert_version(wix_version)?;
    let mut out = String::with_capacity(manifest_content.len() + 48);
    let mut inserted = false;
    for line in manifest_content.lines() {
        out.push_str(line);
        out.push('\n');
        if !inserted && line.trim() == "[package.metadata.packager]" {
            out.push_str(&format!("version = \"{wix_version}\"\n"));
            inserted = true;
        }
    }
    if !inserted {
        return Err("Missing [package.metadata.packager] section in GUI manifest".to_string());
    }
    let parsed: toml::Value = toml::from_str(&out)
        .map_err(|e| format!("Transient WiX manifest override produced invalid TOML: {e}"))?;
    let actual = parsed
        .get("package")
        .and_then(|p| p.get("metadata"))
        .and_then(|m| m.get("packager"))
        .and_then(|pk| pk.get("version"))
        .and_then(toml::Value::as_str);
    if actual != Some(wix_version) {
        return Err("Transient WiX manifest override failed verification".to_string());
    }
    Ok(out)
}

fn parse_cli_args(args: &[String]) -> Result<CliOptions, String> {
    let mut opts = CliOptions::default();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--require-tag" => opts.require_tag = true,
            "--tag" if idx + 1 < args.len() => {
                opts.tag = Some(args[idx + 1].clone());
                idx += 1;
            }
            arg if arg.starts_with("--tag=") => {
                opts.tag = Some(arg["--tag=".len()..].to_string());
            }
            "--github-output" if idx + 1 < args.len() => {
                opts.github_output = Some(PathBuf::from(&args[idx + 1]));
                idx += 1;
            }
            "--apply-wix-override" if idx + 1 < args.len() => {
                opts.apply_wix_override = Some(PathBuf::from(&args[idx + 1]));
                idx += 1;
            }
            other => return Err(format!("Unknown check-release-tag argument: '{other}'")),
        }
        idx += 1;
    }
    Ok(opts)
}

fn write_github_outputs(path: &Path, outcome: &ReleaseTagOutcome) -> Result<(), String> {
    if path.as_os_str().is_empty() {
        return Ok(());
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    writeln!(file, "is_prerelease={}", outcome.is_prerelease)
        .and_then(|()| writeln!(file, "wix_version={}", outcome.wix_version))
        .map_err(|e| format!("Failed writing outputs to {}: {e}", path.display()))
}

pub(crate) fn run_check_release_tag_in_dir(
    root: &Path,
    args: &[String],
    release_tag_env: Option<&str>,
    git_ref_env: Option<&str>,
) -> Result<String, String> {
    let opts = parse_cli_args(args)?;
    let ws_version = read_workspace_package_version(&root.join("Cargo.toml"))?;
    let effective_tag = resolve_effective_tag(
        opts.tag.as_deref(),
        release_tag_env,
        git_ref_env,
        opts.require_tag,
    )?;
    let outcome = evaluate_release_tag(&ws_version, effective_tag.as_deref())?;
    if let Some(manifest_path) = opts.apply_wix_override.as_deref()
        && outcome.is_prerelease
    {
        let raw = fs::read_to_string(manifest_path)
            .map_err(|e| format!("Failed reading {}: {e}", manifest_path.display()))?;
        let updated = inject_transient_wix_packager_version(&raw, &outcome.wix_version)?;
        fs::write(manifest_path, updated)
            .map_err(|e| format!("Failed writing {}: {e}", manifest_path.display()))?;
    }
    if let Some(out_path) = opts.github_output.as_deref() {
        write_github_outputs(out_path, &outcome)?;
    }
    let header = if let Some(tag) = &outcome.release_tag {
        format!(
            "[+] Verified release tag parity: {tag} == {} (WiX ProductVersion: {})",
            outcome.workspace_version, outcome.wix_product_version
        )
    } else {
        format!(
            "[+] Verified [workspace.package].version = {} (no active v* release tag)",
            outcome.workspace_version
        )
    };
    Ok(format!(
        "{header}\nis_prerelease={}\nwix_version={}\n",
        outcome.is_prerelease, outcome.wix_version
    ))
}

/// CLI entrypoint for `unified_verification check-release-tag`.
pub(crate) fn run_check_release_tag_cli(args: &[String]) -> bool {
    let root = crate::manifest_cache::workspace_root_dir();
    let release_tag_env = env::var("RELEASE_TAG").ok();
    let git_ref_env = env::var("GIT_REF").ok();
    match run_check_release_tag_in_dir(
        root,
        args,
        release_tag_env.as_deref(),
        git_ref_env.as_deref(),
    ) {
        Ok(output) => {
            print!("{output}");
            true
        }
        Err(err) => {
            eprintln!("[!] {err}");
            false
        }
    }
}

#[cfg(test)]
mod tests;
