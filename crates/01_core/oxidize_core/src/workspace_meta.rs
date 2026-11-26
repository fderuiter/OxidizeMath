//! Pure-`std` `[package.metadata.oxidize]` schema and workspace package discovery helpers.
//!
//! Shared between `oxidize_core::workspace` and `apps/xtask` (via `build.rs` staging) so
//! workspace tools remain free of hardcoded directory paths without adding external crates
//! to `xtask`.

use std::fs;
use std::path::Path;

/// Canonical architectural tier declared in `[package.metadata.oxidize].tier`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TierKind {
    /// Tier 0 (`crates/00_foundation`): zero-dependency / foundational primitives and proc-macros.
    Foundation00,
    /// Tier 1 (`crates/01_core`): core diagnostics, registries, VFS, and AST infrastructure.
    Core01,
    /// Tier 2 (`crates/02_engines`): `#![no_std]`-capable mathematical and numerical engines.
    Engines02,
    /// Tier 3 (`crates/03_domains`): scientific and pure-mathematics domain crates.
    Domains03,
    /// Tier 4 (`crates/04_bindings`): library facades, doctest harnesses, and verification bindings.
    Bindings04,
    /// Application tier (`apps/`): CLI, GUI, verification runner, and developer task binaries.
    Apps,
    /// Vendored third-party crate patched via `[patch.crates-io]`.
    Vendored,
}

impl TierKind {
    /// Parses a canonical tier identifier string.
    #[must_use]
    pub fn parse_str(raw: &str) -> Option<Self> {
        match raw.trim() {
            "00_foundation" => Some(Self::Foundation00),
            "01_core" => Some(Self::Core01),
            "02_engines" => Some(Self::Engines02),
            "03_domains" => Some(Self::Domains03),
            "04_bindings" => Some(Self::Bindings04),
            "apps" => Some(Self::Apps),
            "vendored" => Some(Self::Vendored),
            _ => None,
        }
    }

    /// Returns the canonical manifest string representation for this tier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Foundation00 => "00_foundation",
            Self::Core01 => "01_core",
            Self::Engines02 => "02_engines",
            Self::Domains03 => "03_domains",
            Self::Bindings04 => "04_bindings",
            Self::Apps => "apps",
            Self::Vendored => "vendored",
        }
    }

    /// Returns the monotonic dependency rank (`0..=5`) for first-party tiers.
    #[must_use]
    pub const fn rank(self) -> u8 {
        match self {
            Self::Foundation00 => 0,
            Self::Core01 => 1,
            Self::Engines02 => 2,
            Self::Domains03 => 3,
            Self::Bindings04 => 4,
            Self::Apps => 5,
            Self::Vendored => 0,
        }
    }

    /// Expected directory prefix (`crates/<tier>/` or `apps/`) for relocated first-party crates.
    #[must_use]
    pub const fn expected_dir_prefix(self) -> Option<&'static str> {
        match self {
            Self::Foundation00 => Some("crates/00_foundation/"),
            Self::Core01 => Some("crates/01_core/"),
            Self::Engines02 => Some("crates/02_engines/"),
            Self::Domains03 => Some("crates/03_domains/"),
            Self::Bindings04 => Some("crates/04_bindings/"),
            Self::Apps => Some("apps/"),
            Self::Vendored => None,
        }
    }

    /// Infers the tier implied by a relocated member directory path (`crates/<tier>/<pkg>` or `apps/<pkg>`), if any.
    #[must_use]
    pub fn from_member_path(member_dir: &str) -> Option<Self> {
        let norm = member_dir.trim().trim_start_matches("./");
        if let Some(rest) = norm.strip_prefix("crates/") {
            let first = rest.split('/').next()?;
            return Self::parse_str(first);
        }
        if norm.starts_with("apps/") {
            return Some(Self::Apps);
        }
        None
    }
}

/// Release versioning role declared in `[package.metadata.oxidize].release_role`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReleaseRole {
    /// First-party published/packaged crate inheriting `version.workspace = true`.
    Release,
    /// Internal tooling, proc-macro, or verification harness pinned to `0.1.0`.
    Internal,
    /// Vendored upstream crate patched via `[patch.crates-io]`.
    Vendored,
}

impl ReleaseRole {
    /// Parses a `release_role` string value.
    #[must_use]
    pub fn parse_str(raw: &str) -> Option<Self> {
        match raw.trim() {
            "release" => Some(Self::Release),
            "internal" => Some(Self::Internal),
            "vendored" => Some(Self::Vendored),
            _ => None,
        }
    }

    /// Returns the canonical TOML string representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Internal => "internal",
            Self::Vendored => "vendored",
        }
    }
}

/// Public facade participation role declared in `[package.metadata.oxidize].facade_role`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FacadeRole {
    /// Crate is not part of the public mathematical facade surface.
    #[default]
    None,
    /// Tier 0 / Tier 1 foundational crate re-exported directly on the facade.
    Foundation,
    /// Tier 2 numerical engine crate re-exported with error parity on the facade.
    Engine,
    /// Tier 3 scientific or pure-math domain crate feature-gated on the facade.
    Domain,
    /// Primary library facade crate (`oxidizemath`, or `math_explorer` prior to split).
    Facade,
    /// Transitional compatibility facade re-exporting the primary facade (`math_explorer`).
    CompatFacade,
}

impl FacadeRole {
    /// Parses a `facade_role` string value.
    #[must_use]
    pub fn parse_str(raw: &str) -> Option<Self> {
        match raw.trim() {
            "none" => Some(Self::None),
            "foundation" => Some(Self::Foundation),
            "engine" => Some(Self::Engine),
            "domain" => Some(Self::Domain),
            "facade" => Some(Self::Facade),
            "compat_facade" => Some(Self::CompatFacade),
            _ => None,
        }
    }

    /// Returns the canonical TOML string representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Foundation => "foundation",
            Self::Engine => "engine",
            Self::Domain => "domain",
            Self::Facade => "facade",
            Self::CompatFacade => "compat_facade",
        }
    }
}

/// Parsed `[package.metadata.oxidize]` block from a workspace member `Cargo.toml`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OxidizeMetadata {
    /// Architectural tier (`00_foundation`..`04_bindings`, `apps`, or `vendored`).
    pub tier: Option<TierKind>,
    /// Release versioning policy (`release`, `internal`, or `vendored`).
    pub release_role: Option<ReleaseRole>,
    /// Role in the public library facade (`none`, `foundation`, `engine`, `domain`, `facade`, `compat_facade`).
    pub facade_role: FacadeRole,
    /// Inner module / feature name exposed on the facade for `domain` crates (e.g. `"ai"`, `"pure_math"`).
    pub facade_module: Option<String>,
    /// Whether this crate participates in `[cite:...]` and `#[verified]` traceability scanning.
    pub traceability: Option<bool>,
    /// Optional crate-relative source subdirectory scanned by traceability (defaults to `"src"`).
    pub traceability_subdir: Option<String>,
    /// Whether public computational functions must carry 4-part mathematical rustdoc contracts.
    pub math_contract: Option<bool>,
}

/// Discovered workspace member package paired with its `[package.metadata.oxidize]` declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspacePackageMeta {
    /// Workspace-relative member directory path from `[workspace].members`.
    pub member_path: String,
    /// Cargo `[package].name` identifier.
    pub package_name: String,
    /// Parsed `[package.metadata.oxidize]` fields.
    pub metadata: OxidizeMetadata,
    /// Whether this crate is patched under `[patch.crates-io]` in root `Cargo.toml`.
    pub is_vendored_patch: bool,
}

impl WorkspacePackageMeta {
    /// Effective architectural tier, defaulting `[patch.crates-io]` crates to [`TierKind::Vendored`].
    #[must_use]
    pub fn effective_tier(&self) -> Option<TierKind> {
        self.metadata
            .tier
            .or_else(|| self.is_vendored_patch.then_some(TierKind::Vendored))
    }

    /// Effective release role, defaulting `[patch.crates-io]` crates to [`ReleaseRole::Vendored`].
    #[must_use]
    pub fn effective_release_role(&self) -> Option<ReleaseRole> {
        self.metadata
            .release_role
            .or_else(|| self.is_vendored_patch.then_some(ReleaseRole::Vendored))
    }

    /// Returns `true` when this crate participates in AST traceability scanning.
    #[must_use]
    pub fn participates_in_traceability(&self) -> bool {
        self.metadata
            .traceability
            .unwrap_or(!self.is_vendored_patch)
    }

    /// Returns the crate-relative source subdirectory scanned by traceability (`"src"` by default).
    #[must_use]
    pub fn traceability_subdir(&self) -> &str {
        self.metadata
            .traceability_subdir
            .as_deref()
            .unwrap_or("src")
    }

    /// Returns `true` when 4-part mathematical rustdoc contracts are enforced on this crate.
    #[must_use]
    pub fn has_math_contract(&self) -> bool {
        self.metadata.math_contract.unwrap_or(matches!(
            self.metadata.facade_role,
            FacadeRole::Engine | FacadeRole::Domain
        ))
    }
}

fn strip_line_comment(line: &str) -> &str {
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
        } else if ch == '"' && !in_single {
            in_double = !in_double;
        } else if ch == '\'' && !in_double {
            in_single = !in_single;
        } else if ch == '#' && !in_double && !in_single {
            return &line[..idx];
        }
    }
    line
}

fn unquote_str(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.len() >= 2
        && ((trimmed.starts_with('"') && trimmed.ends_with('"'))
            || (trimmed.starts_with('\'') && trimmed.ends_with('\'')))
    {
        Some(trimmed[1..trimmed.len() - 1].to_string())
    } else {
        None
    }
}

fn parse_bool_str(raw: &str) -> Option<bool> {
    match raw.trim() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn section_body_std(content: &str, section_name: &str) -> Option<String> {
    let header = format!("[{section_name}]");
    let mut in_sec = false;
    let mut out = String::new();
    for raw in content.lines() {
        let line = strip_line_comment(raw).trim();
        if line.starts_with('[') && line.ends_with(']') {
            if in_sec {
                break;
            }
            if line == header {
                in_sec = true;
            }
            continue;
        }
        if in_sec {
            out.push_str(line);
            out.push('\n');
        }
    }
    in_sec.then_some(out)
}

/// Extracts `[package].name` from a `Cargo.toml` string using only `std`.
#[must_use]
pub fn parse_package_name_std(manifest_content: &str) -> Option<String> {
    let body = section_body_std(manifest_content, "package")?;
    for line in body.lines() {
        if let Some((k, v)) = line.split_once('=')
            && k.trim() == "name"
        {
            return unquote_str(v);
        }
    }
    None
}

/// Extracts package names patched under `[patch.crates-io]` in a root `Cargo.toml`.
#[must_use]
pub fn parse_patched_crates_std(root_manifest_content: &str) -> Vec<String> {
    let Some(body) = section_body_std(root_manifest_content, "patch.crates-io") else {
        return Vec::new();
    };
    body.lines()
        .filter_map(|line| {
            let (k, _) = line.split_once('=')?;
            let key = k.trim().trim_matches('"').trim_matches('\'');
            (!key.is_empty()).then(|| key.to_string())
        })
        .collect()
}

/// Parses `[package.metadata.oxidize]` from a `Cargo.toml` string using only `std`.
#[must_use]
pub fn parse_oxidize_metadata_std(manifest_content: &str) -> OxidizeMetadata {
    let Some(body) = section_body_std(manifest_content, "package.metadata.oxidize") else {
        return OxidizeMetadata::default();
    };
    let mut meta = OxidizeMetadata::default();
    for line in body.lines() {
        let Some((raw_k, raw_v)) = line.split_once('=') else {
            continue;
        };
        match raw_k.trim() {
            "tier" => meta.tier = unquote_str(raw_v).and_then(|s| TierKind::parse_str(&s)),
            "release_role" => {
                meta.release_role = unquote_str(raw_v).and_then(|s| ReleaseRole::parse_str(&s));
            }
            "facade_role" => {
                if let Some(role) = unquote_str(raw_v).and_then(|s| FacadeRole::parse_str(&s)) {
                    meta.facade_role = role;
                }
            }
            "facade_module" => meta.facade_module = unquote_str(raw_v),
            "traceability" => meta.traceability = parse_bool_str(raw_v),
            "traceability_subdir" => meta.traceability_subdir = unquote_str(raw_v),
            "math_contract" => meta.math_contract = parse_bool_str(raw_v),
            _ => {}
        }
    }
    meta
}

fn parse_members_list_std(root_toml: &str) -> Result<Vec<String>, String> {
    let ws_body = section_body_std(root_toml, "workspace")
        .ok_or_else(|| "Could not find [workspace] section in Cargo.toml".to_string())?;
    for (idx, line) in ws_body.lines().enumerate() {
        if let Some((k, rest)) = line.split_once('=')
            && k.trim() == "members"
        {
            let tail: String = std::iter::once(rest)
                .chain(ws_body.lines().skip(idx + 1))
                .collect::<Vec<_>>()
                .join("\n");
            let open = tail
                .find('[')
                .ok_or_else(|| "Malformed workspace.members array".to_string())?;
            let close = tail[open + 1..]
                .find(']')
                .ok_or_else(|| "Unterminated workspace.members array".to_string())?;
            let mut out = Vec::new();
            let mut chars = tail[open + 1..open + 1 + close].chars();
            while let Some(ch) = chars.next() {
                if ch == '"' || ch == '\'' {
                    let quote = ch;
                    let mut item = String::new();
                    for next in chars.by_ref() {
                        if next == quote {
                            out.push(item);
                            break;
                        }
                        item.push(next);
                    }
                }
            }
            return Ok(out);
        }
    }
    Err("Could not find workspace.members array in Cargo.toml".to_string())
}

/// Discovers all workspace member packages and their `[package.metadata.oxidize]` declarations using only `std`.
pub fn discover_workspace_packages_std(
    workspace_root: &Path,
) -> Result<Vec<WorkspacePackageMeta>, String> {
    let root_manifest = workspace_root.join("Cargo.toml");
    let root_toml = fs::read_to_string(&root_manifest)
        .map_err(|e| format!("Failed to read {}: {e}", root_manifest.display()))?;
    let members = parse_members_list_std(&root_toml)?;
    let patched = parse_patched_crates_std(&root_toml);
    let mut pkgs = Vec::with_capacity(members.len());
    for member in members {
        let norm_member = member
            .replace('\\', "/")
            .trim_start_matches("./")
            .trim_end_matches('/')
            .to_string();
        let manifest_path = workspace_root.join(&norm_member).join("Cargo.toml");
        let manifest = fs::read_to_string(&manifest_path)
            .map_err(|e| format!("Failed to read {}: {e}", manifest_path.display()))?;
        let package_name = parse_package_name_std(&manifest)
            .ok_or_else(|| format!("Missing package.name in {}", manifest_path.display()))?;
        let metadata = parse_oxidize_metadata_std(&manifest);
        let is_vendored_patch = patched.iter().any(|p| p == &package_name);
        pkgs.push(WorkspacePackageMeta {
            member_path: norm_member,
            package_name,
            metadata,
            is_vendored_patch,
        });
    }
    Ok(pkgs)
}
