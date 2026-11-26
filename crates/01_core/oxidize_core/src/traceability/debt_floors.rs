//! Ratcheted assertion density floor loader for workspace traceability gates.

use std::fmt;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

/// Typed error returned when parsing `[verification_debt]` density floors from TOML.
#[derive(Clone, Debug, PartialEq)]
pub enum DensityFloorsError {
    /// Underlying TOML document syntax failed to parse.
    InvalidToml(String),
    /// `[verification_debt]` exists in the document but is not a TOML table.
    InvalidVerificationDebtTable,
    /// One or more configured density floor values were negative or non-finite.
    InvalidDensityValue {
        /// Parsed `min_verified_density` floor.
        min_verified_density: f64,
        /// Parsed `min_total_density` floor.
        min_total_density: f64,
        /// Optional parsed `min_unverified_density` floor.
        min_unverified_density: Option<f64>,
    },
}

impl fmt::Display for DensityFloorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidToml(reason) => {
                write!(f, "Invalid TOML in verification_whitelist.toml: {reason}")
            }
            Self::InvalidVerificationDebtTable => {
                write!(f, "[verification_debt] must be a TOML table")
            }
            Self::InvalidDensityValue {
                min_verified_density,
                min_total_density,
                min_unverified_density,
            } => match min_unverified_density {
                Some(unverified) => write!(
                    f,
                    "[verification_debt] density floors must be finite and >= 0.0 (verified={min_verified_density}, total={min_total_density}, unverified={unverified})"
                ),
                None => write!(
                    f,
                    "[verification_debt] density floors must be finite and >= 0.0 (verified={min_verified_density}, total={min_total_density})"
                ),
            },
        }
    }
}

impl std::error::Error for DensityFloorsError {}

/// Minimum assertion density floors enforced during traceability evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityFloors {
    /// Minimum required assertion density (`verified_asserts / verified_funcs`) across verified functions.
    pub min_verified_density: f64,
    /// Minimum required overall assertion density (`total_asserts / total_funcs`) across production functions.
    pub min_total_density: f64,
    /// Optional minimum required assertion density across unverified functions.
    pub min_unverified_density: Option<f64>,
}

impl Default for DensityFloors {
    fn default() -> Self {
        Self {
            min_verified_density: 0.4,
            min_total_density: 0.0,
            min_unverified_density: None,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn find_workspace_whitelist_file() -> Option<PathBuf> {
    let local = Path::new("verification_whitelist.toml");
    if local.is_file() {
        return Some(local.to_path_buf());
    }
    crate::workspace::find_workspace_root()
        .map(|root| root.join("verification_whitelist.toml"))
        .filter(|p| p.is_file())
}

/// Parses `[verification_debt]` density floors from `verification_whitelist.toml` TOML content.
pub fn parse_density_floors_toml(
    content: &str,
) -> Result<Option<DensityFloors>, DensityFloorsError> {
    let parsed: toml::Table = content
        .parse()
        .map_err(|err: toml::de::Error| DensityFloorsError::InvalidToml(err.to_string()))?;
    let Some(debt_val) = parsed.get("verification_debt") else {
        return Ok(None);
    };
    let Some(debt) = debt_val.as_table() else {
        return Err(DensityFloorsError::InvalidVerificationDebtTable);
    };
    let min_verified_density = debt
        .get("min_verified_density")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .unwrap_or(0.4);
    let min_total_density = debt
        .get("min_total_density")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .unwrap_or(0.0);
    let min_unverified_density = debt
        .get("min_unverified_density")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)));
    if !min_verified_density.is_finite()
        || min_verified_density < 0.0
        || !min_total_density.is_finite()
        || min_total_density < 0.0
        || min_unverified_density.is_some_and(|u| !u.is_finite() || u < 0.0)
    {
        return Err(DensityFloorsError::InvalidDensityValue {
            min_verified_density,
            min_total_density,
            min_unverified_density,
        });
    }
    Ok(Some(DensityFloors {
        min_verified_density,
        min_total_density,
        min_unverified_density,
    }))
}

/// Loads ratcheted [`DensityFloors`] from `verification_whitelist.toml` when present,
/// falling back to [`DensityFloors::default`] outside a configured workspace.
pub fn load_workspace_density_floors() -> Result<DensityFloors, std::io::Error> {
    #[cfg(target_arch = "wasm32")]
    {
        Ok(DensityFloors::default())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let Some(path) = find_workspace_whitelist_file() else {
            return Ok(DensityFloors::default());
        };
        let content = std::fs::read_to_string(&path)?;
        let parsed = parse_density_floors_toml(&content)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string()))?;
        Ok(parsed.unwrap_or_default())
    }
}
