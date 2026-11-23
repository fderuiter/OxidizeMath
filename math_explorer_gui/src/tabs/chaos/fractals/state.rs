#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{COORD_MAX, COORD_MIN, ZOOM_MAX, ZOOM_MIN};
use std::collections::BTreeMap;

const fn default_linked_explorer() -> bool {
    true
}

const fn is_default_linked_explorer(v: &bool) -> bool {
    *v
}

/// Fractal family selector.
#[derive(PartialEq, Eq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum FractalMode {
    /// Mandelbrot set $z_{n+1} = z_n^2 + c$ from $z_0 = 0$.
    Mandelbrot,
    /// Julia set $z_{n+1} = z_n^2 + c$ for fixed complex constant $c$.
    Julia,
    /// Custom c-plane escape fractal $z_{n+1} = f(z_n, c)$ from $z_0 = 0$.
    CustomMandelbrot,
    /// Custom Julia escape fractal $z_{n+1} = f(z_n, c)$ for fixed complex constant $c$.
    CustomJulia,
}

impl FractalMode {
    /// Returns `true` when evaluating a user-authored [`super::CompiledFractalJob`].
    #[must_use]
    pub const fn is_custom(self) -> bool {
        matches!(self, Self::CustomMandelbrot | Self::CustomJulia)
    }

    /// Returns `true` when the complex constant $c$ slider applies.
    #[must_use]
    pub const fn uses_julia_c(self) -> bool {
        matches!(self, Self::Julia | Self::CustomJulia)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FractalState {
    pub(super) center_re: f64,
    pub(super) center_im: f64,
    pub(super) zoom: f64,
    pub(super) max_iter: u32,
    pub(super) mode: FractalMode,
    pub(super) julia_re: f64,
    pub(super) julia_im: f64,
    #[serde(
        default = "default_linked_explorer",
        skip_serializing_if = "is_default_linked_explorer"
    )]
    pub(super) linked_explorer: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(super) custom_params: BTreeMap<String, f64>,
}

impl FractalState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let coords_valid = [self.center_re, self.center_im, self.julia_re, self.julia_im]
            .iter()
            .all(|v| v.is_finite() && (COORD_MIN..=COORD_MAX).contains(v));
        let zoom_valid = self.zoom.is_finite() && (ZOOM_MIN..=ZOOM_MAX).contains(&self.zoom);
        let iter_valid =
            (10..=math_commons::registry::MAX_ITERATIONS as u32).contains(&self.max_iter);
        let params_valid = self.custom_params.values().all(|v| v.is_finite());
        if !coords_valid || !zoom_valid || !iter_valid || !params_valid {
            return Err("Fractal view parameters out of bounds".into());
        }
        Ok(())
    }
}
