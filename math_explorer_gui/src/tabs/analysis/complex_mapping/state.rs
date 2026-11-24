#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::framework::shader_foundry::{bind_complex_shader_params, validate_complex_shader_vars};
use crate::framework::MAX_SYNTHESIZED_PARAMS;
use math_commons::ast::{LexError, MAX_EXPR_BYTES};
use math_commons::math_kernel::colormap::ShaderPalette;
use std::collections::BTreeMap;

pub(super) fn reject_invalid_formula(err: impl std::fmt::Display) -> String {
    let msg = format!(
        "Invalid complex mapping formula: {err}. Simplify or fix the expression and retry."
    );
    crate::accessibility::announce_status_with_priority(&msg, "assertive");
    msg
}

const fn default_domain_coloring() -> bool {
    true
}

const fn is_true(v: &bool) -> bool {
    *v
}

fn default_phase_palette_slug() -> String {
    ShaderPalette::PhaseWheel.as_slug().to_string()
}

fn is_default_phase_palette(slug: &str) -> bool {
    ShaderPalette::from_slug(slug) == Some(ShaderPalette::PhaseWheel)
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ComplexMappingState {
    pub(super) function: MappingFunction,
    pub(super) grid_min: f64,
    pub(super) grid_max: f64,
    pub(super) grid_density: usize,
    pub(super) inspection_point: [f64; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(super) custom_params: BTreeMap<String, f64>,
    #[serde(default = "default_domain_coloring", skip_serializing_if = "is_true")]
    pub(super) domain_coloring: bool,
    #[serde(
        default = "default_phase_palette_slug",
        skip_serializing_if = "is_default_phase_palette"
    )]
    pub(super) palette: String,
}

impl ComplexMappingState {
    fn validate_bounds(&self) -> Result<(), String> {
        if !self.grid_min.is_finite()
            || !self.grid_max.is_finite()
            || !(COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX).contains(&self.grid_min)
            || !(COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX).contains(&self.grid_max)
            || self.grid_min >= self.grid_max
            || !(2..=math_commons::registry::MAX_GRID_SIZE).contains(&self.grid_density)
            || !self
                .inspection_point
                .iter()
                .all(|v| v.is_finite() && (COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX).contains(v))
        {
            return Err("Complex mapping bounds or density out of valid range".into());
        }
        if self.custom_params.len() > MAX_SYNTHESIZED_PARAMS
            || !self.custom_params.iter().all(|(k, &v)| {
                !k.is_empty() && k.len() <= 64 && v.is_finite() && (-1.0e4..=1.0e4).contains(&v)
            })
        {
            return Err("Custom complex parameters out of valid range".into());
        }
        Ok(())
    }

    pub(super) fn validate(&self) -> Result<ShaderPalette, String> {
        self.validate_bounds()?;
        let palette = ShaderPalette::from_slug(&self.palette)
            .ok_or_else(|| format!("Unknown complex mapping palette '{}'", self.palette))?;
        if let Some(raw) = &self.custom_expr {
            if raw.len() > MAX_EXPR_BYTES {
                return Err(reject_invalid_formula(
                    LexError::input_too_long(raw.len()).user_message(),
                ));
            }
        }
        if self.function == MappingFunction::Custom {
            let raw = self
                .custom_expr
                .as_deref()
                .unwrap_or(DEFAULT_CUSTOM_MAPPING_EXPR);
            let mut probe = default_complex_math_input();
            probe.set_expression(raw);
            probe.set_params_map(&self.custom_params);
            if let Some(err) = probe.error_message() {
                return Err(reject_invalid_formula(err));
            }
            let expr = probe.expr().expect("validated custom complex AST");
            let bound = bind_complex_shader_params(expr, &probe.params_map(), &["z"])
                .map_err(reject_invalid_formula)?;
            validate_complex_shader_vars(&bound).map_err(reject_invalid_formula)?;
        }
        Ok(palette)
    }
}

impl ComplexMappingTool {
    pub(super) fn to_serializable_state(&self) -> ComplexMappingState {
        let is_custom = self.function == MappingFunction::Custom;
        let custom_expr = (is_custom
            || self.math_input.raw_text.trim() != DEFAULT_CUSTOM_MAPPING_EXPR)
            .then(|| self.math_input.raw_text.clone());
        let custom_params = if is_custom {
            self.math_input.params_map()
        } else {
            BTreeMap::new()
        };
        ComplexMappingState {
            function: self.function,
            grid_min: self.grid_min,
            grid_max: self.grid_max,
            grid_density: self.grid_density,
            inspection_point: self.inspection_point,
            custom_expr,
            custom_params,
            domain_coloring: self.domain_coloring,
            palette: self.palette.as_slug().to_string(),
        }
    }
}

impl scientific_metadata::theory::TheoryDescribable for ComplexMappingTool {
    fn theory_description(&self) -> String {
        let fn_desc = match self.function {
            MappingFunction::ZSquared => "quadratic power map $w = z^2$",
            MappingFunction::Exponential => "entire exponential map $w = e^z$",
            MappingFunction::Sine => "complex trigonometric map $w = \\sin(z)$",
            MappingFunction::Inverse => "inversion map $w = 1/z$",
            MappingFunction::Custom => "custom holomorphic map $w = f(z)$",
        };
        format!(
            "Conformal complex mapping {} preserving angles and orientation under Cauchy-Riemann equations with phase-modulus domain coloring and local scaling factor $|f'(z)|$.",
            fn_desc
        )
    }

    fn phonetic_description(&self) -> String {
        "Conformal complex mapping and domain coloring theory".into()
    }

    fn theory_citation(&self) -> String {
        "Visual Complex Analysis (Needham, 1997); Wegert, E. (2012). Visual Complex Functions: An Introduction with Phase Portraits.".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "ZSquared".into(),
            "w = z^2 doubling angles at the origin".into(),
        );
        map.insert(
            "Exponential".into(),
            "w = e^z mapping vertical lines to circles".into(),
        );
        map.insert(
            "Sine".into(),
            "w = sin(z) mapping strips to upper half-plane".into(),
        );
        map.insert(
            "Inverse".into(),
            "w = 1/z conformal inversion through unit circle".into(),
        );
        map
    }
}
