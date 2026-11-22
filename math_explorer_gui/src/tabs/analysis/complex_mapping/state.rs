#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::framework::shader_foundry::validate_complex_shader_vars;
use math_commons::ast::parse_equation;
use math_commons::math_kernel::colormap::ShaderPalette;

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
    #[serde(default = "default_domain_coloring", skip_serializing_if = "is_true")]
    pub(super) domain_coloring: bool,
    #[serde(
        default = "default_phase_palette_slug",
        skip_serializing_if = "is_default_phase_palette"
    )]
    pub(super) palette: String,
}

impl ComplexMappingState {
    pub(super) fn validate(&self) -> Result<ShaderPalette, String> {
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
        let palette = ShaderPalette::from_slug(&self.palette)
            .ok_or_else(|| format!("Unknown complex mapping palette '{}'", self.palette))?;
        if let Some(raw) = &self.custom_expr {
            if raw.len() > MAX_CUSTOM_EXPR_BYTES {
                return Err("Custom complex expression exceeds 512-byte limit".into());
            }
            let (_, expr) = parse_equation(raw.trim())
                .map_err(|e| format!("Invalid custom complex expression: {e}"))?;
            validate_complex_shader_vars(&expr)?;
        }
        Ok(palette)
    }
}

impl ComplexMappingTool {
    pub(super) fn to_serializable_state(&self) -> ComplexMappingState {
        let custom_expr = (self.function == MappingFunction::Custom
            || self.math_input.raw_text.trim() != DEFAULT_CUSTOM_MAPPING_EXPR)
            .then(|| self.math_input.raw_text.clone());
        ComplexMappingState {
            function: self.function,
            grid_min: self.grid_min,
            grid_max: self.grid_max,
            grid_density: self.grid_density,
            inspection_point: self.inspection_point,
            custom_expr,
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
