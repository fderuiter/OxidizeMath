#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::framework::shader_foundry::validate_complex_shader_vars;
use crate::framework::MAX_SYNTHESIZED_PARAMS;
use math_commons::ast::{
    is_universal_constant, parse_equation, BinaryOp, CompiledExpr, Expr, Func, UnaryOp,
    MAX_EVAL_DEPTH,
};
use math_commons::math_kernel::colormap::ShaderPalette;
use std::collections::BTreeMap;

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

enum BindFrame<'a> {
    Visit { expr: &'a Expr, depth: usize },
    Unary(UnaryOp),
    Binary(BinaryOp),
    Func { func: Func, arity: usize },
}

fn visit_bind_node<'a>(
    expr: &'a Expr,
    depth: usize,
    params: &BTreeMap<String, f64>,
    work: &mut Vec<BindFrame<'a>>,
    out: &mut Vec<Expr>,
) -> Result<(), String> {
    if depth > MAX_EVAL_DEPTH {
        return Err(format!(
            "Custom complex expression exceeds depth limit {MAX_EVAL_DEPTH}"
        ));
    }
    match expr {
        Expr::Number(v) => out.push(Expr::Number(*v)),
        Expr::Variable(name) => {
            let is_reserved = name.eq_ignore_ascii_case("z")
                || name.eq_ignore_ascii_case("i")
                || is_universal_constant(name);
            if !is_reserved {
                if let Some(&val) = params.get(name) {
                    if !val.is_finite() {
                        return Err(format!("Non-finite parameter '{name}'"));
                    }
                    out.push(Expr::Number(val));
                    return Ok(());
                }
            }
            out.push(Expr::Variable(name.clone()));
        }
        Expr::Unary { op, expr: inner } => {
            work.push(BindFrame::Unary(*op));
            work.push(BindFrame::Visit {
                expr: inner,
                depth: depth + 1,
            });
        }
        Expr::Binary { op, left, right } => {
            work.push(BindFrame::Binary(*op));
            work.push(BindFrame::Visit {
                expr: right,
                depth: depth + 1,
            });
            work.push(BindFrame::Visit {
                expr: left,
                depth: depth + 1,
            });
        }
        Expr::Function { func, args } => {
            work.push(BindFrame::Func {
                func: *func,
                arity: args.len(),
            });
            for arg in args.iter().rev() {
                work.push(BindFrame::Visit {
                    expr: arg,
                    depth: depth + 1,
                });
            }
        }
    }
    Ok(())
}

pub(super) fn bind_complex_shader_params(
    root: &Expr,
    params: &BTreeMap<String, f64>,
) -> Result<Expr, String> {
    let mut work = vec![BindFrame::Visit {
        expr: root,
        depth: 0,
    }];
    let mut out = Vec::new();
    while let Some(frame) = work.pop() {
        match frame {
            BindFrame::Visit { expr, depth } => {
                visit_bind_node(expr, depth, params, &mut work, &mut out)?;
            }
            BindFrame::Unary(op) => {
                let inner = out.pop().ok_or("Shader AST stack underflow")?;
                out.push(Expr::unary(op, inner));
            }
            BindFrame::Binary(op) => {
                let rhs = out.pop().ok_or("Shader AST stack underflow")?;
                let lhs = out.pop().ok_or("Shader AST stack underflow")?;
                out.push(Expr::binary(op, lhs, rhs));
            }
            BindFrame::Func { func, arity } => {
                let split = out
                    .len()
                    .checked_sub(arity)
                    .ok_or("Shader AST stack underflow")?;
                let args = out.split_off(split);
                out.push(Expr::function(func, args));
            }
        }
    }
    out.pop().ok_or_else(|| "Empty shader AST".to_string())
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
            if raw.len() > MAX_CUSTOM_EXPR_BYTES {
                return Err("Custom complex expression exceeds 512-byte limit".into());
            }
            let (_, expr) = parse_equation(raw.trim())
                .map_err(|e| format!("Invalid custom complex expression: {e}"))?;
            let compiled = CompiledExpr::compile_complex(&expr, &["z"])
                .map_err(|e| format!("Invalid custom complex expression: {e}"))?;
            if compiled.param_names().len() > MAX_SYNTHESIZED_PARAMS {
                return Err(format!(
                    "Custom complex expression exceeds {MAX_SYNTHESIZED_PARAMS} parameters"
                ));
            }
            let mut effective = BTreeMap::new();
            for name in compiled.param_names() {
                effective.insert(
                    name.clone(),
                    self.custom_params.get(name).copied().unwrap_or(1.0),
                );
            }
            let bound = bind_complex_shader_params(&expr, &effective)?;
            validate_complex_shader_vars(&bound)?;
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
