//! LaTeX, Rust, and Python code exporters and overlay actions for [`crate::fit::FitResult`].

use crate::a11y::{AccessibleTooltipExt as _, focusable_action_button};
#[cfg(not(test))]
use crate::export::save_file_dialog;
use crate::fit::{FitResult, FitType};
use egui::{Button, Ui};

fn format_py_f64(val: f64) -> String {
    let raw = format!("{val}");
    if !raw.contains('.') && !raw.contains('e') && !raw.contains('E') {
        format!("{raw}.0")
    } else {
        raw
    }
}

fn format_rs_f64(val: f64) -> String {
    format!("{}_f64", format_py_f64(val))
}

fn dataset_slug(name: &str) -> String {
    let slug: String = name
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let trimmed = slug.trim_matches('_');
    if trimmed.is_empty() {
        "regression".to_string()
    } else {
        trimmed.to_string()
    }
}

fn required_coeff_count(fit_type: FitType) -> usize {
    match fit_type {
        FitType::Linear | FitType::Polynomial(_) => 1,
        FitType::Exponential | FitType::PowerLaw | FitType::Logarithmic => 2,
        FitType::Logistic => 3,
        FitType::Sinusoidal => 4,
    }
}

fn format_tex_num(val: f64) -> String {
    let raw = format_py_f64(val);
    raw.strip_suffix(".0").unwrap_or(&raw).to_string()
}

impl FitResult {
    /// Returns `true` when the regression contains non-empty finite coefficients and finite $R^2$.
    #[must_use]
    pub fn has_finite_coefficients(&self) -> bool {
        let centered_ok = self.poly_centered.as_ref().is_none_or(|(m, s, c)| {
            m.is_finite()
                && s.is_finite()
                && *s > 0.0
                && !c.is_empty()
                && c.iter().all(|v| v.is_finite())
        });
        self.coefficients.len() >= required_coeff_count(self.fit_type)
            && self.coefficients.iter().all(|c| c.is_finite())
            && self.r_squared.is_finite()
            && centered_ok
    }

    fn validate_exportable(&self) -> Result<(), String> {
        if self.coefficients.len() < required_coeff_count(self.fit_type) {
            return Err("FitResult contains insufficient model coefficients.".to_string());
        }
        if !self.has_finite_coefficients() {
            return Err("FitResult contains non-finite coefficients or R².".to_string());
        }
        Ok(())
    }

    /// Formats the fitted model as a calibrated `\begin{aligned}` LaTeX block.
    pub fn to_latex(&self) -> Result<String, String> {
        self.validate_exportable()?;
        let eq_line = match self.fit_type {
            FitType::Linear | FitType::Polynomial(_) => {
                let u_sym = if self.log_x {
                    r"\log_{10}\left(x\right)"
                } else {
                    "x"
                };
                let poly_latex = format_latex_polynomial(&self.coefficients, u_sym, self.log_x);
                if self.log_y {
                    format!(r"y(x) &= 10^{{{poly_latex}}}")
                } else {
                    format!("y(x) &= {poly_latex}")
                }
            }
            _ => format!(
                "y(x) &= {}",
                format_nonlinear_latex(self.fit_type, &self.coefficients)
            ),
        };
        Ok(format!(
            "\\begin{{aligned}}\n{eq_line} \\\\\nR^2 &= {:.6}\n\\end{{aligned}}",
            self.r_squared
        ))
    }

    /// Formats the fitted model as a standalone zero-dependency Rust 2024 source file.
    pub fn to_rust_fn(&self) -> Result<String, String> {
        self.validate_exportable()?;
        let sample_x =
            if self.log_x || matches!(self.fit_type, FitType::PowerLaw | FitType::Logarithmic) {
                "2.0_f64"
            } else {
                "1.0_f64"
            };
        let body = match self.fit_type {
            FitType::Linear | FitType::Polynomial(_) => format_poly_rust_body(self),
            _ => format_nonlinear_rust_body(self.fit_type, &self.coefficients),
        };
        Ok(format!(
            "pub fn fit_eval(x: f64) -> f64 {{\n\
             {body}\n\
             }}\n\n\
             fn main() {{\n\
             \x20   let y = fit_eval({sample_x});\n\
             \x20   println!(\"y({sample_x}) = {{y:.9}}\");\n\
             }}\n"
        ))
    }

    /// Formats the fitted model as a runnable `numpy` Python evaluator script.
    pub fn to_python_fn(&self) -> Result<String, String> {
        self.validate_exportable()?;
        let sample_x =
            if self.log_x || matches!(self.fit_type, FitType::PowerLaw | FitType::Logarithmic) {
                "2.0"
            } else {
                "1.0"
            };
        let body = match self.fit_type {
            FitType::Linear | FitType::Polynomial(_) => format_poly_python_body(self),
            _ => format_nonlinear_python_body(self.fit_type, &self.coefficients),
        };
        Ok(format!(
            "import numpy as np\n\n\n\
             def fit_eval(x: float) -> float:\n\
             {body}\n\n\n\
             if __name__ == \"__main__\":\n\
             \x20   print(f\"y({sample_x}) = {{fit_eval({sample_x}):.9f}}\")\n"
        ))
    }
}

fn format_poly_rust_body(fit: &FitResult) -> String {
    let (z_binding, active_coeffs): (String, &[f64]) =
        if let Some((u_mean, u_scale, ref centered)) = fit.poly_centered {
            (
                format!(
                    "    let z = (u - {}) / {};",
                    format_rs_f64(u_mean),
                    format_rs_f64(u_scale)
                ),
                centered.as_slice(),
            )
        } else {
            ("    let z = u;".to_string(), fit.coefficients.as_slice())
        };
    let coeffs_literal = active_coeffs
        .iter()
        .map(|&c| format_rs_f64(c))
        .collect::<Vec<_>>()
        .join(", ");
    let u_binding = if fit.log_x {
        "    if !x.is_finite() || x <= 0.0 {\n        return f64::NAN;\n    }\n    let u = x.log10();"
    } else {
        "    let u = x;"
    };
    let ret_expr = if fit.log_y {
        "    10.0_f64.powf(poly.clamp(-300.0, 300.0)).clamp(1e-300, 1e300)"
    } else {
        "    poly"
    };
    format!(
        "{u_binding}\n\
         {z_binding}\n\
         \x20   let coeffs: [f64; {}] = [{coeffs_literal}];\n\
         \x20   let mut poly = 0.0_f64;\n\
         \x20   for coeff in coeffs.into_iter().rev() {{\n\
         \x20       poly = poly * z + coeff;\n\
         \x20   }}\n\
         {ret_expr}",
        active_coeffs.len()
    )
}

fn format_poly_python_body(fit: &FitResult) -> String {
    let (z_line, active_coeffs): (String, &[f64]) =
        if let Some((u_mean, u_scale, ref centered)) = fit.poly_centered {
            (
                format!(
                    "    z = (u - {}) / {}",
                    format_py_f64(u_mean),
                    format_py_f64(u_scale)
                ),
                centered.as_slice(),
            )
        } else {
            ("    z = u".to_string(), fit.coefficients.as_slice())
        };
    let coeffs_literal = active_coeffs
        .iter()
        .map(|&c| format_py_f64(c))
        .collect::<Vec<_>>()
        .join(", ");
    let u_lines = if fit.log_x {
        "    if not np.isfinite(x) or x <= 0.0:\n        return float(\"nan\")\n    u = float(np.log10(x))"
    } else {
        "    u = float(x)"
    };
    let ret_line = if fit.log_y {
        "    return float(np.clip(10.0 ** np.clip(poly, -300.0, 300.0), 1e-300, 1e300))"
    } else {
        "    return float(poly)"
    };
    format!(
        "{u_lines}\n\
         {z_line}\n\
         \x20   coeffs = [{coeffs_literal}]\n\
         \x20   poly = 0.0\n\
         \x20   for coeff in reversed(coeffs):\n\
         \x20       poly = poly * z + coeff\n\
         {ret_line}"
    )
}

fn format_nonlinear_latex(fit_type: FitType, c: &[f64]) -> String {
    match fit_type {
        FitType::Exponential => {
            format!(
                r"{} \cdot \exp\left({} \cdot x\right)",
                format_tex_num(c[0]),
                format_tex_num(c[1])
            )
        }
        FitType::PowerLaw => {
            format!(
                r"{} \cdot x^{{{}}}",
                format_tex_num(c[0]),
                format_tex_num(c[1])
            )
        }
        FitType::Logarithmic => {
            format!(
                r"{} \cdot \ln\left(x\right) + {}",
                format_tex_num(c[0]),
                format_tex_num(c[1])
            )
        }
        FitType::Logistic => format!(
            r"\frac{{{}}}{{1 + {} \cdot \exp\left(-{} \cdot x\right)}}",
            format_tex_num(c[0]),
            format_tex_num(c[1]),
            format_tex_num(c[2])
        ),
        FitType::Sinusoidal => format!(
            r"{} \cdot \sin\left({} \cdot x + {}\right) + {}",
            format_tex_num(c[0]),
            format_tex_num(c[1]),
            format_tex_num(c[2]),
            format_tex_num(c[3])
        ),
        FitType::Linear | FitType::Polynomial(_) => String::new(),
    }
}

fn format_nonlinear_rust_body(fit_type: FitType, c: &[f64]) -> String {
    match fit_type {
        FitType::Exponential => format!(
            "    let arg = ({}) * x;\n    if !x.is_finite() || !arg.is_finite() || arg > 700.0_f64 {{\n        return f64::NAN;\n    }}\n    ({}) * arg.exp()",
            format_rs_f64(c[1]),
            format_rs_f64(c[0])
        ),
        FitType::PowerLaw => format!(
            "    if !x.is_finite() || x <= 0.0_f64 {{\n        return f64::NAN;\n    }}\n    ({}) * x.powf({})",
            format_rs_f64(c[0]),
            format_rs_f64(c[1])
        ),
        FitType::Logarithmic => format!(
            "    if !x.is_finite() || x <= 0.0_f64 {{\n        return f64::NAN;\n    }}\n    ({}) * x.ln() + ({})",
            format_rs_f64(c[0]),
            format_rs_f64(c[1])
        ),
        FitType::Logistic => format!(
            "    if !x.is_finite() {{\n        return f64::NAN;\n    }}\n    let denom = 1.0_f64 + ({}) * (-({}) * x).clamp(-700.0_f64, 700.0_f64).exp();\n    if !denom.is_finite() || denom.abs() <= 1e-15_f64 {{\n        return f64::NAN;\n    }}\n    ({}) / denom",
            format_rs_f64(c[1]),
            format_rs_f64(c[2]),
            format_rs_f64(c[0])
        ),
        FitType::Sinusoidal => format!(
            "    let phase = ({}) * x + ({});\n    if !x.is_finite() || !phase.is_finite() {{\n        return f64::NAN;\n    }}\n    ({}) * phase.sin() + ({})",
            format_rs_f64(c[1]),
            format_rs_f64(c[2]),
            format_rs_f64(c[0]),
            format_rs_f64(c[3])
        ),
        FitType::Linear | FitType::Polynomial(_) => "    f64::NAN".to_string(),
    }
}

fn format_nonlinear_python_body(fit_type: FitType, c: &[f64]) -> String {
    match fit_type {
        FitType::Exponential => format!(
            "    arg = ({}) * float(x)\n    if not np.isfinite(x) or not np.isfinite(arg) or arg > 700.0:\n        return float(\"nan\")\n    return float(({}) * np.exp(arg))",
            format_py_f64(c[1]),
            format_py_f64(c[0])
        ),
        FitType::PowerLaw => format!(
            "    if not np.isfinite(x) or x <= 0.0:\n        return float(\"nan\")\n    return float(({}) * (float(x) ** ({})))",
            format_py_f64(c[0]),
            format_py_f64(c[1])
        ),
        FitType::Logarithmic => format!(
            "    if not np.isfinite(x) or x <= 0.0:\n        return float(\"nan\")\n    return float(({}) * np.log(float(x)) + ({}))",
            format_py_f64(c[0]),
            format_py_f64(c[1])
        ),
        FitType::Logistic => format!(
            "    if not np.isfinite(x):\n        return float(\"nan\")\n    denom = 1.0 + ({}) * float(np.exp(np.clip(-({}) * float(x), -700.0, 700.0)))\n    if not np.isfinite(denom) or abs(denom) <= 1e-15:\n        return float(\"nan\")\n    return float(({}) / denom)",
            format_py_f64(c[1]),
            format_py_f64(c[2]),
            format_py_f64(c[0])
        ),
        FitType::Sinusoidal => format!(
            "    phase = ({}) * float(x) + ({})\n    if not np.isfinite(x) or not np.isfinite(phase):\n        return float(\"nan\")\n    return float(({}) * np.sin(phase) + ({}))",
            format_py_f64(c[1]),
            format_py_f64(c[2]),
            format_py_f64(c[0]),
            format_py_f64(c[3])
        ),
        FitType::Linear | FitType::Polynomial(_) => "    return float(\"nan\")".to_string(),
    }
}

fn format_latex_polynomial(coeffs: &[f64], u_sym: &str, wrap_u: bool) -> String {
    let mut terms = Vec::new();
    for (power, &coeff) in coeffs.iter().enumerate().rev() {
        if coeff == 0.0 && coeffs.len() > 1 && !terms.is_empty() {
            continue;
        }
        let is_first = terms.is_empty();
        let sign = if is_first {
            if coeff < 0.0 { "-" } else { "" }
        } else if coeff < 0.0 {
            " - "
        } else {
            " + "
        };
        let mag = format_py_f64(coeff.abs());
        let mag_clean = mag.strip_suffix(".0").unwrap_or(&mag);
        let term = match power {
            0 => format!("{sign}{mag_clean}"),
            1 => format!(r"{sign}{mag_clean} \cdot {u_sym}"),
            p => {
                let base = if wrap_u {
                    format!(r"\left({u_sym}\right)")
                } else {
                    u_sym.to_string()
                };
                format!(r"{sign}{mag_clean} \cdot {{{base}}}^{{{p}}}")
            }
        };
        terms.push(term);
    }
    if terms.is_empty() {
        "0".to_string()
    } else {
        terms.join("")
    }
}

pub(crate) fn render_fit_overlay_actions(
    ui: &mut Ui,
    fit_res: &FitResult,
    toast: &mut Option<(String, bool)>,
) {
    let enabled = fit_res.has_finite_coefficients();
    let slug = dataset_slug(&fit_res.dataset_name);

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        if focusable_action_button(
            ui,
            enabled,
            Button::new("📋 .tex").small(),
            Some("Copy Fit LaTeX"),
        )
        .accessible_tooltip("Copy fitted LaTeX equation to clipboard")
        .clicked()
        {
            handle_fit_copy(ui, fit_res.to_latex(), "LaTeX fit equation", toast);
        }
        if focusable_action_button(
            ui,
            enabled,
            Button::new("📋 .rs").small(),
            Some("Copy Fit Rust"),
        )
        .accessible_tooltip("Copy standalone Rust fit evaluator to clipboard")
        .clicked()
        {
            handle_fit_copy(ui, fit_res.to_rust_fn(), "Rust fit evaluator", toast);
        }
        if focusable_action_button(
            ui,
            enabled,
            Button::new("📋 .py").small(),
            Some("Copy Fit Python"),
        )
        .accessible_tooltip("Copy Python numpy fit evaluator to clipboard")
        .clicked()
        {
            handle_fit_copy(ui, fit_res.to_python_fn(), "Python fit evaluator", toast);
        }
    });

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        if focusable_action_button(
            ui,
            enabled,
            Button::new("💾 .tex").small(),
            Some("Save Fit .tex"),
        )
        .accessible_tooltip("Save fitted LaTeX equation as .tex file")
        .clicked()
        {
            handle_fit_save(
                ui,
                fit_res.to_latex(),
                &format!("{slug}_fit.tex"),
                "LaTeX Equation",
                "tex",
                toast,
            );
        }
        if focusable_action_button(
            ui,
            enabled,
            Button::new("💾 .rs").small(),
            Some("Save Fit .rs"),
        )
        .accessible_tooltip("Save standalone Rust fit evaluator as .rs file")
        .clicked()
        {
            handle_fit_save(
                ui,
                fit_res.to_rust_fn(),
                &format!("{slug}_fit.rs"),
                "Rust Source",
                "rs",
                toast,
            );
        }
        if focusable_action_button(
            ui,
            enabled,
            Button::new("💾 .py").small(),
            Some("Save Fit .py"),
        )
        .accessible_tooltip("Save Python numpy fit evaluator as .py file")
        .clicked()
        {
            handle_fit_save(
                ui,
                fit_res.to_python_fn(),
                &format!("{slug}_fit.py"),
                "Python Script",
                "py",
                toast,
            );
        }
    });
}

fn handle_fit_copy(
    ui: &Ui,
    payload: Result<String, String>,
    label: &str,
    toast: &mut Option<(String, bool)>,
) {
    match payload {
        Ok(text) => {
            ui.ctx().copy_text(text.clone());
            #[cfg(not(test))]
            let _ = crate::export::copy_text_to_clipboard(&text);
            let msg = format!("Copied {label} to clipboard.");
            *toast = Some((msg.clone(), false));
            crate::emit_plot_live_message(ui.ctx(), msg, false);
        }
        Err(err) => {
            *toast = Some((err.clone(), true));
            crate::emit_plot_live_message(ui.ctx(), err, true);
        }
    }
}

fn handle_fit_save(
    ui: &Ui,
    payload: Result<String, String>,
    filename: &str,
    filter_name: &str,
    ext: &str,
    toast: &mut Option<(String, bool)>,
) {
    match payload {
        Ok(text) => {
            #[cfg(test)]
            let _ = (&text, filter_name, ext);
            #[cfg(not(test))]
            save_file_dialog(filename, text.as_bytes(), filter_name, &[ext]);
            let msg = format!("Saved {filename}.");
            *toast = Some((msg.clone(), false));
            crate::emit_plot_live_message(ui.ctx(), msg, false);
        }
        Err(err) => {
            *toast = Some((err.clone(), true));
            crate::emit_plot_live_message(ui.ctx(), err, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::fit_dataset_scaled;
    #[cfg(not(target_arch = "wasm32"))]
    use std::process::Command;

    #[test]
    fn test_fit_badge_overlay_copy_and_save_actions_emit_live_region() {
        let pts = vec![[1.0, 2.0], [2.0, 5.0], [3.0, 10.0], [4.0, 17.0]];
        let fit = fit_dataset_scaled(&pts, 2, "Sensor Curve", false, false).expect("fit");
        let ctx = egui::Context::default();
        ctx.enable_accesskit();

        let screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(800.0, 400.0),
        ));
        let mut toast = None;
        let out = ctx.run(
            egui::RawInput {
                screen_rect,
                ..Default::default()
            },
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    render_fit_overlay_actions(ui, &fit, &mut toast);
                });
            },
        );
        let ak = out.platform_output.accesskit_update.as_ref().unwrap();
        let rust_pos = ak
            .nodes
            .iter()
            .find_map(|(_, n)| {
                (n.role() == egui::accesskit::Role::Button && n.label() == Some("Copy Fit Rust"))
                    .then(|| n.bounds().unwrap())
            })
            .map(|b| egui::pos2(((b.x0 + b.x1) * 0.5) as f32, ((b.y0 + b.y1) * 0.5) as f32))
            .expect("Copy Fit Rust button");

        for pressed in [true, false] {
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect,
                    events: vec![
                        egui::Event::PointerMoved(rust_pos),
                        egui::Event::PointerButton {
                            pos: rust_pos,
                            button: egui::PointerButton::Primary,
                            pressed,
                            modifiers: egui::Modifiers::NONE,
                        },
                    ],
                    ..Default::default()
                },
                |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        render_fit_overlay_actions(ui, &fit, &mut toast);
                    });
                },
            );
        }
        assert_eq!(
            toast.as_ref().map(|(m, err)| (m.as_str(), *err)),
            Some(("Copied Rust fit evaluator to clipboard.", false))
        );
        assert!(
            ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")))
                .is_some()
        );
    }

    #[test]
    fn test_fit_export_rejects_empty_or_nonfinite_coefficients() {
        let mut bad = FitResult {
            fit_type: FitType::Linear,
            coefficients: vec![1.0, f64::NAN],
            r_squared: 0.99,
            rmse: 0.0,
            residuals: Vec::new(),
            degree: 1,
            formula: "y = x".to_string(),
            dataset_name: "Bad".to_string(),
            log_x: false,
            log_y: false,
            log_mode: false,
            poly_centered: None,
        };
        assert!(!bad.has_finite_coefficients());
        assert!(bad.to_latex().is_err());
        assert!(bad.to_rust_fn().is_err());
        assert!(bad.to_python_fn().is_err());

        bad.coefficients.clear();
        assert!(bad.to_latex().is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_fit_export_linear_semilog_and_loglog_parity_and_rustc_2024() {
        let pts = vec![
            [1.0, 2.5],
            [2.0, 7.8],
            [5.0, 38.4],
            [10.0, 145.2],
            [20.0, 560.9],
        ];
        let sample_xs = [0.5_f64, 1.0, 3.5, 12.0, 25.0];

        for (deg, log_x, log_y) in [
            (2, false, false),
            (2, true, false),
            (1, false, true),
            (2, true, true),
        ] {
            let fit = fit_dataset_scaled(&pts, deg, "Sensor Series", log_x, log_y)
                .expect("fit must succeed");
            let latex = fit.to_latex().expect("latex export");
            assert!(latex.contains(r"\begin{aligned}") && latex.contains("R^2 &="));
            if log_x {
                assert!(latex.contains(r"\log_{10}"));
            }
            if log_y {
                assert!(latex.contains("10^{"));
            }

            verify_fit_rust_parity(&fit, &sample_xs);
            verify_fit_python_parity(&fit, &sample_xs);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn verify_fit_rust_parity(fit: &FitResult, sample_xs: &[f64]) {
        static NEXT_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let run_id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "fit_export_rs_{}_{}_{}_{run_id}",
            std::process::id(),
            fit.log_x as u8,
            fit.log_y as u8
        ));
        let _ = std::fs::create_dir_all(&dir);
        let src_path = dir.join("fit_model.rs");
        let lib_rlib = dir.join("libfit_model.rlib");
        let harness_path = dir.join("harness.rs");
        let exe_ext = std::env::consts::EXE_SUFFIX;
        let standalone_bin = dir.join(format!("standalone_bin{exe_ext}"));
        let bin_path = dir.join(format!("harness_bin{exe_ext}"));
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());

        let rs_src = fit.to_rust_fn().expect("rust fit source");
        std::fs::write(&src_path, &rs_src).expect("write fit_model.rs");

        // Verify standalone binary compilation on Rust 2024
        let bin_check = Command::new(&rustc)
            .args(["--edition", "2024", "-D", "warnings"])
            .arg(&src_path)
            .arg("-o")
            .arg(&standalone_bin)
            .output()
            .expect("rustc standalone");
        assert!(
            bin_check.status.success(),
            "standalone fit .rs must compile on Rust 2024:\n{}",
            String::from_utf8_lossy(&bin_check.stderr)
        );

        // Also compile as library and verify numerical parity against FitResult::eval
        let lib_check = Command::new(&rustc)
            .args([
                "--edition",
                "2024",
                "--crate-type",
                "rlib",
                "--crate-name",
                "fit_model",
                "-A",
                "dead_code",
            ])
            .arg(&src_path)
            .arg("-o")
            .arg(&lib_rlib)
            .output()
            .expect("rustc rlib");
        assert!(
            lib_check.status.success(),
            "fit rlib failed:\n{}",
            String::from_utf8_lossy(&lib_check.stderr)
        );

        let mut checks = String::from("extern crate fit_model;\nfn main() {\n");
        for &x in sample_xs {
            let expected = fit.eval(x);
            checks.push_str(&format!(
                "    let v = fit_model::fit_eval({x:?}_f64);\n\
                 \x20   assert!(((v - {expected:?}_f64).abs() / {expected:?}_f64.abs().max(1.0)) < 1e-9, \"Rust fit mismatch: {{v}} vs {expected:?}\");\n"
            ));
        }
        checks.push_str("}\n");
        std::fs::write(&harness_path, checks).expect("write harness");
        let h_out = Command::new(&rustc)
            .args(["--edition", "2024", "--extern"])
            .arg(format!("fit_model={}", lib_rlib.display()))
            .arg(&harness_path)
            .arg("-o")
            .arg(&bin_path)
            .output()
            .expect("compile harness");
        assert!(
            h_out.status.success(),
            "fit harness failed:\n{}",
            String::from_utf8_lossy(&h_out.stderr)
        );
        assert!(Command::new(&bin_path).status().unwrap().success());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn verify_fit_python_parity(fit: &FitResult, sample_xs: &[f64]) {
        let Some(py_cmd) = ["python3", "python"].into_iter().find(|cmd| {
            Command::new(cmd)
                .args(["-c", "import numpy"])
                .output()
                .is_ok_and(|o| o.status.success())
        }) else {
            return;
        };
        let mut script = fit.to_python_fn().expect("python fit source");
        script.push('\n');
        for &x in sample_xs {
            let expected = fit.eval(x);
            script.push_str(&format!(
                "assert abs(fit_eval({x:?}) - {expected:?}) / max(abs({expected:?}), 1.0) < 1e-9\n"
            ));
        }
        let status = Command::new(py_cmd)
            .args(["-c", &script])
            .status()
            .expect("python fit parity");
        assert!(status.success(), "Python fit_eval failed 1e-9 parity");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_nonlinear_fit_types_export_parity_and_rustc_2024() {
        let pos_pts = vec![
            [1.0, 2.2],
            [2.0, 4.9],
            [3.0, 9.8],
            [4.0, 17.5],
            [5.0, 28.0],
            [6.0, 39.2],
        ];
        let xs = [1.0_f64, 2.5, 4.5];
        for ft in [
            FitType::Exponential,
            FitType::PowerLaw,
            FitType::Logarithmic,
            FitType::Logistic,
            FitType::Sinusoidal,
        ] {
            let fit =
                crate::fit::fit_model(&pos_pts, ft, "Nonlinear", false).expect("nonlinear fit");
            assert!(fit.to_latex().unwrap().contains("R^2 &="));
            verify_fit_rust_parity(&fit, &xs);
            verify_fit_python_parity(&fit, &xs);
        }
    }
}
