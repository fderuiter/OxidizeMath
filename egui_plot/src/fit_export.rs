//! LaTeX, Rust, and Python code exporters and overlay actions for [`crate::fit::FitResult`].

use crate::a11y::{AccessibleTooltipExt as _, focusable_action_button};
#[cfg(not(test))]
use crate::export::save_file_dialog;
use crate::fit::FitResult;
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

impl FitResult {
    /// Returns `true` when the regression contains non-empty finite coefficients and finite $R^2$.
    #[must_use]
    pub fn has_finite_coefficients(&self) -> bool {
        !self.coefficients.is_empty()
            && self.coefficients.iter().all(|c| c.is_finite())
            && self.r_squared.is_finite()
    }

    fn validate_exportable(&self) -> Result<(), String> {
        if self.coefficients.is_empty() {
            return Err("FitResult contains no polynomial coefficients.".to_string());
        }
        if !self.coefficients.iter().all(|c| c.is_finite()) || !self.r_squared.is_finite() {
            return Err("FitResult contains non-finite coefficients or R².".to_string());
        }
        Ok(())
    }

    /// Formats the fitted model as a calibrated `\begin{aligned}` LaTeX block.
    pub fn to_latex(&self) -> Result<String, String> {
        self.validate_exportable()?;
        let u_sym = if self.log_x {
            r"\log_{10}\left(x\right)"
        } else {
            "x"
        };
        let poly_latex = format_latex_polynomial(&self.coefficients, u_sym, self.log_x);
        let eq_line = if self.log_y {
            format!(r"y(x) &= 10^{{{poly_latex}}}")
        } else {
            format!("y(x) &= {poly_latex}")
        };
        Ok(format!(
            "\\begin{{aligned}}\n{eq_line} \\\\\nR^2 &= {:.6}\n\\end{{aligned}}",
            self.r_squared
        ))
    }

    /// Formats the fitted model as a standalone zero-dependency Rust 2024 source file.
    pub fn to_rust_fn(&self) -> Result<String, String> {
        self.validate_exportable()?;
        let coeffs_literal = self
            .coefficients
            .iter()
            .map(|&c| format_rs_f64(c))
            .collect::<Vec<_>>()
            .join(", ");
        let u_binding = if self.log_x {
            "    if !x.is_finite() || x <= 0.0 {\n        return f64::NAN;\n    }\n    let u = x.log10();"
        } else {
            "    let u = x;"
        };
        let ret_expr = if self.log_y {
            "    10.0_f64.powf(poly.clamp(-300.0, 300.0)).clamp(1e-300, 1e300)"
        } else {
            "    poly"
        };
        let sample_x = if self.log_x { "10.0_f64" } else { "1.0_f64" };
        Ok(format!(
            "pub fn fit_eval(x: f64) -> f64 {{\n\
             {u_binding}\n\
             \x20   let coeffs: [f64; {}] = [{coeffs_literal}];\n\
             \x20   let mut poly = 0.0_f64;\n\
             \x20   let mut u_pow = 1.0_f64;\n\
             \x20   for coeff in coeffs {{\n\
             \x20       poly += coeff * u_pow;\n\
             \x20       u_pow *= u;\n\
             \x20   }}\n\
             {ret_expr}\n\
             }}\n\n\
             fn main() {{\n\
             \x20   let y = fit_eval({sample_x});\n\
             \x20   println!(\"y({sample_x}) = {{y:.9}}\");\n\
             }}\n",
            self.coefficients.len()
        ))
    }

    /// Formats the fitted model as a runnable `numpy` Python evaluator script.
    pub fn to_python_fn(&self) -> Result<String, String> {
        self.validate_exportable()?;
        let coeffs_literal = self
            .coefficients
            .iter()
            .map(|&c| format_py_f64(c))
            .collect::<Vec<_>>()
            .join(", ");
        let u_lines = if self.log_x {
            "    if not np.isfinite(x) or x <= 0.0:\n        return float(\"nan\")\n    u = float(np.log10(x))"
        } else {
            "    u = float(x)"
        };
        let ret_line = if self.log_y {
            "    return float(np.clip(10.0 ** np.clip(poly, -300.0, 300.0), 1e-300, 1e300))"
        } else {
            "    return float(poly)"
        };
        let sample_x = if self.log_x { "10.0" } else { "1.0" };
        Ok(format!(
            "import numpy as np\n\n\n\
             def fit_eval(x: float) -> float:\n\
             {u_lines}\n\
             \x20   coeffs = [{coeffs_literal}]\n\
             \x20   poly = 0.0\n\
             \x20   u_pow = 1.0\n\
             \x20   for coeff in coeffs:\n\
             \x20       poly += coeff * u_pow\n\
             \x20       u_pow *= u\n\
             {ret_line}\n\n\n\
             if __name__ == \"__main__\":\n\
             \x20   print(f\"y({sample_x}) = {{fit_eval({sample_x}):.9f}}\")\n"
        ))
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
            coefficients: vec![1.0, f64::NAN],
            r_squared: 0.99,
            degree: 1,
            formula: "y = x".to_string(),
            dataset_name: "Bad".to_string(),
            log_x: false,
            log_y: false,
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
        let dir = std::env::temp_dir().join(format!(
            "fit_export_rs_{}_{}_{}",
            std::process::id(),
            fit.log_x as u8,
            fit.log_y as u8
        ));
        let _ = std::fs::create_dir_all(&dir);
        let src_path = dir.join("fit_model.rs");
        let lib_rlib = dir.join("libfit_model.rlib");
        let harness_path = dir.join("harness.rs");
        let bin_path = dir.join("harness_bin");

        let rs_src = fit.to_rust_fn().expect("rust fit source");
        std::fs::write(&src_path, &rs_src).expect("write fit_model.rs");

        // Verify standalone binary compilation on Rust 2024
        let bin_check = Command::new("rustc")
            .args(["--edition", "2024", "-D", "warnings"])
            .arg(&src_path)
            .arg("-o")
            .arg(dir.join("standalone_bin"))
            .status()
            .expect("rustc standalone");
        assert!(
            bin_check.success(),
            "standalone fit .rs must compile on Rust 2024"
        );

        // Also compile as library and verify numerical parity against FitResult::eval
        let lib_check = Command::new("rustc")
            .args([
                "--edition",
                "2024",
                "--crate-type",
                "lib",
                "-A",
                "dead_code",
            ])
            .arg(&src_path)
            .arg("-o")
            .arg(&lib_rlib)
            .status()
            .expect("rustc rlib");
        assert!(lib_check.success());

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
        let h_status = Command::new("rustc")
            .args(["--edition", "2024", "--extern"])
            .arg(format!("fit_model={}", lib_rlib.display()))
            .arg(&harness_path)
            .arg("-o")
            .arg(&bin_path)
            .status()
            .expect("compile harness");
        assert!(h_status.success());
        assert!(Command::new(&bin_path).status().unwrap().success());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn verify_fit_python_parity(fit: &FitResult, sample_xs: &[f64]) {
        let Ok(check) = Command::new("python3")
            .args(["-c", "import numpy"])
            .status()
        else {
            return;
        };
        if !check.success() {
            return;
        }
        let mut script = fit.to_python_fn().expect("python fit source");
        script.push('\n');
        for &x in sample_xs {
            let expected = fit.eval(x);
            script.push_str(&format!(
                "assert abs(fit_eval({x:?}) - {expected:?}) / max(abs({expected:?}), 1.0) < 1e-9\n"
            ));
        }
        let status = Command::new("python3")
            .args(["-c", &script])
            .status()
            .expect("python3 fit parity");
        assert!(status.success(), "Python fit_eval failed 1e-9 parity");
    }
}
