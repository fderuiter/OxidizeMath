use super::*;
use egui::{Color32, FontFamily, FontId, TextStyle};

fn audit_font() -> FontId {
    FontId::new(14.0, FontFamily::Proportional)
}

fn collect_box_texts(b: &MathBox, out: &mut String) {
    match b {
        MathBox::Text { text, .. } => out.push_str(text),
        MathBox::Row(children) => {
            for c in children {
                collect_box_texts(c, out);
            }
        }
        MathBox::Fraction {
            numerator,
            denominator,
        } => {
            collect_box_texts(numerator, out);
            collect_box_texts(denominator, out);
        }
        MathBox::Sqrt(radicand) => collect_box_texts(radicand, out),
        MathBox::Script { base, sup, sub } => {
            if let Some(base) = base {
                collect_box_texts(base, out);
            }
            if let Some(sup) = sup {
                collect_box_texts(sup, out);
            }
            if let Some(sub) = sub {
                collect_box_texts(sub, out);
            }
        }
        MathBox::Delimited {
            open, body, close, ..
        } => {
            out.push_str(open.glyph());
            collect_box_texts(body, out);
            out.push_str(close.glyph());
        }
        MathBox::Accent { base, .. } => collect_box_texts(base, out),
    }
}

fn assert_no_leaked_latex_words(latex: &str) {
    let font = audit_font();
    let parsed = parse_latex_to_box(latex, &font, Color32::WHITE)
        .unwrap_or_else(|()| panic!("Failed to parse LaTeX: {latex}"));
    let mut box_text = String::new();
    collect_box_texts(&parsed, &mut box_text);
    let box_sanitized = box_text.replace("right-censored", "");

    let job = parse_latex_with_font(latex, font, Color32::WHITE);
    let job_sanitized = job.text.replace("right-censored", "");
    for leaked in [
        "left",
        "right",
        "dot",
        "dots",
        "odot",
        "mathbb",
        "mathrm",
        "mathbf",
        "mathcal",
        "operatorname",
    ] {
        assert!(
            !box_sanitized.contains(leaked),
            "2D MathBox text leaked '{leaked}' for {latex:?}: got {box_text:?}"
        );
        assert!(
            !job_sanitized.contains(leaked),
            "LayoutJob text leaked '{leaked}' for {latex:?}: got {:?}",
            job.text
        );
    }

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let resp = render_math_label(ui, latex, TextStyle::Body, Color32::WHITE);
            assert!(resp.rect.width() > 0.0 && resp.rect.height() > 0.0);
        });
    });
}

#[test]
fn test_audited_theory_and_ode_equations_render_without_leaked_commands() {
    let audited_formulas = [
        // ode.rs / ode/ui.rs / ode/systems.rs
        r"\frac{dy}{dt} = k y",
        r"\frac{d^2 y}{dt^2} = -k y",
        r"\frac{dy}{dt} = r y (1 - \frac{y}{K})",
        r"\frac{dy}{dt} = r \cdot y \cdot \left(1 - \frac{y}{cap\_k}\right)",
        r"r=1.00, K=10.00, \Delta t=0.05",
        // favoritism/theory.rs
        r"F = B_{\text{order}} \cdot M_{\text{crisis}} \int_0^T \frac{C(t)\,\left(G_p + G_e + P\right)}{1 + \alpha x_0}\,dt",
        // q_table_inspector.rs
        "Q-learning temporal difference control: $Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right]$, converging to the optimal action-value function.",
        // capacity_fade.rs
        "Lithium-ion capacity fade power-law degradation: remaining capacity $C(N) = C_0 \\left(1 - 0.3 (N / N_{70})^z\\right)$ at 80.0% depth of discharge.",
        // survival.rs
        "Kaplan-Meier non-parametric maximum likelihood estimator of survival function $S(t) = \\prod_{t_i \\le t} \\left(1 - \\frac{d_i}{n_i}\\right)$ accounting for right-censored data.",
        // attractors/runner.rs
        "Lorenz attractor dynamical system: $\\dot{x} = \\sigma(y - x)$, $\\dot{y} = x(\\rho - z) - y$, $\\dot{z} = xy - \\beta z$ with parameters $\\sigma = 10.00$, $\\rho = 28.00$, $\\beta = 2.67$.",
        // replicator.rs
        "Multi-strategy evolutionary replicator dynamics: $\\dot{x}_i = x_i \\left( (A x)_i - x^T A x \\right)$, where strategy frequency $x_i$ increases.",
        // vietoris_rips/mod.rs
        "Topological Data Analysis (TDA) via Vietoris-Rips simplicial filtration and exact $\\mathbb{Z}_2$ persistent homology: $\\sigma = [v_0, \\dots, v_k]$ enters at $\\max_{i,j} d(v_i, v_j) \\le \\epsilon$ ($\\epsilon = 0.50$, $\\beta_0 = 1$, $\\beta_1 = 1$, 24 points in $\\mathbb{R}^2$).",
        // volumetric_studio/mod.rs
        "3D Volumetric Field Studio (Hydrogenic 1s + 2pz) evolves hydrogenic superpositions $\\Psi(\\mathbf{r}, t) = c_1(t) R_{n_1 \\ell_1}(r) Y_{\\ell_1 m_1}(\\theta,\\phi) + c_2(t) R_{n_2 \\ell_2}(r) Y_{\\ell_2 m_2}(\\theta,\\phi)$ and 3D Schnakenberg reaction-diffusion $\\partial_t u = D_u \\nabla^2_{3D} u + a - u + u^2 v$ across a 28^3 VoxelGrid using MarchingCubes and EWA Gaussian3D splatting.",
        // lenia/presets.rs
        r"A_{t+\Delta t} = \operatorname{clamp}\!\left(A_t + \Delta t \, G\!\left(\mathcal{F}^{-1}(\mathcal{F}(A_t)\odot \hat{K});\,\mu,\sigma\right),\,0,\,1\right)",
    ];

    for formula in audited_formulas {
        assert_no_leaked_latex_words(formula);
    }
}

#[test]
fn test_all_ast_emitted_functions_and_custom_presets_render_cleanly() {
    let ast_latex_samples = [
        r"\sin\left(t\right)",
        r"\cos\left(t\right)",
        r"\tan\left(t\right)",
        r"\exp\left(-x\right)",
        r"\ln\left(x\right)",
        r"\log\left(x\right)",
        r"\sqrt{x^2 + 1}",
        r"\left|\frac{y}{t}\right|",
        r"\sinh\left(x\right)",
        r"\cosh\left(x\right)",
        r"\tanh\left(x\right)",
        r"\arcsin\left(x\right)",
        r"\arccos\left(x\right)",
        r"\arctan\left(x\right)",
        r"\min\left(a, b\right)",
        r"\max\left(a, b\right)",
        r"\mathrm{clamp}\left(x, -1, 1\right)",
        r"\mathrm{step}\left(x\right)",
        r"\mathrm{rect}\left(x\right)",
        r"\mathrm{atan2}\left(y, x\right)",
        r"-k \cdot y \cdot \sin\left(t\right)",
        r"z + \frac{a^2}{z}",
        r"z^3 - \frac{1}{z}",
        r"v_0 \cdot \exp\left(-\left(\frac{x}{w}\right)^4\right)",
    ];

    for sample in ast_latex_samples {
        assert_no_leaked_latex_words(sample);
    }
}

#[test]
fn test_delimiter_height_scales_to_match_enclosed_fraction() {
    let font = audit_font();
    let single_line = parse_latex_to_box(r"\left(1 - y\right)", &font, Color32::WHITE).unwrap();
    let frac_only = parse_latex_to_box(r"\frac{d_i}{n_i}", &font, Color32::WHITE).unwrap();
    let wrapped_left_right =
        parse_latex_to_box(r"\left(1 - \frac{d_i}{n_i}\right)", &font, Color32::WHITE).unwrap();
    let wrapped_brackets =
        parse_latex_to_box(r"\left[1 + \frac{d_i}{n_i}\right]", &font, Color32::WHITE).unwrap();
    let wrapped_pipes =
        parse_latex_to_box(r"\left|\frac{d_i}{n_i}\right|", &font, Color32::WHITE).unwrap();
    let wrapped_plain_parens =
        parse_latex_to_box(r"(1 - \frac{d_i}{n_i})", &font, Color32::WHITE).unwrap();

    assert!(matches!(wrapped_left_right, MathBox::Delimited { .. }));
    assert!(matches!(wrapped_brackets, MathBox::Delimited { .. }));
    assert!(matches!(wrapped_pipes, MathBox::Delimited { .. }));
    assert!(matches!(wrapped_plain_parens, MathBox::Delimited { .. }));

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let g_single = measure_math_box(ui, &single_line);
            let g_frac = measure_math_box(ui, &frac_only);
            let g_parens = measure_math_box(ui, &wrapped_left_right);
            let g_brackets = measure_math_box(ui, &wrapped_brackets);
            let g_pipes = measure_math_box(ui, &wrapped_pipes);
            let g_plain = measure_math_box(ui, &wrapped_plain_parens);

            assert!(
                g_parens.size.y >= g_frac.size.y && g_parens.size.y > g_single.size.y * 1.4,
                "Scaled \\left(...\\right) height ({}) must match fraction height ({}) > single line ({})",
                g_parens.size.y,
                g_frac.size.y,
                g_single.size.y
            );
            assert!(g_brackets.size.y >= g_frac.size.y);
            assert!(g_pipes.size.y >= g_frac.size.y);
            assert!(g_plain.size.y >= g_frac.size.y);
        });
    });
}

#[test]
fn test_overdot_ellipses_and_blackboard_substitutions() {
    let font = audit_font();
    let dot_job = parse_latex_with_font(r"\dot{x}_i + \ddot{y}", font.clone(), Color32::WHITE);
    assert!(dot_job.text.contains("x\u{0307}"));
    assert!(dot_job.text.contains("y\u{0308}"));
    assert!(!dot_job.text.contains("dot"));

    let dots_job = parse_latex_with_font(
        r"v_0, \dots, v_k, \ldots, \cdots, \mathbb{Z}_2, \mathbb{R}^2",
        font.clone(),
        Color32::WHITE,
    );
    assert!(dots_job.text.contains('…'));
    assert!(dots_job.text.contains('⋯'));
    assert!(dots_job.text.contains('ℤ'));
    assert!(dots_job.text.contains('ℝ'));

    let floor_ceil_norm = parse_latex_with_font(
        r"\left\lfloor \frac{x}{2} \right\rfloor + \left\lceil \frac{y}{2} \right\rceil + \left\| z \right\|",
        font.clone(),
        Color32::WHITE,
    );
    assert!(floor_ceil_norm.text.contains('⌊') && floor_ceil_norm.text.contains('⌋'));
    assert!(floor_ceil_norm.text.contains('⌈') && floor_ceil_norm.text.contains('⌉'));
    assert!(floor_ceil_norm.text.contains('‖'));

    let prose_job = parse_prose_with_font(
        r"Logistic rate $\frac{dy}{dt} = r y \left(1 - \frac{y}{K}\right)$ with $\dot{x}_i$ in $\mathbb{R}^n$.",
        font,
        Color32::WHITE,
    );
    assert!(prose_job.text.contains("dy⁄dt"));
    assert!(prose_job.text.contains("x\u{0307}"));
    assert!(prose_job.text.contains('ℝ'));
    assert!(!prose_job.text.contains("left") && !prose_job.text.contains("right"));
}
