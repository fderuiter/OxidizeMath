#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{PopulationViewMode, MAX_AGE_BINS, MIN_AGE_BINS};
use super::tool::PopulationStructureTool;
use crate::accessibility::{AccessibleHoverText, AccessibleTheoryHover};
use crate::framework::{
    render_recovery_banner, FiniteStateGuard, InteractiveTool, PlaybackAction, PlaybackTransport,
    RecoveryAction,
};
use eframe::egui;
use egui_plot::{Legend, Line, Plot, Points};

/// Renders [`PopulationStructureTool`] inside a scoped parent [`egui::Ui`].
pub fn render_tool_inside(tool: &mut PopulationStructureTool, parent_ui: &mut egui::Ui) {
    let ctx = parent_ui.ctx().clone();
    let is_hc = crate::app::ThemeMode::from_visuals(parent_ui.visuals())
        == crate::app::ThemeMode::HighContrast;
    if tool.high_contrast != is_hc {
        tool.high_contrast = is_hc;
        tool.lexis_dirty = true;
    }
    if !tool.paused && !tool.diverged() {
        tool.advance_frame_budgeted();
        ctx.request_repaint();
    }

    render_controls_panel(tool, parent_ui);

    egui::CentralPanel::default().show_inside(parent_ui, |ui| {
        if tool.diverged() {
            if let Some(action) = render_recovery_banner(ui, tool.diagnostic_report()) {
                apply_recovery(tool, action);
            }
            ui.add_space(6.0);
        }
        render_active_viewport(tool, ui, &ctx);
    });
}

fn apply_recovery(tool: &mut PopulationStructureTool, action: RecoveryAction) {
    match action {
        RecoveryAction::AutoScaleStepSize => tool.auto_scale_step_size(),
        RecoveryAction::ClampParameters => {
            <PopulationStructureTool as FiniteStateGuard>::clamp_parameters(tool)
        }
        RecoveryAction::ResetParameters => tool.reset_parameters(),
    }
}

fn render_controls_panel(tool: &mut PopulationStructureTool, parent_ui: &mut egui::Ui) {
    let panel_id = crate::framework::scoped_panel_id(parent_ui, "cannibalism_controls");
    crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
        ui.heading("Population Structure");
        if let Some(action) = PlaybackTransport::in_thread(
            "cannibalism_transport",
            tool.name(),
            !tool.paused && !tool.diverged(),
        )
        .show(ui)
        {
            handle_playback_action(tool, action);
        }
        render_courant_and_metrics(tool, ui);
        ui.separator();
        render_view_selector(tool, ui);
        ui.separator();
        let mut rebuild = false;
        egui::ScrollArea::vertical().show(ui, |ui| {
            rebuild |= render_grid_sliders(tool, ui);
            rebuild |= render_kernel_sliders(tool, ui);
        });
        if rebuild {
            tool.config = tool.config.clamped();
            tool.rebuild_and_warmup();
        }
    });
}

fn handle_playback_action(tool: &mut PopulationStructureTool, action: PlaybackAction) {
    match action {
        PlaybackAction::Play => {
            if !tool.diverged() {
                tool.paused = false;
            }
        }
        PlaybackAction::Pause => tool.paused = true,
        PlaybackAction::Step => {
            tool.paused = true;
            tool.advance_one_step();
        }
        PlaybackAction::Reset => {
            tool.paused = true;
            tool.rebuild_and_warmup();
        }
    }
}

fn render_courant_and_metrics(tool: &PopulationStructureTool, ui: &mut egui::Ui) {
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    let lambda = tool.courant_number();
    let substeps = lambda.ceil().max(1.0) as usize;
    let badge_color = if lambda <= 1.0 {
        palette.success
    } else {
        palette.warning
    };
    ui.horizontal_wrapped(|ui| {
        ui.colored_label(
            badge_color,
            format!("CFL λ = Δt/Δa = {lambda:.2} ({substeps}× upwind)"),
        );
    });
    ui.small(format!(
        "t = {:.2}  |  N(t) = {:.2}  |  J = {:.2}, A = {:.2}",
        tool.mvf_solver.time(),
        tool.mvf_solver.total_population(),
        tool.ja_solver.total_juveniles(),
        tool.ja_solver.adults(),
    ));
}

fn render_view_selector(tool: &mut PopulationStructureTool, ui: &mut egui::Ui) {
    ui.label("Viewport Mode");
    let modes = [
        PopulationViewMode::Dashboard,
        PopulationViewMode::LexisHeatmap,
        PopulationViewMode::AgePyramid,
        PopulationViewMode::PhasePortrait,
        PopulationViewMode::OdeComparison,
    ];
    egui::ComboBox::from_id_salt("cannibalism_view_mode")
        .selected_text(tool.config.view_mode.label())
        .show_ui(ui, |ui| {
            for mode in modes {
                ui.selectable_value(&mut tool.config.view_mode, mode, mode.label());
            }
        });
}

fn render_grid_sliders(tool: &mut PopulationStructureTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label("Age Grid & Time Step");
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.config.age_bins, MIN_AGE_BINS..=MAX_AGE_BINS)
                    .text("Age Bins"),
            )
            .accessible_hover_text("Uniform age-grid node count (16 to 512)")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut tool.config.max_age, 2.0..=20.0).text("Max Age a_max"))
            .changed();
        let mat_cap = tool.config.max_age.min(10.0);
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.config.maturation_age, 0.5..=mat_cap)
                    .text("Maturation α"),
            )
            .changed();
        let vuln_cap = tool.config.maturation_age.min(8.0);
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.config.vulnerable_age_max, 0.2..=vuln_cap)
                    .text("Vulnerable a_vuln"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut tool.config.dt, 0.002..=0.5).text("Step Δt"))
            .changed();
    });
    changed
}

fn render_kernel_sliders(tool: &mut PopulationStructureTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label("Cannibalism & Lifeboat Rates");
        changed |= ui
            .add(egui::Slider::new(&mut tool.config.attack_rate, 0.0..=2.0).text("Attack Rate"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.config.saturation_rate, 0.0..=5.0)
                    .text("Ricker Saturation σ"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.config.fecundity_rate, 0.5..=50.0)
                    .text("Adult Fecundity β"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.config.base_mortality, 0.01..=2.0)
                    .text("Base Mortality ν₀"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut tool.config.energy_gain, 0.0..=1.0).text("Lifeboat Gain γ"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut tool.config.ode_phi, 0.0..=1.0).text("2D ODE Loss φ"))
            .changed();
    });
    changed
}

fn render_active_viewport(
    tool: &mut PopulationStructureTool,
    ui: &mut egui::Ui,
    ctx: &egui::Context,
) {
    match tool.config.view_mode {
        PopulationViewMode::Dashboard => render_dashboard_grid(tool, ui, ctx),
        PopulationViewMode::LexisHeatmap => {
            ui.heading("Lexis (a, t) Cohort Density Heatmap");
            render_lexis_panel(tool, ui, ctx, None);
        }
        PopulationViewMode::AgePyramid => {
            ui.heading("Age Pyramid n(t, a)");
            render_age_pyramid_plot(tool, ui, None);
        }
        PopulationViewMode::PhasePortrait => {
            ui.heading("Juvenile-Adult Hopf Phase Portrait (J(t), A(t))");
            render_hopf_phase_plot(tool, ui, None);
        }
        PopulationViewMode::OdeComparison => {
            ui.heading("2D Lumped ODE vs Age-Structured PDE Trajectories");
            render_ode_comparison_plot(tool, ui, None);
        }
    }
}

fn render_dashboard_grid(
    tool: &mut PopulationStructureTool,
    ui: &mut egui::Ui,
    ctx: &egui::Context,
) {
    let row_h = ((ui.available_height() - 56.0) * 0.5).max(150.0);
    ui.columns(2, |cols| {
        cols[0].strong("Lexis (a, t) Cohort Heatmap");
        render_lexis_panel(tool, &mut cols[0], ctx, Some(row_h));
        cols[1].strong("Age Pyramid n(t, a)");
        render_age_pyramid_plot(tool, &mut cols[1], Some(row_h));
    });
    ui.add_space(8.0);
    ui.columns(2, |cols| {
        cols[0].strong("Hopf Phase Portrait (J(t), A(t))");
        render_hopf_phase_plot(tool, &mut cols[0], Some(row_h));
        cols[1].strong("2D Cannibal-Normal ODE vs PDE");
        render_ode_comparison_plot(tool, &mut cols[1], Some(row_h));
    });
}

fn render_lexis_panel(
    tool: &mut PopulationStructureTool,
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    fixed_height: Option<f32>,
) {
    if tool.lexis_texture.is_none() || tool.lexis_dirty {
        let img = tool.compute_lexis_image(tool.high_contrast);
        let opts = egui::TextureOptions::NEAREST;
        if let Some(tex) = &mut tool.lexis_texture {
            tex.set(img, opts);
        } else {
            tool.lexis_texture = Some(ctx.load_texture("cannibalism_lexis", img, opts));
        }
        tool.lexis_dirty = false;
    }
    if let Some(texture) = &tool.lexis_texture {
        let avail = ui.available_size();
        let target = egui::Vec2::new(
            avail.x.max(120.0),
            fixed_height.unwrap_or_else(|| (avail.y - 8.0).max(160.0)),
        );
        let alt = format!(
            "Lexis age-time cohort density heatmap at t = {:.2}, total N = {:.2}",
            tool.mvf_solver.time(),
            tool.mvf_solver.total_population()
        );
        let response = ui
            .add(
                egui::Image::new(texture)
                    .fit_to_exact_size(target)
                    .alt_text(alt)
                    .sense(egui::Sense::focusable_noninteractive()),
            )
            .accessible_theory_hover(tool.theory_model());
        crate::accessibility::paint_canvas_focus_ring(ui, &response);
    }
}

fn render_age_pyramid_plot(
    tool: &PopulationStructureTool,
    ui: &mut egui::Ui,
    fixed_height: Option<f32>,
) {
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    let mvf_line: Vec<[f64; 2]> = tool
        .mvf_solver
        .ages()
        .iter()
        .zip(tool.mvf_solver.density().iter())
        .map(|(&a, &n)| [a, n])
        .collect();
    let ja_line: Vec<[f64; 2]> = tool
        .ja_solver
        .ages()
        .iter()
        .zip(tool.ja_solver.juvenile_density().iter())
        .map(|(&a, &n)| [a, n])
        .collect();
    let mut plot = Plot::new("cannibalism_age_pyramid")
        .legend(Legend::default())
        .x_axis_label("Age a")
        .y_axis_label("Density n(t, a)");
    if let Some(h) = fixed_height {
        plot = plot.height(h);
    }
    plot.show(ui, |plot_ui| {
        plot_ui.line(
            Line::new("McKendrick n(t, a)", mvf_line)
                .color(palette.series_color(0))
                .fill(0.0_f32),
        );
        plot_ui.line(Line::new("Juvenile n_J(t, a)", ja_line).color(palette.series_color(1)));
    });
}

fn render_hopf_phase_plot(
    tool: &PopulationStructureTool,
    ui: &mut egui::Ui,
    fixed_height: Option<f32>,
) {
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    let curve: Vec<[f64; 2]> = tool.ja_history.iter().map(|s| [s[1], s[2]]).collect();
    let head: Vec<[f64; 2]> = tool
        .ja_history
        .back()
        .map(|s| vec![[s[1], s[2]]])
        .unwrap_or_default();
    let mut plot = Plot::new("cannibalism_hopf_phase")
        .legend(Legend::default())
        .x_axis_label("Juvenile Biomass J(t)")
        .y_axis_label("Adult Density A(t)");
    if let Some(h) = fixed_height {
        plot = plot.height(h);
    }
    plot.show(ui, |plot_ui| {
        plot_ui.line(Line::new("Cohort Orbit (J, A)", curve).color(palette.series_color(2)));
        plot_ui.points(
            Points::new("Current Cohort State", head)
                .radius(4.5_f32)
                .color(palette.accent),
        );
    });
}

fn render_ode_comparison_plot(
    tool: &PopulationStructureTool,
    ui: &mut egui::Ui,
    fixed_height: Option<f32>,
) {
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    let ode_c: Vec<[f64; 2]> = tool.ode_history.iter().map(|s| [s[0], s[1]]).collect();
    let ode_n: Vec<[f64; 2]> = tool.ode_history.iter().map(|s| [s[0], s[2]]).collect();
    let pde_a: Vec<[f64; 2]> = tool.ja_history.iter().map(|s| [s[0], s[2]]).collect();
    let mut plot = Plot::new("cannibalism_ode_comparison")
        .legend(Legend::default())
        .x_axis_label("Time t")
        .y_axis_label("Population");
    if let Some(h) = fixed_height {
        plot = plot.height(h);
    }
    plot.show(ui, |plot_ui| {
        plot_ui.line(Line::new("2D ODE Cannibals C(t)", ode_c).color(palette.series_color(0)));
        plot_ui.line(Line::new("2D ODE Normals N(t)", ode_n).color(palette.series_color(1)));
        plot_ui.line(Line::new("PDE Adults A(t)", pde_a).color(palette.series_color(3)));
    });
}
