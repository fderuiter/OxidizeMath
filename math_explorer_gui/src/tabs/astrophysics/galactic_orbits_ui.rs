#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::galactic_orbits_tool::{GalacticOrbitSimulator, TRAIL_CAPACITY};
use super::HaloKindChoice;
use crate::framework::{PlaybackAction, PlaybackTransport};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, LineStyle, MarkerShape, Plot, PlotPoints, Points};
use math_explorer::physics::astrophysics::{
    decompose_rotation_curve, DwarfMorphology, OrbitParticleKind,
};

impl GalacticOrbitSimulator {
    pub(super) fn render_transport_and_metrics(&mut self, ui: &mut egui::Ui) {
        if let Some(action) = PlaybackTransport::in_thread(
            "galactic_orbits_transport",
            "Galactic Orbit Simulator",
            self.cfg.is_playing,
        )
        .reset_label("↺ Reset Orbits")
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.cfg.is_playing = true,
                PlaybackAction::Pause => self.cfg.is_playing = false,
                PlaybackAction::Step => {
                    self.cfg.is_playing = false;
                    self.advance_burst();
                }
                PlaybackAction::Reset => {
                    self.cfg.is_playing = false;
                    self.rebuild_system(0);
                }
                PlaybackAction::StepBack | PlaybackAction::Scrub(_) => {}
            }
            self.revision = self.revision.wrapping_add(1).max(1);
        }

        let outer_v = decompose_rotation_curve(
            4.0 * self.cfg.disk_scale_kpc,
            &self.system.baryons,
            &self.system.halo,
        )
        .v_circ_kms;
        render_copyable_metric(
            ui,
            "Elapsed Galactic Time",
            &format!("{:.1} Myr", self.system.elapsed_myr()),
        );
        render_copyable_metric(ui, "Outer v_circ (4 R_d)", &format!("{:.2} km/s", outer_v));
        render_copyable_metric(
            ui,
            "Relative Energy Drift |ΔE/E0|",
            &format!("{:.2e}", self.system.relative_energy_drift()),
        );
        render_copyable_metric(
            ui,
            "Angular Momentum Drift |ΔLz/Lz0|",
            &format!("{:.2e}", self.system.relative_angular_momentum_drift()),
        );
    }

    pub(super) fn render_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut rebuild = self.render_potential_controls(ui);
        ui.separator();
        ui.heading("Symplectic Leapfrog");
        if ui
            .add(egui::Slider::new(&mut self.cfg.dt_myr, 0.1..=10.0).text("Leapfrog Step dt (Myr)"))
            .changed()
            || ui
                .add(
                    egui::Slider::new(&mut self.cfg.substeps_per_frame, 1..=16)
                        .text("Substeps / Frame"),
                )
                .changed()
        {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        rebuild |= ui
            .add(egui::Slider::new(&mut self.cfg.disk_tracers, 4..=48).text("Disk Tracer Stars"))
            .changed();
        rebuild |= ui
            .add(egui::Slider::new(&mut self.cfg.tidal_tracers, 0..=24).text("Tidal Stream Stars"))
            .changed();
        rebuild |= ui
            .add(
                egui::Slider::new(&mut self.cfg.tidal_tangential_ratio, 0.2..=1.2)
                    .text("Stream v_tan / v_circ"),
            )
            .changed();
        rebuild
    }

    fn render_potential_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let c = &mut self.cfg;
        let mut rebuild = false;
        ui.heading("Galactic Potential");
        egui::ComboBox::from_label("Galaxy Morphology")
            .selected_text(c.morphology.label())
            .show_ui(ui, |ui| {
                for m in [
                    DwarfMorphology::GeneralIrregular,
                    DwarfMorphology::TypeCode10,
                    DwarfMorphology::TypeCode95To99,
                ] {
                    rebuild |= ui
                        .selectable_value(&mut c.morphology, m, m.label())
                        .changed();
                }
            });
        egui::ComboBox::from_label("Dark Halo Profile")
            .selected_text(c.halo_kind.label())
            .show_ui(ui, |ui| {
                for h in [
                    HaloKindChoice::BaryonicOnly,
                    HaloKindChoice::NfwCusp,
                    HaloKindChoice::BurkertCore,
                ] {
                    rebuild |= ui
                        .selectable_value(&mut c.halo_kind, h, h.label())
                        .changed();
                }
            });
        rebuild |= ui
            .add(egui::Slider::new(&mut c.distance_mpc, 0.5..=250.0).text("Distance D (Mpc)"))
            .changed();
        rebuild |= ui
            .add(egui::Slider::new(&mut c.gas_fraction, 0.0..=0.95).text("HI Gas Fraction"))
            .changed();
        rebuild |= ui
            .add(egui::Slider::new(&mut c.disk_scale_kpc, 0.2..=5.0).text("Disk Scale R_d (kpc)"))
            .changed();
        rebuild |= ui
            .add(
                egui::Slider::new(&mut c.halo_log10_mass, 8.0..=12.0)
                    .text("Halo log10(M200 / M_sun)"),
            )
            .changed();
        rebuild |= ui
            .add(
                egui::Slider::new(&mut c.halo_scale_kpc, 0.3..=10.0)
                    .text("Halo Scale / Core r_s (kpc)"),
            )
            .changed();
        rebuild |= ui
            .add(egui::Slider::new(&mut c.halo_concentration, 4.0..=25.0).text("Concentration c"))
            .changed();
        rebuild
    }

    pub(super) fn render_orbit_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let (mut disk_pts, mut stream_pts) = (Vec::new(), Vec::new());
        for p in self.system.particles() {
            match p.kind {
                OrbitParticleKind::DiskTracer => disk_pts.push([p.x_kpc, p.y_kpc]),
                OrbitParticleKind::TidalStream => stream_pts.push([p.x_kpc, p.y_kpc]),
            }
        }
        let scale_ring = self.scale_ring_points(64);
        Plot::new("galactic_orbits_midplane_plot")
            .data_aspect(1.0)
            .view_aspect(1.35)
            .x_axis_label("Galactocentric x (kpc)")
            .y_axis_label("Galactocentric y (kpc)")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Halo Scale / Core Radius", PlotPoints::new(scale_ring))
                        .color(palette.info)
                        .style(LineStyle::Dashed { length: 6.0 }),
                );
                let mut trail_buf = [[0.0_f64; 2]; TRAIL_CAPACITY];
                for (idx, p) in self.system.particles().iter().enumerate() {
                    if idx % 2 != 0 && p.kind == OrbitParticleKind::DiskTracer {
                        continue;
                    }
                    let n = self.system.copy_particle_trail(idx, &mut trail_buf);
                    if n >= 2 {
                        let col = if p.kind == OrbitParticleKind::DiskTracer {
                            palette.axis_z
                        } else {
                            palette.axis_x
                        };
                        plot_ui.line(
                            Line::new("", PlotPoints::new(trail_buf[..n].to_vec()))
                                .color(col)
                                .width(1.2_f32),
                        );
                    }
                }
                plot_ui.points(
                    Points::new("Disk Tracer Stars", PlotPoints::new(disk_pts))
                        .color(palette.accent)
                        .radius(4.0_f32)
                        .shape(MarkerShape::Circle),
                );
                if !stream_pts.is_empty() {
                    plot_ui.points(
                        Points::new("Tidal Stream Stars", PlotPoints::new(stream_pts))
                            .color(palette.axis_x)
                            .radius(4.5_f32)
                            .shape(MarkerShape::Diamond),
                    );
                }
            });
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]
