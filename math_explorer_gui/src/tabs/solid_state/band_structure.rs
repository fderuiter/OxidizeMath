//! 1D Tight-Binding and Free-Electron Brillouin-zone band structure workbench.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::datasheet::DatasheetExportState;
use crate::accessibility::{
    paint_canvas_focus_ring, register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, ExportCapabilities, InteractiveTool, PaneSidePanel};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui::{self, accesskit::Role};
use egui_plot::{Corner, Legend, Line, LineStyle, Plot, PlotPoints};
use math_explorer::physics::solid_state::band_theory::{free_electron_1d, tight_binding_1d};
use std::f64::consts::PI;
use std::fmt::Write as _;

const BAND_GRID_POINTS: usize = 200;

/// Interactive 1D electronic band-structure dispersion explorer.
pub struct BandStructureTool {
    e0: f64,
    t: f64,
    a: f64,
    hbar: f64,
    m: f64,
    show_free_electron: bool,
    tb_points: Vec<[f64; 2]>,
    fe_points: Vec<[f64; 2]>,
    revision: u64,
    datasheet_export: DatasheetExportState,
}

impl Default for BandStructureTool {
    fn default() -> Self {
        let mut tool = Self {
            e0: 0.0,
            t: 1.0,
            a: 1.0,
            hbar: 1.0,
            m: 1.0,
            show_free_electron: false,
            tb_points: Vec::with_capacity(BAND_GRID_POINTS),
            fe_points: Vec::with_capacity(BAND_GRID_POINTS),
            revision: 1,
            datasheet_export: DatasheetExportState::default(),
        };
        tool.recalculate();
        tool
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BandStructureState {
    e0: f64,
    t: f64,
    a: f64,
    hbar: f64,
    m: f64,
    show_free_electron: bool,
}

impl BandStructureTool {
    /// Recomputes Brillouin-zone dispersion buffers without reallocating storage.
    pub fn recalculate(&mut self) {
        self.tb_points.clear();
        self.fe_points.clear();
        let k_min = -PI / self.a;
        let k_max = PI / self.a;
        let k_step = (k_max - k_min) / ((BAND_GRID_POINTS - 1) as f64);
        for i in 0..BAND_GRID_POINTS {
            let k = k_min + (i as f64) * k_step;
            self.tb_points
                .push([k, tight_binding_1d(k, self.e0, self.t, self.a)]);
            self.fe_points
                .push([k, free_electron_1d(k, self.hbar, self.m)]);
        }
    }

    /// Bandwidth $W = 4t$ of the 1D nearest-neighbor cosine band in eV.
    #[must_use]
    pub fn bandwidth_ev(&self) -> f64 {
        4.0 * self.t
    }

    /// Zone-boundary free-electron kinetic energy at $k = \pi / a$.
    #[must_use]
    pub fn zone_edge_free_energy(&self) -> f64 {
        free_electron_1d(PI / self.a, self.hbar, self.m)
    }

    /// Formats a standalone Markdown band-structure datasheet.
    #[must_use]
    pub fn format_markdown_card(&self) -> String {
        let mut out = String::with_capacity(512);
        let _ = writeln!(out, "# 1D Electronic Band Structure Datasheet\n");
        let _ = writeln!(out, "| Parameter / Metric | Value | Unit |");
        let _ = writeln!(out, "| :--- | :--- | :--- |");
        let _ = writeln!(out, "| On-site Energy (E_0) | {:.4} | eV |", self.e0);
        let _ = writeln!(out, "| Hopping Integral (t) | {:.4} | eV |", self.t);
        let _ = writeln!(out, "| Lattice Constant (a) | {:.4} | Å |", self.a);
        let _ = writeln!(
            out,
            "| Tight-Binding Bandwidth (4t) | {:.4} | eV |",
            self.bandwidth_ev()
        );
        let _ = writeln!(
            out,
            "| Band Minimum (E_0 - 2t) | {:.4} | eV |",
            self.e0 - 2.0 * self.t
        );
        let _ = writeln!(
            out,
            "| Band Maximum (E_0 + 2t) | {:.4} | eV |",
            self.e0 + 2.0 * self.t
        );
        let _ = writeln!(
            out,
            "| Zone-Edge Free-Electron Energy | {:.4} | arb. eV |",
            self.zone_edge_free_energy()
        );
        out
    }

    /// Formats a standalone CSV band-structure datasheet.
    #[must_use]
    pub fn format_csv_card(&self) -> String {
        let mut out = String::with_capacity(512);
        let _ = writeln!(out, "section,key,value,unit");
        let _ = writeln!(out, "parameter,e0,{:.6},eV", self.e0);
        let _ = writeln!(out, "parameter,hopping_t,{:.6},eV", self.t);
        let _ = writeln!(out, "parameter,lattice_a,{:.6},angstrom", self.a);
        let _ = writeln!(out, "parameter,hbar,{:.6},scaled", self.hbar);
        let _ = writeln!(out, "parameter,effective_mass,{:.6},scaled", self.m);
        let _ = writeln!(out, "metric,bandwidth_4t,{:.6},eV", self.bandwidth_ev());
        let _ = writeln!(out, "metric,band_min,{:.6},eV", self.e0 - 2.0 * self.t);
        let _ = writeln!(out, "metric,band_max,{:.6},eV", self.e0 + 2.0 * self.t);
        let _ = writeln!(
            out,
            "metric,zone_edge_free_energy,{:.6},eV",
            self.zone_edge_free_energy()
        );
        out
    }

    fn render_sidebar(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        ui.heading("Band Parameters");
        ui.separator();
        ui.label(egui::RichText::new("1D Tight-Binding Model").strong());
        changed |= ui
            .add(
                egui::Slider::new(&mut self.e0, -5.0..=5.0)
                    .text("E₀ (On-site eV)")
                    .step_by(0.1),
            )
            .accessible_hover_text("Atomic orbital on-site energy E_0 in eV")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.t, 0.0..=5.0)
                    .text("t (Hopping eV)")
                    .step_by(0.1),
            )
            .accessible_hover_text("Nearest-neighbor hopping overlap integral t in eV")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.a, 0.1..=5.0)
                    .text("a (Lattice Å)")
                    .step_by(0.1),
            )
            .accessible_hover_text("1D Bravais lattice constant a in Angstroms")
            .changed();
        changed |= self.render_free_electron_controls(ui);
        if changed {
            self.recalculate();
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        ui.separator();
        render_copyable_metric(
            ui,
            "Bandwidth 4t",
            &format!("{:.2} eV", self.bandwidth_ev()),
        );
        render_copyable_metric(ui, "BZ Edge π/a", &format!("±{:.3} Å⁻¹", PI / self.a));
        ui.add_space(4.0);
        let md = self.format_markdown_card();
        let csv = self.format_csv_card();
        self.datasheet_export.render_controls(
            ui,
            "Band Structure Datasheets",
            "band_structure_datasheet.md",
            "band_structure_datasheet.csv",
            move || md.clone(),
            move || csv.clone(),
        );
    }

    fn render_free_electron_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = ui
            .checkbox(
                &mut self.show_free_electron,
                "Overlay Free-Electron Parabola",
            )
            .accessible_hover_text(
                "Overlay parabolic free-electron dispersion E(k) = hbar^2 k^2 / (2m)",
            )
            .changed();
        if self.show_free_electron {
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.hbar, 0.1..=5.0)
                        .text("ħ (Scaled Planck)")
                        .step_by(0.1),
                )
                .accessible_hover_text("Reduced Planck constant scaling factor")
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.m, 0.1..=5.0)
                        .text("m* (Effective Mass)")
                        .step_by(0.1),
                )
                .accessible_hover_text("Carrier effective mass scaling factor")
                .changed();
        }
        changed
    }

    fn render_plot(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Brillouin-Zone E(k) Dispersion");
        let plot_id = scoped_panel_id(ui, "band_structure_plot");
        let tb_pts = PlotPoints::new(self.tb_points.clone());
        let fe_pts = self
            .show_free_electron
            .then(|| PlotPoints::new(self.fe_points.clone()));
        let resp = Plot::new(plot_id)
            .x_axis_label("Wavevector k (Å⁻¹)")
            .y_axis_label("Energy E(k) (eV)")
            .legend(Legend::default().position(Corner::LeftTop))
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Tight Binding", tb_pts)
                        .color(palette.series_color(0))
                        .width(2.2_f32),
                );
                if let Some(fe) = fe_pts {
                    plot_ui.line(
                        Line::new("Free Electron", fe)
                            .color(palette.series_color(1))
                            .style(LineStyle::Dashed { length: 7.0 }),
                    );
                }
            });
        let summary = format!(
            "1D Brillouin zone [-pi/a, pi/a] with E0 = {:.2} eV, bandwidth 4t = {:.2} eV, lattice a = {:.2} A",
            self.e0,
            self.bandwidth_ev(),
            self.a
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "1D Electronic Band Structure E(k) Dispersion Plot",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }
}

impl InteractiveTool for BandStructureTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Band Structure"
    }
    fn domain(&self) -> &'static str {
        "solid_state"
    }
    fn state_revision(&self) -> u64 {
        self.revision
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut ds = vec![(
            "Tight Binding E(k) (eV)".to_string(),
            self.tb_points.clone(),
        )];
        if self.show_free_electron {
            ds.push((
                "Free Electron E(k) (eV)".to_string(),
                self.fe_points.clone(),
            ));
        }
        ds
    }

    fn has_finite_parameters(&self) -> bool {
        self.e0.is_finite()
            && self.t.is_finite()
            && self.a.is_finite()
            && self.a > 0.0
            && self.hbar.is_finite()
            && self.m.is_finite()
            && self.m > 0.0
    }

    fn clamp_parameters(&mut self) -> bool {
        self.e0 = if self.e0.is_finite() {
            self.e0.clamp(-5.0, 5.0)
        } else {
            0.0
        };
        self.t = if self.t.is_finite() {
            self.t.clamp(0.0, 5.0)
        } else {
            1.0
        };
        self.a = if self.a.is_finite() {
            self.a.clamp(0.1, 5.0)
        } else {
            1.0
        };
        self.hbar = if self.hbar.is_finite() {
            self.hbar.clamp(0.1, 5.0)
        } else {
            1.0
        };
        self.m = if self.m.is_finite() {
            self.m.clamp(0.1, 5.0)
        } else {
            1.0
        };
        self.recalculate();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&BandStructureState {
            e0: self.e0,
            t: self.t,
            a: self.a,
            hbar: self.hbar,
            m: self.m,
            show_free_electron: self.show_free_electron,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: BandStructureState =
            serde_json::from_str(state).map_err(|e| format!("Invalid BandStructure state: {e}"))?;
        if !s.e0.is_finite() || !(-5.0..=5.0).contains(&s.e0) {
            return Err(format!("e0 out of [-5.0, 5.0]: {}", s.e0));
        }
        if !s.t.is_finite() || !(0.0..=5.0).contains(&s.t) {
            return Err(format!("t out of [0.0, 5.0]: {}", s.t));
        }
        if !s.a.is_finite()
            || !(0.1..=5.0).contains(&s.a)
            || !s.hbar.is_finite()
            || !(0.1..=5.0).contains(&s.hbar)
            || !s.m.is_finite()
            || !(0.1..=5.0).contains(&s.m)
        {
            return Err("a, hbar, and m must be finite in [0.1, 5.0]".to_string());
        }
        self.e0 = s.e0;
        self.t = s.t;
        self.a = s.a;
        self.hbar = s.hbar;
        self.m = s.m;
        self.show_free_electron = s.show_free_electron;
        self.recalculate();
        self.revision = self.revision.wrapping_add(1).max(1);
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.datasheet_export.poll();
        let panel_id = scoped_panel_id(parent_ui, "band_structure_controls");
        PaneSidePanel::right(panel_id)
            .resizable(false)
            .min_width(260.0)
            .show_inside(parent_ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.render_sidebar(ui);
                });
            });
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.render_plot(ui);
        });
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BandStructureTool",
        display_name: "Band Structure",
        domain: "solid_state",
        tags: &["tight-binding", "brillouin zone", "dispersion", "bloch", "effective mass"],
        build: || Box::new(BandStructureTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for BandStructureTool {
    fn theory_description(&self) -> String {
        let model_str = if self.show_free_electron {
            format!(
                "parabolic free-electron dispersion $E(k) = \\frac{{\\hbar^2 k^2}}{{2m}}$ with effective mass $m = {:.2}$",
                self.m
            )
        } else {
            format!(
                "1D tight-binding nearest-neighbor dispersion $E(k) = E_0 - 2t \\cos(ka)$ with $E_0 = {:.2}$, hopping $t = {:.2}$, lattice constant $a = {:.2}$",
                self.e0, self.t, self.a
            )
        };
        format!(
            "Solid state electronic band structure dispersion relation in the first Brillouin zone $[-\\pi/a, \\pi/a]$: {model_str}."
        )
    }
    fn phonetic_description(&self) -> String {
        "Electronic band structure and tight-binding dispersion theory".into()
    }
    fn theory_citation(&self) -> String {
        "Introduction to Solid State Physics (Kittel, 2004); Solid State Physics (Ashcroft & Mermin, 1976)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "TightBinding".into(),
            "E(k) = E_0 - 2t cos(ka) atomic orbital overlap model".into(),
        );
        map.insert(
            "FreeElectron".into(),
            "E(k) = hbar^2 k^2 / (2m) parabolic free Fermi gas dispersion".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Electronic Band Structure",
    "domain": "solid_state",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "e0": 0.0,
    "t": 1.5,
    "a": 1.2,
    "hbar": 1.0,
    "m": 1.0,
    "show_free_electron": true
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "band_structure_starter",
        title: "Electronic Band Structure",
        description: "1D tight-binding E(k) = E₀ - 2t·cos(ka) and free-electron Brillouin zone dispersion.",
        category: "Condensed Matter",
        icon: "📉",
        domain: "solid_state",
        tool_name: Some("BandStructureTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:mmwave_radiotherapy_setup]
