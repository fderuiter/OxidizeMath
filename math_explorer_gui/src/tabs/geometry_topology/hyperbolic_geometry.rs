#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod canvas;
mod canvas_half_plane;
mod export;
mod presets;
#[cfg(test)]
mod tests;
mod ui;

use crate::framework::{ExportCapabilities, InteractiveTool};
use crate::web_io::PlatformInstant;
use eframe::egui;
use math_explorer::pure_math::hyperbolic::{
    BilliardDomainKind, DiskPoint, GeodesicSegment, HalfPlanePoint, HyperbolicTilingBuffer,
    MAX_BILLIARD_SEGMENTS, MAX_POLYGON_VERTICES, MAX_TILING_DEPTH, MAX_TILING_POLYGONS, Psl2R,
    Psu11, TiledPolygon, TilingKind, TwinBilliardSystem, gauss_bonnet_polygon_area,
    validate_hyperbolic_schlafli,
};

/// Primary laboratory mode inside [`HyperbolicGeometryTool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LabMode {
    /// Interactive Schläfli `{p, q}` and `{infinity, q}` Coxeter tilings with Möbius dragging.
    TilingExplorer,
    /// Chaotic geodesic twin-ray billiards with Lyapunov separation $\ln(\delta_t / \delta_0)$.
    ChaoticBilliards,
}

/// Synchronized conformal chart layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ChartLayout {
    /// Side-by-side Poincaré disk $\mathbb{D}$ and upper half-plane $\mathbb{H}$ synchronized via the Cayley map.
    DualSynchronized,
    /// Single full-width Poincaré disk $\mathbb{D}$ viewport.
    PoincareDiskOnly,
    /// Single full-width Poincaré upper half-plane $\mathbb{H}$ viewport.
    UpperHalfPlaneOnly,
}

/// Curated hyperbolic tiling presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TilingPreset {
    /// Klein-quartic heptagonal `{7, 3}` tiling.
    Heptagonal73,
    /// Right-angled pentagonal `{5, 4}` tiling.
    Pentagonal54,
    /// Bolza-surface octagonal `{8, 3}` tiling.
    Octagonal83,
    /// Right-angled hexagonal `{6, 4}` tiling.
    Hexagonal64,
    /// Modular group `{infinity, 3}` apeirogonal tiling.
    IdealModularInf3,
    /// Hecke `{infinity, 4}` apeirogonal tiling.
    IdealHeckeInf4,
    /// User-specified Schläfli `{p, q}` symbol.
    CustomSchlafli,
}

/// Curated chaotic billiard table domains.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BilliardPreset {
    /// Compact regular `{8, 3}` hyperbolic octagon.
    CompactOctagon83,
    /// Non-compact ideal triangle `{3}` with cusps on $\partial\mathbb{D}$.
    IdealTriangle,
    /// Compact regular `{7, 3}` hyperbolic heptagon.
    CompactHeptagon73,
    /// Non-compact ideal quadrilateral `{4}` with cusps on $\partial\mathbb{D}$.
    IdealQuad,
}

/// Serializable parameter configuration for [`HyperbolicGeometryTool`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperbolicGeometryState {
    /// Active laboratory mode (`TilingExplorer` or `ChaoticBilliards`).
    pub mode: LabMode,
    /// Active conformal chart layout.
    pub chart_layout: ChartLayout,
    /// Selected tiling preset.
    pub tiling_preset: TilingPreset,
    /// Polygon side count $p$ for compact `{p, q}` tilings (`3..=16`).
    pub schlafli_p: u32,
    /// Vertex valence $q$ (`3..=16`).
    pub schlafli_q: u32,
    /// Coxeter reflection BFS depth (`1..=5`).
    pub tiling_depth: usize,
    /// Whether to render a tangent horocycle overlay in $\mathbb{D}$ and $\mathbb{H}$.
    pub show_horocycles: bool,
    /// Euclidean radius $r \in [0.05, 0.90]$ of the reference horocycle in $\mathbb{D}$.
    pub horocycle_radius: f64,
    /// Selected billiard table preset.
    pub billiard_preset: BilliardPreset,
    /// Deterministic `OxidizeRng` seed for ray launch conditions.
    pub billiard_seed: u64,
    /// Base-10 logarithm of initial twin-ray angular perturbation $\log_{10}(\delta_0)$.
    pub perturbation_log10: f64,
    /// Hyperbolic arc-length speed multiplier (`0.1..=10.0`).
    pub billiard_speed: f64,
    /// Whether real-time billiard integration is currently playing.
    pub billiards_running: bool,
    /// Current `PSU(1,1)` Möbius camera/drag isometry on $\mathbb{D}$.
    pub disk_isometry: Psu11,
    /// Horizontal half-span visible in the upper half-plane chart (`0.5..=25.0`).
    pub half_plane_span: f64,
}

impl Default for HyperbolicGeometryState {
    fn default() -> Self {
        Self {
            mode: LabMode::TilingExplorer,
            chart_layout: ChartLayout::DualSynchronized,
            tiling_preset: TilingPreset::Heptagonal73,
            schlafli_p: 7,
            schlafli_q: 3,
            tiling_depth: 3,
            show_horocycles: true,
            horocycle_radius: 0.28,
            billiard_preset: BilliardPreset::CompactOctagon83,
            billiard_seed: 42,
            perturbation_log10: -6.0,
            billiard_speed: 1.6,
            billiards_running: false,
            disk_isometry: Psu11::identity(),
            half_plane_span: 3.0,
        }
    }
}

impl HyperbolicGeometryState {
    pub(super) fn resolved_tiling_kind(&self) -> TilingKind {
        match self.tiling_preset {
            TilingPreset::Heptagonal73 => TilingKind::Regular { p: 7, q: 3 },
            TilingPreset::Pentagonal54 => TilingKind::Regular { p: 5, q: 4 },
            TilingPreset::Octagonal83 => TilingKind::Regular { p: 8, q: 3 },
            TilingPreset::Hexagonal64 => TilingKind::Regular { p: 6, q: 4 },
            TilingPreset::IdealModularInf3 => TilingKind::Ideal { q: 3 },
            TilingPreset::IdealHeckeInf4 => TilingKind::Ideal { q: 4 },
            TilingPreset::CustomSchlafli => TilingKind::Regular {
                p: self.schlafli_p,
                q: self.schlafli_q,
            },
        }
    }

    pub(super) fn resolved_billiard_kind(&self) -> BilliardDomainKind {
        match self.billiard_preset {
            BilliardPreset::CompactOctagon83 => BilliardDomainKind::RegularPolygon { p: 8, q: 3 },
            BilliardPreset::IdealTriangle => BilliardDomainKind::IdealPolygon { sides: 3 },
            BilliardPreset::CompactHeptagon73 => BilliardDomainKind::RegularPolygon { p: 7, q: 3 },
            BilliardPreset::IdealQuad => BilliardDomainKind::IdealPolygon { sides: 4 },
        }
    }

    pub(super) fn perturbation_rad(&self) -> f64 {
        10.0_f64.powf(self.perturbation_log10.clamp(-12.0, -1.0))
    }

    fn is_valid(&self) -> bool {
        let det = self.disk_isometry.determinant();
        let finite_scalars = self.horocycle_radius.is_finite()
            && (0.02..=0.95).contains(&self.horocycle_radius)
            && self.perturbation_log10.is_finite()
            && (-12.0..=-1.0).contains(&self.perturbation_log10)
            && self.billiard_speed.is_finite()
            && (0.05..=20.0).contains(&self.billiard_speed)
            && self.half_plane_span.is_finite()
            && (0.25..=50.0).contains(&self.half_plane_span)
            && det.is_finite()
            && (0.5..=2.0).contains(&det)
            && (1..=MAX_TILING_DEPTH).contains(&self.tiling_depth);
        if !finite_scalars {
            return false;
        }
        match self.resolved_tiling_kind() {
            TilingKind::Regular { p, q } => {
                (p as usize) <= MAX_POLYGON_VERTICES && validate_hyperbolic_schlafli(p, q).is_ok()
            }
            TilingKind::Ideal { q } => (3..=MAX_POLYGON_VERTICES as u32).contains(&q),
        }
    }
}

/// Interactive Poincaré disk tiling and chaotic geodesic billiard laboratory.
pub struct HyperbolicGeometryTool {
    pub(super) state: HyperbolicGeometryState,
    pub(super) tiling_buffer: HyperbolicTilingBuffer,
    pub(super) transformed_tiles: Vec<TiledPolygon>,
    pub(super) transformed_tile_count: usize,
    pub(super) billiards: TwinBilliardSystem,
    pub(super) primary_seg_scratch: Vec<GeodesicSegment>,
    pub(super) twin_seg_scratch: Vec<GeodesicSegment>,
    pub(super) edge_hit_scratch: Vec<u8>,
    pub(super) sep_series_scratch: Vec<[f64; 2]>,
    pub(super) last_frame_instant: Option<PlatformInstant>,
    pub(super) focused_canvas: Option<egui::Id>,
    pub(super) drag_disk_last: Option<DiskPoint>,
    pub(super) drag_hp_last: Option<HalfPlanePoint>,
    pub(super) revision: u64,
}

impl Default for HyperbolicGeometryTool {
    fn default() -> Self {
        let state = HyperbolicGeometryState::default();
        let tiling_buffer =
            HyperbolicTilingBuffer::new(state.resolved_tiling_kind(), state.tiling_depth)
                .expect("default {7,3} tiling must succeed");
        let mut transformed_tiles = vec![TiledPolygon::default(); MAX_TILING_POLYGONS];
        let transformed_tile_count =
            tiling_buffer.transform_into(state.disk_isometry, &mut transformed_tiles);
        let mut billiards = TwinBilliardSystem::new(
            state.resolved_billiard_kind(),
            state.billiard_seed,
            state.perturbation_rad(),
        )
        .expect("default octagon billiard must succeed");
        billiards.advance_reflections(18);

        Self {
            state,
            tiling_buffer,
            transformed_tiles,
            transformed_tile_count,
            billiards,
            primary_seg_scratch: vec![GeodesicSegment::default(); MAX_BILLIARD_SEGMENTS],
            twin_seg_scratch: vec![GeodesicSegment::default(); MAX_BILLIARD_SEGMENTS],
            edge_hit_scratch: vec![0u8; MAX_BILLIARD_SEGMENTS],
            sep_series_scratch: vec![[0.0; 2]; MAX_BILLIARD_SEGMENTS],
            last_frame_instant: None,
            focused_canvas: None,
            drag_disk_last: None,
            drag_hp_last: None,
            revision: 1,
        }
    }
}

impl HyperbolicGeometryTool {
    /// Applies a zero-allocation `PSU(1,1)` drag isometry between `from` and `to` on $\mathbb{D}$.
    pub fn apply_disk_drag(&mut self, from: DiskPoint, to: DiskPoint) {
        let delta = Psu11::drag_between(from, to);
        self.state.disk_isometry = delta.compose(self.state.disk_isometry);
        self.state.disk_isometry.renormalize();
        self.refresh_transformed_tiles();
        self.bump_revision();
    }

    /// Applies a zero-allocation `PSL(2,R)` drag isometry on $\mathbb{H}$, synchronizing $\mathbb{D}$ via Cayley conjugation.
    pub fn apply_half_plane_drag(&mut self, from: HalfPlanePoint, to: HalfPlanePoint) {
        let delta_hp = Psl2R::drag_between(from, to);
        let mut next_hp = delta_hp.compose(self.state.disk_isometry.to_psl2r());
        next_hp.renormalize();
        self.state.disk_isometry = next_hp.to_psu11();
        self.state.disk_isometry.renormalize();
        self.refresh_transformed_tiles();
        self.bump_revision();
    }

    pub(super) fn refresh_transformed_tiles(&mut self) {
        self.transformed_tile_count = self
            .tiling_buffer
            .transform_into(self.state.disk_isometry, &mut self.transformed_tiles);
    }

    pub(super) fn rebuild_tiling(&mut self) {
        let kind = self.state.resolved_tiling_kind();
        if self
            .tiling_buffer
            .rebuild(kind, self.state.tiling_depth)
            .is_ok()
        {
            self.refresh_transformed_tiles();
        }
    }

    pub(super) fn rebuild_billiards(&mut self) {
        let kind = self.state.resolved_billiard_kind();
        if self
            .billiards
            .reconfigure(kind, self.state.billiard_seed, self.state.perturbation_rad())
            .is_ok()
        {
            self.billiards.advance_reflections(18);
        }
    }

    pub(super) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    pub(super) fn central_tile_area(&self) -> Option<f64> {
        let first = self.transformed_tiles.first()?;
        if first.vertex_count < 3 || matches!(self.tiling_buffer.kind(), TilingKind::Ideal { .. }) {
            return None;
        }
        gauss_bonnet_polygon_area(first.active_vertices()).ok()
    }
}

impl InteractiveTool for HyperbolicGeometryTool {
    fn name(&self) -> &'static str {
        "Hyperbolic Geometry"
    }

    fn domain(&self) -> &'static str {
        "geometry_topology"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.tick_realtime(ui.ctx());
        let panel_id = crate::framework::scoped_panel_id(ui, "hyperbolic_geometry_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .resizable(true)
            .default_width(292.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.render_controls_panel(ui);
                });
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.render_canvases(ui);
        });
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn has_finite_parameters(&self) -> bool {
        self.state.is_valid()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.state.tiling_depth = self.state.tiling_depth.clamp(1, MAX_TILING_DEPTH);
        self.state.schlafli_p = self.state.schlafli_p.clamp(3, MAX_POLYGON_VERTICES as u32);
        self.state.schlafli_q = self.state.schlafli_q.clamp(3, MAX_POLYGON_VERTICES as u32);
        if matches!(self.state.tiling_preset, TilingPreset::CustomSchlafli)
            && validate_hyperbolic_schlafli(self.state.schlafli_p, self.state.schlafli_q).is_err()
        {
            self.state.schlafli_p = 7;
            self.state.schlafli_q = 3;
        }
        self.state.horocycle_radius = if self.state.horocycle_radius.is_finite() {
            self.state.horocycle_radius.clamp(0.05, 0.90)
        } else {
            0.28
        };
        self.state.perturbation_log10 = if self.state.perturbation_log10.is_finite() {
            self.state.perturbation_log10.clamp(-12.0, -1.0)
        } else {
            -6.0
        };
        self.state.billiard_speed = if self.state.billiard_speed.is_finite() {
            self.state.billiard_speed.clamp(0.1, 10.0)
        } else {
            1.6
        };
        self.state.half_plane_span = if self.state.half_plane_span.is_finite() {
            self.state.half_plane_span.clamp(0.5, 25.0)
        } else {
            3.0
        };
        self.state.disk_isometry.renormalize();
        self.rebuild_tiling();
        self.rebuild_billiards();
        self.bump_revision();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state_json: &str) -> Result<(), String> {
        let mut parsed: HyperbolicGeometryState = serde_json::from_str(state_json)
            .map_err(|e| format!("Invalid HyperbolicGeometryTool state JSON: {e}"))?;
        if !parsed.is_valid() {
            return Err("HyperbolicGeometryTool state violates hyperbolic domain bounds".into());
        }
        parsed.disk_isometry.renormalize();
        self.state = parsed;
        self.rebuild_tiling();
        self.rebuild_billiards();
        self.last_frame_instant = None;
        self.bump_revision();
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_export_primitives()
    }

    fn step_simulation(&mut self) {
        match self.state.mode {
            LabMode::TilingExplorer => {
                let step_rot = Psu11::rotation(0.08);
                self.state.disk_isometry = step_rot.compose(self.state.disk_isometry);
                self.state.disk_isometry.renormalize();
                self.refresh_transformed_tiles();
            }
            LabMode::ChaoticBilliards => {
                let ds = 0.12 * self.state.billiard_speed.clamp(0.1, 10.0);
                self.billiards.step(ds);
            }
        }
        self.bump_revision();
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}
