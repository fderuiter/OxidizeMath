#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Compact `< 4 KiB` serializable parameter schema and validation for [`super::FabricationStudioTool`].

use egui_plot::{
    FabricationBedPreset, FabricationSvgOptions, MAX_FABRICATION_BED_MM, MIN_FABRICATION_BED_MM,
};
use math_commons::registry::MAX_GRID_SIZE;
use math_explorer::pure_math::tilings::{TilingKind, MAX_TILING_DEPTH};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::hash::{Hash, Hasher};

/// Maximum flow elements stored in a single studio preset so share URLs stay well below `4 KiB`.
pub const MAX_STUDIO_FLOW_ELEMENTS: usize = 16;

/// Maximum deterministic Turing grid width/height exposed in the interactive studio.
pub const MAX_STUDIO_TURING_GRID: usize = 128;

/// Geometry family selected inside [`super::FabricationStudioTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum FabricationSourceKind {
    /// 2D Schnakenberg Turing reaction-diffusion field with marching-squares cut/score contours.
    TuringField,
    /// 2D potential-flow superposition with RK4 streamlines and dividing obstacle contours.
    PotentialFlow,
    /// Aperiodic Penrose (`P2`, `P3`) or Einstein monotile (`Hat`, `Spectre`) patch.
    #[default]
    AperiodicTiling,
}

impl FabricationSourceKind {
    /// Display label shown in UI selectors.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::TuringField => "Turing Reaction-Diffusion",
            Self::PotentialFlow => "Potential Flow Streamlines",
            Self::AperiodicTiling => "Aperiodic Tiling",
        }
    }
}

/// Serializable selector mirroring [`TilingKind`] for URL and JSON presets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum FabricationTilingSelector {
    /// Penrose `P2` Kite & Dart tiling.
    PenroseP2,
    /// Penrose `P3` Thick & Thin Rhomb tiling.
    PenroseP3,
    /// Smith–Myers–Kaplan–Goodman-Strauss `Hat` (`Tile(1, √3)`) monotile.
    #[default]
    EinsteinHat,
    /// Chiral `Spectre` (`Tile(1, 1)`) monotile.
    EinsteinSpectre,
}

impl FabricationTilingSelector {
    /// Converts the UI selector to [`TilingKind`].
    #[must_use]
    pub const fn to_domain_kind(self) -> TilingKind {
        match self {
            Self::PenroseP2 => TilingKind::PenroseP2,
            Self::PenroseP3 => TilingKind::PenroseP3,
            Self::EinsteinHat => TilingKind::EinsteinHat,
            Self::EinsteinSpectre => TilingKind::EinsteinSpectre,
        }
    }

    /// Display label for tiling radio/combo controls.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::PenroseP2 => "Penrose P2 (Kite & Dart)",
            Self::PenroseP3 => "Penrose P3 (Rhomb)",
            Self::EinsteinHat => "Einstein Hat Monotile",
            Self::EinsteinSpectre => "Einstein Spectre (Chiral)",
        }
    }
}

/// Physical sheet / laser-bed mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum StudioBedMode {
    /// ISO `A4` (`210 × 297 mm`).
    #[default]
    A4,
    /// ANSI `US Letter` (`215.9 × 279.4 mm`).
    UsLetter,
    /// User-defined bed dimensions (`20..=1200 mm`).
    Custom,
}

/// Elementary potential-flow singularity or freestream type for [`StudioFlowElement`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum StudioFlowElementKind {
    /// Uniform freestream flow at `angle_deg`.
    Uniform,
    /// Radial point source (`strength > 0`).
    #[default]
    Source,
    /// Radial point sink.
    Sink,
    /// Point circulation vortex.
    Vortex,
    /// Point dipole / doublet.
    Doublet,
}

/// Compact serializable potential-flow element specification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudioFlowElement {
    /// Element singularity kind.
    pub kind: StudioFlowElementKind,
    /// Element strength or freestream speed (`-1000.0..=1000.0`).
    pub strength: f64,
    /// Planar X coordinate (`-50.0..=50.0`).
    pub x: f64,
    /// Planar Y coordinate (`-50.0..=50.0`).
    pub y: f64,
    /// Orientation angle in degrees (`-180.0..=180.0`).
    pub angle_deg: f64,
}

impl StudioFlowElement {
    /// Checks finiteness and boundedness of all element fields.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.strength.is_finite()
            && (-1_000.0..=1_000.0).contains(&self.strength)
            && self.x.is_finite()
            && (-50.0..=50.0).contains(&self.x)
            && self.y.is_finite()
            && (-50.0..=50.0).contains(&self.y)
            && self.angle_deg.is_finite()
            && (-180.0..=180.0).contains(&self.angle_deg)
    }

    fn hash_into(&self, h: &mut impl Hasher) {
        self.kind.hash(h);
        self.strength.to_bits().hash(h);
        self.x.to_bits().hash(h);
        self.y.to_bits().hash(h);
        self.angle_deg.to_bits().hash(h);
    }
}

/// Validated serializable state for [`super::FabricationStudioTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FabricationStudioState {
    /// Active mathematical generator family.
    pub source: FabricationSourceKind,
    /// Sheet / laser bed preset mode.
    pub bed_mode: StudioBedMode,
    /// Custom bed width in millimeters (`20.0..=1200.0`).
    pub custom_width_mm: f64,
    /// Custom bed height in millimeters (`20.0..=1200.0`).
    pub custom_height_mm: f64,
    /// Uniform safe inset margin in millimeters (`0.0..=100.0`).
    pub margin_mm: f64,
    /// Reorders paths and reverses open endpoints via nearest-neighbor heuristic.
    pub optimize_travel: bool,
    /// Renders rapid non-cutting pen-lift travel segments in the `egui_plot` preview.
    pub show_pen_travel: bool,
    /// Schnakenberg activator rate `a` (`0.02..=0.5`).
    pub turing_a: f64,
    /// Schnakenberg inhibitor rate `b` (`0.2..=1.8`).
    pub turing_b: f64,
    /// Activator diffusion `D_u` (`0.2..=4.0`).
    pub turing_d_u: f64,
    /// Inhibitor diffusion `D_v` (`10.0..=150.0`).
    pub turing_d_v: f64,
    /// Reaction-diffusion step size `dt` (`0.005..=0.08`).
    pub turing_dt: f64,
    /// Square reaction-diffusion grid width/height (`16..=128`).
    pub turing_grid_size: usize,
    /// Number of deterministic integration steps (`10..=400`).
    pub turing_steps: usize,
    /// Number of concentric marching-squares score isolines (`1..=8`).
    pub turing_score_levels: usize,
    /// Emits a circular coaster outer cut perimeter around the Turing field.
    pub turing_coaster_border: bool,
    /// Potential-flow element list (`1..=16`).
    pub flow_elements: Vec<StudioFlowElement>,
    /// Half-width of the square potential-flow domain (`2.0..=50.0`).
    pub flow_view_range: f64,
    /// Number of RK4 streamlines traced across the flow domain (`4..=48`).
    pub flow_streamline_count: usize,
    /// Selected aperiodic tiling family.
    pub tiling_kind: FabricationTilingSelector,
    /// Substitution / deflation depth (`0..=6`).
    pub tiling_depth: usize,
}

impl Default for FabricationStudioState {
    fn default() -> Self {
        let kappa = 2.0 * PI * 10.0 * 4.0;
        Self {
            source: FabricationSourceKind::AperiodicTiling,
            bed_mode: StudioBedMode::A4,
            custom_width_mm: 200.0,
            custom_height_mm: 200.0,
            margin_mm: 10.0,
            optimize_travel: true,
            show_pen_travel: true,
            turing_a: 0.10,
            turing_b: 0.90,
            turing_d_u: 1.0,
            turing_d_v: 45.0,
            turing_dt: 0.04,
            turing_grid_size: 48,
            turing_steps: 140,
            turing_score_levels: 4,
            turing_coaster_border: true,
            flow_elements: vec![
                StudioFlowElement {
                    kind: StudioFlowElementKind::Uniform,
                    strength: 10.0,
                    x: 0.0,
                    y: 0.0,
                    angle_deg: 0.0,
                },
                StudioFlowElement {
                    kind: StudioFlowElementKind::Doublet,
                    strength: kappa,
                    x: 0.0,
                    y: 0.0,
                    angle_deg: 0.0,
                },
            ],
            flow_view_range: 10.0,
            flow_streamline_count: 16,
            tiling_kind: FabricationTilingSelector::EinsteinHat,
            tiling_depth: 1,
        }
    }
}

impl FabricationStudioState {
    /// Resolves the active [`FabricationBedPreset`].
    #[must_use]
    pub const fn bed_preset(&self) -> FabricationBedPreset {
        match self.bed_mode {
            StudioBedMode::A4 => FabricationBedPreset::A4,
            StudioBedMode::UsLetter => FabricationBedPreset::UsLetter,
            StudioBedMode::Custom => FabricationBedPreset::Custom {
                width_mm: self.custom_width_mm,
                height_mm: self.custom_height_mm,
            },
        }
    }

    /// Builds [`FabricationSvgOptions`] for millimeter serialization.
    #[must_use]
    pub fn svg_options(&self) -> FabricationSvgOptions {
        FabricationSvgOptions {
            bed: self.bed_preset(),
            margin_mm: self.margin_mm,
            ..FabricationSvgOptions::default()
        }
    }

    /// Validates all domain, grid, tiling-depth, and physical-bed guardrails.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_bed_and_margins()?;
        self.validate_turing_params()?;
        self.validate_flow_and_tiling_params()
    }

    fn validate_bed_and_margins(&self) -> Result<(), String> {
        if !self.custom_width_mm.is_finite()
            || !self.custom_height_mm.is_finite()
            || !(MIN_FABRICATION_BED_MM..=MAX_FABRICATION_BED_MM).contains(&self.custom_width_mm)
            || !(MIN_FABRICATION_BED_MM..=MAX_FABRICATION_BED_MM).contains(&self.custom_height_mm)
        {
            return Err(format!(
                "Custom bed dimensions ({} x {} mm) must lie within [{MIN_FABRICATION_BED_MM}, {MAX_FABRICATION_BED_MM}] mm",
                self.custom_width_mm, self.custom_height_mm
            ));
        }
        let (w_mm, h_mm) = self.bed_preset().dimensions_mm()?;
        if !self.margin_mm.is_finite()
            || self.margin_mm < 0.0
            || self.margin_mm > 100.0
            || 2.0 * self.margin_mm >= w_mm.min(h_mm)
        {
            return Err(format!(
                "Fabrication margin {} mm is invalid for {w_mm} x {h_mm} mm bed",
                self.margin_mm
            ));
        }
        Ok(())
    }

    fn validate_turing_params(&self) -> Result<(), String> {
        let checks = [
            ("turing_a", self.turing_a, 0.02, 0.5),
            ("turing_b", self.turing_b, 0.2, 1.8),
            ("turing_d_u", self.turing_d_u, 0.2, 4.0),
            ("turing_d_v", self.turing_d_v, 10.0, 150.0),
            ("turing_dt", self.turing_dt, 0.005, 0.08),
        ];
        for (name, val, lo, hi) in checks {
            if !val.is_finite() || !(lo..=hi).contains(&val) {
                return Err(format!(
                    "Fabrication parameter '{name}' = {val} must lie within [{lo}, {hi}]"
                ));
            }
        }
        let max_grid = MAX_STUDIO_TURING_GRID.min(MAX_GRID_SIZE);
        if !(16..=max_grid).contains(&self.turing_grid_size)
            || !(10..=400).contains(&self.turing_steps)
            || !(1..=8).contains(&self.turing_score_levels)
        {
            return Err("Fabrication Turing grid, steps, or score levels out of range".to_string());
        }
        Ok(())
    }

    fn validate_flow_and_tiling_params(&self) -> Result<(), String> {
        if self.flow_elements.is_empty()
            || self.flow_elements.len() > MAX_STUDIO_FLOW_ELEMENTS
            || !self.flow_elements.iter().all(StudioFlowElement::is_valid)
        {
            return Err("Invalid or oversized potential-flow element list".to_string());
        }
        if !self.flow_view_range.is_finite()
            || !(2.0..=50.0).contains(&self.flow_view_range)
            || !(4..=48).contains(&self.flow_streamline_count)
        {
            return Err("Potential-flow view range or streamline count out of bounds".to_string());
        }
        if self.tiling_depth > MAX_TILING_DEPTH {
            return Err(format!(
                "Tiling depth {} exceeds MAX_TILING_DEPTH ({MAX_TILING_DEPTH})",
                self.tiling_depth
            ));
        }
        Ok(())
    }

    /// Clamps any out-of-range numeric parameter back into safe finite bounds.
    pub fn clamp_in_place(&mut self) {
        self.custom_width_mm = clamp_finite(
            self.custom_width_mm,
            MIN_FABRICATION_BED_MM,
            MAX_FABRICATION_BED_MM,
            200.0,
        );
        self.custom_height_mm = clamp_finite(
            self.custom_height_mm,
            MIN_FABRICATION_BED_MM,
            MAX_FABRICATION_BED_MM,
            200.0,
        );
        let (w, h) = self.bed_preset().clamped_dimensions_mm();
        let max_margin = (0.4 * w.min(h)).min(100.0);
        self.margin_mm = clamp_finite(self.margin_mm, 0.0, max_margin, 10.0);
        self.turing_a = clamp_finite(self.turing_a, 0.02, 0.5, 0.10);
        self.turing_b = clamp_finite(self.turing_b, 0.2, 1.8, 0.90);
        self.turing_d_u = clamp_finite(self.turing_d_u, 0.2, 4.0, 1.0);
        self.turing_d_v = clamp_finite(self.turing_d_v, 10.0, 150.0, 45.0);
        self.turing_dt = clamp_finite(self.turing_dt, 0.005, 0.08, 0.04);
        self.turing_grid_size = self
            .turing_grid_size
            .clamp(16, MAX_STUDIO_TURING_GRID.min(MAX_GRID_SIZE));
        self.turing_steps = self.turing_steps.clamp(10, 400);
        self.turing_score_levels = self.turing_score_levels.clamp(1, 8);
        self.flow_view_range = clamp_finite(self.flow_view_range, 2.0, 50.0, 10.0);
        self.flow_streamline_count = self.flow_streamline_count.clamp(4, 48);
        self.tiling_depth = self.tiling_depth.min(MAX_TILING_DEPTH);
        self.flow_elements.truncate(MAX_STUDIO_FLOW_ELEMENTS);
        if self.flow_elements.is_empty() {
            self.flow_elements = Self::default().flow_elements;
        }
        for el in &mut self.flow_elements {
            el.strength = clamp_finite(el.strength, -1_000.0, 1_000.0, 10.0);
            el.x = clamp_finite(el.x, -50.0, 50.0, 0.0);
            el.y = clamp_finite(el.y, -50.0, 50.0, 0.0);
            el.angle_deg = clamp_finite(el.angle_deg, -180.0, 180.0, 0.0);
        }
    }

    /// Computes a deterministic hash across all persisted studio parameters.
    pub fn hash_into(&self, h: &mut impl Hasher) {
        self.source.hash(h);
        self.bed_mode.hash(h);
        self.custom_width_mm.to_bits().hash(h);
        self.custom_height_mm.to_bits().hash(h);
        self.margin_mm.to_bits().hash(h);
        self.optimize_travel.hash(h);
        self.show_pen_travel.hash(h);
        self.turing_a.to_bits().hash(h);
        self.turing_b.to_bits().hash(h);
        self.turing_d_u.to_bits().hash(h);
        self.turing_d_v.to_bits().hash(h);
        self.turing_dt.to_bits().hash(h);
        self.turing_grid_size.hash(h);
        self.turing_steps.hash(h);
        self.turing_score_levels.hash(h);
        self.turing_coaster_border.hash(h);
        self.flow_elements.len().hash(h);
        for el in &self.flow_elements {
            el.hash_into(h);
        }
        self.flow_view_range.to_bits().hash(h);
        self.flow_streamline_count.hash(h);
        self.tiling_kind.hash(h);
        self.tiling_depth.hash(h);
    }
}

fn clamp_finite(val: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(lo, hi)
    } else {
        fallback
    }
}
