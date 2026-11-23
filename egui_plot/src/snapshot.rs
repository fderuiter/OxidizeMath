//! Serializable per-plot overlay snapshots and scoped persistence synchronization.

mod scope;

use std::{
    cmp::Ordering,
    collections::BTreeSet,
    hash::{Hash, Hasher},
};

use egui::{Id, Vec2b};

use crate::{AccessState, FitType, ImportedDataset, PlotBounds, PlotMemory, SavedCallout};

pub(crate) use scope::{
    AfterShowSyncParams, BeforeShowSyncParams, sync_plot_after_show, sync_plot_before_show,
};
pub use scope::{begin_plot_persistence_scope, end_plot_persistence_scope};

/// Maximum number of imported empirical datasets persisted per plot snapshot.
pub const MAX_SNAPSHOT_DATASETS: usize = 8;

/// Maximum total imported CSV points persisted across all datasets in a single plot snapshot.
pub const MAX_SNAPSHOT_TOTAL_POINTS: usize = 900;

/// Maximum number of user callouts persisted per plot snapshot.
pub const MAX_SNAPSHOT_CALLOUTS: usize = 64;

/// Maximum characters retained in dataset names and callout labels.
pub const MAX_SNAPSHOT_LABEL_CHARS: usize = 128;

/// Maximum UTF-8 bytes retained in custom curve-fit expressions and seed strings.
pub const MAX_SNAPSHOT_FORMULA_BYTES: usize = 256;

/// Default minimum grid line spacing in pixels.
pub const DEFAULT_GRID_DENSITY: f32 = 8.0;

/// Default custom nonlinear regression formula.
pub const DEFAULT_CUSTOM_FIT_FORMULA: &str = "a * exp(-b * x) + c";

/// Wrapper around [`egui::Id`] providing a deterministic `u64` ordering for [`BTreeSet`] storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct OrderedId(pub Id);

impl PartialOrd for OrderedId {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedId {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.value().cmp(&other.0.value())
    }
}

#[cfg(feature = "serde")]
const fn default_grid_density() -> f32 {
    DEFAULT_GRID_DENSITY
}
#[cfg(feature = "serde")]
const fn default_true() -> bool {
    true
}
#[cfg(feature = "serde")]
const fn default_show_grid() -> [bool; 2] {
    [true, true]
}
#[cfg(feature = "serde")]
fn default_custom_fit_formula() -> String {
    DEFAULT_CUSTOM_FIT_FORMULA.to_owned()
}
#[cfg(feature = "serde")]
const fn is_false(v: &bool) -> bool {
    !*v
}
#[cfg(feature = "serde")]
const fn is_true(v: &bool) -> bool {
    *v
}
#[cfg(feature = "serde")]
const fn is_zero_usize(v: &usize) -> bool {
    *v == 0
}
#[cfg(feature = "serde")]
fn is_default_grid_density(v: &f32) -> bool {
    (*v - DEFAULT_GRID_DENSITY).abs() <= f32::EPSILON
}
#[cfg(feature = "serde")]
const fn is_default_show_grid(v: &[bool; 2]) -> bool {
    v[0] && v[1]
}
#[cfg(feature = "serde")]
fn is_default_custom_fit_formula(s: &str) -> bool {
    s == DEFAULT_CUSTOM_FIT_FORMULA
}

/// Serializable user-configured overlay, viewport, grid, and analysis state for a single [`crate::Plot`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct PlotOverlaySnapshot {
    /// Axis auto-bounds flags `[x, y]` when manual bounds are active.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub auto_bounds: Option<[bool; 2]>,
    /// Persisted 2D plot bounds when manual pan/zoom has overridden default bounds.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub bounds: Option<PlotBounds>,
    /// Deterministically ordered legend item IDs hidden by the user.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "BTreeSet::is_empty")
    )]
    pub hidden_items: BTreeSet<OrderedId>,
    /// Imported empirical CSV datasets overlaid on this plot.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub imported_datasets: Vec<ImportedDataset>,
    /// User-pinned callout annotations overlaid on this plot.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub callouts: Vec<SavedCallout>,
    /// Logarithmic X-axis toggle.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_false"))]
    pub log_x: bool,
    /// Logarithmic Y-axis toggle.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_false"))]
    pub log_y: bool,
    /// Minimum grid line spacing in pixels (`4.0..=100.0`).
    #[cfg_attr(
        feature = "serde",
        serde(
            default = "default_grid_density",
            skip_serializing_if = "is_default_grid_density"
        )
    )]
    pub grid_density: f32,
    /// Visibility of X-axis tick labels.
    #[cfg_attr(
        feature = "serde",
        serde(default = "default_true", skip_serializing_if = "is_true")
    )]
    pub show_x_labels: bool,
    /// Visibility of Y-axis tick labels.
    #[cfg_attr(
        feature = "serde",
        serde(default = "default_true", skip_serializing_if = "is_true")
    )]
    pub show_y_labels: bool,
    /// Clamp grid lines to actual data range.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_false"))]
    pub clamp_grid: bool,
    /// Grid line visibility `[x, y]`.
    #[cfg_attr(
        feature = "serde",
        serde(
            default = "default_show_grid",
            skip_serializing_if = "is_default_show_grid"
        )
    )]
    pub show_grid: [bool; 2],
    /// Whether the accessible Data Table modal is open.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_false"))]
    pub show_table: bool,
    /// Active regression model selected in `Fit Curve`.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub fit_model: Option<FitType>,
    /// Optional legacy/shorthand polynomial degree (`1..=4`), normalized into `fit_model`.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub fit_degree: Option<usize>,
    /// Whether curve fitting evaluates in log-transformed space.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_false"))]
    pub fit_log_mode: bool,
    /// Whether residual stem markers are overlaid for the active curve fit.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_false"))]
    pub show_fit_residuals: bool,
    /// Target series index for curve fitting.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "is_zero_usize")
    )]
    pub fit_dataset_idx: usize,
    /// Whether the active curve-fit line and formula badge are visible.
    #[cfg_attr(
        feature = "serde",
        serde(default = "default_true", skip_serializing_if = "is_true")
    )]
    pub show_fit_overlay: bool,
    /// Custom Levenberg-Marquardt fit expression.
    #[cfg_attr(
        feature = "serde",
        serde(
            default = "default_custom_fit_formula",
            skip_serializing_if = "is_default_custom_fit_formula"
        )
    )]
    pub custom_fit_formula: String,
    /// Custom Levenberg-Marquardt initial parameter seeds string.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "String::is_empty")
    )]
    pub custom_fit_seeds_text: String,
}

impl Default for PlotOverlaySnapshot {
    fn default() -> Self {
        Self {
            auto_bounds: None,
            bounds: None,
            hidden_items: BTreeSet::new(),
            imported_datasets: Vec::new(),
            callouts: Vec::new(),
            log_x: false,
            log_y: false,
            grid_density: DEFAULT_GRID_DENSITY,
            show_x_labels: true,
            show_y_labels: true,
            clamp_grid: false,
            show_grid: [true, true],
            show_table: false,
            fit_model: None,
            fit_degree: None,
            fit_log_mode: false,
            show_fit_residuals: false,
            fit_dataset_idx: 0,
            show_fit_overlay: true,
            custom_fit_formula: DEFAULT_CUSTOM_FIT_FORMULA.to_owned(),
            custom_fit_seeds_text: String::new(),
        }
    }
}

fn norm_f64_bits(v: f64) -> u64 {
    if v == 0.0 {
        0.0_f64.to_bits()
    } else {
        v.to_bits()
    }
}

fn norm_f32_bits(v: f32) -> u32 {
    if v == 0.0 {
        0.0_f32.to_bits()
    } else {
        v.to_bits()
    }
}

fn truncate_chars(s: &mut String, max_chars: usize) {
    if s.chars().count() > max_chars {
        *s = s.chars().take(max_chars).collect();
    }
}

fn truncate_utf8_bytes(s: &mut String, max_bytes: usize) {
    if s.len() <= max_bytes {
        return;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
}

fn downsample_points(points: &[[f64; 2]], cap: usize) -> Vec<[f64; 2]> {
    if cap == 0 || points.is_empty() {
        return Vec::new();
    }
    if points.len() <= cap {
        return points.to_vec();
    }
    if cap == 1 {
        return vec![points[0]];
    }
    let last = points.len() - 1;
    (0..cap).map(|i| points[(i * last) / (cap - 1)]).collect()
}

fn cap_datasets_in_place(datasets: &mut Vec<ImportedDataset>) {
    if datasets.len() > MAX_SNAPSHOT_DATASETS {
        datasets.truncate(MAX_SNAPSHOT_DATASETS);
    }
    let total_pts: usize = datasets.iter().map(|d| d.points.len()).sum();
    if total_pts <= MAX_SNAPSHOT_TOTAL_POINTS {
        return;
    }
    let count = datasets.len().max(1);
    let per_ds_cap = (MAX_SNAPSHOT_TOTAL_POINTS / count).max(2);
    let mut remaining = MAX_SNAPSHOT_TOTAL_POINTS;
    for ds in datasets.iter_mut() {
        let budget = per_ds_cap.min(remaining);
        ds.points = downsample_points(&ds.points, budget);
        remaining = remaining.saturating_sub(ds.points.len());
    }
}

impl PlotOverlaySnapshot {
    /// Returns `true` if every field matches [`PlotOverlaySnapshot::default`].
    #[must_use]
    pub fn is_default(&self) -> bool {
        self.auto_bounds.is_none()
            && self.bounds.is_none()
            && self.hidden_items.is_empty()
            && self.imported_datasets.is_empty()
            && self.callouts.is_empty()
            && !self.log_x
            && !self.log_y
            && (self.grid_density - DEFAULT_GRID_DENSITY).abs() <= f32::EPSILON
            && self.show_x_labels
            && self.show_y_labels
            && !self.clamp_grid
            && self.show_grid == [true, true]
            && !self.show_table
            && self.fit_model.is_none()
            && self.fit_degree.is_none()
            && !self.fit_log_mode
            && !self.show_fit_residuals
            && self.fit_dataset_idx == 0
            && self.show_fit_overlay
            && self.custom_fit_formula == DEFAULT_CUSTOM_FIT_FORMULA
            && self.custom_fit_seeds_text.is_empty()
    }

    /// Resets any persisted manual 2D viewport bounds so the plot re-enters auto-bounds framing.
    pub fn clear_manual_bounds(&mut self) {
        self.auto_bounds = None;
        self.bounds = None;
    }

    /// Validates finiteness of all coordinates/bounds/offsets, clamps `fit_degree` to `1..=4`,
    /// and enforces dataset/point/callout count caps.
    pub fn sanitize_and_validate(&mut self) -> Result<(), String> {
        if !self.grid_density.is_finite() {
            return Err("PlotOverlaySnapshot grid_density must be finite".to_owned());
        }
        self.grid_density = self.grid_density.clamp(4.0, 100.0);

        if let Some(b) = self.bounds {
            if !b.is_finite() || b.min()[0] > b.max()[0] || b.min()[1] > b.max()[1] {
                return Err("PlotOverlaySnapshot bounds must be finite with min <= max".to_owned());
            }
            if self.auto_bounds == Some([true, true]) {
                self.bounds = None;
                self.auto_bounds = None;
            } else if self.auto_bounds.is_none() {
                self.auto_bounds = Some([false, false]);
            }
        } else {
            self.auto_bounds = None;
        }

        if let Some(deg) = self.fit_degree.take()
            && self.fit_model.is_none()
        {
            self.fit_model = Some(match deg.clamp(1, 4) {
                1 => FitType::Linear,
                2 => FitType::Quadratic,
                3 => FitType::Cubic,
                _ => FitType::Quartic,
            });
        }

        truncate_utf8_bytes(&mut self.custom_fit_formula, MAX_SNAPSHOT_FORMULA_BYTES);
        truncate_utf8_bytes(&mut self.custom_fit_seeds_text, MAX_SNAPSHOT_FORMULA_BYTES);
        self.validate_overlays()
    }

    fn validate_overlays(&mut self) -> Result<(), String> {
        for callout in &mut self.callouts {
            if !callout.point[0].is_finite()
                || !callout.point[1].is_finite()
                || !callout.offset[0].is_finite()
                || !callout.offset[1].is_finite()
            {
                return Err(
                    "PlotOverlaySnapshot callout coordinates and offsets must be finite".to_owned(),
                );
            }
            truncate_chars(&mut callout.label, MAX_SNAPSHOT_LABEL_CHARS);
            if callout.label.trim().is_empty() {
                callout.label = format!("({:.2}, {:.2})", callout.point[0], callout.point[1]);
            }
        }
        if self.callouts.len() > MAX_SNAPSHOT_CALLOUTS {
            let drop_count = self.callouts.len() - MAX_SNAPSHOT_CALLOUTS;
            self.callouts.drain(0..drop_count);
        }

        for ds in &mut self.imported_datasets {
            if ds
                .points
                .iter()
                .any(|p| !p[0].is_finite() || !p[1].is_finite())
            {
                return Err(
                    "PlotOverlaySnapshot imported_datasets points must be finite".to_owned(),
                );
            }
            truncate_chars(&mut ds.name, MAX_SNAPSHOT_LABEL_CHARS);
        }
        cap_datasets_in_place(&mut self.imported_datasets);
        Ok(())
    }

    /// Feeds a deterministic fingerprint of this snapshot into `hasher`.
    pub fn hash_state(&self, hasher: &mut impl Hasher) {
        self.auto_bounds.hash(hasher);
        if let Some(b) = self.bounds {
            1_u8.hash(hasher);
            norm_f64_bits(b.min()[0]).hash(hasher);
            norm_f64_bits(b.min()[1]).hash(hasher);
            norm_f64_bits(b.max()[0]).hash(hasher);
            norm_f64_bits(b.max()[1]).hash(hasher);
        } else {
            0_u8.hash(hasher);
        }
        self.hidden_items.len().hash(hasher);
        for id in &self.hidden_items {
            id.0.value().hash(hasher);
        }
        self.hash_overlays_and_settings(hasher);
    }

    fn hash_overlays_and_settings(&self, hasher: &mut impl Hasher) {
        self.imported_datasets.len().hash(hasher);
        for ds in &self.imported_datasets {
            ds.name.hash(hasher);
            ds.visible.hash(hasher);
            ds.show_points.hash(hasher);
            ds.show_lines.hash(hasher);
            ds.points.len().hash(hasher);
            for pt in &ds.points {
                norm_f64_bits(pt[0]).hash(hasher);
                norm_f64_bits(pt[1]).hash(hasher);
            }
        }
        self.callouts.len().hash(hasher);
        for c in &self.callouts {
            norm_f64_bits(c.point[0]).hash(hasher);
            norm_f64_bits(c.point[1]).hash(hasher);
            c.label.hash(hasher);
            norm_f32_bits(c.offset[0]).hash(hasher);
            norm_f32_bits(c.offset[1]).hash(hasher);
        }
        self.log_x.hash(hasher);
        self.log_y.hash(hasher);
        norm_f32_bits(self.grid_density).hash(hasher);
        self.show_x_labels.hash(hasher);
        self.show_y_labels.hash(hasher);
        self.clamp_grid.hash(hasher);
        self.show_grid.hash(hasher);
        self.show_table.hash(hasher);
        self.fit_model.hash(hasher);
        self.fit_log_mode.hash(hasher);
        self.show_fit_residuals.hash(hasher);
        self.fit_dataset_idx.hash(hasher);
        self.show_fit_overlay.hash(hasher);
        self.custom_fit_formula.hash(hasher);
        self.custom_fit_seeds_text.hash(hasher);
    }

    pub(crate) fn capture(
        mem: &PlotMemory,
        access: &AccessState,
        default_auto_bounds: Vec2b,
        min_auto_bounds: PlotBounds,
        has_data_aspect: bool,
        allow_bounds_persist: bool,
        allow_2d_analysis: bool,
    ) -> Self {
        let (auto_bounds, bounds) = if allow_bounds_persist {
            capture_bounds(mem, default_auto_bounds, min_auto_bounds, has_data_aspect)
        } else {
            (None, None)
        };
        let mut datasets: Vec<ImportedDataset> = mem
            .imported_datasets
            .iter()
            .filter(|d| {
                d.points
                    .iter()
                    .all(|p| p[0].is_finite() && p[1].is_finite())
            })
            .cloned()
            .collect();
        cap_datasets_in_place(&mut datasets);

        let mut callouts: Vec<SavedCallout> = if allow_2d_analysis {
            mem.callouts
                .iter()
                .filter(|c| {
                    c.point[0].is_finite()
                        && c.point[1].is_finite()
                        && c.offset[0].is_finite()
                        && c.offset[1].is_finite()
                })
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        if callouts.len() > MAX_SNAPSHOT_CALLOUTS {
            let drop_count = callouts.len() - MAX_SNAPSHOT_CALLOUTS;
            callouts.drain(0..drop_count);
        }

        let mut snap = Self {
            auto_bounds,
            bounds,
            hidden_items: mem.hidden_items.iter().copied().map(OrderedId).collect(),
            imported_datasets: datasets,
            callouts,
            log_x: mem.log_x,
            log_y: mem.log_y,
            grid_density: mem.grid_density.clamp(4.0, 100.0),
            show_x_labels: mem.show_x_labels,
            show_y_labels: mem.show_y_labels,
            clamp_grid: mem.clamp_grid,
            show_grid: [mem.show_grid.x, mem.show_grid.y],
            show_table: access.show_table,
            fit_model: if allow_2d_analysis {
                access.fit_model
            } else {
                None
            },
            fit_degree: None,
            fit_log_mode: access.fit_log_mode,
            show_fit_residuals: access.show_fit_residuals,
            fit_dataset_idx: access.fit_dataset_idx,
            show_fit_overlay: access.show_fit_overlay,
            custom_fit_formula: access.custom_fit_formula.clone(),
            custom_fit_seeds_text: access.custom_fit_seeds_text.clone(),
        };
        let _ = snap.sanitize_and_validate();
        snap
    }
}

fn capture_bounds(
    mem: &PlotMemory,
    default_auto_bounds: Vec2b,
    min_auto_bounds: PlotBounds,
    has_data_aspect: bool,
) -> (Option<[bool; 2]>, Option<PlotBounds>) {
    let b = *mem.bounds();
    if !b.is_valid() || !b.is_finite() {
        return (None, None);
    }
    let modified_axis = (0..2).any(|d| {
        if default_auto_bounds[d] {
            !mem.auto_bounds[d]
        } else if !has_data_aspect
            && min_auto_bounds.min()[d].is_finite()
            && min_auto_bounds.max()[d].is_finite()
        {
            !mem.auto_bounds[d]
                && ((b.min()[d] - min_auto_bounds.min()[d]).abs() > 1e-6
                    || (b.max()[d] - min_auto_bounds.max()[d]).abs() > 1e-6)
        } else {
            false
        }
    });
    if modified_axis {
        (Some([mem.auto_bounds.x, mem.auto_bounds.y]), Some(b))
    } else {
        (None, None)
    }
}
