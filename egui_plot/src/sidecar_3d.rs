//! World-coordinate 3D series sidecar for [`crate::Plot`] viewports.
//!
//! Maps projected 2D [`crate::PlotItem`] vertices back to immutable world
//! `[f64; 3]` coordinates so hover tooltips, keyboard point navigation,
//! `Table View`, CSV exports, and JSON exports remain invariant to 3D camera pose.

use std::cell::RefCell;
use std::sync::Arc;

use ahash::HashMap;
use egui::{Color32, Id};

/// A logical 3D series pairing projected 2D canvas points with immutable world `(x, y, z)` coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Plot3DSeries {
    /// Display name of the 3D series.
    pub name: String,
    /// Projected 2D viewport coordinates `[proj_x, proj_y]` matching `world` 1:1.
    pub projected: Vec<[f64; 2]>,
    /// Immutable world-space coordinates `[x, y, z]`.
    pub world: Vec<[f64; 3]>,
    /// Representative series stroke or marker color.
    pub color: Color32,
}

impl Plot3DSeries {
    /// Creates a new logical 3D series.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        projected: Vec<[f64; 2]>,
        world: Vec<[f64; 3]>,
        color: Color32,
    ) -> Self {
        Self {
            name: name.into(),
            projected,
            world,
            color,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct SidecarInner {
    item_vertices: HashMap<Id, Arc<Vec<[f64; 3]>>>,
    fallback_vertices: Vec<([f64; 2], [f64; 3])>,
    series: Vec<Plot3DSeries>,
}

/// Sidecar attached to [`crate::PlotUi`] storing per-item 3D vertex coordinates and logical 3D series.
#[derive(Clone, Debug, Default)]
pub struct Plot3DSeriesSidecar {
    inner: Arc<SidecarInner>,
}

impl Plot3DSeriesSidecar {
    /// Creates an empty 3D series sidecar.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears all stored vertices and series while retaining allocated capacities when uniquely owned.
    pub fn clear(&mut self) {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.item_vertices.clear();
            inner.fallback_vertices.clear();
            inner.series.clear();
        } else {
            self.inner = Arc::default();
        }
    }

    /// Associates a [`crate::PlotItem`]'s `Id` with its per-vertex world `[x, y, z]` coordinates.
    pub fn register_item_vertices(&mut self, item_id: Id, world_vertices: Vec<[f64; 3]>) {
        self.register_shared_item_vertices(item_id, Arc::new(world_vertices));
    }

    /// Associates a [`crate::PlotItem`]'s `Id` with shared per-vertex world `[x, y, z]` coordinates.
    pub fn register_shared_item_vertices(
        &mut self,
        item_id: Id,
        world_vertices: Arc<Vec<[f64; 3]>>,
    ) {
        Arc::make_mut(&mut self.inner)
            .item_vertices
            .insert(item_id, world_vertices);
    }

    /// Registers a projected-to-world coordinate pair for fallback hover lookup.
    pub fn register_projected_pair(&mut self, projected: [f64; 2], world: [f64; 3]) {
        Arc::make_mut(&mut self.inner)
            .fallback_vertices
            .push((projected, world));
    }

    /// Appends a logical 3D series for `Table View`, keyboard navigation, and dataset exports.
    pub fn push_series(&mut self, series: Plot3DSeries) {
        if !series.world.is_empty() {
            Arc::make_mut(&mut self.inner).series.push(series);
        }
    }

    /// Returns `true` when at least one non-empty logical 3D series is registered.
    #[must_use]
    pub fn has_series(&self) -> bool {
        self.inner.series.iter().any(|s| !s.world.is_empty())
    }

    /// Returns `true` when any 3D hover vertices or series are registered.
    #[must_use]
    pub fn has_any_data(&self) -> bool {
        !self.inner.item_vertices.is_empty()
            || !self.inner.fallback_vertices.is_empty()
            || self.has_series()
    }

    /// Returns the registered logical 3D series.
    #[must_use]
    pub fn series(&self) -> &[Plot3DSeries] {
        &self.inner.series
    }

    /// Extracts `(name, world_points)` datasets for 3D CSV and JSON serialization.
    #[must_use]
    pub fn export_datasets_3d(&self) -> Vec<(String, Vec<[f64; 3]>)> {
        self.inner
            .series
            .iter()
            .filter(|s| !s.world.is_empty())
            .map(|s| (s.name.clone(), s.world.clone()))
            .collect()
    }

    /// Extracts `(name, projected_points, color)` tuples for 2D compatibility consumers.
    #[must_use]
    pub fn accessible_datasets_2d(&self) -> Vec<(String, Vec<[f64; 2]>, Color32)> {
        self.inner
            .series
            .iter()
            .filter(|s| !s.projected.is_empty())
            .map(|s| (s.name.clone(), s.projected.clone(), s.color))
            .collect()
    }

    /// Resolves the immutable world `[x, y, z]` coordinate for a hovered [`crate::PlotItem`] vertex.
    #[must_use]
    pub fn world_point(&self, item_id: Id, index: usize, projected: [f64; 2]) -> Option<[f64; 3]> {
        if let Some(world) = self
            .inner
            .item_vertices
            .get(&item_id)
            .and_then(|verts| verts.get(index))
            .copied()
        {
            return Some(world);
        }
        self.lookup_by_projected(projected)
    }

    fn lookup_by_projected(&self, projected: [f64; 2]) -> Option<[f64; 3]> {
        const MAX_DIST_SQ: f64 = 1e-12;
        for (proj, world) in &self.inner.fallback_vertices {
            let dx = proj[0] - projected[0];
            let dy = proj[1] - projected[1];
            if dx * dx + dy * dy <= MAX_DIST_SQ {
                return Some(*world);
            }
        }
        for s in &self.inner.series {
            for (proj, world) in s.projected.iter().zip(&s.world) {
                let dx = proj[0] - projected[0];
                let dy = proj[1] - projected[1];
                if dx * dx + dy * dy <= MAX_DIST_SQ {
                    return Some(*world);
                }
            }
        }
        None
    }
}

/// Formats a single world-space coordinate with deterministic camera-invariant precision.
#[must_use]
pub fn format_world_coord(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            "inf".to_owned()
        } else {
            "-inf".to_owned()
        };
    }
    if value == 0.0 {
        return "0.0000".to_owned();
    }
    if !(1e-3..1e6).contains(&value.abs()) {
        format!("{value:.4e}")
    } else {
        format!("{value:.4}")
    }
}

#[derive(Clone)]
enum Deferred3DDataset {
    Sidecar(Plot3DSeriesSidecar),
    Owned(String, Vec<[f64; 3]>),
}

thread_local! {
    static LAST_PLOT_DATASETS_3D: RefCell<(Vec<Deferred3DDataset>, u64)> =
        const { RefCell::new((Vec::new(), 0)) };
}

/// Clears the thread-local 3D plot datasets recorded during the current frame.
pub fn clear_last_plot_datasets_3d() {
    LAST_PLOT_DATASETS_3D.with(|d| {
        let mut guard = d.borrow_mut();
        guard.0.clear();
        guard.1 = 0;
    });
}

/// Records a shared 3D series sidecar without cloning world coordinate buffers.
pub fn append_last_plot_sidecar_3d(sidecar: &Plot3DSeriesSidecar) {
    if !sidecar.has_series() {
        return;
    }
    LAST_PLOT_DATASETS_3D.with(|d| {
        use std::hash::{Hash as _, Hasher as _};
        let mut guard = d.borrow_mut();
        let mut hasher = ahash::AHasher::default();
        guard.1.hash(&mut hasher);
        (Arc::as_ptr(&sidecar.inner) as usize).hash(&mut hasher);
        guard.0.push(Deferred3DDataset::Sidecar(sidecar.clone()));
        guard.1 = hasher.finish().max(1);
    });
}

/// Appends 3D world-coordinate datasets rendered during the current frame.
pub fn append_last_plot_datasets_3d(datasets: &[(String, Vec<[f64; 3]>)]) {
    if datasets.is_empty() {
        return;
    }
    LAST_PLOT_DATASETS_3D.with(|d| {
        use std::hash::{Hash as _, Hasher as _};
        let mut guard = d.borrow_mut();
        let mut hasher = ahash::AHasher::default();
        guard.1.hash(&mut hasher);
        for (name, pts) in datasets {
            name.hash(&mut hasher);
            pts.len().hash(&mut hasher);
            if let Some(first) = pts.first() {
                for v in first {
                    v.to_bits().hash(&mut hasher);
                }
            }
            guard
                .0
                .push(Deferred3DDataset::Owned(name.clone(), pts.clone()));
        }
        guard.1 = hasher.finish().max(1);
    });
}

/// Returns a non-zero fingerprint when 3D datasets were captured in the active frame (`0` if empty).
#[must_use]
pub fn last_plot_3d_capture_hash() -> u64 {
    LAST_PLOT_DATASETS_3D.with(|d| d.borrow().1)
}

/// Returns all 3D world-coordinate datasets rendered during the current frame.
#[must_use]
pub fn get_last_plot_datasets_3d() -> Vec<(String, Vec<[f64; 3]>)> {
    crate::record_materialized_plot_datasets();
    LAST_PLOT_DATASETS_3D.with(|d| {
        let guard = d.borrow();
        let mut out = Vec::new();
        for entry in &guard.0 {
            match entry {
                Deferred3DDataset::Sidecar(sidecar) => {
                    out.extend(sidecar.export_datasets_3d());
                }
                Deferred3DDataset::Owned(name, pts) => {
                    out.push((name.clone(), pts.clone()));
                }
            }
        }
        out
    })
}

#[cfg(test)]
#[path = "sidecar_3d_tests.rs"]
mod tests;
