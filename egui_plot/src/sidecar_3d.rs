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
struct Plot3DSeriesSidecarInner {
    item_vertices: HashMap<Id, Vec<[f64; 3]>>,
    fallback_vertices: Vec<([f64; 2], [f64; 3])>,
    series: Vec<Plot3DSeries>,
    series_projected_shared: Vec<crate::SharedPoints>,
}

/// Sidecar attached to [`crate::PlotUi`] storing per-item 3D vertex coordinates and logical 3D series.
///
/// Cloning [`Plot3DSeriesSidecar`] shares an internal reference-counted snapshot in $O(1)$ without
/// duplicating vertex maps or series buffers across idle frames; mutating methods detach via copy-on-write.
#[derive(Clone, Debug, Default)]
pub struct Plot3DSeriesSidecar {
    inner: Arc<Plot3DSeriesSidecarInner>,
}

impl Plot3DSeriesSidecar {
    /// Creates an empty 3D series sidecar.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the raw pointer address of the shared backing storage for zero-copy identity checks.
    #[must_use]
    pub fn shared_ptr(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    /// Returns `true` when `self` and `other` share the same underlying allocation.
    #[must_use]
    pub fn shares_allocation_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Clears all stored vertices and series, reusing capacity if uniquely owned or allocating a fresh empty snapshot if shared.
    pub fn clear(&mut self) {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.item_vertices.clear();
            inner.fallback_vertices.clear();
            inner.series.clear();
            inner.series_projected_shared.clear();
        } else {
            self.inner = Arc::default();
        }
    }

    /// Associates a [`crate::PlotItem`]'s `Id` with its per-vertex world `[x, y, z]` coordinates.
    pub fn register_item_vertices(&mut self, item_id: Id, world_vertices: Vec<[f64; 3]>) {
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
            let shared = crate::SharedPoints::from_slice(&series.projected);
            let inner = Arc::make_mut(&mut self.inner);
            inner.series_projected_shared.push(shared);
            inner.series.push(series);
        }
    }

    /// Returns the registered logical 3D series.
    #[must_use]
    pub fn series(&self) -> &[Plot3DSeries] {
        &self.inner.series
    }

    /// Returns `true` if any registered 3D series contains world points.
    #[must_use]
    pub fn has_export_points(&self) -> bool {
        self.inner.series.iter().any(|s| !s.world.is_empty())
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

    /// Extracts `(name, shared_projected_points, color)` tuples for zero-copy 2D plot consumers.
    #[must_use]
    pub fn accessible_shared_datasets_2d(&self) -> Vec<(String, crate::SharedPoints, Color32)> {
        self.inner
            .series
            .iter()
            .zip(&self.inner.series_projected_shared)
            .filter(|(s, _)| !s.projected.is_empty())
            .map(|(s, shared)| (s.name.clone(), shared.clone(), s.color))
            .collect()
    }

    pub(crate) fn push_deferred_2d_datasets(&self, out: &mut Vec<crate::DeferredDatasetEntry>) {
        for (s, shared) in self
            .inner
            .series
            .iter()
            .zip(&self.inner.series_projected_shared)
        {
            if !shared.is_empty() {
                out.push(crate::DeferredDatasetEntry::SharedCoords {
                    name: s.name.clone(),
                    points: shared.clone(),
                });
            }
        }
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
enum DeferredDatasets3DEntry {
    Sidecar(Plot3DSeriesSidecar),
    Owned(Vec<(String, Vec<[f64; 3]>)>),
}

thread_local! {
    static LAST_PLOT_DATASETS_3D: RefCell<Vec<DeferredDatasets3DEntry>> =
        const { RefCell::new(Vec::new()) };
}

/// Clears the thread-local 3D plot datasets recorded during the current frame.
pub fn clear_last_plot_datasets_3d() {
    LAST_PLOT_DATASETS_3D.with(|d| d.borrow_mut().clear());
}

/// Appends a shared [`Plot3DSeriesSidecar`] handle without cloning vertex arrays during the frame.
pub fn append_last_plot_sidecar_3d(sidecar: &Plot3DSeriesSidecar) {
    if !sidecar.has_export_points() {
        return;
    }
    LAST_PLOT_DATASETS_3D.with(|d| {
        d.borrow_mut()
            .push(DeferredDatasets3DEntry::Sidecar(sidecar.clone()));
    });
}

/// Appends 3D world-coordinate datasets rendered during the current frame.
pub fn append_last_plot_datasets_3d(datasets: &[(String, Vec<[f64; 3]>)]) {
    if datasets.is_empty() {
        return;
    }
    LAST_PLOT_DATASETS_3D.with(|d| {
        d.borrow_mut()
            .push(DeferredDatasets3DEntry::Owned(datasets.to_vec()));
    });
}

/// Returns all 3D world-coordinate datasets rendered during the current frame.
#[must_use]
pub fn get_last_plot_datasets_3d() -> Vec<(String, Vec<[f64; 3]>)> {
    LAST_PLOT_DATASETS_3D.with(|d| {
        let mut out = Vec::new();
        for entry in d.borrow().iter() {
            match entry {
                DeferredDatasets3DEntry::Sidecar(sidecar) => {
                    out.extend(sidecar.export_datasets_3d());
                }
                DeferredDatasets3DEntry::Owned(items) => {
                    out.extend_from_slice(items);
                }
            }
        }
        out
    })
}

#[cfg(test)]
#[path = "sidecar_3d_tests.rs"]
mod tests;
