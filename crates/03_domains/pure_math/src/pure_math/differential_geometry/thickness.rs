//! Uniform CSR spatial grids (`FlatSpatialGrid`, `SpatialGrid`), preallocated
//! `ThicknessWorkspace`, and zero-allocation single-vertex and batch cortical thickness kernels.
// [cite:pure_math]

#[path = "thickness_grid.rs"]
mod thickness_grid;

use super::cortical_mesh::Surface;
use thickness_grid::linear_scan_distance;
pub use thickness_grid::{FlatSpatialGrid, SpatialGrid, ThicknessWorkspace};

/// Calculates the shortest Euclidean distance from `point` to `surface` via a direct scan.
#[verified_engine::verified]
fn point_to_surface_distance(point: &[f64; 3], surface: &Surface) -> f64 {
    linear_scan_distance(point, &surface.vertices)
}

/// Calculates symmetric paired surface thickness at `(v_w, v_p)` via direct vertex scan:
///
/// $$T(\mathbf{v}_w, \mathbf{v}_p) = \frac{1}{2}\left(\min_{\mathbf{u} \in S_p} \|\mathbf{v}_w - \mathbf{u}\|_2 + \min_{\mathbf{w} \in S_w} \|\mathbf{v}_p - \mathbf{w}\|_2\right)$$
///
/// Reference: Fischl & Dale (2000), *PNAS* 97(20), 11050–11055 (`freesurfer.tex`); see
/// [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Exact $O(V)$ Euclidean nearest-vertex scan with zero dynamic heap allocation.
///
/// # Domain of Validity
/// Non-empty surfaces with finite 3D coordinates (returns `f64::INFINITY` on empty surfaces and `f64::NAN` on `NaN` query coordinates).
#[verified_engine::verified]
pub fn cortical_thickness(
    v_w: &[f64; 3],
    v_p: &[f64; 3],
    white_surface: &Surface,
    pial_surface: &Surface,
) -> f64 {
    let dist_w_to_p = point_to_surface_distance(v_w, pial_surface);
    let dist_p_to_w = point_to_surface_distance(v_p, white_surface);
    0.5 * (dist_w_to_p + dist_p_to_w)
}

/// Evaluates symmetric cortical thickness at `(v_w, v_p)` in expected $O(1)$ time using
/// prebuilt [`FlatSpatialGrid`] indices without dynamic heap allocation.
#[must_use]
#[verified_engine::verified]
pub fn cortical_thickness_indexed(
    v_w: &[f64; 3],
    v_p: &[f64; 3],
    white_grid: &FlatSpatialGrid<'_>,
    pial_grid: &FlatSpatialGrid<'_>,
) -> f64 {
    let dist_w_to_p = pial_grid.shortest_distance(v_w);
    let dist_p_to_w = white_grid.shortest_distance(v_p);
    0.5 * (dist_w_to_p + dist_p_to_w)
}

/// Calculates cortical thickness at a vertex pair using pre-built [`SpatialGrid`] indices
/// with zero post-initialization heap allocations.
#[must_use]
#[verified_engine::verified]
pub fn cortical_thickness_with_grid(
    v_w: &[f64; 3],
    v_p: &[f64; 3],
    white_grid: &SpatialGrid,
    pial_grid: &SpatialGrid,
) -> f64 {
    cortical_thickness_indexed(
        v_w,
        v_p,
        &white_grid.as_flat_grid(),
        &pial_grid.as_flat_grid(),
    )
}

/// Computes vertex-wise cortical thickness across `white_surface` and `pial_surface` into
/// caller-allocated `out` using pre-built [`SpatialGrid`] indices with zero heap allocations.
///
/// Evaluates `count = white_surface.vertices.len().min(pial_surface.vertices.len()).min(out.len())`
/// corresponding vertex pairs and zeros any trailing elements in `out[count..]`.
#[verified_engine::verified]
pub fn cortical_thickness_into(
    white_surface: &Surface,
    pial_surface: &Surface,
    white_grid: &SpatialGrid,
    pial_grid: &SpatialGrid,
    out: &mut [f64],
) {
    let count = white_surface
        .vertices
        .len()
        .min(pial_surface.vertices.len())
        .min(out.len());
    for ((dst, v_w), v_p) in out[..count]
        .iter_mut()
        .zip(&white_surface.vertices[..count])
        .zip(&pial_surface.vertices[..count])
    {
        *dst = cortical_thickness_with_grid(v_w, v_p, white_grid, pial_grid);
    }
    out[count..].fill(0.0);
}

/// Computes symmetric cortical thickness across all corresponding `(white, pial)` vertex pairs
/// into `thicknesses_out` using `workspace` with zero dynamic heap allocations.
///
/// Returns the number of vertex thicknesses written, or `0` when either surface is empty or when
/// `workspace` or `thicknesses_out` lacks capacity.
#[verified_engine::verified]
pub fn cortical_thickness_map_into(
    white_surface: &Surface,
    pial_surface: &Surface,
    workspace: &mut ThicknessWorkspace,
    thicknesses_out: &mut [f64],
) -> usize {
    let count = white_surface
        .vertices
        .len()
        .min(pial_surface.vertices.len());
    if count == 0 || thicknesses_out.len() < count {
        return 0;
    }
    let Some((white_grid, pial_grid)) = workspace.build_grids(white_surface, pial_surface) else {
        return 0;
    };
    for (idx, out) in thicknesses_out.iter_mut().enumerate().take(count) {
        *out = cortical_thickness_indexed(
            &white_surface.vertices[idx],
            &pial_surface.vertices[idx],
            &white_grid,
            &pial_grid,
        );
    }
    count
}

/// Computes symmetric cortical thickness across all corresponding `(white, pial)` vertex pairs,
/// constructing a uniform CSR [`FlatSpatialGrid`] once per surface mesh.
#[must_use]
pub fn cortical_thickness_map(white_surface: &Surface, pial_surface: &Surface) -> Vec<f64> {
    let count = white_surface
        .vertices
        .len()
        .min(pial_surface.vertices.len());
    if count == 0 {
        return Vec::new();
    }
    let max_verts = white_surface
        .vertices
        .len()
        .max(pial_surface.vertices.len());
    let mut workspace = ThicknessWorkspace::with_max_vertices(max_verts);
    let mut thicknesses = vec![0.0; count];
    let written = cortical_thickness_map_into(
        white_surface,
        pial_surface,
        &mut workspace,
        &mut thicknesses,
    );
    thicknesses.truncate(written);
    thicknesses
}

/// Evaluates symmetric white-to-pial cortical thickness across paired vertices into `out`.
pub fn cortical_thickness_batch_into(
    white_surface: &Surface,
    pial_surface: &Surface,
    out: &mut [f64],
) {
    let (nw, np) = (white_surface.vertices.len(), pial_surface.vertices.len());
    let count = nw.min(np).min(out.len());
    if count > 0 {
        let mut ws = ThicknessWorkspace::with_max_vertices(nw.max(np));
        let _ =
            cortical_thickness_map_into(white_surface, pial_surface, &mut ws, &mut out[..count]);
    }
}

/// Evaluates symmetric white-to-pial cortical thickness across all paired vertices via [`cortical_thickness_map`]:
///
/// $$T_i = \frac{1}{2}\left(\min_{\mathbf{u} \in S_p} \|\mathbf{v}_{w,i} - \mathbf{u}\|_2 + \min_{\mathbf{w} \in S_w} \|\mathbf{v}_{p,i} - \mathbf{w}\|_2\right)$$
///
/// Reference: Fischl & Dale (2000), *PNAS* 97(20), 11050–11055 (`freesurfer.tex`); see
/// [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Exact concentric Chebyshev-shell pruning over CSR voxel cells matches $O(V^2)$ brute-force Euclidean distances bit-for-bit.
///
/// # Domain of Validity
/// Paired triangulated or point-cloud [`Surface`] meshes in $\mathbb{R}^3$ (evaluates `min(V_w, V_p)` corresponding vertices).
#[must_use]
pub fn cortical_thickness_batch(white_surface: &Surface, pial_surface: &Surface) -> Vec<f64> {
    cortical_thickness_map(white_surface, pial_surface)
}

/// Computes vertex-wise cortical thickness across an entire cortical mesh pair by constructing
/// one reusable CSR spatial grid per surface.
#[must_use]
pub fn cortical_thickness_mesh(white_surface: &Surface, pial_surface: &Surface) -> Vec<f64> {
    cortical_thickness_map(white_surface, pial_surface)
}

#[cfg(test)]
#[path = "thickness_tests.rs"]
mod tests;
