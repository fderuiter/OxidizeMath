//! Aperiodic 2D tilings: Penrose (`P2` Kite & Dart, `P3` Thick & Thin Rhomb) and
//! Einstein monotiles (`Hat` and chiral `Spectre`) with shared interior edge deduplication.
//!
//! # Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::tilings::{
//!     generate_aperiodic_tiling, TilingKind, MAX_TILING_DEPTH,
//! };
//!
//! let patch = generate_aperiodic_tiling(TilingKind::PenroseP3, 2).unwrap();
//! assert!(!patch.tiles.is_empty());
//! assert!(patch.deduplicated_edges.len() < patch.raw_edge_count);
//! assert!(generate_aperiodic_tiling(TilingKind::EinsteinHat, MAX_TILING_DEPTH + 1).is_err());
//! ```

mod affine;
mod hat;
mod penrose;
mod spectre;

#[cfg(test)]
mod tests;

use math_commons::contours::{
    ContourPath, GridBounds2D, Point2D, Segment2D, chain_segments, deduplicate_segments,
    order_paths_nearest_neighbor,
};
use math_commons::error::{MathError, MathResult};
use std::collections::HashMap;

/// Maximum recursion / substitution depth permitted for aperiodic tiling generation.
pub const MAX_TILING_DEPTH: usize = 6;

/// Supported aperiodic 2D tiling families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TilingKind {
    /// Penrose `P2` Kite and Dart aperiodic tiling via Robinson triangle deflation.
    PenroseP2,
    /// Penrose `P3` Thick and Thin Rhomb aperiodic tiling via Robinson triangle deflation.
    PenroseP3,
    /// Smith–Myers–Kaplan–Goodman-Strauss `Hat` (`Tile(1, √3)`) polykite monotile patch.
    EinsteinHat,
    /// Chiral `Spectre` (`Tile(1, 1)`) monotile supertile substitution patch.
    EinsteinSpectre,
}

/// Sub-species classification for an individual tile polygon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileRole {
    /// Penrose `P2` convex kite (`72°–72°–72°–144°`).
    Kite,
    /// Penrose `P2` non-convex dart (`36°–72°–36°–216°`).
    Dart,
    /// Penrose `P3` thick rhomb (`72°–108°`).
    ThickRhomb,
    /// Penrose `P3` thin rhomb (`36°–144°`).
    ThinRhomb,
    /// Unreflected Einstein `Hat` (`Tile(1, √3)`).
    Hat,
    /// Reflected Einstein `Anti-Hat` (`Tile(1, √3)` mirror image).
    AntiHat,
    /// Chiral Einstein `Spectre` (`Tile(1, 1)`).
    Spectre,
}

/// Closed 2D polygon representing a single tile in an aperiodic patch.
#[derive(Debug, Clone, PartialEq)]
pub struct TilePolygon {
    /// Ordered exterior polygon vertices (without repeating vertex `0` at the end).
    pub vertices: Vec<Point2D>,
    /// Geometric tile classification.
    pub role: TileRole,
}

/// Deduplicated geometric fabrication output for an aperiodic tiling patch.
#[derive(Debug, Clone)]
pub struct TilingPatch {
    /// Individual tile polygons in the patch.
    pub tiles: Vec<TilePolygon>,
    /// Total undirected polygon edges before shared-border deduplication.
    pub raw_edge_count: usize,
    /// Unique undirected segments across the entire patch (interior edges appear once).
    pub deduplicated_edges: Vec<Segment2D>,
    /// Chained exterior perimeter contours (incidence degree 1), ordered for minimal pen travel.
    pub boundary_paths: Vec<ContourPath>,
    /// Chained shared interior border contours (incidence degree >= 2), ordered for minimal travel.
    pub interior_paths: Vec<ContourPath>,
    /// Axis-aligned bounding box enclosing all vertices in the patch.
    pub bounds: GridBounds2D,
}

/// Generates a bounded aperiodic tiling patch of `kind` at recursion `depth` (`0..=6`).
///
/// # Errors
///
/// Returns [`MathError::NumericalError`] when `depth > MAX_TILING_DEPTH` (`6`).
#[verified_engine::verified]
pub fn generate_aperiodic_tiling(kind: TilingKind, depth: usize) -> MathResult<TilingPatch> {
    if depth > MAX_TILING_DEPTH {
        return Err(MathError::NumericalError {
            reason: format!("Tiling depth {depth} exceeds MAX_TILING_DEPTH ({MAX_TILING_DEPTH})"),
        });
    }
    let tiles = match kind {
        TilingKind::PenroseP2 => penrose::generate_penrose_p2(depth),
        TilingKind::PenroseP3 => penrose::generate_penrose_p3(depth),
        TilingKind::EinsteinHat => hat::generate_einstein_hat(depth),
        TilingKind::EinsteinSpectre => spectre::generate_einstein_spectre(depth),
    };
    assemble_tiling_patch(tiles, 1e-4)
}

pub(crate) type QuantizedVertex = (i64, i64);
pub(crate) type QuantizedEdgeKey = (QuantizedVertex, QuantizedVertex);
type EdgeCountEntry = (usize, Segment2D);

#[verified_engine::verified]
pub(crate) fn assemble_tiling_patch(
    tiles: Vec<TilePolygon>,
    tolerance: f64,
) -> MathResult<TilingPatch> {
    let inv_tol = 1.0 / tolerance.max(1e-7);
    let mut all_edges = Vec::new();
    let mut edge_counts: HashMap<QuantizedEdgeKey, EdgeCountEntry> = HashMap::new();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for tile in &tiles {
        let n = tile.vertices.len();
        if n < 3 {
            continue;
        }
        for idx in 0..n {
            let p0 = tile.vertices[idx];
            let p1 = tile.vertices[(idx + 1) % n];
            min_x = min_x.min(p0[0]);
            max_x = max_x.max(p0[0]);
            min_y = min_y.min(p0[1]);
            max_y = max_y.max(p0[1]);
            all_edges.push([p0, p1]);

            let k0 = quantize_vertex(p0, inv_tol);
            let k1 = quantize_vertex(p1, inv_tol);
            if k0 == k1 {
                continue;
            }
            let key = if k0 <= k1 { (k0, k1) } else { (k1, k0) };
            edge_counts
                .entry(key)
                .and_modify(|e| e.0 += 1)
                .or_insert((1, [p0, p1]));
        }
    }

    let raw_edge_count = all_edges.len();
    let deduplicated_edges = deduplicate_segments(&all_edges, tolerance);
    let mut boundary_segs = Vec::new();
    let mut interior_segs = Vec::new();
    for (count, seg) in edge_counts.into_values() {
        if count == 1 {
            boundary_segs.push(seg);
        } else {
            interior_segs.push(seg);
        }
    }

    let boundary_paths = order_paths_nearest_neighbor(&chain_segments(&boundary_segs, tolerance));
    let interior_paths = order_paths_nearest_neighbor(&chain_segments(&interior_segs, tolerance));
    let bounds = GridBounds2D::new(min_x, max_x, min_y, max_y)?;

    Ok(TilingPatch {
        tiles,
        raw_edge_count,
        deduplicated_edges,
        boundary_paths,
        interior_paths,
        bounds,
    })
}

#[verified_engine::verified]
pub(crate) fn quantize_vertex(pt: Point2D, inv_tol: f64) -> (i64, i64) {
    let qx = (pt[0] * inv_tol)
        .round()
        .clamp(i64::MIN as f64, i64::MAX as f64) as i64;
    let qy = (pt[1] * inv_tol)
        .round()
        .clamp(i64::MIN as f64, i64::MAX as f64) as i64;
    (qx, qy)
}

// [cite:graph_parameters_rust]
