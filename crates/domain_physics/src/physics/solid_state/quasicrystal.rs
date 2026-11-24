//! Zero-allocation 2D cut-and-project quasicrystal hyperlattice kernels.
//!
//! Projects integer hyperlattices $\mathbb{Z}^N \to E_\parallel \oplus E_\perp$ onto 2D physical
//! space $E_\parallel$ and 2D internal perpendicular space $E_\perp$ for:
//! - **5D Penrose** (`N = 5`, $C_5 / D_{10}$, golden ratio $\varphi = \frac{1 + \sqrt{5}}{2}$),
//! - **4D Ammann-Beenker** (`N = 4`, $D_8$, silver ratio $\delta_S = 1 + \sqrt{2}$),
//! - **6D Stampfli / Gähler** (`N = 6`, $D_{12}$, dodecagonal ratio $2 + \sqrt{3}$).
//!
//! # Examples
//!
//! ```
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::solid_state::quasicrystal::{
//!     GOLDEN_RATIO, QuasicrystalKind, QuasicrystalParams, QuasicrystalWorkspace,
//! };
//!
//! let mut ws = QuasicrystalWorkspace::new();
//! let params = QuasicrystalParams::new(QuasicrystalKind::Penrose5D, 1.0, 24.0);
//! let stats = ws.project_into(&params).expect("valid Penrose projection");
//! assert!(stats.vertex_count > 200);
//! assert!((stats.empirical_ratio - GOLDEN_RATIO).abs() / GOLDEN_RATIO < 0.01);
//! ```

// [cite:solid_state]

#[path = "quasicrystal/penrose_ammann.rs"]
mod penrose_ammann;
#[path = "quasicrystal/stampfli.rs"]
mod stampfli;

use crate::error::SolidStateError;
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, SQRT_2};

/// Golden ratio $\varphi = \frac{1 + \sqrt{5}}{2}$ governing 5D Penrose icosahedral/decagonal scaling.
pub const GOLDEN_RATIO: f64 = std::f64::consts::GOLDEN_RATIO;

/// Silver ratio $\delta_S = 1 + \sqrt{2}$ governing 4D Ammann-Beenker octagonal scaling.
pub const SILVER_RATIO: f64 = 1.0 + SQRT_2;

/// Stampfli dodecagonal inflation ratio $2 + \sqrt{3}$ governing 6D 12-fold self-similarity.
pub const STAMPFLI_RATIO: f64 = 3.732_050_807_568_877;

const PENROSE_GAMMA0: f64 = 0.12;
const NON_SINGULAR_BIAS: [f64; 2] = [1.13e-7, -1.79e-7];

/// Supported higher-dimensional cut-and-project 2D quasicrystal families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum QuasicrystalKind {
    /// 5D hypercubic projection ($\mathbb{Z}^5 \to E_\parallel \oplus E_\perp$) yielding $D_{10}$ Penrose rhombs.
    #[default]
    Penrose5D,
    /// 4D hypercubic projection ($\mathbb{Z}^4 \to E_\parallel \oplus E_\perp$) yielding $D_8$ Ammann-Beenker tiles.
    AmmannBeenker4D,
    /// 6D dodecagonal projection ($\mathbb{Z}^6 \to E_\parallel \oplus E_\perp$) yielding $D_{12}$ Stampfli-Gähler tiles.
    Stampfli6D,
}

impl QuasicrystalKind {
    /// Human-readable quasicrystal family label.
    #[must_use]
    #[verified_engine::verified]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Penrose5D => "5D Penrose (C5 / D10)",
            Self::AmmannBeenker4D => "4D Ammann-Beenker (D8)",
            Self::Stampfli6D => "6D Stampfli Dodecagonal (D12)",
        }
    }

    /// Ambient integer hyperlattice dimension $N$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn hyperlattice_dim(self) -> usize {
        match self {
            Self::Penrose5D => 5,
            Self::AmmannBeenker4D => 4,
            Self::Stampfli6D => 6,
        }
    }

    /// Dominant Bragg diffraction rotational symmetry order $n$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn rotational_order(self) -> usize {
        match self {
            Self::Penrose5D => 10,
            Self::AmmannBeenker4D => 8,
            Self::Stampfli6D => 12,
        }
    }

    /// Characteristic algebraic irrational ratio of the quasicrystal family.
    #[must_use]
    #[verified_engine::verified]
    pub const fn theoretical_ratio(self) -> f64 {
        match self {
            Self::Penrose5D => GOLDEN_RATIO,
            Self::AmmannBeenker4D => SILVER_RATIO,
            Self::Stampfli6D => STAMPFLI_RATIO,
        }
    }

    /// Outer acceptance window circumradius in perpendicular space $E_\perp$.
    #[must_use]
    #[verified_engine::verified]
    pub fn window_radius(self) -> f64 {
        match self {
            Self::Penrose5D => 0.5 * GOLDEN_RATIO * GOLDEN_RATIO / (PI / 5.0).cos(),
            Self::AmmannBeenker4D => 0.5 * SILVER_RATIO / (PI / 8.0).cos(),
            Self::Stampfli6D => 1.0,
        }
    }
}

/// Prototile classification across 5D Penrose, 4D Ammann-Beenker, and 6D Stampfli tilings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuasicrystalTileKind {
    /// Thick $72^\circ / 108^\circ$ Penrose rhomb.
    PenroseThickRhomb,
    /// Thin $36^\circ / 144^\circ$ Penrose rhomb.
    PenroseThinRhomb,
    /// $45^\circ / 135^\circ$ Ammann-Beenker rhomb.
    AmmannRhomb,
    /// $90^\circ$ Ammann-Beenker square.
    AmmannSquare,
    /// $60^\circ$ Stampfli equilateral triangle.
    StampfliTriangle,
    /// $90^\circ$ Stampfli square.
    StampfliSquare,
    /// $90^\circ / 150^\circ$ hexagonal Stampfli-Gähler shield.
    StampfliShield,
}

/// Input parameters for a zero-allocation 2D cut-and-project evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuasicrystalParams {
    /// Hyperlattice family (`Penrose5D`, `AmmannBeenker4D`, or `Stampfli6D`).
    pub kind: QuasicrystalKind,
    /// Tile edge length $a > 0$ in direct space.
    pub edge_length: f64,
    /// Circular spatial aperture radius $R > 0$ in direct space.
    pub aperture_radius: f64,
    /// Active perpendicular-space phason shift $(w_x, w_y) \in E_\perp$.
    pub phason: [f64; 2],
    /// Reference phason shift $(w_{x,0}, w_{y,0})$ used to detect phason-flipped tiles.
    pub reference_phason: [f64; 2],
}

impl QuasicrystalParams {
    /// Creates baseline cut-and-project parameters with zero phason shift.
    #[must_use]
    #[verified_engine::verified]
    pub const fn new(kind: QuasicrystalKind, edge_length: f64, aperture_radius: f64) -> Self {
        Self {
            kind,
            edge_length,
            aperture_radius,
            phason: [0.0, 0.0],
            reference_phason: [0.0, 0.0],
        }
    }
}

/// Single projected quasicrystal vertex in physical space $E_\parallel$ and internal space $E_\perp$.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ProjectedVertex2D {
    /// Direct-space coordinates $(x, y) \in E_\parallel$.
    pub parallel: [f64; 2],
    /// Relative perpendicular-space coordinates $(y_x, y_y) = \pi_\perp(\mathbf{n}) - \mathbf{w} \in E_\perp$.
    pub perp: [f64; 2],
    /// Hyperplane index ($h \in \{1..5\}$ for Penrose, $0$ otherwise).
    pub hyperplane: i8,
}

/// Projected 2D prototile polygon (3, 4, or 6 vertices) with phason-flip telemetry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedTile2D {
    /// Prototile variant.
    pub kind: QuasicrystalTileKind,
    /// Ordered perimeter 2D vertices in $E_\parallel$.
    pub vertices: [[f64; 2]; 6],
    /// Active vertex count (`3` for triangles, `4` for quads, `6` for shields).
    pub vertex_count: u8,
    /// Whether this tile was newly created (flipped) relative to `reference_phason`.
    pub phason_flipped: bool,
}

/// Summary counts and algebraic scaling invariants from a cut-and-project pass.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct QuasicrystalStats {
    /// Number of accepted vertices inside the spatial aperture.
    pub vertex_count: usize,
    /// Total number of accepted tiles inside the spatial aperture.
    pub tile_count: usize,
    /// Primary prototile count (thick rhombs, $45^\circ$ rhombs, or $60^\circ$ triangles).
    pub primary_tile_count: usize,
    /// Secondary prototile count (thin rhombs or $90^\circ$ squares).
    pub secondary_tile_count: usize,
    /// Tertiary prototile count (hexagonal shields for Stampfli, $0$ otherwise).
    pub tertiary_tile_count: usize,
    /// Number of tiles flipped between `reference_phason` and `phason`.
    pub flipped_tile_count: usize,
    /// Empirical tile frequency ratio matching `theoretical_ratio`.
    pub empirical_ratio: f64,
    /// Exact theoretical algebraic ratio ($\varphi$, $\delta_S$, or $2 + \sqrt{3}$).
    pub theoretical_ratio: f64,
    /// Geometric silver/golden span ($1 + d_{\text{sq}} / a = 1 + \sqrt{2}$ for square tilings, $\varphi$ for Penrose).
    pub secondary_geometric_ratio: f64,
}

/// Pre-allocated workspace executing zero-allocation $\mathbb{Z}^N \to E_\parallel \oplus E_\perp$ projections.
#[derive(Debug, Clone)]
pub struct QuasicrystalWorkspace {
    vertices: Vec<ProjectedVertex2D>,
    positions: Vec<[f64; 2]>,
    tiles: Vec<ProjectedTile2D>,
    last_stats: QuasicrystalStats,
}

impl Default for QuasicrystalWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl QuasicrystalWorkspace {
    /// Allocates a workspace sized for studio apertures up to `16_384` vertices and `20_480` tiles.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(16_384, 20_480)
    }

    /// Allocates a workspace with explicit vertex and tile capacities.
    #[must_use]
    pub fn with_capacity(max_vertices: usize, max_tiles: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(max_vertices),
            positions: Vec::with_capacity(max_vertices),
            tiles: Vec::with_capacity(max_tiles),
            last_stats: QuasicrystalStats::default(),
        }
    }

    /// Borrowed slice of projected vertices from the latest [`Self::project_into`] call.
    #[must_use]
    #[verified_engine::verified]
    pub fn vertices(&self) -> &[ProjectedVertex2D] {
        &self.vertices
    }

    /// Borrowed slice of 2D vertex coordinates $(x, y)$ suitable for Bragg diffraction.
    #[must_use]
    #[verified_engine::verified]
    pub fn positions_xy(&self) -> &[[f64; 2]] {
        &self.positions
    }

    /// Borrowed slice of projected prototiles from the latest [`Self::project_into`] call.
    #[must_use]
    #[verified_engine::verified]
    pub fn tiles(&self) -> &[ProjectedTile2D] {
        &self.tiles
    }

    /// Latest summary statistics.
    #[must_use]
    #[verified_engine::verified]
    pub const fn stats(&self) -> QuasicrystalStats {
        self.last_stats
    }

    /// Projects the configured hyperlattice into pre-allocated buffers with zero heap allocations.
    #[verified_engine::verified]
    pub fn project_into(
        &mut self,
        params: &QuasicrystalParams,
    ) -> Result<QuasicrystalStats, SolidStateError> {
        if !params.edge_length.is_finite()
            || params.edge_length <= 0.0
            || !params.aperture_radius.is_finite()
            || params.aperture_radius <= 0.0
            || !params.phason[0].is_finite()
            || !params.phason[1].is_finite()
            || !params.reference_phason[0].is_finite()
            || !params.reference_phason[1].is_finite()
        {
            return Err(SolidStateError::InvalidParameter(
                "quasicrystal edge_length, aperture_radius, and phason must be finite and valid"
                    .into(),
            ));
        }
        self.vertices.clear();
        self.positions.clear();
        self.tiles.clear();
        let stats = match params.kind {
            QuasicrystalKind::Penrose5D => self.project_penrose(params),
            QuasicrystalKind::AmmannBeenker4D => self.project_ammann_beenker(params),
            QuasicrystalKind::Stampfli6D => self.project_stampfli(params),
        };
        self.last_stats = stats;
        Ok(stats)
    }

    #[verified_engine::verified]
    fn push_vertex(&mut self, parallel: [f64; 2], perp: [f64; 2], hyperplane: i8) {
        if self.vertices.len() < self.vertices.capacity()
            && self.positions.len() < self.positions.capacity()
        {
            self.vertices.push(ProjectedVertex2D {
                parallel,
                perp,
                hyperplane,
            });
            self.positions.push(parallel);
        }
    }

    #[verified_engine::verified]
    fn push_tile(&mut self, tile: ProjectedTile2D) {
        if self.tiles.len() < self.tiles.capacity() {
            self.tiles.push(tile);
        }
    }
}
