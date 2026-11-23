//! Seeded point-cloud shape generators and starter gallery presets for the Vietoris-Rips TDA workbench.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::pure_math::statistics::tda::core::Point2D;
use math_explorer::pure_math::statistics::tda::MAX_TDA_POINTS;
use rand::Rng;
use std::f64::consts::TAU;

/// Default point count for seeded shape presets.
pub const DEFAULT_PRESET_POINTS: usize = 48;

/// Default deterministic PRNG seed for shape generators.
pub const DEFAULT_PRESET_SEED: u64 = 42;

/// Default radial/coordinate noise amplitude for shape generators.
pub const DEFAULT_PRESET_NOISE: f64 = 0.06;

/// Seeded synthetic point-cloud families exposing canonical $H_0$ and $H_1$ topological features.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShapePreset {
    /// Noisy circular ring centered at `(5, 5)` with one prominent $H_1$ loop.
    Annulus,
    /// Two separated circles of distinct radii producing two $H_1$ features with distinct persistence.
    TwoCircles,
    /// Three separated clusters producing three prominent $H_0$ connected components.
    NoisyClusters,
    /// Two touching loops (figure-eight / wedge $S^1 \vee S^1$) with $\beta_0 = 1$ and $\beta_1 = 2$.
    FigureEight,
    /// Uniformly distributed random points in `[0, 10] × [0, 10]`.
    UniformRandom,
}

impl ShapePreset {
    /// Ordered list of all available seeded presets for UI selection.
    pub const ALL: [Self; 5] = [
        Self::Annulus,
        Self::TwoCircles,
        Self::NoisyClusters,
        Self::FigureEight,
        Self::UniformRandom,
    ];

    /// Human-readable label shown in the preset selector.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Annulus => "Annulus (1 Loop)",
            Self::TwoCircles => "Two Circles (Multi-Scale H₁)",
            Self::NoisyClusters => "Noisy Clusters (3 Components)",
            Self::FigureEight => "Figure Eight (β₁ = 2)",
            Self::UniformRandom => "Uniform Random",
        }
    }

    /// Recommended initial filtration scale $\epsilon$ highlighting the preset's primary topology.
    #[must_use]
    pub const fn default_radius(self) -> f64 {
        match self {
            Self::Annulus => 1.5,
            Self::TwoCircles => 1.25,
            Self::NoisyClusters => 1.4,
            Self::FigureEight => 1.15,
            Self::UniformRandom => 1.8,
        }
    }
}

/// Compact URL-safe descriptor reproducing up to `MAX_TDA_POINTS` points deterministically.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorDescriptor {
    /// Shape family to sample.
    pub preset: ShapePreset,
    /// Number of points (`1..=250`).
    pub count: usize,
    /// Deterministic `OxidizeRng` seed.
    pub seed: u64,
    /// Isotropic jitter amplitude (`0.0..=1.0`).
    pub noise: f64,
}

impl Default for GeneratorDescriptor {
    fn default() -> Self {
        Self {
            preset: ShapePreset::Annulus,
            count: DEFAULT_PRESET_POINTS,
            seed: DEFAULT_PRESET_SEED,
            noise: DEFAULT_PRESET_NOISE,
        }
    }
}

impl GeneratorDescriptor {
    /// Validates generator bounds for untrusted JSON/URL deserialization.
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=MAX_TDA_POINTS).contains(&self.count) {
            return Err(format!(
                "generator count must be in 1..={MAX_TDA_POINTS} (got {})",
                self.count
            ));
        }
        if !self.noise.is_finite() || !(0.0..=1.0).contains(&self.noise) {
            return Err(format!(
                "generator noise must be finite in [0.0, 1.0] (got {})",
                self.noise
            ));
        }
        Ok(())
    }
}

/// Generates a deterministic 2D point cloud from `desc`.
#[must_use]
pub fn generate_preset_points(desc: &GeneratorDescriptor) -> Vec<Point2D> {
    let count = desc.count.clamp(1, MAX_TDA_POINTS);
    let noise = if desc.noise.is_finite() {
        desc.noise.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut rng = oxidize_core::rng::OxidizeRng::new(desc.seed);
    match desc.preset {
        ShapePreset::Annulus => sample_annulus(&mut rng, count, noise),
        ShapePreset::TwoCircles => sample_two_circles(&mut rng, count, noise),
        ShapePreset::NoisyClusters => sample_noisy_clusters(&mut rng, count, noise),
        ShapePreset::FigureEight => sample_figure_eight(&mut rng, count, noise),
        ShapePreset::UniformRandom => sample_uniform_random(&mut rng, count),
    }
}

fn jitter(rng: &mut oxidize_core::rng::OxidizeRng, scale: f64) -> f64 {
    if scale <= 0.0 {
        0.0
    } else {
        rng.gen_range(-scale..=scale)
    }
}

fn sample_annulus(
    rng: &mut oxidize_core::rng::OxidizeRng,
    count: usize,
    noise: f64,
) -> Vec<Point2D> {
    let radius = 3.2;
    let radial_band = 0.25 + noise * 0.8;
    (0..count)
        .map(|i| {
            let theta = TAU * (i as f64) / (count as f64) + jitter(rng, 0.05 + noise * 0.2);
            let r = radius + jitter(rng, radial_band);
            Point2D::new(5.0 + r * theta.cos(), 5.0 + r * theta.sin())
        })
        .collect()
}

fn sample_two_circles(
    rng: &mut oxidize_core::rng::OxidizeRng,
    count: usize,
    noise: f64,
) -> Vec<Point2D> {
    let first_n = count.div_ceil(2);
    let second_n = count.saturating_sub(first_n);
    let mut pts = Vec::with_capacity(count);
    for i in 0..first_n {
        let theta = TAU * (i as f64) / (first_n.max(1) as f64);
        let r = 1.45 + jitter(rng, noise * 0.5);
        pts.push(Point2D::new(2.6 + r * theta.cos(), 5.0 + r * theta.sin()));
    }
    for i in 0..second_n {
        let theta = TAU * (i as f64) / (second_n.max(1) as f64);
        let r = 2.55 + jitter(rng, noise * 0.5);
        pts.push(Point2D::new(8.2 + r * theta.cos(), 5.0 + r * theta.sin()));
    }
    pts
}

fn sample_noisy_clusters(
    rng: &mut oxidize_core::rng::OxidizeRng,
    count: usize,
    noise: f64,
) -> Vec<Point2D> {
    let centers = [(2.2, 2.5), (7.8, 2.8), (5.0, 7.8)];
    let spread = 0.45 + noise * 0.9;
    (0..count)
        .map(|i| {
            let (cx, cy) = centers[i % centers.len()];
            let angle = rng.gen_range(0.0..TAU);
            let dist = rng.gen_range(0.0..spread);
            Point2D::new(cx + dist * angle.cos(), cy + dist * angle.sin())
        })
        .collect()
}

fn sample_figure_eight(
    rng: &mut oxidize_core::rng::OxidizeRng,
    count: usize,
    noise: f64,
) -> Vec<Point2D> {
    let lobe_radius = 2.1;
    let left_n = count.div_ceil(2);
    let right_n = count.saturating_sub(left_n);
    let mut pts = Vec::with_capacity(count);
    for i in 0..left_n {
        let theta = TAU * (i as f64) / (left_n.max(1) as f64);
        pts.push(Point2D::new(
            (5.0 - lobe_radius) + lobe_radius * theta.cos() + jitter(rng, noise * 0.4),
            5.0 + lobe_radius * theta.sin() + jitter(rng, noise * 0.4),
        ));
    }
    for i in 0..right_n {
        let theta = TAU * ((i as f64) + 0.5) / (right_n.max(1) as f64);
        pts.push(Point2D::new(
            (5.0 + lobe_radius) + lobe_radius * theta.cos() + jitter(rng, noise * 0.4),
            5.0 + lobe_radius * theta.sin() + jitter(rng, noise * 0.4),
        ));
    }
    pts
}

fn sample_uniform_random(rng: &mut oxidize_core::rng::OxidizeRng, count: usize) -> Vec<Point2D> {
    (0..count)
        .map(|_| Point2D::new(rng.gen_range(0.0..10.0), rng.gen_range(0.0..10.0)))
        .collect()
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Vietoris-Rips Filtration",
    "domain": "geometry_topology",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "radius": 2.2,
    "points": [
      [5.0, 8.0], [6.5, 7.6], [7.6, 6.5], [8.0, 5.0],
      [7.6, 3.5], [6.5, 2.4], [5.0, 2.0], [3.5, 2.4],
      [2.4, 3.5], [2.0, 5.0], [2.4, 6.5], [3.5, 7.6]
    ]
  }
}"#;

const STARTER_PRESET_LORENZ_3D_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "3D Lorenz Strange Attractor Homology",
    "domain": "geometry_topology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "mode": "lorenz_3d",
    "radius": 7.5,
    "chaos": {
      "sigma": 10.0,
      "rho": 28.0,
      "beta": 2.6666666666666665,
      "dt": 0.01,
      "landmark_count": 64,
      "takens_lag": 8
    },
    "camera": {
      "pitch": 0.35,
      "yaw": 0.55,
      "zoom": 1.0
    }
  }
}"#;

const STARTER_PRESET_TAKENS_3D_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Takens Delay Attractor Reconstruction",
    "domain": "geometry_topology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "mode": "takens_3d",
    "radius": 7.0,
    "chaos": {
      "sigma": 10.0,
      "rho": 28.0,
      "beta": 2.6666666666666665,
      "dt": 0.01,
      "landmark_count": 64,
      "takens_lag": 10
    },
    "camera": {
      "pitch": 0.45,
      "yaw": -0.6,
      "zoom": 1.0
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vietoris_rips_starter",
        title: "Vietoris-Rips Filtration",
        description: "Simplicial complex filtration, exact Z₂ Betti numbers, and persistence diagram across ε.",
        category: "Algebraic Topology",
        icon: "🕸️",
        domain: "geometry_topology",
        tool_name: Some("VietorisRipsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vietoris_rips_lorenz_3d",
        title: "3D Lorenz Strange Attractor Homology",
        description: "64-landmark 3D Lorenz butterfly filtration locking onto β₁ = 2 persistent loops alongside λ₁ and D₂.",
        category: "Algebraic Topology",
        icon: "🦋",
        domain: "geometry_topology",
        tool_name: Some("VietorisRipsTool"),
        preset_generator: || STARTER_PRESET_LORENZ_3D_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vietoris_rips_takens_3d",
        title: "Takens Delay Attractor Reconstruction",
        description: "3D delay-coordinate embedding (x(t), x(t-τ), x(t-2τ)) reconstructing strange-attractor topology from scalar observations.",
        category: "Algebraic Topology",
        icon: "🌀",
        domain: "geometry_topology",
        tool_name: Some("VietorisRipsTool"),
        preset_generator: || STARTER_PRESET_TAKENS_3D_JSON,
    }
}

// [cite:graph_parameters_rust]
