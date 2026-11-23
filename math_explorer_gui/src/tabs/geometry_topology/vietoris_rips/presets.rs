//! Seeded point-cloud shape generators and starter gallery preset for the Vietoris-Rips TDA workbench.

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

pub(super) struct SerializedPoints2D<'a> {
    pub(super) pts: &'a [Point2D],
    pub(super) quantize: bool,
}

impl serde::Serialize for SerializedPoints2D<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.pts.len()))?;
        for pt in self.pts {
            let pair = if self.quantize {
                [
                    super::io::quantize_url_coord(pt.x),
                    super::io::quantize_url_coord(pt.y),
                ]
            } else {
                [pt.x, pt.y]
            };
            seq.serialize_element(&pair)?;
        }
        seq.end()
    }
}

fn is_zero(val: &f64) -> bool {
    *val == 0.0
}

#[derive(serde::Serialize)]
pub(super) struct VietorisRipsSaveState<'a> {
    pub(super) radius: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) points: Option<SerializedPoints2D<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) generator: Option<&'a GeneratorDescriptor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) points_nd: Option<&'a [Vec<f64>]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) axis_x: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) axis_y: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) plot_mode: Option<super::plots::PersistencePlotMode>,
    #[serde(skip_serializing_if = "is_zero")]
    pub(super) min_persistence: f64,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct VietorisRipsLoadState {
    pub(super) radius: f64,
    #[serde(default)]
    pub(super) points: Option<Vec<[f64; 2]>>,
    #[serde(default)]
    pub(super) generator: Option<GeneratorDescriptor>,
    #[serde(default)]
    pub(super) points_nd: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub(super) axis_x: Option<usize>,
    #[serde(default)]
    pub(super) axis_y: Option<usize>,
    #[serde(default)]
    pub(super) plot_mode: super::plots::PersistencePlotMode,
    #[serde(default)]
    pub(super) min_persistence: f64,
}

const MAX_ALLOWED_RADIUS: f64 = 1_000_000.0;

pub(super) fn serialize_tool_state(tool: &super::VietorisRipsTool) -> Option<String> {
    let mode_opt = (!tool.plot_mode.is_default()).then_some(tool.plot_mode);
    let state = match &tool.source {
        super::CloudSource::Custom2D(pts) => VietorisRipsSaveState {
            radius: tool.radius,
            points: Some(SerializedPoints2D {
                pts,
                quantize: pts.len() <= super::io::MAX_URL_CUSTOM_POINTS_2D,
            }),
            generator: None,
            points_nd: None,
            axis_x: None,
            axis_y: None,
            plot_mode: mode_opt,
            min_persistence: tool.min_persistence,
        },
        super::CloudSource::Generator(desc) => VietorisRipsSaveState {
            radius: tool.radius,
            points: None,
            generator: Some(desc),
            points_nd: None,
            axis_x: None,
            axis_y: None,
            plot_mode: mode_opt,
            min_persistence: tool.min_persistence,
        },
        super::CloudSource::CustomNd {
            cloud,
            axis_x,
            axis_y,
        } => VietorisRipsSaveState {
            radius: tool.radius,
            points: None,
            generator: None,
            points_nd: Some(cloud.points()),
            axis_x: Some(*axis_x),
            axis_y: Some(*axis_y),
            plot_mode: mode_opt,
            min_persistence: tool.min_persistence,
        },
    };
    serde_json::to_string(&state).ok()
}

pub(super) fn deserialize_tool_state(
    tool: &mut super::VietorisRipsTool,
    state: &str,
) -> Result<(), String> {
    let loaded: VietorisRipsLoadState =
        serde_json::from_str(state).map_err(|e| format!("Invalid Vietoris-Rips state: {e}"))?;
    if !loaded.radius.is_finite() || !(0.0..=MAX_ALLOWED_RADIUS).contains(&loaded.radius) {
        return Err(format!(
            "radius out of [0.0, {MAX_ALLOWED_RADIUS}]: {}",
            loaded.radius
        ));
    }
    if !loaded.min_persistence.is_finite()
        || !(0.0..=MAX_ALLOWED_RADIUS).contains(&loaded.min_persistence)
    {
        return Err(format!(
            "invalid min_persistence: {}",
            loaded.min_persistence
        ));
    }
    apply_loaded_cloud(tool, loaded)
}

fn apply_loaded_cloud(
    tool: &mut super::VietorisRipsTool,
    loaded: VietorisRipsLoadState,
) -> Result<(), String> {
    use math_explorer::pure_math::statistics::tda::core::NdPointCloud;
    let sources = usize::from(loaded.points.is_some())
        + usize::from(loaded.generator.is_some())
        + usize::from(loaded.points_nd.is_some());
    if sources != 1 {
        return Err(
            "Vietoris-Rips state must specify one of 'points', 'generator', or 'points_nd'"
                .to_string(),
        );
    }
    if let Some(raw_pts) = loaded.points {
        if !(1..=super::io::MAX_VIETORIS_RIPS_POINTS).contains(&raw_pts.len())
            || !raw_pts.iter().all(|p| p[0].is_finite() && p[1].is_finite())
        {
            return Err(format!(
                "points must contain 1..={} finite points",
                super::io::MAX_VIETORIS_RIPS_POINTS
            ));
        }
        let quantize = raw_pts.len() <= super::io::MAX_URL_CUSTOM_POINTS_2D;
        let pts: Vec<Point2D> = raw_pts
            .into_iter()
            .map(|[x, y]| {
                if quantize {
                    Point2D::new(
                        super::io::quantize_url_coord(x),
                        super::io::quantize_url_coord(y),
                    )
                } else {
                    Point2D::new(x, y)
                }
            })
            .collect();
        tool.rebuild_from_2d(super::CloudSource::Custom2D(pts), Vec::new(), loaded.radius)?;
    } else if let Some(desc) = loaded.generator {
        desc.validate()?;
        let pts = generate_preset_points(&desc);
        tool.rebuild_from_2d(super::CloudSource::Generator(desc), pts, loaded.radius)?;
    } else if let Some(nd_rows) = loaded.points_nd {
        let nd = NdPointCloud::new(nd_rows).map_err(|e| format!("{e}"))?;
        let ax = loaded.axis_x.unwrap_or(0);
        let ay = loaded
            .axis_y
            .unwrap_or(1.min(nd.dimension().saturating_sub(1)));
        tool.rebuild_from_nd(nd, ax, ay, loaded.radius)?;
    }
    tool.plot_mode = loaded.plot_mode;
    tool.min_persistence = loaded.min_persistence;
    tool.refresh_url_budget_flag();
    Ok(())
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

// [cite:graph_parameters_rust]
