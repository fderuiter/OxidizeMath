#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Shared dynamic volumetric `Scene` builders for `QuantumTab`, `AiTab`, and `MorphogenesisTab`.
// [cite:gaussian_splatting]
// [cite:quantum_mechanics]
// [cite:favorite_child]

use super::{extract_voxel_splats_into, VolumetricSplatWorkspace, MAX_DYNAMIC_SPLATS};
use math_explorer::ai::gaussian_splatting::{Gaussian3D, Scene};
use math_explorer::applied::isosurface::{Point3D, VoxelGrid};
use math_explorer::biology::morphogenesis::{DiffusionCoeff, SchnakenbergKinetics, TuringSystem};
use math_explorer::math_kernel::types::{flatten_2d_index, flatten_3d_index, Dimension, StepSize};
use math_explorer::physics::quantum::{
    HydrogenicSuperposition3D, OrbitalQuantumNumbers, QuantumFieldObservable,
};
use nalgebra::{Point3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

/// Deterministic PCG-style PRNG used to seed reproducible initial perturbations across 2D/3D grids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Creates a new deterministic PRNG seeded with `seed`.
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Samples a uniform `f64` in `[0.0, 1.0]`.
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = self.state;
        let x = x ^ (x >> 18);
        let rot = (x >> 27) as u32;
        let val = (x as u32).rotate_right(rot);
        (val as f64) / (u32::MAX as f64)
    }

    /// Samples a uniform `f64` in `[min, max]`.
    pub fn range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }
}

type SplatSpec = ([f64; 3], [f64; 3], [f64; 3], f64, [f64; 3], f64);

const CANONICAL_SCENE: [SplatSpec; 8] = [
    (
        [-0.65, 0.45, -0.70],
        [0.28, 0.16, 0.20],
        [0.0, 0.0, 0.45],
        0.90,
        [0.96, 0.32, 0.28],
        0.24,
    ),
    (
        [0.60, 0.42, -0.50],
        [0.18, 0.30, 0.22],
        [0.2, -0.1, -0.55],
        0.86,
        [0.22, 0.82, 0.56],
        0.19,
    ),
    (
        [0.0, -0.15, -1.00],
        [0.36, 0.24, 0.26],
        [0.15, 0.25, 0.10],
        0.82,
        [0.28, 0.58, 0.98],
        0.31,
    ),
    (
        [-0.48, -0.58, -0.10],
        [0.58, 0.32, 0.28],
        [-0.2, 0.1, 0.35],
        0.78,
        [0.95, 0.76, 0.22],
        0.16,
    ),
    (
        [0.55, -0.52, 0.10],
        [0.25, 0.22, 0.20],
        [0.0, 0.2, -0.30],
        0.80,
        [0.78, 0.36, 0.94],
        0.06,
    ),
    (
        [0.02, 0.68, 0.40],
        [0.34, 0.20, 0.24],
        [0.1, -0.15, 0.05],
        0.74,
        [0.20, 0.86, 0.90],
        0.04,
    ),
    (
        [-0.15, 0.08, 0.80],
        [0.65, 0.52, 0.40],
        [0.05, 0.1, -0.20],
        0.52,
        [0.92, 0.46, 0.68],
        0.07,
    ),
    (
        [0.85, 0.82, 1.30],
        [0.22, 0.18, 0.18],
        [0.0, 0.0, 0.0],
        0.02,
        [0.50, 0.55, 0.65],
        0.14,
    ),
];

/// Selectable 3D Gaussian splat scene source shared across `GaussianSplattingTool` and studio views.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VolumetricSplatPreset {
    /// Classic 8-Gaussian benchmark radiance field.
    #[default]
    Canonical,
    /// Dynamic 3D `1s + 2pz` hydrogenic dipole superposition cloud.
    HydrogenDipole1s2pz,
    /// Dynamic 3D `3dz²` polar-cone & equatorial-ring orbital cloud.
    HydrogenCone3dz2,
    /// Dynamic 3D Schnakenberg Turing reaction-diffusion gyroid cloud.
    TuringGyroid3d,
}

impl VolumetricSplatPreset {
    /// Returns `true` when set to the default 8-splat canonical scene.
    pub fn is_canonical(&self) -> bool {
        matches!(self, Self::Canonical)
    }

    /// Human-readable label for UI combo boxes.
    pub fn label(self) -> &'static str {
        match self {
            Self::Canonical => "Canonical 8-Splat Radiance",
            Self::HydrogenDipole1s2pz => "Quantum 1s + 2pz Dipole Cloud",
            Self::HydrogenCone3dz2 => "Quantum 3dz² Nodal Cloud",
            Self::TuringGyroid3d => "3D Turing Gyroid Morphogenesis",
        }
    }

    /// Ordered preset list for UI iteration.
    pub const ALL: [Self; 4] = [
        Self::Canonical,
        Self::HydrogenDipole1s2pz,
        Self::HydrogenCone3dz2,
        Self::TuringGyroid3d,
    ];
}

/// Builds the `(Scene, grad_magnitudes)` pair for any [`VolumetricSplatPreset`], normalized into `[-1.8, 1.8]^3`.
pub fn build_preset_scene_and_gradients(preset: VolumetricSplatPreset) -> (Scene, Vec<f64>) {
    match preset {
        VolumetricSplatPreset::Canonical => build_canonical_scene(),
        VolumetricSplatPreset::HydrogenDipole1s2pz => {
            let a = OrbitalQuantumNumbers::clamped(1, 0, 0);
            let b = OrbitalQuantumNumbers::clamped(2, 1, 0);
            build_orbital_splat_scene(a, b, 0.5, 6.0, 1.75)
        }
        VolumetricSplatPreset::HydrogenCone3dz2 => {
            let a = OrbitalQuantumNumbers::clamped(3, 2, 0);
            build_orbital_splat_scene(a, a, 0.0, 11.5, 1.75)
        }
        VolumetricSplatPreset::TuringGyroid3d => build_turing_gyroid_splat_scene(20, 1.75),
    }
}

fn build_canonical_scene() -> (Scene, Vec<f64>) {
    let mut gaussians = Vec::with_capacity(CANONICAL_SCENE.len());
    let mut grads = Vec::with_capacity(CANONICAL_SCENE.len());
    for &(pos, base_scale, axis_angle, opacity, color, grad_mag) in &CANONICAL_SCENE {
        gaussians.push(Gaussian3D {
            mean: Point3::new(pos[0], pos[1], pos[2]),
            scale: Vector3::new(base_scale[0], base_scale[1], base_scale[2]),
            rotation: UnitQuaternion::new(Vector3::new(
                axis_angle[0],
                axis_angle[1],
                axis_angle[2],
            )),
            opacity,
            color: Vector3::new(color[0], color[1], color[2]),
        });
        grads.push(grad_mag);
    }
    (Scene { gaussians }, grads)
}

fn build_orbital_splat_scene(
    orb_a: OrbitalQuantumNumbers,
    orb_b: OrbitalQuantumNumbers,
    mix: f64,
    phys_extent: f64,
    target_extent: f64,
) -> (Scene, Vec<f64>) {
    let dim = 22;
    let mut engine = HydrogenicSuperposition3D::new(orb_a, orb_b, dim, dim, phys_extent, 0.1);
    engine.set_superposition(mix, 0.0);
    let mut voxels = vec![0.0_f32; dim * dim * dim];
    engine.write_field_into(QuantumFieldObservable::SignedRealWavefunction, &mut voxels);
    let step = (2.0 * target_extent / ((dim - 1) as f64)) as f32;
    let orig = -(target_extent as f32);
    let grid = VoxelGrid::builder()
        .dimensions(dim, dim, dim)
        .data(voxels)
        .voxel_size(Point3D::new(step, step, step))
        .origin(Point3D::new(orig, orig, orig))
        .build()
        .expect("valid orbital voxel grid");
    let mut ws = VolumetricSplatWorkspace::with_capacity(MAX_DYNAMIC_SPLATS);
    extract_voxel_splats_into(&grid, 1e-4, true, None, &mut ws);
    (ws.scene, ws.grad_magnitudes)
}

/// Seeds a deterministic 3D Schnakenberg gyroid field using [`SimpleRng`] and triply periodic harmonics.
pub fn seed_3d_turing_gyroid(
    sys: &mut TuringSystem<
        2,
        SchnakenbergKinetics,
        math_explorer::biology::diffusion::FiniteDifference3D,
    >,
    dim: usize,
    a: f64,
    b: f64,
    seed: u64,
) {
    let u_ss = (a + b).max(0.05);
    let v_ss = b / (u_ss * u_ss);
    let mut rng = SimpleRng::new(seed);
    let k = 2.0 * std::f64::consts::TAU / (dim.max(2) as f64);
    for z in 0..dim {
        let kz = (z as f64) * k;
        for y in 0..dim {
            let ky = (y as f64) * k;
            for x in 0..dim {
                let kx = (x as f64) * k;
                let idx = flatten_3d_index(x, y, z, dim, dim);
                let gyroid = kx.sin() * ky.cos() + ky.sin() * kz.cos() + kz.sin() * kx.cos();
                let jitter = (rng.next_f64() - 0.5) * 0.04;
                sys.u_mut()[idx] = (u_ss + 0.22 * gyroid + jitter).max(0.02);
                sys.v_mut()[idx] = (v_ss - 0.12 * gyroid - 0.5 * jitter).max(0.02);
            }
        }
    }
    for _ in 0..6 {
        sys.step(0.008);
    }
}

fn build_turing_gyroid_splat_scene(dim: usize, target_extent: f64) -> (Scene, Vec<f64>) {
    let a = 0.10;
    let b = 0.90;
    let mut sys = TuringSystem::new_3d(
        (Dimension(dim), Dimension(dim), Dimension(dim)),
        DiffusionCoeff(1.0),
        DiffusionCoeff(18.0),
        StepSize(1.0),
        SchnakenbergKinetics::new(a, b),
    );
    seed_3d_turing_gyroid(&mut sys, dim, a, b, 0x517C_C1B7_2722_0A95);
    let u_ss = (a + b) as f32;
    let mut voxels = vec![0.0_f32; dim * dim * dim];
    sys.write_activator_f32_into(&mut voxels);
    for v in &mut voxels {
        *v -= u_ss;
    }
    let step = (2.0 * target_extent / ((dim - 1) as f64)) as f32;
    let orig = -(target_extent as f32);
    let grid = VoxelGrid::builder()
        .dimensions(dim, dim, dim)
        .data(voxels)
        .voxel_size(Point3D::new(step, step, step))
        .origin(Point3D::new(orig, orig, orig))
        .build()
        .expect("valid Turing gyroid voxel grid");
    let mut ws = VolumetricSplatWorkspace::with_capacity(MAX_DYNAMIC_SPLATS);
    extract_voxel_splats_into(&grid, 0.05, false, None, &mut ws);
    (ws.scene, ws.grad_magnitudes)
}

/// Converts a 2D `MorphogenesisTab` activator concentration slice into a 3D extruded `Gaussian3D` `Scene`.
pub fn build_scene_from_2d_morphogenesis_slice(
    u_slice: &[f64],
    width: usize,
    height: usize,
) -> Scene {
    if width < 2 || height < 2 || u_slice.len() < width * height {
        return Scene {
            gaussians: Vec::new(),
        };
    }
    let mean = u_slice.iter().take(width * height).sum::<f64>() / ((width * height) as f64);
    let depth = 5usize;
    let mut data = vec![0.0_f32; width * height * depth];
    for z in 0..depth {
        let z_env = (-0.6 * ((z as f32) - 2.0).powi(2)).exp();
        for y in 0..height {
            for x in 0..width {
                let idx_2d = flatten_2d_index(x, y, width);
                let idx_3d = flatten_3d_index(x, y, z, width, height);
                let delta = (u_slice[idx_2d] - mean) as f32;
                data[idx_3d] = delta * z_env;
            }
        }
    }
    let sx = 3.2 / ((width - 1) as f32);
    let sy = 3.2 / ((height - 1) as f32);
    let Ok(grid) = VoxelGrid::builder()
        .dimensions(width, height, depth)
        .data(data)
        .voxel_size(Point3D::new(sx, sy, 0.35))
        .origin(Point3D::new(-1.6, -1.6, -0.7))
        .build()
    else {
        return Scene {
            gaussians: Vec::new(),
        };
    };
    let mut ws = VolumetricSplatWorkspace::with_capacity(192);
    extract_voxel_splats_into(&grid, 0.02, false, None, &mut ws);
    ws.scene
}
