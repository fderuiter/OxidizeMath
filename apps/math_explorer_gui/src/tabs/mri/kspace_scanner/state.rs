//! Validated state schema, enums, and bounded hex bitmask serialization for [`super::tool::MriKSpaceScannerTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{KSPACE_DIM, KSPACE_LEN};
use crate::framework::ToolStateBounds;
use serde::{Deserialize, Serialize};

/// Tight JSON payload bounds (`8 KiB`) enforced prior to deserializing [`MriKSpaceScannerState`].
pub const SCANNER_STATE_BOUNDS: ToolStateBounds = ToolStateBounds::new(8 * 1024, 8, 128);

const HEX_CHARS_LEN: usize = KSPACE_LEN / 4;

/// Spatial tissue phantom generator feeding the 2D spin-echo MRI forward model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhantomSource {
    /// Multi-compartment analytic Shepp-Logan head phantom ellipses.
    #[default]
    SheppLogan,
    /// Cross-domain 2D Schnakenberg Turing morphogenesis activator field (`biology` feature; falls back cleanly).
    TuringMorphogenesis,
}

impl PhantomSource {
    /// Human-readable display label for UI selectors and accessibility announcements.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SheppLogan => "Shepp-Logan Head Phantom",
            Self::TuringMorphogenesis => "Turing Morphogenesis Spots",
        }
    }
}

/// 2D k-space readout trajectory family mapped via physical gradient integrals.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KSpaceTrajectory {
    /// Fully sampled `64x64` Nyquist Cartesian grid (`100%` k-space coverage).
    #[default]
    FullCartesian,
    /// Regular phase-encode Cartesian line skipping demonstrating coherent Nyquist ghosting.
    CartesianLines,
    /// Uniform angular spokes passing through the DC k-space origin.
    RadialSpokes,
    /// Continuous outward Archimedean spiral trajectory from DC to `k_max`.
    ArchimedeanSpiral,
    /// Chaotic 3D `LorenzSystem` trajectory projected into variable-density 2D k-space.
    LorenzChaotic,
}

impl KSpaceTrajectory {
    /// Human-readable display label for trajectory selectors.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::FullCartesian => "Full 64x64 Cartesian",
            Self::CartesianLines => "Undersampled Cartesian Lines",
            Self::RadialSpokes => "Radial Spokes",
            Self::ArchimedeanSpiral => "Archimedean Spiral",
            Self::LorenzChaotic => "Chaotic Lorenz Attractor",
        }
    }
}

/// Frequency-domain window or stochastic undersampling mask combined with the trajectory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KSpaceMaskFilter {
    /// Retains all bins visited by the selected [`KSpaceTrajectory`].
    #[default]
    TrajectoryOnly,
    /// Circular low-pass keyhole mask preserving bulk DC contrast around `(32, 32)`.
    Keyhole,
    /// Circular high-pass mask erasing the DC center to isolate sharp tissue boundaries.
    HighPass,
    /// Deterministic Bernoulli mask seeded via `OxidizeRng` for incoherent undersampling.
    SeededRandom,
}

impl KSpaceMaskFilter {
    /// Human-readable display label for mask filter selectors.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::TrajectoryOnly => "Trajectory Only",
            Self::Keyhole => "Keyhole Low-Pass Disk",
            Self::HighPass => "High-Pass Center Erase",
            Self::SeededRandom => "Seeded Random Mask",
        }
    }
}

/// Tactile pointer brush action applied when painting or dragging across the 2D k-space canvas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KSpaceBrushMode {
    /// Erases (`zeros`) k-space bins inside the brush radius.
    #[default]
    Erase,
    /// Paints (`acquires`) k-space bins inside the brush radius.
    Paint,
}

impl KSpaceBrushMode {
    /// Human-readable display label for brush mode controls.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Erase => "Erase Frequencies",
            Self::Paint => "Paint Frequencies",
        }
    }
}

/// Complete persisted configuration for [`super::tool::MriKSpaceScannerTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MriKSpaceScannerState {
    /// Spin-echo repetition time `TR` in seconds (`0.1..=5.0`).
    pub tr_s: f64,
    /// Spin-echo echo time `TE` in seconds (`0.005..=0.5`).
    pub te_s: f64,
    /// Multiplier scaling compartment `T1` relaxation constants (`0.25..=3.0`).
    pub t1_scale: f64,
    /// Multiplier scaling compartment `T2` relaxation constants (`0.25..=3.0`).
    pub t2_scale: f64,
    /// Active spatial phantom generator.
    pub phantom: PhantomSource,
    /// Active k-space acquisition trajectory.
    pub trajectory: KSpaceTrajectory,
    /// Active frequency-space filter mask.
    pub mask_filter: KSpaceMaskFilter,
    /// Active pointer brush mode (`Paint` or `Erase`).
    pub brush_mode: KSpaceBrushMode,
    /// Circular brush radius in k-space bins (`1..=16`).
    pub brush_radius: usize,
    /// Phase-encode stride for [`KSpaceTrajectory::CartesianLines`] (`1..=8`).
    pub cartesian_stride: usize,
    /// Number of angular spokes for [`KSpaceTrajectory::RadialSpokes`] (`4..=128`).
    pub radial_spokes: usize,
    /// Outward revolutions for [`KSpaceTrajectory::ArchimedeanSpiral`] (`2.0..=32.0`).
    pub spiral_turns: f64,
    /// RK4 integration steps for [`KSpaceTrajectory::LorenzChaotic`] (`64..=8192`).
    pub lorenz_steps: usize,
    /// Lorenz Rayleigh parameter `rho` (`10.0..=60.0`).
    pub lorenz_rho: f64,
    /// Radial cutoff in k-space bins for `Keyhole` and `HighPass` masks (`1..=45`).
    pub cutoff_radius: usize,
    /// Retention probability for [`KSpaceMaskFilter::SeededRandom`] (`0.05..=1.0`).
    pub random_keep_ratio: f64,
    /// Deterministic `OxidizeRng` seed (`u64`).
    pub rng_seed: u64,
    /// Reaction-diffusion evolution steps for the Turing phantom (`0..=2000`).
    pub turing_steps: usize,
    /// Optional `"0x"`-prefixed 1024-hex-nybble bitmask storing custom brush/reticle edits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_mask_hex: Option<String>,
}

impl Default for MriKSpaceScannerState {
    fn default() -> Self {
        Self {
            tr_s: 1.5,
            te_s: 0.06,
            t1_scale: 1.0,
            t2_scale: 1.0,
            phantom: PhantomSource::SheppLogan,
            trajectory: KSpaceTrajectory::FullCartesian,
            mask_filter: KSpaceMaskFilter::TrajectoryOnly,
            brush_mode: KSpaceBrushMode::Erase,
            brush_radius: 3,
            cartesian_stride: 2,
            radial_spokes: 24,
            spiral_turns: 14.0,
            lorenz_steps: 1800,
            lorenz_rho: 28.0,
            cutoff_radius: 10,
            random_keep_ratio: 0.35,
            rng_seed: 42,
            turing_steps: 120,
            custom_mask_hex: None,
        }
    }
}

impl MriKSpaceScannerState {
    /// Validates finiteness, physical parameter ranges, and optional custom mask hex encoding.
    pub fn validate(&self) -> Result<(), String> {
        let float_checks = [
            ("tr_s", self.tr_s, 0.1, 5.0),
            ("te_s", self.te_s, 0.005, 0.5),
            ("t1_scale", self.t1_scale, 0.25, 3.0),
            ("t2_scale", self.t2_scale, 0.25, 3.0),
            ("spiral_turns", self.spiral_turns, 2.0, 32.0),
            ("lorenz_rho", self.lorenz_rho, 10.0, 60.0),
            ("random_keep_ratio", self.random_keep_ratio, 0.05, 1.0),
        ];
        for (name, val, min, max) in float_checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "k-Space scanner parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        let int_checks = [
            ("brush_radius", self.brush_radius, 1, 16),
            ("cartesian_stride", self.cartesian_stride, 1, 8),
            ("radial_spokes", self.radial_spokes, 4, 128),
            ("lorenz_steps", self.lorenz_steps, 64, 8192),
            ("cutoff_radius", self.cutoff_radius, 1, 45),
            ("turing_steps", self.turing_steps, 0, 2000),
        ];
        for (name, val, min, max) in int_checks {
            if val < min || val > max {
                return Err(format!(
                    "k-Space scanner integer '{name}' = {val} must be within [{min}, {max}]"
                ));
            }
        }
        if let Some(ref hex) = self.custom_mask_hex {
            let _ = decode_mask_hex(hex)?;
        }
        debug_assert_eq!(KSPACE_DIM, 64);
        Ok(())
    }

    /// Clamps out-of-range numeric fields back into valid domain bounds.
    pub fn clamp_in_place(&mut self) {
        self.tr_s = if self.tr_s.is_finite() {
            self.tr_s.clamp(0.1, 5.0)
        } else {
            1.5
        };
        self.te_s = if self.te_s.is_finite() {
            self.te_s.clamp(0.005, 0.5)
        } else {
            0.06
        };
        self.t1_scale = if self.t1_scale.is_finite() {
            self.t1_scale.clamp(0.25, 3.0)
        } else {
            1.0
        };
        self.t2_scale = if self.t2_scale.is_finite() {
            self.t2_scale.clamp(0.25, 3.0)
        } else {
            1.0
        };
        self.spiral_turns = if self.spiral_turns.is_finite() {
            self.spiral_turns.clamp(2.0, 32.0)
        } else {
            14.0
        };
        self.lorenz_rho = if self.lorenz_rho.is_finite() {
            self.lorenz_rho.clamp(10.0, 60.0)
        } else {
            28.0
        };
        self.random_keep_ratio = if self.random_keep_ratio.is_finite() {
            self.random_keep_ratio.clamp(0.05, 1.0)
        } else {
            0.35
        };
        self.brush_radius = self.brush_radius.clamp(1, 16);
        self.cartesian_stride = self.cartesian_stride.clamp(1, 8);
        self.radial_spokes = self.radial_spokes.clamp(4, 128);
        self.lorenz_steps = self.lorenz_steps.clamp(64, 8192);
        self.cutoff_radius = self.cutoff_radius.clamp(1, 45);
        self.turing_steps = self.turing_steps.clamp(0, 2000);
        if self
            .custom_mask_hex
            .as_deref()
            .is_some_and(|h| decode_mask_hex(h).is_err())
        {
            self.custom_mask_hex = None;
        }
    }
}

/// Encodes a `4096`-bin boolean k-space mask into a `"0x"`-prefixed 1024-nybble hex string.
#[must_use]
pub fn encode_mask_hex(mask: &[bool]) -> String {
    assert_eq!(mask.len(), KSPACE_LEN);
    let mut out = String::with_capacity(2 + HEX_CHARS_LEN);
    out.push_str("0x");
    for chunk in mask.as_chunks::<4>().0 {
        let mut nybble = 0_u8;
        for (bit, &active) in chunk.iter().enumerate() {
            if active {
                nybble |= 1_u8 << bit;
            }
        }
        let ch = char::from_digit(u32::from(nybble), 16).unwrap_or('0');
        out.push(ch);
    }
    out
}

/// Decodes a `"0x"`-prefixed 1024-nybble hex string into a `4096`-bin boolean k-space mask.
pub fn decode_mask_hex(hex: &str) -> Result<Vec<bool>, String> {
    let body = hex
        .strip_prefix("0x")
        .or_else(|| hex.strip_prefix("0X"))
        .unwrap_or(hex);
    if body.len() != HEX_CHARS_LEN {
        return Err(format!(
            "Custom k-space mask hex length ({}) must equal {HEX_CHARS_LEN}",
            body.len()
        ));
    }
    let mut mask = Vec::with_capacity(KSPACE_LEN);
    for ch in body.chars() {
        let nybble = ch
            .to_digit(16)
            .ok_or_else(|| format!("Invalid hex character '{ch}' in custom k-space mask"))?
            as u8;
        for bit in 0..4 {
            mask.push((nybble & (1_u8 << bit)) != 0);
        }
    }
    Ok(mask)
}
