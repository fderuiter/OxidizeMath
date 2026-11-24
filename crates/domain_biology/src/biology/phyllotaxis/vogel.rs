//! Pre-allocated Vogel primordia coordinate engine ($r_n = c\sqrt{n}, \theta_n = n\alpha$).

use super::parastichy::{ParastichyAnalysis, analyze_parastichies, fill_parastichy_chain};
use crate::error::BotanicalError;

/// Hard upper bound on primordia points in [`VogelPhyllotaxisEngine`] ($2^{12} = 4{,}096$).
pub const MAX_PRIMORDIA: usize = 4_096;

/// Square root of 5 ($\sqrt{5}$) used to derive exact golden-angle constants.
const SQRT_5: f64 = 2.236_067_977_499_79;

/// Golden ratio $\varphi = (1 + \sqrt{5}) / 2$.
pub const GOLDEN_RATIO: f64 = 0.5 * (1.0 + SQRT_5);

/// Golden divergence angle $\alpha_{\varphi} = 360^\circ (2 - \varphi) \approx 137.507764^\circ$.
pub const GOLDEN_ANGLE_DEG: f64 = 360.0 * (2.0 - GOLDEN_RATIO);

/// Golden divergence angle in radians $\alpha_{\varphi} = 2\pi (2 - \varphi) \approx 2.39996323\text{ rad}$.
pub const GOLDEN_ANGLE_RAD: f64 = core::f64::consts::TAU * (2.0 - GOLDEN_RATIO);

/// Canonical irrational and rational phyllotactic divergence angles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceAnglePreset {
    /// Golden angle $2\pi(2 - \varphi) \approx 137.5078^\circ$ yielding Fibonacci $(F_k, F_{k+1})$ parastichies.
    GoldenAngle,
    /// Lucas noble angle $2\pi / (3 + 1/\varphi) \approx 99.5016^\circ$ yielding Lucas $(3, 4, 7, 11, 18)$ spirals.
    LucasAngle,
    /// Rational $1/3$ turn ($120^\circ$) collapsing primordia into 3 radial orthostichy spokes.
    RationalOneThird,
    /// Rational $2/5$ turn ($144^\circ$) collapsing primordia into 5 radial spokes.
    RationalTwoFifths,
    /// Rational $3/8$ turn ($135^\circ$) collapsing primordia into 8 radial spokes.
    RationalThreeEighths,
    /// Rational $5/13$ turn ($138.4615^\circ$) approximating the golden angle at low $N$ before forming 13 spokes.
    RationalFiveThirteenths,
}

impl DivergenceAnglePreset {
    /// All canonical divergence angle presets.
    pub const ALL: [Self; 6] = [
        Self::GoldenAngle,
        Self::LucasAngle,
        Self::RationalOneThird,
        Self::RationalTwoFifths,
        Self::RationalThreeEighths,
        Self::RationalFiveThirteenths,
    ];

    /// Human-readable display label.
    #[verified_engine::verified]
    pub fn label(self) -> &'static str {
        match self {
            Self::GoldenAngle => "Golden Angle (137.5078°)",
            Self::LucasAngle => "Lucas Noble Angle (99.5016°)",
            Self::RationalOneThird => "Rational 1/3 Turn (120.0°)",
            Self::RationalTwoFifths => "Rational 2/5 Turn (144.0°)",
            Self::RationalThreeEighths => "Rational 3/8 Turn (135.0°)",
            Self::RationalFiveThirteenths => "Rational 5/13 Turn (138.4615°)",
        }
    }

    /// Divergence angle in degrees.
    #[verified_engine::verified]
    pub fn angle_deg(self) -> f64 {
        match self {
            Self::GoldenAngle => GOLDEN_ANGLE_DEG,
            Self::LucasAngle => 360.0 / (3.0 + 1.0 / GOLDEN_RATIO),
            Self::RationalOneThird => 120.0,
            Self::RationalTwoFifths => 144.0,
            Self::RationalThreeEighths => 135.0,
            Self::RationalFiveThirteenths => 360.0 * 5.0 / 13.0,
        }
    }

    /// Divergence angle in radians.
    #[verified_engine::verified]
    pub fn angle_rad(self) -> f64 {
        self.angle_deg().to_radians()
    }
}

/// Coordinates of primordium $n \in [0, N)$ on the planar Vogel spiral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrimordiumPoint {
    /// Zero-based primordium index $n$.
    pub index: usize,
    /// Polar radius $r_n = c\sqrt{n}$.
    pub radius: f64,
    /// Cumulative polar angle $\theta_n = n\alpha$ in radians.
    pub theta: f64,
    /// Cartesian abscissa $x_n = r_n \cos\theta_n$.
    pub x: f64,
    /// Cartesian ordinate $y_n = r_n \sin\theta_n$.
    pub y: f64,
}

/// Zero-allocation Vogel phyllotaxis coordinate and parastichy analysis engine.
#[derive(Debug, Clone)]
pub struct VogelPhyllotaxisEngine {
    count: usize,
    divergence_rad: f64,
    scale: f64,
    points: Vec<PrimordiumPoint>,
    analysis: ParastichyAnalysis,
    spiral_scratch: Vec<usize>,
}

impl VogelPhyllotaxisEngine {
    /// Constructs a [`VogelPhyllotaxisEngine`] with pre-allocated buffers for up to
    /// [`MAX_PRIMORDIA`] ($4{,}096$) primordia and computes the initial primordia layout.
    #[verified_engine::verified]
    pub fn new(count: usize, divergence_rad: f64, scale: f64) -> Result<Self, BotanicalError> {
        validate_phyllotaxis_params(count, divergence_rad, scale)?;
        let mut engine = Self {
            count,
            divergence_rad,
            scale,
            points: Vec::with_capacity(MAX_PRIMORDIA),
            analysis: ParastichyAnalysis::default(),
            spiral_scratch: Vec::with_capacity(MAX_PRIMORDIA),
        };
        engine.recompute();
        Ok(engine)
    }

    /// Convenience constructor initialized at the golden angle $\alpha_{\varphi} \approx 137.5078^\circ$.
    #[verified_engine::verified]
    pub fn golden(count: usize, scale: f64) -> Result<Self, BotanicalError> {
        Self::new(count, GOLDEN_ANGLE_RAD, scale)
    }

    /// Updates `(count, divergence_rad, scale)` and recomputes all Vogel coordinates and
    /// parastichy counts in place with zero heap allocations.
    #[verified_engine::verified]
    pub fn set_parameters(
        &mut self,
        count: usize,
        divergence_rad: f64,
        scale: f64,
    ) -> Result<(), BotanicalError> {
        validate_phyllotaxis_params(count, divergence_rad, scale)?;
        self.count = count;
        self.divergence_rad = divergence_rad;
        self.scale = scale;
        self.recompute();
        Ok(())
    }

    /// Recomputes $r_n = c\sqrt{n}$, $\theta_n = n\alpha$, Cartesian $(x_n, y_n)$, and opposing
    /// parastichy spiral metrics into the pre-allocated buffers with zero heap allocations.
    #[verified_engine::verified]
    pub fn recompute(&mut self) {
        self.points.clear();
        for n in 0..self.count {
            let nf = n as f64;
            let radius = self.scale * nf.sqrt();
            let theta = nf * self.divergence_rad;
            let (sin_t, cos_t) = theta.sin_cos();
            self.points.push(PrimordiumPoint {
                index: n,
                radius,
                theta,
                x: radius * cos_t,
                y: radius * sin_t,
            });
        }
        self.analysis = analyze_parastichies(&self.points, self.divergence_rad, self.scale);
    }

    /// Populates the internal pre-allocated scratch buffer with primordia indices belonging to
    /// `arm_index` (`0..parastichy_step`) for a spiral family of step `parastichy_step`, returning
    /// the borrowed index slice without allocating.
    #[verified_engine::verified]
    pub fn trace_parastichy_arm(&mut self, arm_index: usize, parastichy_step: usize) -> &[usize] {
        fill_parastichy_chain(
            self.count,
            arm_index,
            parastichy_step,
            &mut self.spiral_scratch,
        );
        &self.spiral_scratch
    }

    /// Active primordia slice of length `count` ($\le 4{,}096$).
    #[inline]
    #[verified_engine::verified]
    pub fn points(&self) -> &[PrimordiumPoint] {
        &self.points
    }

    /// Current opposing parastichy counts and packing uniformity analysis.
    #[inline]
    #[verified_engine::verified]
    pub fn analysis(&self) -> ParastichyAnalysis {
        self.analysis
    }

    /// Active primordia count $N$.
    #[inline]
    #[verified_engine::verified]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Active divergence angle $\alpha$ in radians.
    #[inline]
    #[verified_engine::verified]
    pub fn divergence_rad(&self) -> f64 {
        self.divergence_rad
    }

    /// Active radial scale $c$.
    #[inline]
    #[verified_engine::verified]
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

#[verified_engine::verified]
fn validate_phyllotaxis_params(
    count: usize,
    divergence_rad: f64,
    scale: f64,
) -> Result<(), BotanicalError> {
    if count == 0 || count > MAX_PRIMORDIA {
        return Err(BotanicalError::InvalidPrimordiaCount {
            count,
            max: MAX_PRIMORDIA,
        });
    }
    if !divergence_rad.is_finite() {
        return Err(BotanicalError::InvalidDivergenceAngle {
            angle: divergence_rad,
        });
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(BotanicalError::InvalidScaleFactor { scale });
    }
    Ok(())
}
