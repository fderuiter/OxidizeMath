use std::f64::consts::PI;

/// Represents the 1D Wave Equation parameters.
/// $\frac{\partial^2 y}{\partial x^2} = \frac{1}{v^2} \frac{\partial^2 y}{\partial t^2}$
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WaveEquation1D {
    #[allow(missing_docs)]
    pub wave_speed: f64, // v or c
}

impl WaveEquation1D {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(wave_speed: f64) -> Self {
        Self { wave_speed }
    }

    /// D'Alembert's general solution for traveling waves.
    /// $u(x, t) = f(x - ct) + g(x + ct)$
    ///
    /// # Arguments
    /// * `f` - Function representing the right-traveling component (or part of d'Alembert's IVP solution).
    /// * `g` - Function representing the left-traveling component.
    /// * `x` - Position.
    /// * `t` - Time.
    #[verified_engine::verified]
    pub fn dalembert_solution<F, G>(&self, f: F, g: G, x: f64, t: f64) -> f64
    where
        F: Fn(f64) -> f64,
        G: Fn(f64) -> f64,
    {
        let c = self.wave_speed;
        f(x - c * t) + g(x + c * t)
    }

    /// Computes a single separated variable mode.
    /// $u(x, t) = (A \cos kx + B \sin kx)(C \cos \omega t + D \sin \omega t)$
    /// where $\omega = c k$.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn separated_mode(
        &self,
        k: f64,
        params: (f64, f64, f64, f64), // A, B, C, D
        x: f64,
        t: f64,
    ) -> f64 {
        let (a, b, c_coef, d) = params;
        let omega = self.wave_speed * k;
        let spatial = a * (k * x).cos() + b * (k * x).sin();
        let temporal = c_coef * (omega * t).cos() + d * (omega * t).sin();
        spatial * temporal
    }
}

/// Represents a standing wave solution on a string of length L with fixed ends.
/// $u(x, t) = \sum_{n=1}^\infty (A_n \cos \omega_n t + B_n \sin \omega_n t) \sin(n \pi x / L)$
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StringWaveSolution {
    #[allow(missing_docs)]
    pub length: f64,
    #[allow(missing_docs)]
    pub wave_speed: f64,
    #[allow(missing_docs)]
    pub harmonics: Vec<(f64, f64)>, // (A_n, B_n) coefficients for n=1, 2, ...
}

impl StringWaveSolution {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn evaluate(&self, x: f64, t: f64) -> f64 {
        let mut u = 0.0;
        let c = self.wave_speed;
        let l = self.length;

        for (n_idx, &(an, bn)) in self.harmonics.iter().enumerate() {
            let n = (n_idx + 1) as f64;
            let k = n * PI / l;
            let omega = c * k;

            u += (an * (omega * t).cos() + bn * (omega * t).sin()) * (k * x).sin();
        }
        u
    }
}
