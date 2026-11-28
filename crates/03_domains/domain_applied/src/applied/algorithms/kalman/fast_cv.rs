//! Zero-allocation 2-state Constant-Velocity scalar Kalman filter (`FastCvKalman`).

/// Stack-resident 2-state Constant-Velocity Kalman filter for real-time respiratory and kinematic tracking.
///
/// Propagates `[position, velocity]` and its $2 \times 2$ covariance matrix on stack registers
/// without dynamic heap allocations.
///
/// # Examples
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::algorithms::kalman::FastCvKalman;
///
/// let mut kf = FastCvKalman::with_unit_covariance(0.0, 0.025, 3.0e-4, 2.0e-5);
/// kf.predict();
/// kf.update(0.004);
/// assert!(kf.state[0] > 0.0);
/// assert!(kf.predict_ahead(0.05).is_finite());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FastCvKalman {
    /// State vector `[amplitude_m, velocity_m_s]`.
    pub state: [f64; 2],
    /// State error covariance `[[p00, p01], [p10, p11]]`.
    pub cov: [[f64; 2]; 2],
    /// Discretization interval in seconds.
    pub dt: f64,
    /// Process noise variance $Q$.
    pub process_var: f64,
    /// Measurement noise variance $R$.
    pub meas_var: f64,
}

impl FastCvKalman {
    /// Constructs a [`FastCvKalman`] from explicit initial `[position, velocity]` and $2 \times 2$ covariance.
    #[must_use]
    #[verified_engine::verified]
    pub const fn new(
        state: [f64; 2],
        cov: [[f64; 2]; 2],
        dt: f64,
        process_var: f64,
        meas_var: f64,
    ) -> Self {
        Self {
            state,
            cov,
            dt,
            process_var,
            meas_var,
        }
    }

    /// Constructs a [`FastCvKalman`] at `[initial_amplitude, 0.0]` with identity $2 \times 2$ initial covariance.
    #[must_use]
    #[verified_engine::verified]
    pub const fn with_unit_covariance(
        initial_amplitude: f64,
        dt: f64,
        process_var: f64,
        meas_var: f64,
    ) -> Self {
        Self {
            state: [initial_amplitude, 0.0],
            cov: [[1.0, 0.0], [0.0, 1.0]],
            dt,
            process_var,
            meas_var,
        }
    }

    /// Executes the exact constant-velocity predict step on stack scalars.
    #[verified_engine::verified]
    pub fn predict(&mut self) {
        debug_assert!(self.dt.is_finite());
        let dt = self.dt;
        self.state[0] += dt * self.state[1];
        let (p00, p01, p10, p11) = (
            self.cov[0][0],
            self.cov[0][1],
            self.cov[1][0],
            self.cov[1][1],
        );
        self.cov[0][0] = p00 + dt * (p10 + p01) + dt * dt * p11 + self.process_var;
        self.cov[0][1] = p01 + dt * p11;
        self.cov[1][0] = p10 + dt * p11;
        self.cov[1][1] = p11 + self.process_var;
    }

    /// Executes the exact scalar position measurement update (`H = [1, 0]`) on stack scalars.
    #[verified_engine::verified]
    pub fn update(&mut self, z_m: f64) {
        let s = self.cov[0][0] + self.meas_var;
        if !z_m.is_finite() || !s.is_finite() || s.abs() <= 1.0e-18 {
            return;
        }
        let y = z_m - self.state[0];
        let (k0, k1) = (self.cov[0][0] / s, self.cov[1][0] / s);
        self.state[0] += k0 * y;
        self.state[1] += k1 * y;
        let (p00, p01, p10, p11) = (
            self.cov[0][0],
            self.cov[0][1],
            self.cov[1][0],
            self.cov[1][1],
        );
        self.cov[0][0] = (1.0 - k0) * p00;
        self.cov[0][1] = (1.0 - k0) * p01;
        self.cov[1][0] = p10 - k1 * p00;
        self.cov[1][1] = p11 - k1 * p01;
    }

    /// Extrapolates position `horizon_s` seconds ahead using constant-velocity kinematics.
    #[must_use]
    #[verified_engine::verified]
    pub fn predict_ahead(&self, horizon_s: f64) -> f64 {
        debug_assert!(horizon_s.is_finite());
        self.state[0] + self.state[1] * horizon_s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn fast_cv_kalman_tracks_linear_ramp_zero_alloc() {
        let _guard = crate::VERIFIED_ENGINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut kf = FastCvKalman::with_unit_covariance(0.0, 0.025, 3.0e-4, 2.0e-5);
        let (pred, _) = VerifiedEngine::run_verified(|| {
            for k in 1..=40 {
                let z = 0.02 * f64::from(k) * 0.025;
                kf.predict();
                kf.update(z);
            }
            kf.predict_ahead(0.05)
        });
        assert!((kf.state[1] - 0.02).abs() < 2.0e-3);
        assert!(pred > kf.state[0]);
    }
}
