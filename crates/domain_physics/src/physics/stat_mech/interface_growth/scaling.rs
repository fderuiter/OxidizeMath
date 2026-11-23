//! Pre-allocated interface width $W(L, t)$, Family-Vicsek dynamic scaling collapse,
//! and ordinary least-squares log-log $\beta$ growth exponent regression.

// [cite:interface_growth]

/// Exact 1D Edwards-Wilkinson growth exponent $\beta_{\text{EW}} = 1/4$.
pub const EW_BETA_1D: f64 = 0.25;
/// Exact 1D Edwards-Wilkinson roughness exponent $\alpha_{\text{EW}} = 1/2$.
pub const EW_ALPHA_1D: f64 = 0.5;
/// Exact 1D Edwards-Wilkinson dynamic exponent $z_{\text{EW}} = 2$.
pub const EW_DYNAMIC_Z_1D: f64 = 2.0;

/// Exact 1D Kardar-Parisi-Zhang growth exponent $\beta_{\text{KPZ}} = 1/3$.
pub const KPZ_BETA_1D: f64 = 1.0 / 3.0;
/// Exact 1D Kardar-Parisi-Zhang roughness exponent $\alpha_{\text{KPZ}} = 1/2$.
pub const KPZ_ALPHA_1D: f64 = 0.5;
/// Exact 1D Kardar-Parisi-Zhang dynamic exponent $z_{\text{KPZ}} = 3/2$.
pub const KPZ_DYNAMIC_Z_1D: f64 = 1.5;

/// Universality class selector for Family-Vicsek dynamic scaling collapse $W(L, t) = L^\alpha f(t / L^z)$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceGrowthKind {
    /// Linear diffusive roughening ($\lambda = 0$): 1D $(\alpha, \beta, z) = (1/2, 1/4, 2)$.
    EdwardsWilkinson,
    /// Nonlinear ballistic/lateral roughening ($\lambda \ne 0$): 1D $(\alpha, \beta, z) = (1/2, 1/3, 3/2)$.
    Kpz,
}

impl InterfaceGrowthKind {
    /// Theoretical `(alpha, beta, z)` scaling exponents in spatial dimension `dim` (`1` or `2`).
    #[verified_engine::verified]
    pub fn scaling_exponents(self, dim: usize) -> (f64, f64, f64) {
        assert!(
            dim == 1 || dim == 2,
            "Supported interface dimensions are 1 and 2"
        );
        match (self, dim) {
            (Self::EdwardsWilkinson, 1) => (EW_ALPHA_1D, EW_BETA_1D, EW_DYNAMIC_Z_1D),
            // 2D EW shows logarithmic roughness (alpha -> 0, z = 2).
            (Self::EdwardsWilkinson, _) => (0.0, 0.0, 2.0),
            (Self::Kpz, 1) => (KPZ_ALPHA_1D, KPZ_BETA_1D, KPZ_DYNAMIC_Z_1D),
            // Numerical Kim-Kosterlitz / high-precision 2+1D KPZ universality exponents:
            (Self::Kpz, _) => (0.3869, 0.2398, 1.6131),
        }
    }
}

/// Zero-allocation fixed-capacity telemetry recorder for $W(L, t)$ and Family-Vicsek rescaled series.
#[derive(Clone, Debug)]
pub struct InterfaceScalingTelemetry {
    capacity: usize,
    len: usize,
    times: Vec<f64>,
    widths: Vec<f64>,
    rescaled_times: Vec<f64>,
    rescaled_widths: Vec<f64>,
}

impl InterfaceScalingTelemetry {
    /// Pre-allocates scaling telemetry buffers for up to `capacity` samples (`capacity >= 4`).
    #[verified_engine::verified]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity >= 4, "Scaling telemetry capacity must be >= 4");
        Self {
            capacity,
            len: 0,
            times: vec![0.0; capacity],
            widths: vec![0.0; capacity],
            rescaled_times: vec![0.0; capacity],
            rescaled_widths: vec![0.0; capacity],
        }
    }

    /// Clears recorded samples in place without deallocating storage.
    #[verified_engine::verified]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Number of valid recorded scaling samples.
    #[verified_engine::verified]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether zero scaling samples have been recorded.
    #[verified_engine::verified]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum sample capacity before in-place decimation occurs.
    #[verified_engine::verified]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Recorded physical times $t$.
    #[verified_engine::verified]
    pub fn times(&self) -> &[f64] {
        &self.times[..self.len]
    }

    /// Recorded RMS interface widths $W(L, t)$.
    #[verified_engine::verified]
    pub fn widths(&self) -> &[f64] {
        &self.widths[..self.len]
    }

    /// Recorded Family-Vicsek rescaled times $u = t / L^z$.
    #[verified_engine::verified]
    pub fn rescaled_times(&self) -> &[f64] {
        &self.rescaled_times[..self.len]
    }

    /// Recorded Family-Vicsek rescaled widths $w = W(L, t) / L^\alpha$.
    #[verified_engine::verified]
    pub fn rescaled_widths(&self) -> &[f64] {
        &self.rescaled_widths[..self.len]
    }

    /// Appends `(time, width)` and computes Family-Vicsek `(t / L^z, W / L^alpha)` with zero heap allocations.
    ///
    /// When the buffer reaches `capacity`, every second historical point is compacted in place so
    /// early-time power-law growth history is retained without dynamic reallocation.
    #[verified_engine::verified]
    pub fn record_sample(
        &mut self,
        time: f64,
        width: f64,
        system_size: f64,
        alpha: f64,
        dynamic_z: f64,
    ) {
        assert!(
            time.is_finite() && time >= 0.0 && width.is_finite() && width >= 0.0,
            "Sampled time and interface width must be finite and non-negative"
        );
        assert!(
            system_size.is_finite() && system_size > 0.0,
            "System size L must be finite and positive"
        );
        if self.len == self.capacity {
            decimate_in_place(&mut self.times, &mut self.widths, &mut self.len);
            self.recompute_family_vicsek(system_size, alpha, dynamic_z);
        }
        let idx = self.len;
        self.times[idx] = time;
        self.widths[idx] = width;
        self.rescaled_times[idx] = time / system_size.powf(dynamic_z);
        self.rescaled_widths[idx] = width / system_size.powf(alpha);
        self.len += 1;
    }

    /// Convenience helper recording a 1D KPZ sample ($\alpha = 1/2, z = 3/2$).
    #[verified_engine::verified]
    pub fn record_kpz_1d(&mut self, time: f64, width: f64, system_size: f64) {
        self.record_sample(time, width, system_size, KPZ_ALPHA_1D, KPZ_DYNAMIC_Z_1D);
    }

    /// Convenience helper recording a 1D Edwards-Wilkinson sample ($\alpha = 1/2, z = 2$).
    #[verified_engine::verified]
    pub fn record_ew_1d(&mut self, time: f64, width: f64, system_size: f64) {
        self.record_sample(time, width, system_size, EW_ALPHA_1D, EW_DYNAMIC_Z_1D);
    }

    /// Re-evaluates `(t / L^z, W / L^alpha)` across all stored samples in place without heap allocation.
    #[verified_engine::verified]
    pub fn recompute_family_vicsek(&mut self, system_size: f64, alpha: f64, dynamic_z: f64) {
        assert!(
            system_size.is_finite() && system_size > 0.0,
            "System size L must be finite and positive"
        );
        let inv_lz = 1.0 / system_size.powf(dynamic_z);
        let inv_la = 1.0 / system_size.powf(alpha);
        for i in 0..self.len {
            self.rescaled_times[i] = self.times[i] * inv_lz;
            self.rescaled_widths[i] = self.widths[i] * inv_la;
        }
    }

    /// Fits the log-log growth slope $\beta$ in $\ln W(L, t) \approx \beta \ln t + c$ over $t \in [t_{\min}, t_{\max}]$.
    ///
    /// Returns `Some((beta, ln_intercept))` when at least 2 strictly positive samples lie in the window.
    #[verified_engine::verified]
    pub fn fit_growth_exponent_beta(&self, t_min: f64, t_max: f64) -> Option<(f64, f64)> {
        fit_log_log_slope(
            &self.times[..self.len],
            &self.widths[..self.len],
            t_min,
            t_max,
        )
    }
}

fn decimate_in_place(times: &mut [f64], widths: &mut [f64], len: &mut usize) {
    let mut write = 0usize;
    let mut read = 0usize;
    while read < *len {
        times[write] = times[read];
        widths[write] = widths[read];
        write += 1;
        read += 2;
    }
    *len = write;
}

/// Zero-allocation ordinary least-squares log-log slope regression over `t in [t_min, t_max]`.
#[verified_engine::verified]
pub fn fit_log_log_slope(
    times: &[f64],
    widths: &[f64],
    t_min: f64,
    t_max: f64,
) -> Option<(f64, f64)> {
    assert_eq!(
        times.len(),
        widths.len(),
        "Time and width slices must match"
    );
    let mut count = 0usize;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;

    for (&t, &w) in times.iter().zip(widths.iter()) {
        if t >= t_min && t <= t_max && t > 0.0 && w > 1e-15 && t.is_finite() && w.is_finite() {
            let x = t.ln();
            let y = w.ln();
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
            count += 1;
        }
    }

    if count < 2 {
        return None;
    }
    let n = count as f64;
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() <= 1e-14 {
        return None;
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    Some((slope, intercept))
}
