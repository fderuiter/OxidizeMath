//! MUSIC (Multiple Signal Classification) Algorithm for FMCW Radar.
//!
//! This module implements the "Math" Solution: Super-Resolution processing.
//! It allows distinguishing between reflection points (e.g., skin vs. clothes) closer than the
//! physical range resolution limit ($\Delta R = c/2B$) by exploiting the eigen-structure of the
//! signal covariance matrix.

use crate::error::RadarError;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use std::collections::VecDeque;
use std::f64::consts::PI;

/// Implements the MUSIC algorithm for high-resolution range estimation.
pub struct MusicEstimator {
    /// Number of signal snapshots (chirps) to estimate covariance.
    snapshot_count: usize,
    /// Number of samples per chirp ($N$).
    samples_per_chirp: usize,
    /// Estimated number of targets/signals ($P$).
    signal_subspace_dim: usize,
    /// Signal covariance matrix buffer.
    snapshots: VecDeque<DVector<Complex<f64>>>,
}

impl MusicEstimator {
    /// Creates a new MUSIC Estimator.
    ///
    /// # Arguments
    ///
    /// * `samples_per_chirp` - Length of the FMCW chirp ($N$).
    /// * `smoothing_factor` - Number of snapshots to average for covariance matrix stability (e.g., 10).
    /// * `num_targets` - Expected number of reflecting surfaces (e.g., 1 for just skin, 2 for skin+clothes).
    ///
    /// # Errors
    /// Returns `RadarError::InvalidConfiguration` if parameters are invalid.
    #[verified_engine::verified]
    pub fn new(
        samples_per_chirp: usize,
        smoothing_factor: usize,
        num_targets: usize,
    ) -> Result<Self, RadarError> {
        if samples_per_chirp == 0 {
            return Err(RadarError::InvalidConfiguration(
                "samples_per_chirp must be > 0".into(),
            ));
        }
        if smoothing_factor == 0 {
            return Err(RadarError::InvalidConfiguration(
                "smoothing_factor must be > 0".into(),
            ));
        }
        if num_targets == 0 || num_targets >= samples_per_chirp {
            return Err(RadarError::InvalidConfiguration(format!(
                "num_targets {} must be > 0 and < samples_per_chirp {}",
                num_targets, samples_per_chirp
            )));
        }

        Ok(Self {
            snapshot_count: smoothing_factor,
            samples_per_chirp,
            signal_subspace_dim: num_targets,
            snapshots: VecDeque::with_capacity(smoothing_factor),
        })
    }

    /// Adds a chirp snapshot to the estimator.
    ///
    /// Once enough snapshots are collected, the covariance matrix is robust.
    /// Ideally, call `compute_spectrum` after filling the buffer.
    #[verified_engine::verified]
    pub fn add_snapshot(&mut self, chirp: &[Complex<f64>]) -> Result<(), RadarError> {
        if chirp.len() != self.samples_per_chirp {
            return Err(RadarError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(self.samples_per_chirp),
                    actual: math_commons::math_kernel::types::Dimension(chirp.len()),
                },
            ));
        }
        let vec = DVector::from_iterator(chirp.len(), chirp.iter().cloned());

        if self.snapshots.len() >= self.snapshot_count {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(vec);
        Ok(())
    }

    /// Computes the MUSIC Pseudospectrum over a range of distances.
    ///
    /// $$ P(R) = \frac{1}{a(R)^H E_n E_n^H a(R)} $$
    ///
    /// # Arguments
    ///
    /// * `start_range` - Start distance in meters.
    /// * `end_range` - End distance in meters.
    /// * `step_range` - Resolution step in meters (e.g., 0.001 for 1mm).
    /// * `bandwidth` - Radar bandwidth ($B$) in Hz.
    /// * `c` - Speed of light (default ~3e8).
    ///
    /// # Returns
    ///
    /// A vector of `(Range, Power)` tuples.
    ///
    /// # Panics
    ///
    /// This method performs an `unwrap()` when sorting eigenvalues descending. This is provably
    /// infallible because the algorithm explicitly checks and returns an error (`RadarError::Math(NumericalError)`)
    /// if any eigenvalues contain `NaN` prior to sorting.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use domain_physics::physics::medical::radar_gating::super_resolution::MusicEstimator;
    /// use num_complex::Complex;
    ///
    /// // Initialize estimator for 1 target with 4 samples per chirp and 2 snapshots smoothing.
    /// let mut estimator = MusicEstimator::new(4, 2, 1).unwrap();
    ///
    /// // Provide simulated chirp data (e.g., reflections)
    /// let chirp1 = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0), Complex::new(-1.0, 0.0), Complex::new(0.0, -1.0)];
    /// let chirp2 = vec![Complex::new(0.9, 0.1), Complex::new(-0.1, 0.9), Complex::new(-0.9, -0.1), Complex::new(0.1, -0.9)];
    ///
    /// estimator.add_snapshot(&chirp1).unwrap();
    /// estimator.add_snapshot(&chirp2).unwrap();
    ///
    /// // Compute the spectrum
    /// let bandwidth = 1.5e9; // 1.5 GHz
    /// let c = 3e8;           // Speed of light
    /// let spectrum = estimator.compute_spectrum(0.0, 1.0, 0.1, bandwidth, c).unwrap();
    ///
    /// assert_eq!(spectrum.len(), 11); // 0.0 to 1.0 with 0.1 step inclusive
    /// ```
    #[verified_engine::verified]
    pub fn compute_spectrum(
        &self,
        start_range: f64,
        end_range: f64,
        step_range: f64,
        bandwidth: f64,
        c: f64,
    ) -> Result<Vec<(f64, f64)>, RadarError> {
        if self.snapshots.len() < self.snapshot_count {
            return Err(RadarError::InsufficientSnapshots {
                required: self.snapshot_count,
                actual: self.snapshots.len(),
            });
        }
        if step_range <= 0.0 {
            return Err(RadarError::InvalidConfiguration(
                "step_range must be > 0".into(),
            ));
        }
        if start_range > end_range {
            return Err(RadarError::InvalidConfiguration(
                "start_range must be <= end_range".into(),
            ));
        }

        let r_xx = self.estimate_covariance();
        let subspace = self.extract_compact_subspace(r_xx)?;
        let params = RangeSweepParams {
            start_range,
            end_range,
            step_range,
            bandwidth,
            c,
        };

        debug_assert!(subspace.basis.ncols() <= self.samples_per_chirp / 2);
        Ok(subspace.evaluate_range_sweep(&params))
    }

    /// Estimates the sample covariance matrix $R_{xx} = \frac{1}{K} \sum_{k=1}^K x_k x_k^H$.
    #[verified_engine::verified]
    fn estimate_covariance(&self) -> DMatrix<Complex<f64>> {
        let n = self.samples_per_chirp;
        debug_assert!(n > 0);
        debug_assert!(!self.snapshots.is_empty());

        let mut r_xx = DMatrix::<Complex<f64>>::zeros(n, n);
        for snap in &self.snapshots {
            let outer = snap * snap.adjoint();
            r_xx += outer;
        }
        r_xx /= Complex::new(self.snapshots.len() as f64, 0.0);
        r_xx
    }

    /// Extracts a compact $N \times \min(P, N - P)$ orthonormal eigenvector basis.
    #[verified_engine::verified]
    fn extract_compact_subspace(
        &self,
        r_xx: DMatrix<Complex<f64>>,
    ) -> Result<CompactSubspace, RadarError> {
        let n = self.samples_per_chirp;
        debug_assert_eq!(r_xx.nrows(), n);
        debug_assert_eq!(r_xx.ncols(), n);

        let eigen = r_xx.symmetric_eigen();
        if eigen.eigenvalues.iter().any(|v| v.is_nan()) {
            return Err(RadarError::Math(
                math_commons::error::MathError::NumericalError {
                    reason: "Eigenvalues contain NaN".into(),
                },
            ));
        }

        let num_noise_vectors = n.saturating_sub(self.signal_subspace_dim);
        if num_noise_vectors == 0 {
            return Err(RadarError::InvalidSignalSubspace {
                samples: n,
                subspace: self.signal_subspace_dim,
            });
        }

        let mut pairs: Vec<(f64, usize)> = eigen
            .eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i))
            .collect();
        // Sort descending (largest first). Safe to unwrap because we checked for NaNs above.
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let use_signal_complement = self.signal_subspace_dim < num_noise_vectors;
        let selected = if use_signal_complement {
            &pairs[..self.signal_subspace_dim]
        } else {
            &pairs[self.signal_subspace_dim..]
        };

        let mut basis = DMatrix::<Complex<f64>>::zeros(n, selected.len());
        for (col_idx, &(_, eig_idx)) in selected.iter().enumerate() {
            basis
                .column_mut(col_idx)
                .copy_from(&eigen.eigenvectors.column(eig_idx));
        }

        Ok(CompactSubspace {
            basis,
            use_signal_complement,
        })
    }
}

/// Compact column-major orthonormal eigenvector basis of shape $N \times \min(P, N - P)$.
struct CompactSubspace {
    basis: DMatrix<Complex<f64>>,
    use_signal_complement: bool,
}

/// Range sweep configuration for evaluating the MUSIC pseudospectrum.
struct RangeSweepParams {
    start_range: f64,
    end_range: f64,
    step_range: f64,
    bandwidth: f64,
    c: f64,
}

impl CompactSubspace {
    /// Evaluates $P(R) = (d(R) + 10^{-12})^{-1}$ across the range grid using low-rank projection.
    #[verified_engine::verified]
    fn evaluate_range_sweep(&self, params: &RangeSweepParams) -> Vec<(f64, f64)> {
        let n = self.basis.nrows();
        debug_assert!(n > 0);
        debug_assert!(params.step_range > 0.0);

        let steps = ((params.end_range - params.start_range) / params.step_range).round() as usize;
        let mut spectrum = Vec::with_capacity(steps + 1);
        let constant_factor = (4.0 * PI * params.bandwidth) / (params.c * n as f64);
        let mut a_vec = DVector::from_element(n, Complex::new(0.0, 0.0));
        let mut r = params.start_range;

        for _ in 0..=steps {
            let alpha = constant_factor * r;
            let (sin_alpha, cos_alpha) = alpha.sin_cos();
            let step_phasor = Complex::new(cos_alpha, sin_alpha);

            a_vec[0] = Complex::new(1.0, 0.0);
            for k in 1..n {
                a_vec[k] = a_vec[k - 1] * step_phasor;
            }

            let mut proj_norm_sq = 0.0;
            for col in self.basis.column_iter() {
                proj_norm_sq += col.dotc(&a_vec).norm_sqr();
            }

            let den = if self.use_signal_complement {
                (a_vec.norm_squared() - proj_norm_sq).max(0.0)
            } else {
                proj_norm_sq.max(0.0)
            };

            let power = 1.0 / (den + 1e-12);
            spectrum.push((r, power));
            r += params.step_range;
        }

        spectrum
    }
}
