use super::surface::MriVolume;
use std::f64::consts::TAU;

/// Maximum number of simultaneous tissue classes supported in stack-allocated MRF segmentation.
pub const MAX_TISSUE_CLASSES: usize = 16;

/// Gaussian intensity likelihood and spatial prior parameters for an anatomical tissue class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TissueClass {
    /// Expected MRI intensity $\mu_k$ for class $k$.
    pub mean: f64,
    /// Intensity standard deviation $\sigma_k > 0$ for class $k$.
    pub std_dev: f64,
    /// Baseline spatial atlas prior probability $P(L_k) \ge 0$.
    pub prior: f64,
}

impl TissueClass {
    /// Constructs a [`TissueClass`] descriptor with Gaussian intensity parameters and prior weight.
    #[must_use]
    pub const fn new(mean: f64, std_dev: f64, prior: f64) -> Self {
        Self {
            mean,
            std_dev,
            prior,
        }
    }

    /// Evaluates Gaussian intensity likelihood $P(I \mid L_k)$ at `intensity`.
    #[must_use]
    #[verified_engine::verified]
    pub fn gaussian_likelihood(&self, intensity: f64) -> f64 {
        if !intensity.is_finite() || !self.mean.is_finite() || !self.std_dev.is_finite() {
            return 0.0;
        }
        let sigma = self.std_dev.max(1e-6);
        let z = (intensity - self.mean) / sigma;
        let norm = 1.0 / (TAU.sqrt() * sigma);
        norm * (-0.5 * z * z).exp()
    }
}

/// Calculates the unnormalized posterior weight $P(I \mid L) \cdot P(L)$ for a single voxel label.
///
/// # Arguments
/// * `likelihood` - $P(I \mid L)$: probability density of intensity given the label.
/// * `prior` - $P(L)$: prior probability of the label at this location.
#[verified_engine::verified]
pub fn bayesian_classification(likelihood: f64, prior: f64) -> f64 {
    likelihood * prior
}

/// Writes normalized multi-label Bayes posteriors $P(L_k \mid I)$ into `posteriors_out` without
/// heap allocation, falling back to uniform $1/K$ probabilities when total posterior mass is zero.
#[verified_engine::verified]
pub fn normalized_posteriors_into(
    likelihoods: &[f64],
    priors: &[f64],
    posteriors_out: &mut [f64],
) -> usize {
    let k_len = likelihoods
        .len()
        .min(priors.len())
        .min(posteriors_out.len());
    if k_len == 0 {
        return 0;
    }
    let mut sum = 0.0;
    for k in 0..k_len {
        let lk = if likelihoods[k].is_finite() {
            likelihoods[k].max(0.0)
        } else {
            0.0
        };
        let pr = if priors[k].is_finite() {
            priors[k].max(0.0)
        } else {
            0.0
        };
        let unnorm = bayesian_classification(lk, pr);
        posteriors_out[k] = unnorm;
        sum += unnorm;
    }
    if !sum.is_finite() || sum <= 0.0 {
        let uniform = 1.0 / (k_len as f64);
        for out in posteriors_out.iter_mut().take(k_len) {
            *out = uniform;
        }
    } else {
        let inv_sum = 1.0 / sum;
        for out in posteriors_out.iter_mut().take(k_len) {
            *out *= inv_sum;
        }
    }
    k_len
}

/// Computes normalized posterior probabilities across all candidate labels.
///
/// Returns uniform probabilities when the unnormalized sum is zero or non-finite.
#[must_use]
pub fn normalized_posteriors(likelihoods: &[f64], priors: &[f64]) -> Vec<f64> {
    let k_len = likelihoods.len().min(priors.len());
    let mut out = vec![0.0; k_len];
    let _ = normalized_posteriors_into(likelihoods, priors, &mut out);
    out
}

#[verified_engine::verified]
fn count_six_connected_neighbors(
    dims: [usize; 3],
    labels: &[u8],
    coord: [usize; 3],
    k_len: usize,
    counts: &mut [u8; MAX_TISSUE_CLASSES],
) {
    *counts = [0; MAX_TISSUE_CLASSES];
    let [nx, ny, nz] = dims;
    let [ix, iy, iz] = coord;
    let mut push_label = |x: usize, y: usize, z: usize| {
        let idx = (z * ny + y) * nx + x;
        if let Some(&lbl) = labels.get(idx) {
            let c = usize::from(lbl);
            if c < k_len {
                counts[c] += 1;
            }
        }
    };
    if ix > 0 {
        push_label(ix - 1, iy, iz);
    }
    if ix + 1 < nx {
        push_label(ix + 1, iy, iz);
    }
    if iy > 0 {
        push_label(ix, iy - 1, iz);
    }
    if iy + 1 < ny {
        push_label(ix, iy + 1, iz);
    }
    if iz > 0 {
        push_label(ix, iy, iz - 1);
    }
    if iz + 1 < nz {
        push_label(ix, iy, iz + 1);
    }
}

/// Evaluates normalized Potts MRF voxel posteriors
/// $P(L_v = k \mid I_v, L_{\mathcal{N}_6}) \propto P(I_v \mid L_k)\,P(L_k)\,\exp(\beta n_k(v))$
/// into `posteriors_out` without dynamic heap allocation.
#[verified_engine::verified]
pub fn mrf_voxel_posteriors(
    intensity: f64,
    classes: &[TissueClass],
    neighbor_counts: &[u8],
    potts_beta: f64,
    posteriors_out: &mut [f64],
) -> usize {
    let k_len = classes
        .len()
        .min(neighbor_counts.len())
        .min(posteriors_out.len())
        .min(MAX_TISSUE_CLASSES);
    if k_len == 0 {
        return 0;
    }
    let beta = if potts_beta.is_finite() {
        potts_beta.clamp(0.0, 20.0)
    } else {
        0.0
    };
    let mut likelihoods = [0.0_f64; MAX_TISSUE_CLASSES];
    let mut mrf_priors = [0.0_f64; MAX_TISSUE_CLASSES];
    for k in 0..k_len {
        likelihoods[k] = classes[k].gaussian_likelihood(intensity);
        let spatial_prior = if classes[k].prior.is_finite() {
            classes[k].prior.max(0.0)
        } else {
            0.0
        };
        mrf_priors[k] = spatial_prior * (beta * f64::from(neighbor_counts[k])).exp();
    }
    normalized_posteriors_into(&likelihoods[..k_len], &mrf_priors[..k_len], posteriors_out)
}

#[verified_engine::verified]
fn initialize_ml_labels(
    volume: &MriVolume,
    classes: &[TissueClass],
    k_len: usize,
    labels: &mut [u8],
) {
    let mut likelihoods = [0.0_f64; MAX_TISSUE_CLASSES];
    let mut priors = [0.0_f64; MAX_TISSUE_CLASSES];
    let mut post = [0.0_f64; MAX_TISSUE_CLASSES];
    for k in 0..k_len {
        priors[k] = classes[k].prior.max(0.0);
    }
    let total = volume.intensities.len().min(labels.len());
    for (lbl_out, &val) in labels.iter_mut().zip(&volume.intensities).take(total) {
        for (lk, cls) in likelihoods.iter_mut().zip(&classes[..k_len]) {
            *lk = cls.gaussian_likelihood(val);
        }
        let _ =
            normalized_posteriors_into(&likelihoods[..k_len], &priors[..k_len], &mut post[..k_len]);
        let mut best_k = 0_usize;
        for k in 1..k_len {
            if post[k] > post[best_k] {
                best_k = k;
            }
        }
        *lbl_out = best_k as u8;
    }
}

#[verified_engine::verified]
fn sweep_checkerboard_parity(
    volume: &MriVolume,
    classes: &[TissueClass],
    k_len: usize,
    potts_beta: f64,
    parity: usize,
    labels: &mut [u8],
) -> usize {
    let [nx, ny, nz] = volume.dims;
    let mut counts = [0_u8; MAX_TISSUE_CLASSES];
    let mut post = [0.0_f64; MAX_TISSUE_CLASSES];
    let mut changed = 0_usize;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                if ((ix + iy + iz) & 1) != parity {
                    continue;
                }
                let idx = (iz * ny + iy) * nx + ix;
                count_six_connected_neighbors(
                    volume.dims,
                    labels,
                    [ix, iy, iz],
                    k_len,
                    &mut counts,
                );
                let _ = mrf_voxel_posteriors(
                    volume.intensities[idx],
                    &classes[..k_len],
                    &counts[..k_len],
                    potts_beta,
                    &mut post[..k_len],
                );
                let mut best_k = usize::from(labels[idx]).min(k_len - 1);
                for k in 0..k_len {
                    if post[k] > post[best_k] + 1e-15 {
                        best_k = k;
                    }
                }
                if labels[idx] != best_k as u8 {
                    labels[idx] = best_k as u8;
                    changed += 1;
                }
            }
        }
    }
    changed
}

/// Segments `volume` into tissue labels `0..classes.len()` in-place using zero-allocation
/// red-black Iterated Conditional Modes (ICM) with 6-connected Potts Markov Random Field priors.
///
/// Returns the number of ICM sweeps executed prior to convergence or `max_iters`.
#[verified_engine::verified]
pub fn mrf_segment_volume(
    volume: &MriVolume,
    classes: &[TissueClass],
    potts_beta: f64,
    max_iters: usize,
    labels: &mut [u8],
) -> usize {
    let k_len = classes.len().min(MAX_TISSUE_CLASSES);
    let total = volume.dims[0]
        .saturating_mul(volume.dims[1])
        .saturating_mul(volume.dims[2]);
    if k_len == 0 || total == 0 || labels.len() < total || volume.intensities.len() < total {
        return 0;
    }

    initialize_ml_labels(volume, classes, k_len, labels);
    let bounded_iters = max_iters.min(256);
    let mut executed = 0_usize;
    while executed < bounded_iters {
        let even_changes = sweep_checkerboard_parity(volume, classes, k_len, potts_beta, 0, labels);
        let odd_changes = sweep_checkerboard_parity(volume, classes, k_len, potts_beta, 1, labels);
        executed += 1;
        if even_changes + odd_changes == 0 {
            break;
        }
    }
    executed
}
