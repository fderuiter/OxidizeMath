//! Birth-death queueing models ($M/M/1$ and $M/M/c$ Erlang-C) with exact steady-state metrics,
//! truncated CTMC generator construction, and zero-allocation event trajectories.

use crate::pure_math::statistics::markov::ctmc::ContinuousMarkovChain;
use nalgebra::DMatrix;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;
use thiserror::Error;

/// Errors returned by $M/M/1$ and $M/M/c$ birth-death queueing models.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum QueueingError {
    /// Arrival or service rate is non-positive or non-finite.
    #[error("Invalid queueing rate ({name}): {value} (must be > 0 and finite)")]
    InvalidRate {
        /// Parameter name.
        name: &'static str,
        /// Invalid numeric value.
        value: f64,
    },
    /// Server count must be at least 1.
    #[error("Invalid server count: {servers} (must be >= 1)")]
    InvalidServerCount {
        /// Requested server count.
        servers: usize,
    },
    /// Stability condition $\rho = \lambda / (c \mu) < 1$ violated.
    #[error("Unstable queue: utilization rho = {utilization:.4} >= 1.0")]
    UnstableQueue {
        /// Computed server utilization $\rho$.
        utilization: f64,
    },
    /// Finite truncation capacity is too small.
    #[error("Invalid truncation capacity {capacity} (must be >= {min_required})")]
    InvalidCapacity {
        /// Requested capacity.
        capacity: usize,
        /// Minimum allowed capacity.
        min_required: usize,
    },
}

/// Steady-state Little's Law and Erlang-C performance metrics for a stable queue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueueMetrics {
    /// Per-server traffic utilization $\rho = \lambda / (c \mu) \in (0, 1)$.
    pub utilization: f64,
    /// Probability $p_0$ that the system is completely empty.
    pub idle_probability: f64,
    /// Probability $P(W_q > 0)$ that an arriving customer must wait (Erlang-C $C(c, a)$).
    pub prob_wait: f64,
    /// Expected customers in system $L = L_q + \lambda / \mu$.
    pub mean_in_system: f64,
    /// Expected customers waiting in queue $L_q$.
    pub mean_in_queue: f64,
    /// Expected total time in system $W = L / \lambda$.
    pub mean_sojourn_time: f64,
    /// Expected waiting time in queue $W_q = L_q / \lambda$.
    pub mean_waiting_time: f64,
}

/// Single-server Markovian birth-death queue ($M/M/1$).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mm1Queue {
    arrival_rate: f64,
    service_rate: f64,
}

impl Mm1Queue {
    /// Creates a stable $M/M/1$ queue ($\lambda > 0, \mu > 0, \lambda < \mu$).
    ///
    /// # Errors
    ///
    /// Returns [`QueueingError`] if rates are invalid or $\rho = \lambda / \mu \ge 1$.
    #[verified_engine::verified]
    pub fn new(arrival_rate: f64, service_rate: f64) -> Result<Self, QueueingError> {
        validate_rates(arrival_rate, service_rate)?;
        let rho = arrival_rate / service_rate;
        if rho >= 1.0 {
            return Err(QueueingError::UnstableQueue { utilization: rho });
        }
        Ok(Self {
            arrival_rate,
            service_rate,
        })
    }

    /// Returns the arrival rate $\lambda$.
    #[must_use]
    #[verified_engine::verified]
    pub fn arrival_rate(&self) -> f64 {
        self.arrival_rate
    }

    /// Returns the service rate $\mu$.
    #[must_use]
    #[verified_engine::verified]
    pub fn service_rate(&self) -> f64 {
        self.service_rate
    }

    /// Returns server utilization $\rho = \lambda / \mu$.
    #[must_use]
    #[verified_engine::verified]
    pub fn utilization(&self) -> f64 {
        self.arrival_rate / self.service_rate
    }

    /// Stationary probability $p_k = (1 - \rho)\rho^k$ of $k$ customers in the system.
    #[must_use]
    #[verified_engine::verified]
    pub fn state_probability(&self, k: usize) -> f64 {
        let rho = self.utilization();
        (1.0 - rho) * rho.powi(k as i32)
    }

    /// Computes exact steady-state Little's Law metrics.
    #[must_use]
    #[verified_engine::verified]
    pub fn metrics(&self) -> QueueMetrics {
        let rho = self.utilization();
        let p0 = 1.0 - rho;
        let l = rho / (1.0 - rho);
        let l_q = (rho * rho) / (1.0 - rho);
        let w = 1.0 / (self.service_rate - self.arrival_rate);
        let w_q = rho / (self.service_rate - self.arrival_rate);
        QueueMetrics {
            utilization: rho,
            idle_probability: p0,
            prob_wait: rho,
            mean_in_system: l,
            mean_in_queue: l_q,
            mean_sojourn_time: w,
            mean_waiting_time: w_q,
        }
    }

    /// Builds a finite-capacity birth-death [`ContinuousMarkovChain`] on states `0..=capacity`.
    ///
    /// # Errors
    ///
    /// Returns [`QueueingError::InvalidCapacity`] if `capacity == 0`.
    #[verified_engine::verified]
    pub fn to_truncated_ctmc(
        &self,
        capacity: usize,
    ) -> Result<ContinuousMarkovChain<f64>, QueueingError> {
        MmcQueue::new(self.arrival_rate, self.service_rate, 1)?.to_truncated_ctmc(capacity)
    }

    /// Simulates a birth-death queue trajectory in `[0, horizon]` with **zero heap allocations**.
    #[verified_engine::verified]
    pub fn simulate_trajectory_into(
        &self,
        initial_customers: usize,
        horizon: f64,
        out_times: &mut [f64],
        out_counts: &mut [usize],
        rng: &mut OxidizeRng,
    ) -> usize {
        simulate_birth_death_into(
            self.arrival_rate,
            self.service_rate,
            1,
            initial_customers,
            horizon,
            out_times,
            out_counts,
            rng,
        )
    }
}

/// Multi-server Markovian birth-death queue ($M/M/c$ Erlang-C model).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MmcQueue {
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
}

impl MmcQueue {
    /// Creates a stable $M/M/c$ queue ($\lambda > 0, \mu > 0, c \ge 1, \lambda < c \mu$).
    ///
    /// # Errors
    ///
    /// Returns [`QueueingError`] if rates/server count are invalid or $\rho = \lambda/(c\mu) \ge 1$.
    #[verified_engine::verified]
    pub fn new(
        arrival_rate: f64,
        service_rate: f64,
        servers: usize,
    ) -> Result<Self, QueueingError> {
        validate_rates(arrival_rate, service_rate)?;
        if servers == 0 {
            return Err(QueueingError::InvalidServerCount { servers });
        }
        let rho = arrival_rate / (servers as f64 * service_rate);
        if rho >= 1.0 {
            return Err(QueueingError::UnstableQueue { utilization: rho });
        }
        Ok(Self {
            arrival_rate,
            service_rate,
            servers,
        })
    }

    /// Returns arrival rate $\lambda$.
    #[must_use]
    #[verified_engine::verified]
    pub fn arrival_rate(&self) -> f64 {
        self.arrival_rate
    }

    /// Returns per-server service rate $\mu$.
    #[must_use]
    #[verified_engine::verified]
    pub fn service_rate(&self) -> f64 {
        self.service_rate
    }

    /// Returns the number of parallel servers $c$.
    #[must_use]
    #[verified_engine::verified]
    pub fn servers(&self) -> usize {
        self.servers
    }

    /// Offered load $a = \lambda / \mu$ (in Erlangs).
    #[must_use]
    #[verified_engine::verified]
    pub fn traffic_intensity(&self) -> f64 {
        self.arrival_rate / self.service_rate
    }

    /// Per-server utilization $\rho = \lambda / (c \mu)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn utilization(&self) -> f64 {
        self.traffic_intensity() / self.servers as f64
    }

    /// Overflow-safe empty-system probability $p_0$ and Erlang-C delay probability $C(c, a)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn idle_and_erlang_c(&self) -> (f64, f64) {
        let a = self.traffic_intensity();
        let rho = self.utilization();
        let mut sum_sub_c = 1.0_f64;
        let mut term = 1.0_f64;
        for k in 1..self.servers {
            term *= a / k as f64;
            sum_sub_c += term;
        }
        let tail_term = (term * (a / self.servers as f64)) / (1.0 - rho);
        let p0 = 1.0 / (sum_sub_c + tail_term);
        let erlang_c = tail_term * p0;
        (p0, erlang_c)
    }

    /// Stationary probability $p_k$ of $k$ customers in the system.
    #[must_use]
    #[verified_engine::verified]
    pub fn state_probability(&self, k: usize) -> f64 {
        let (p0, _) = self.idle_and_erlang_c();
        let a = self.traffic_intensity();
        let rho = self.utilization();
        let mut prob = p0;
        let bound = k.min(self.servers);
        for m in 1..=bound {
            prob *= a / m as f64;
        }
        if k > self.servers {
            prob *= rho.powi((k - self.servers) as i32);
        }
        prob
    }

    /// Computes exact steady-state Erlang-C and Little's Law metrics.
    #[must_use]
    #[verified_engine::verified]
    pub fn metrics(&self) -> QueueMetrics {
        let rho = self.utilization();
        let (p0, erlang_c) = self.idle_and_erlang_c();
        let l_q = erlang_c * rho / (1.0 - rho);
        let w_q = l_q / self.arrival_rate;
        let w = w_q + 1.0 / self.service_rate;
        let l = self.arrival_rate * w;
        QueueMetrics {
            utilization: rho,
            idle_probability: p0,
            prob_wait: erlang_c,
            mean_in_system: l,
            mean_in_queue: l_q,
            mean_sojourn_time: w,
            mean_waiting_time: w_q,
        }
    }

    /// Constructs the birth-death generator $G$ truncated to states `0..=capacity` ($M/M/c/K$).
    ///
    /// # Errors
    ///
    /// Returns [`QueueingError::InvalidCapacity`] if `capacity < self.servers`.
    #[verified_engine::verified]
    pub fn to_truncated_ctmc(
        &self,
        capacity: usize,
    ) -> Result<ContinuousMarkovChain<f64>, QueueingError> {
        if capacity < self.servers {
            return Err(QueueingError::InvalidCapacity {
                capacity,
                min_required: self.servers,
            });
        }
        let n_states = capacity + 1;
        let mut g = DMatrix::zeros(n_states, n_states);
        for k in 0..n_states {
            let birth = if k < capacity { self.arrival_rate } else { 0.0 };
            let death = if k > 0 {
                k.min(self.servers) as f64 * self.service_rate
            } else {
                0.0
            };
            if k < capacity {
                g[(k, k + 1)] = birth;
            }
            if k > 0 {
                g[(k, k - 1)] = death;
            }
            g[(k, k)] = -(birth + death);
        }
        ContinuousMarkovChain::new(g).map_err(|_| QueueingError::InvalidCapacity {
            capacity,
            min_required: self.servers,
        })
    }

    /// Simulates a birth-death queue trajectory in `[0, horizon]` with **zero heap allocations**.
    #[verified_engine::verified]
    pub fn simulate_trajectory_into(
        &self,
        initial_customers: usize,
        horizon: f64,
        out_times: &mut [f64],
        out_counts: &mut [usize],
        rng: &mut OxidizeRng,
    ) -> usize {
        simulate_birth_death_into(
            self.arrival_rate,
            self.service_rate,
            self.servers,
            initial_customers,
            horizon,
            out_times,
            out_counts,
            rng,
        )
    }
}

#[verified_engine::verified]
fn validate_rates(arrival_rate: f64, service_rate: f64) -> Result<(), QueueingError> {
    if !arrival_rate.is_finite() || arrival_rate <= 0.0 {
        return Err(QueueingError::InvalidRate {
            name: "arrival_rate",
            value: arrival_rate,
        });
    }
    if !service_rate.is_finite() || service_rate <= 0.0 {
        return Err(QueueingError::InvalidRate {
            name: "service_rate",
            value: service_rate,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn simulate_birth_death_into(
    arrival_rate: f64,
    service_rate: f64,
    servers: usize,
    initial_customers: usize,
    horizon: f64,
    out_times: &mut [f64],
    out_counts: &mut [usize],
    rng: &mut OxidizeRng,
) -> usize {
    let cap = out_times.len().min(out_counts.len());
    if cap == 0 || !horizon.is_finite() || horizon <= 0.0 {
        return 0;
    }

    out_times[0] = 0.0;
    out_counts[0] = initial_customers;
    let mut written = 1_usize;
    let mut n = initial_customers;
    let mut t = 0.0_f64;

    while t < horizon {
        let death_rate = n.min(servers) as f64 * service_rate;
        let total_rate = arrival_rate + death_rate;
        if total_rate <= 0.0 {
            break;
        }
        let u1 = rng.r#gen::<f64>().max(f64::MIN_POSITIVE);
        t += -u1.ln() / total_rate;
        if t > horizon {
            break;
        }
        if rng.r#gen::<f64>() * total_rate < arrival_rate {
            n += 1;
        } else {
            n = n.saturating_sub(1);
        }
        if written < cap {
            out_times[written] = t;
            out_counts[written] = n;
            written += 1;
        }
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_mm1_and_mmc_consistency_and_littles_law() {
        let mm1 = Mm1Queue::new(2.0, 3.0).unwrap();
        let m1 = mm1.metrics();
        assert_relative_eq!(m1.utilization, 2.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(m1.mean_in_system, 2.0, epsilon = 1e-12);
        assert_relative_eq!(m1.mean_sojourn_time, 1.0, epsilon = 1e-12);

        // M/M/c with c = 1 must match M/M/1 identically
        let mmc1 = MmcQueue::new(2.0, 3.0, 1).unwrap();
        let mc1 = mmc1.metrics();
        assert_relative_eq!(mc1.mean_in_queue, m1.mean_in_queue, epsilon = 1e-12);
        assert_relative_eq!(mc1.prob_wait, m1.prob_wait, epsilon = 1e-12);

        // M/M/3 with lambda = 6, mu = 3 => a = 2, rho = 2/3
        let mmc3 = MmcQueue::new(6.0, 3.0, 3).unwrap();
        let m3 = mmc3.metrics();
        // p0 = 1 / (1 + 2 + 2 + (8/6)/(1/3)) = 1 / (5 + 4) = 1/9
        assert_relative_eq!(m3.idle_probability, 1.0 / 9.0, epsilon = 1e-12);
        assert_relative_eq!(m3.prob_wait, 4.0 / 9.0, epsilon = 1e-12);
        assert_relative_eq!(
            m3.mean_in_system,
            6.0 * m3.mean_sojourn_time,
            epsilon = 1e-12
        );

        // Compare against truncated CTMC steady state at high capacity K = 60
        let ctmc = mmc3.to_truncated_ctmc(60).unwrap();
        let pi = ctmc.steady_state().unwrap();
        for k in 0..10 {
            assert_relative_eq!(pi[k], mmc3.state_probability(k), epsilon = 1e-9);
        }
    }

    #[test]
    fn test_queue_zero_allocation_simulation_verified() {
        let queue = MmcQueue::new(4.0, 2.5, 2).unwrap();
        let mut times = [0.0_f64; 256];
        let mut counts = [0_usize; 256];
        let mut rng = OxidizeRng::new(314);
        let (events, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            queue.simulate_trajectory_into(0, 50.0, &mut times, &mut counts, &mut rng)
        });
        assert!(metrics.function_calls > 0);
        assert!(events >= 1);
    }
}

// [cite:clinical_trials_statistics]
