//! Discrete Chaos (The Logistic Map)
//!
//! This module implements the Logistic Map, a classic example of how complex, chaotic behavior
//! can arise from very simple non-linear dynamical equations.
//!
//! The map is defined by the recurrence relation:
//! $x_{n+1} = r x_n (1 - x_n)$

/// Number of iterations to discard as transient before collecting attractor points.
pub const TRANSIENT_STEPS: usize = 100;
/// Number of points to collect for the attractor visualization.
pub const ATTRACTOR_POINTS: usize = 50;

/// A struct representing the Logistic Map, a classic example of how complex, chaotic behavior
/// can arise from very simple non-linear dynamical equations.
///
/// The map is defined by the recurrence relation:
/// $x_{n+1} = r x_n (1 - x_n)$
pub struct LogisticMap {
    /// The growth rate parameter $r$.
    /// - For $r < 3$, the population eventually settles into a stable value.
    /// - For $3 < r < 3.57$, the population oscillates between 2, 4, 8, ... values (period doubling).
    /// - For $r > 3.57$, the behavior becomes chaotic.
    pub r: f64,
    /// The current state value $x$, where $0 \le x \le 1$.
    pub state: f64,
}

impl LogisticMap {
    /// Creates a new LogisticMap instance.
    #[verified_engine::verified]
    pub fn new(r: f64, initial_state: f64) -> Self {
        LogisticMap {
            r,
            state: initial_state,
        }
    }
}

impl Iterator for LogisticMap {
    type Item = f64;

    #[verified_engine::verified]
    fn next(&mut self) -> Option<Self::Item> {
        self.state = self.r * self.state * (1.0 - self.state);
        Some(self.state)
    }
}

/// Generates points for a bifurcation diagram.
///
/// This function iterates over a range of $r$ values. For each $r$, it iterates the map
/// to discard transients and then captures the attractor states.
///
/// # Arguments
///
/// * `r_start` - The starting value of the parameter $r$.
/// * `r_end` - The ending value of the parameter $r$.
/// * `steps` - The number of discrete steps to take from `r_start` to `r_end`.
///
/// # Returns
///
/// A vector of tuples `(r, x)`, where `r` is the parameter value and `x` is a visited state
/// on the attractor.
#[verified_engine::verified]
pub fn generate_bifurcation_diagram(r_start: f64, r_end: f64, steps: usize) -> Vec<(f64, f64)> {
    let mut points = Vec::new();
    let step_size = (r_end - r_start) / steps as f64;

    for i in 0..=steps {
        let r = r_start + step_size * i as f64;
        let mut map = LogisticMap::new(r, 0.5); // Start with a generic seed like 0.5

        // Discard transient
        for _ in 0..TRANSIENT_STEPS {
            map.next();
        }

        // Collect attractor points
        for _ in 0..ATTRACTOR_POINTS {
            if let Some(x) = map.next() {
                points.push((r, x));
            }
        }
    }

    points
}
