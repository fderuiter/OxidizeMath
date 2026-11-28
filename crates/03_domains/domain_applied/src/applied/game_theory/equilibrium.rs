use crate::error::GameTheoryError;
use nalgebra::DVector;

mod nash_solvers;
pub use engine_opt::nash::{BimatrixNashStats, PolymatrixConfig, PolymatrixNashStats};
pub use nash_solvers::{
    BimatrixBestResponse, bimatrix_nash_workspace_len, polymatrix_nash_workspace_len,
    solve_bimatrix_nash_into, solve_polymatrix_nash_into,
};

/// Represents a compact, convex subset of Euclidean space.
///
/// In Game Theory, the strategy space of a player is often modeled as a convex set.
/// The convexity property is crucial for the application of fixed-point theorems
/// (like Kakutani's) which guarantee the existence of Nash Equilibria.
pub trait ConvexSet {
    /// Checks if a point belongs to the set.
    #[verified_engine::verified]
    fn contains(&self, point: &DVector<f64>) -> bool;

    /// Checks if the set is convex.
    ///
    /// *Note:* This is often a theoretical property assumed by the modeler.
    /// An implementation might perform a randomized check (e.g., is the midpoint of two random points in the set?).
    #[verified_engine::verified]
    fn is_convex(&self) -> bool;
}

/// A simplified representation of a Box constraint set $[min, max]^n$.
///
/// A hyper-rectangle is always convex and compact (closed and bounded), making it
/// a valid domain for standard fixed-point theorems.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxSet {
    /// Coordinate-wise inclusive lower bounds $\mathbf{l} \in \mathbb{R}^n$.
    pub min_bounds: DVector<f64>,
    /// Coordinate-wise inclusive upper bounds $\mathbf{u} \in \mathbb{R}^n$ ($\mathbf{l} \le \mathbf{u}$).
    pub max_bounds: DVector<f64>,
}

impl BoxSet {
    /// Fallibly constructs a `BoxSet` hyper-rectangle $[min, max]^n$.
    ///
    /// Returns [`GameTheoryError::BoundsDimensionMismatch`] if `min_vals` and `max_vals` differ in
    /// length, or [`GameTheoryError::InvalidBoxBounds`] if any coordinate has non-finite bounds or
    /// `min_vals[i] > max_vals[i]`.
    #[verified_engine::verified]
    pub fn try_new(min_vals: Vec<f64>, max_vals: Vec<f64>) -> Result<Self, GameTheoryError> {
        if min_vals.len() != max_vals.len() {
            return Err(GameTheoryError::BoundsDimensionMismatch {
                min_len: min_vals.len(),
                max_len: max_vals.len(),
            });
        }
        for (dimension, (&min, &max)) in min_vals.iter().zip(max_vals.iter()).enumerate() {
            if !min.is_finite() || !max.is_finite() || min > max {
                return Err(GameTheoryError::InvalidBoxBounds {
                    dimension,
                    min,
                    max,
                });
            }
        }
        Ok(Self {
            min_bounds: DVector::from_vec(min_vals),
            max_bounds: DVector::from_vec(max_vals),
        })
    }

    /// Creates a new `BoxSet` representing a hyper-rectangle.
    ///
    /// The resulting `BoxSet` constrains the domain across $n$ dimensions,
    /// where $n$ is the length of the provided vectors.
    ///
    /// # Panics
    ///
    /// Panics if `min_vals` and `max_vals` have different lengths or invalid bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use domain_applied::__doctest_facade as math_explorer;
    /// use math_explorer::applied::game_theory::equilibrium::{BoxSet, ConvexSet};
    /// use math_explorer::nalgebra::DVector;
    ///
    /// let box_set = BoxSet::new(vec![0.0, -1.0], vec![1.0, 1.0]);
    ///
    /// // A point inside the box
    /// assert!(box_set.contains(&DVector::from_vec(vec![0.5, 0.0])));
    /// ```
    #[verified_engine::verified]
    pub fn new(min_vals: Vec<f64>, max_vals: Vec<f64>) -> Self {
        Self::try_new(min_vals, max_vals).unwrap_or_else(|e| panic!("{e}"))
    }
}

impl ConvexSet for BoxSet {
    #[verified_engine::verified]
    fn contains(&self, point: &DVector<f64>) -> bool {
        if point.len() != self.min_bounds.len() {
            return false;
        }
        for i in 0..point.len() {
            if point[i] < self.min_bounds[i] || point[i] > self.max_bounds[i] {
                return false;
            }
        }
        true
    }

    #[verified_engine::verified]
    fn is_convex(&self) -> bool {
        true // A box is always convex
    }
}

/// A set-valued function (correspondence) $\phi: S \to 2^S$.
///
/// In Nash Equilibrium analysis, the "Best Response" function is often a set-valued correspondence
/// (i.e., there may be multiple equally good strategies).
///
/// Kakutani's Fixed Point Theorem states that if $S$ is a non-empty, compact, convex subset of Euclidean space,
/// and $\phi: S \to 2^S$ is upper hemicontinuous with non-empty, convex images, then there exists a fixed point
/// $x^* \in \phi(x^*)$. This fixed point is the Nash Equilibrium.
pub trait Correspondence {
    /// Checks if `target` is in the image set $\phi(\text{source})$.
    /// effectively: `target` $\in \phi(\text{source})$
    #[verified_engine::verified]
    fn is_in_image(&self, source: &DVector<f64>, target: &DVector<f64>) -> bool;
}

/// Checks if $x^* \in \phi(x^*)$.
///
/// If this returns true, `point` represents a Nash Equilibrium (or a fixed point of the dynamical system).
#[verified_engine::verified]
pub fn is_fixed_point<C: Correspondence>(correspondence: &C, point: &DVector<f64>) -> bool {
    correspondence.is_in_image(point, point)
}

/// Example: Best Response correspondence in a simplified 2-player game.
///
/// This is a utility to demonstrate how one might wrap a Nash equilibrium check.
/// For a function $f(x)$, the correspondence is often defined as "all points close to $f(x)$".
pub struct BestResponseCorrespondence {
    /// A mapping function that returns the "ideal" best response.
    #[allow(clippy::type_complexity)]
    pub mapping: Box<dyn Fn(&DVector<f64>) -> DVector<f64>>,
    /// Tolerance for checking set membership (handling floating point inaccuracies).
    pub tolerance: f64,
}

impl Correspondence for BestResponseCorrespondence {
    #[verified_engine::verified]
    fn is_in_image(&self, source: &DVector<f64>, target: &DVector<f64>) -> bool {
        let expected = (self.mapping)(source);
        (target - expected).norm() <= self.tolerance
    }
}

/// Stack-allocated 1D best-response correspondence $\phi: \mathbb{R} \to 2^{\mathbb{R}}$
/// evaluating scalar fixed points without dynamic heap allocations.
///
/// # Examples
///
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::game_theory::equilibrium::{ScalarBestResponse, is_fixed_point};
/// use math_explorer::nalgebra::DVector;
///
/// // Symmetric 2-bidder first-price shading on Uniform(0, 100) with reserve r = 50 at v = 100
/// let best_response = ScalarBestResponse::new(|_b| 62.5, 0.5);
/// let eq_bid = DVector::from_vec(vec![62.5]);
/// let truthful_bid = DVector::from_vec(vec![100.0]);
/// assert!(is_fixed_point(&best_response, &eq_bid));
/// assert!(!is_fixed_point(&best_response, &truthful_bid));
/// ```
pub struct ScalarBestResponse<F> {
    /// Scalar best-response mapping $b^*(b)$ returning the optimal response on $\mathbb{R}$.
    pub mapping: F,
    /// Non-negative tolerance $\varepsilon \ge 0$ for set-membership $|y - b^*(x)| \le \varepsilon$.
    pub tolerance: f64,
}

impl<F: Fn(f64) -> f64> ScalarBestResponse<F> {
    /// Creates a new stack-allocated [`ScalarBestResponse`] with `tolerance` clamped to $\ge 0$.
    #[inline]
    #[verified_engine::verified]
    pub fn new(mapping: F, tolerance: f64) -> Self {
        Self {
            mapping,
            tolerance: if tolerance.is_finite() {
                tolerance.max(0.0)
            } else {
                0.0
            },
        }
    }

    /// Evaluates whether `target` lies within `tolerance` of `(self.mapping)(source)` on the stack.
    #[inline]
    #[verified_engine::verified]
    pub fn is_scalar_in_image(&self, source: f64, target: f64) -> bool {
        if !source.is_finite() || !target.is_finite() {
            return false;
        }
        let expected = (self.mapping)(source);
        expected.is_finite() && (target - expected).abs() <= self.tolerance
    }
}

impl<F: Fn(f64) -> f64> Correspondence for ScalarBestResponse<F> {
    #[verified_engine::verified]
    fn is_in_image(&self, source: &DVector<f64>, target: &DVector<f64>) -> bool {
        if source.len() != 1 || target.len() != 1 {
            return false;
        }
        self.is_scalar_in_image(source[0], target[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    #[verified_engine::verified]
    fn test_box_set_contains() {
        let box_set = BoxSet::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        assert!(box_set.contains(&DVector::from_vec(vec![0.5, 0.5])));
        assert!(!box_set.contains(&DVector::from_vec(vec![1.5, 0.5])));
    }

    #[test]
    #[verified_engine::verified]
    fn test_fixed_point() {
        // Define a simple mapping f(x) = 0.5 * x. Fixed point is 0.
        let correspondence = BestResponseCorrespondence {
            mapping: Box::new(|x| 0.5 * x),
            tolerance: math_commons::registry::TOLERANCE_FAST,
        };

        let point_zero = DVector::from_vec(vec![0.0]);
        let point_one = DVector::from_vec(vec![1.0]);

        // x=0 => f(0)=0. 0 is in {y | |y-0| < eps}. True.
        assert!(is_fixed_point(&correspondence, &point_zero));

        // x=1 => f(1)=0.5. 1 is NOT in {y | |y-0.5| < eps}. False.
        assert!(!is_fixed_point(&correspondence, &point_one));
    }

    #[test]
    #[verified_engine::verified]
    fn test_scalar_best_response_zero_alloc_fixed_point() {
        let eq_bid = DVector::from_vec(vec![62.5]);
        let shaded_off = DVector::from_vec(vec![100.0]);
        let wrong_dim = DVector::from_vec(vec![62.5, 62.5]);

        let _guard = crate::VERIFIED_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ((ok_eq, ok_off, ok_dim), metrics) = VerifiedEngine::run_verified(|| {
            let br = ScalarBestResponse::new(|b| 0.5 * b + 31.25, 1e-6);
            (
                is_fixed_point(&br, &eq_bid),
                is_fixed_point(&br, &shaded_off),
                is_fixed_point(&br, &wrong_dim),
            )
        });
        assert!(ok_eq);
        assert!(!ok_off);
        assert!(!ok_dim);
        assert!(metrics.function_calls >= 3);
    }

    #[test]
    fn test_lemke_howson_exact_fixed_point_and_polymatrix_regret_zero_alloc() {
        use engine_opt::ScalarWorkspace;

        // Asymmetric inspection / Battle-of-Sexes-style 2x3 bimatrix game + 3x3 Rock-Paper-Scissors.
        let a_asym = [3.0, 0.0, 1.0, 0.0, 2.0, 1.0];
        let b_asym = [2.0, 0.0, 3.0, 0.0, 3.0, 1.0];
        let mut sa = [0.0; 2];
        let mut sb = [0.0; 3];
        let mut lh_ws_buf = [0.0; bimatrix_nash_workspace_len(2, 3)];

        let counts = [2usize, 2, 2];
        let mut poly = [0.0; 36];
        // 3-player coordination network game with unique pure/mixed equilibria.
        for (p, q) in [(0usize, 1usize), (1, 2), (2, 0)] {
            poly[(2 * p) * 6 + (2 * q)] = 2.0;
            poly[(2 * p + 1) * 6 + (2 * q + 1)] = 1.0;
        }
        let mut poly_strat = [0.0; 6];
        let mut poly_ws_buf = [0.0; polymatrix_nash_workspace_len(6)];

        let ((lh_stats, poly_stats), _) = {
            let _guard = crate::VERIFIED_TEST_LOCK
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            VerifiedEngine::run_verified(|| {
                let mut ws1 = ScalarWorkspace::new(&mut lh_ws_buf);
                let s1 =
                    solve_bimatrix_nash_into(&a_asym, &b_asym, 2, 3, &mut sa, &mut sb, &mut ws1)
                        .expect("lemke-howson must succeed");

                let mut ws2 = ScalarWorkspace::new(&mut poly_ws_buf);
                let s2 = solve_polymatrix_nash_into(
                    &counts,
                    &poly,
                    &PolymatrixConfig {
                        max_iterations: 2_500,
                        tolerance: 1.0e-6,
                        cfr_plus: true,
                    },
                    &mut poly_strat,
                    &mut ws2,
                )
                .expect("polymatrix regret matching must succeed");
                (s1, s2)
            })
        };

        assert!(lh_stats.max_regret < 1.0e-10);
        let joint = DVector::from_iterator(5, sa.iter().chain(sb.iter()).copied());
        let br = BimatrixBestResponse {
            payoff_a: &a_asym,
            payoff_b: &b_asym,
            rows: 2,
            cols: 3,
            tolerance: 1.0e-10,
        };
        assert!(is_fixed_point(&br, &joint));
        assert!(br.unilateral_regret(&sa, &sb) < 1.0e-10);

        let non_eq = DVector::from_vec(vec![0.5, 0.5, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        assert!(!is_fixed_point(&br, &non_eq));

        assert!(poly_stats.converged);
        assert!(poly_stats.exploitability < 1.0e-5);
    }
}
