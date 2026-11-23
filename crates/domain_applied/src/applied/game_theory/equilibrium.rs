use crate::error::{AppliedError, GameTheoryError};
use nalgebra::DVector;
use pure_math::pure_math::analysis::optimization::{
    BoxBounds, ConstrainedSolution, ProjectedGradientConfig, ProjectedGradientSolver,
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
    #[allow(missing_docs)]
    pub min_bounds: DVector<f64>,
    #[allow(missing_docs)]
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

    /// Converts this strategy [`BoxSet`] into a validated [`BoxBounds`] constraint.
    #[verified_engine::verified]
    pub fn to_box_bounds(&self) -> Result<BoxBounds, AppliedError> {
        Ok(BoxBounds::new(
            self.min_bounds.clone(),
            self.max_bounds.clone(),
        )?)
    }

    /// Projects `point` onto $[min, max]^n$ in-place with zero heap allocations.
    #[verified_engine::verified]
    pub fn project_in_place(&self, point: &mut DVector<f64>) {
        assert_eq!(point.len(), self.min_bounds.len());
        for i in 0..point.len() {
            let low = self.min_bounds[i].min(self.max_bounds[i]);
            let high = self.min_bounds[i].max(self.max_bounds[i]);
            point[i] = point[i].clamp(low, high);
        }
    }

    /// Returns the Euclidean projection $P_{\text{Box}}(\text{point})$ onto this strategy box.
    #[verified_engine::verified]
    pub fn project(&self, point: &DVector<f64>) -> DVector<f64> {
        let mut projected = point.clone();
        self.project_in_place(&mut projected);
        projected
    }

    /// Computes a fixed point $x^* = P_{\text{Box}}(\phi(x^*))$ via Krasnoselskii-Mann
    /// damped projected iteration $x_{k+1} = P_{\text{Box}}((1 - \alpha) x_k + \alpha \phi(x_k))$.
    #[verified_engine::verified]
    pub fn find_projected_fixed_point<F>(
        &self,
        mut response_map: F,
        initial: &DVector<f64>,
        damping: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<DVector<f64>, AppliedError>
    where
        F: FnMut(&DVector<f64>) -> DVector<f64>,
    {
        let bounds = self.to_box_bounds()?;
        let mut x = bounds.project(initial);
        let mut next = DVector::zeros(x.len());
        let alpha = damping.clamp(1e-3, 1.0);
        for _ in 0..max_iterations {
            let br = response_map(&x);
            let mut max_step = 0.0_f64;
            for i in 0..x.len() {
                let proj_br = br[i].clamp(bounds.lower[i], bounds.upper[i]);
                let candidate = (1.0 - alpha) * x[i] + alpha * proj_br;
                max_step = max_step.max((candidate - x[i]).abs());
                next[i] = candidate;
            }
            x.copy_from(&next);
            if max_step <= tolerance {
                break;
            }
        }
        Ok(x)
    }

    /// Minimizes a smooth potential game objective over this [`BoxSet`] via [`ProjectedGradientSolver`].
    #[verified_engine::verified]
    pub fn minimize_potential<F>(
        &self,
        potential_and_grad: F,
        initial: &DVector<f64>,
        config: &ProjectedGradientConfig,
    ) -> Result<ConstrainedSolution, AppliedError>
    where
        F: FnMut(&DVector<f64>, &mut DVector<f64>) -> f64,
    {
        let bounds = self.to_box_bounds()?;
        Ok(ProjectedGradientSolver::new(*config).solve(potential_and_grad, &bounds, initial)?)
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_box_set_projection_and_projected_equilibrium() {
        let box_set = BoxSet::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let outside = DVector::from_vec(vec![-0.4, 1.7]);
        let projected = box_set.project(&outside);
        assert!((projected[0] - 0.0).abs() < 1e-12);
        assert!((projected[1] - 1.0).abs() < 1e-12);
        assert!(box_set.contains(&projected));

        let eq = box_set
            .find_projected_fixed_point(
                |x| DVector::from_vec(vec![1.5 - 0.5 * x[1], 0.8 - 0.25 * x[0]]),
                &DVector::from_vec(vec![0.1, 0.9]),
                0.6,
                200,
                1e-9,
            )
            .expect("fixed point");
        assert!((eq[0] - 1.0).abs() < 1e-6);
        assert!((eq[1] - 0.55).abs() < 1e-6);
    }
}
