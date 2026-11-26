use super::types::FavoritismInputs;
use nalgebra::{DMatrix, DVector};
use pure_math::pure_math::analysis::integration::Integrator;
use rand::{Rng, RngCore};

const EPSILON: f64 = math_commons::registry::TOLERANCE_STANDARD;

/// Context provided to scoring strategies during evaluation.
///
/// We use `dyn RngCore` to be object-safe regarding the RNG, but we must stay generic
/// over `I` because `Integrator` is not object-safe.
pub struct ScoringContext<'a, I: Integrator + ?Sized> {
    /// Mutable pseudo-random generator used for stochastic mood perturbations.
    pub rng: &'a mut dyn RngCore,
    /// Numerical quadrature backend evaluating proximity and sibling-effort integrals.
    pub integrator: &'a I,
    /// Singularity-guarded baseline distance $x_0$ (`|x_0| >= TOLERANCE_STANDARD`).
    pub safe_x0: f64,
    /// Non-negative integration horizon $t \ge 0$.
    pub safe_t: f64,
}

impl<'a, I: Integrator + ?Sized> ScoringContext<'a, I> {
    /// Constructs a sanitized evaluation context clamping near-zero distances and negative times.
    #[verified_engine::verified]
    pub fn new(inputs: &FavoritismInputs, rng: &'a mut dyn RngCore, integrator: &'a I) -> Self {
        let safe_x0 = if inputs.time.x_0.abs() < EPSILON {
            EPSILON
        } else {
            inputs.time.x_0
        };
        let safe_t = inputs.time.t.max(0.0);

        Self {
            rng,
            integrator,
            safe_x0,
            safe_t,
        }
    }
}

/// Multiplicative composite model evaluating the Unified Favoritism Theory (`UFT`) functional.
///
/// Evaluates the quadrature-normalized score ratio over horizon $[0, t]$:
///
/// $$F(t) = \frac{\mathcal{N}(t)\,\mathcal{E}(t)\,\det(\mathbf{G})\,(\mathbf{c}^\top \mathbf{w}_c)\,\ln(1 + f_0)\,\mathcal{P}\,\mathcal{S}\,e^{-\lambda \Delta t}\,\xi}{\int_0^t \sum_{s \in \text{siblings}} d_s(\tau)^{-1}\,\mathrm{d}\tau}$$
///
/// Reference: Clenshaw & Curtis (1960), DOI: 10.1007/BF01386223; see
/// [`docs/math_specs/domain_applied.md`](../../../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Near-zero distances $|x_0|, |d_s| < \epsilon$ and near-zero sibling integrals are clamped to
/// `TOLERANCE_STANDARD` ($10^{-10}$) or `1.0` so neither quadrature nor quotient division overflows.
///
/// # Domain of Validity
/// Finite [`FavoritismInputs`] with $t \ge 0$, $1 + f_0 > 0$, and convergent 1D [`Integrator`] backend `I`.
pub struct UnifiedFavoritismModel<I: Integrator + ?Sized> {
    /// Multiplicative factor strategies multiplied into the score numerator.
    pub numerator_strategies: Vec<Box<dyn ScoringStrategy<I>>>,
    /// Normalization factor strategies multiplied into the score denominator.
    pub denominator_strategies: Vec<Box<dyn ScoringStrategy<I>>>,
}

impl<I: Integrator + ?Sized> Default for UnifiedFavoritismModel<I> {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            numerator_strategies: vec![
                Box::new(ProximityStrategy),
                Box::new(EmotionalSupportStrategy),
                Box::new(GiftStrategy),
                Box::new(ComplimentStrategy),
                Box::new(ContactFrequencyStrategy),
                Box::new(PersonalityStrategy),
                Box::new(SocialMultiplierStrategy),
                Box::new(DecayStrategy),
                Box::new(StochasticStrategy),
            ],
            denominator_strategies: vec![Box::new(SiblingCompetitionStrategy)],
        }
    }
}

impl<I: Integrator + ?Sized> UnifiedFavoritismModel<I> {
    /// Calculates the favoritism score using the composed strategies.
    #[verified_engine::verified]
    pub fn calculate(
        &self,
        inputs: &FavoritismInputs,
        rng: &mut dyn RngCore,
        integrator: &I,
    ) -> f64 {
        let mut ctx = ScoringContext::new(inputs, rng, integrator);

        let mut numerator = 1.0;
        for strategy in &self.numerator_strategies {
            numerator *= strategy.calculate(inputs, &mut ctx);
        }

        let mut denominator = 1.0;
        for strategy in &self.denominator_strategies {
            denominator *= strategy.calculate(inputs, &mut ctx);
        }

        numerator / denominator
    }
}

/// A strategy for calculating a component of the favoritism score.
pub trait ScoringStrategy<I: Integrator + ?Sized> {
    /// Evaluates this multiplicative score factor for `inputs` within `ctx`.
    #[verified_engine::verified]
    fn calculate(&self, inputs: &FavoritismInputs, ctx: &mut ScoringContext<I>) -> f64;
}

// --- Numerator Strategies ---

/// Integrates inverse baseline distance $\int_0^t x_0^{-1}\,\mathrm{d}\tau$ over $[0, t]$.
pub struct ProximityStrategy;
impl<I: Integrator + ?Sized> ScoringStrategy<I> for ProximityStrategy {
    #[verified_engine::verified]
    fn calculate(&self, _inputs: &FavoritismInputs, ctx: &mut ScoringContext<I>) -> f64 {
        ctx.integrator
            .integrate(|_t| 1.0 / ctx.safe_x0, 0.0, ctx.safe_t, EPSILON)
            .value
    }
}

/// Integrates cumulative emotional-support effort over $[0, t] \times [0, 1]$.
pub struct EmotionalSupportStrategy;
impl<I: Integrator + ?Sized> ScoringStrategy<I> for EmotionalSupportStrategy {
    #[verified_engine::verified]
    fn calculate(&self, _inputs: &FavoritismInputs, ctx: &mut ScoringContext<I>) -> f64 {
        ctx.integrator
            .integrate(
                |_t| ctx.integrator.integrate(|_x| 8.0, 0.0, 1.0, EPSILON).value,
                0.0,
                ctx.safe_t,
                EPSILON,
            )
            .value
    }
}

/// Computes the $2 \times 2$ diagonal gift matrix determinant $g_{\mathrm{emotional}} \cdot g_{\mathrm{practical}}$.
pub struct GiftStrategy;
impl<I: Integrator + ?Sized> ScoringStrategy<I> for GiftStrategy {
    #[verified_engine::verified]
    fn calculate(&self, inputs: &FavoritismInputs, _ctx: &mut ScoringContext<I>) -> f64 {
        let gift_matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![
            inputs.gifts.g_emotional,
            inputs.gifts.g_practical,
        ]));
        gift_matrix.determinant()
    }
}

/// Evaluates the weighted inner product $\mathbf{c}^\top \mathbf{w}_c$ of compliment counts and weights.
pub struct ComplimentStrategy;
impl<I: Integrator + ?Sized> ScoringStrategy<I> for ComplimentStrategy {
    #[verified_engine::verified]
    fn calculate(&self, inputs: &FavoritismInputs, _ctx: &mut ScoringContext<I>) -> f64 {
        inputs
            .compliments
            .compliments
            .dot(&inputs.compliments.compliment_weights)
    }
}

/// Evaluates logarithmic diminishing returns $\ln(1 + f_0)$ of contact frequency.
pub struct ContactFrequencyStrategy;
impl<I: Integrator + ?Sized> ScoringStrategy<I> for ContactFrequencyStrategy {
    #[verified_engine::verified]
    fn calculate(&self, inputs: &FavoritismInputs, _ctx: &mut ScoringContext<I>) -> f64 {
        (1.0 + inputs.contact.f_initial).max(EPSILON).ln()
    }
}

/// Computes the linear combination of intelligence, sensitivity, wealth, and talent traits.
pub struct PersonalityStrategy;
impl<I: Integrator + ?Sized> ScoringStrategy<I> for PersonalityStrategy {
    #[verified_engine::verified]
    fn calculate(&self, inputs: &FavoritismInputs, _ctx: &mut ScoringContext<I>) -> f64 {
        inputs.personality.w_i * inputs.personality.intelligence
            + inputs.personality.w_es * inputs.personality.emotional_sensitivity
            + inputs.personality.w_w * inputs.personality.wealth
            + inputs.personality.w_t * inputs.personality.talent
    }
}

/// Applies birth-order, life-event, crisis-assistance, and visibility multipliers.
pub struct SocialMultiplierStrategy;
impl<I: Integrator + ?Sized> ScoringStrategy<I> for SocialMultiplierStrategy {
    #[verified_engine::verified]
    fn calculate(&self, inputs: &FavoritismInputs, _ctx: &mut ScoringContext<I>) -> f64 {
        let mut multiplier = inputs.social.birth_order_weight * inputs.social.major_life_events;

        // h: Crisis Multiplier
        if inputs.social.helped_during_crisis {
            multiplier *= 1.5;
        }

        // s: Visibility Multiplier
        if inputs.social.active_on_social_media {
            multiplier *= 1.3;
        }

        multiplier
    }
}

/// Evaluates exponential memory decay $\exp(-\lambda \Delta t)$ since last contact.
pub struct DecayStrategy;
impl<I: Integrator + ?Sized> ScoringStrategy<I> for DecayStrategy {
    #[verified_engine::verified]
    fn calculate(&self, inputs: &FavoritismInputs, _ctx: &mut ScoringContext<I>) -> f64 {
        (-inputs.contact.decay_constant * inputs.contact.time_since_last_contact).exp()
    }
}

/// Samples a uniform mood-swing perturbation $\xi \sim \mathcal{U}(0.9, 1.1)$.
pub struct StochasticStrategy;
impl<I: Integrator + ?Sized> ScoringStrategy<I> for StochasticStrategy {
    #[verified_engine::verified]
    fn calculate(&self, _inputs: &FavoritismInputs, ctx: &mut ScoringContext<I>) -> f64 {
        // Rng is implemented for &mut dyn RngCore, so we can call gen_range
        // However, gen_range is in the Rng trait which is not object safe?
        // Wait, gen_range is a default method on Rng.
        // If we have `dyn RngCore`, we can wrap it in something that implements `Rng`?
        // Actually, `dyn RngCore` implements `Rng`.
        ctx.rng.gen_range(0.9..1.1)
    }
}

// --- Denominator Strategies ---

/// Integrates aggregate inverse sibling distances $\int_0^t \sum_s d_s^{-1}\,\mathrm{d}\tau$ as competitive normalization.
pub struct SiblingCompetitionStrategy;
impl<I: Integrator + ?Sized> ScoringStrategy<I> for SiblingCompetitionStrategy {
    #[verified_engine::verified]
    fn calculate(&self, inputs: &FavoritismInputs, ctx: &mut ScoringContext<I>) -> f64 {
        let sibling_proximity_integral = ctx
            .integrator
            .integrate(
                |_t| {
                    inputs
                        .family
                        .sibling_distances
                        .iter()
                        .map(|distance| {
                            let safe_distance = if distance.abs() < EPSILON {
                                EPSILON
                            } else {
                                *distance
                            };
                            1.0 / safe_distance
                        })
                        .sum()
                },
                0.0,
                ctx.safe_t,
                EPSILON,
            )
            .value;

        if sibling_proximity_integral.abs() < EPSILON {
            1.0
        } else {
            sibling_proximity_integral
        }
    }
}
