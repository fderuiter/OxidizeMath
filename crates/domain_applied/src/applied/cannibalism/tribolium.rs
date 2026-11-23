//! Discrete stage-structured *Tribolium* Larvae-Pupae-Adults (LPA) cannibalism map.
// [cite:cannibalism]

/// Cushing-Costantino *Tribolium* Larvae-Pupae-Adults (LPA) stage-structured cannibalism model.
///
/// Advances `[L, P, A]` cohorts across discrete 2-week stage intervals under egg cannibalism by
/// larvae ($c_{el}$) and adults ($c_{ea}$) and callow/pupal cannibalism by adults ($c_{pa}$):
///
/// $$L_{t+1} = b A_t \exp(-c_{el} L_t - c_{ea} A_t), \quad P_{t+1} = (1 - \mu_l) L_t, \quad A_{t+1} = P_t \exp(-c_{pa} A_t) + (1 - \mu_a) A_t$$
///
/// # Example
///
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::cannibalism::TriboliumLpaModel;
///
/// let model = TriboliumLpaModel::two_cycle_preset(0.7);
/// let mut state = [1.5, 0.5, 1.0];
/// for _ in 0..200 {
///     state = model.step(state);
/// }
/// assert!(state[2] > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriboliumLpaModel {
    /// Larval recruitment rate per adult ($b \ge 0$).
    pub b: f64,
    /// Egg cannibalism coefficient by feeding larvae ($c_{el} \ge 0$).
    pub c_el: f64,
    /// Egg cannibalism coefficient by adults ($c_{ea} \ge 0$).
    pub c_ea: f64,
    /// Callow/pupal cannibalism coefficient by adults ($c_{pa} \ge 0$).
    pub c_pa: f64,
    /// Natural larval stage mortality fraction $\mu_l \in [0, 1]$.
    pub mu_l: f64,
    /// Natural adult stage mortality fraction $\mu_a \in [0, 1]$.
    pub mu_a: f64,
}

impl TriboliumLpaModel {
    /// Constructs a validated `TriboliumLpaModel` clamping rates to physical domains.
    #[verified_engine::verified]
    pub fn new(b: f64, c_el: f64, c_ea: f64, c_pa: f64, mu_l: f64, mu_a: f64) -> Self {
        Self {
            b: if b.is_finite() { b.max(0.0) } else { 0.0 },
            c_el: if c_el.is_finite() { c_el.max(0.0) } else { 0.0 },
            c_ea: if c_ea.is_finite() { c_ea.max(0.0) } else { 0.0 },
            c_pa: if c_pa.is_finite() { c_pa.max(0.0) } else { 0.0 },
            mu_l: if mu_l.is_finite() {
                mu_l.clamp(0.0, 1.0)
            } else {
                0.2
            },
            mu_a: if mu_a.is_finite() {
                mu_a.clamp(0.0, 1.0)
            } else {
                0.7
            },
        }
    }

    /// Canonical *Tribolium* period-doubling preset exhibiting synchronous 2-cycles at $c_{pa} = 0.7$
    /// (`A` alternating near `[0.88, 2.94]`) and period-doubling chaos for larger $c_{pa}$.
    #[verified_engine::verified]
    pub fn two_cycle_preset(c_pa: f64) -> Self {
        Self::new(11.9, 1.73, 0.50, c_pa, 0.267, 0.70)
    }

    /// Costantino et al. (1997) *Science* experimental *Tribolium castaneum* parameter set
    /// (scaled per 100 beetles) exhibiting invariant-loop and chaotic attractors near $c_{pa} = 0.7$.
    #[verified_engine::verified]
    pub fn costantino_science_1997(c_pa: f64) -> Self {
        Self::new(10.45, 1.731, 1.310, c_pa, 0.20, 0.96)
    }

    /// Advances `[L, P, A]` by one discrete generation step on the stack with zero heap allocations.
    #[verified_engine::verified]
    pub fn step(&self, state: [f64; 3]) -> [f64; 3] {
        let l = if state[0].is_finite() {
            state[0].max(0.0)
        } else {
            0.0
        };
        let p = if state[1].is_finite() {
            state[1].max(0.0)
        } else {
            0.0
        };
        let a = if state[2].is_finite() {
            state[2].max(0.0)
        } else {
            0.0
        };

        let next_l = self.b * a * (-self.c_el * l - self.c_ea * a).exp();
        let next_p = (1.0 - self.mu_l) * l;
        let next_a = p * (-self.c_pa * a).exp() + (1.0 - self.mu_a) * a;
        debug_assert!(next_l >= 0.0 && next_p >= 0.0 && next_a >= 0.0);

        [next_l, next_p, next_a]
    }
}
