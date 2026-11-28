//! Smooth 3D Alexander torus closures, formatting, and trait impls for [`BraidWord`].

// [cite:knot_braid_topology]

use super::BraidWord;
use crate::pure_math::algebra::traits::Semigroup;
use nalgebra::Point3;
use std::f64::consts::{PI, TAU};
use std::fmt;

impl BraidWord {
    /// Evaluates the smooth $C^2$ 3D Alexander torus closure $\hat{\beta}$, returning one closed
    /// polyline per link component.
    #[must_use]
    #[verified_engine::verified]
    pub fn alexander_closure_3d(
        &self,
        major_radius: f64,
        strand_spacing: f64,
        samples_per_turn: usize,
    ) -> Vec<Vec<Point3<f64>>> {
        let r_maj = major_radius.max(0.6);
        let max_spread = (r_maj * 0.55) / (self.strands.max(2) as f64);
        let spacing = strand_spacing.clamp(0.05, max_spread.max(0.06));
        let per_turn = samples_per_turn.max(24);
        let perm = self.strand_permutation();
        let mut visited = vec![false; self.strands];
        let mut components = Vec::new();

        for start_slot in 0..self.strands {
            if visited[start_slot] {
                continue;
            }
            let mut cycle_slots = Vec::new();
            let mut cur = start_slot;
            let mut guard = 0;
            while !visited[cur] && guard <= self.strands {
                visited[cur] = true;
                cycle_slots.push(cur);
                cur = perm.map[cur];
                guard += 1;
            }
            let mut pts = Vec::with_capacity(per_turn * cycle_slots.len());
            for &initial_slot in &cycle_slots {
                for s in 0..per_turn {
                    let frac = (s as f64) / (per_turn as f64);
                    pts.push(self.eval_closure_turn_point(initial_slot, frac, r_maj, spacing));
                }
            }
            components.push(pts);
        }
        components
    }

    #[verified_engine::verified]
    fn eval_closure_turn_point(
        &self,
        initial_slot: usize,
        turn_frac: f64,
        major_radius: f64,
        spacing: f64,
    ) -> Point3<f64> {
        let theta = TAU * turn_frac;
        let center_shift = 0.5 * ((self.strands as f64) - 1.0);
        if self.generators.is_empty() {
            let u = spacing * ((initial_slot as f64) - center_shift);
            let rad = major_radius + u;
            return Point3::new(rad * theta.cos(), rad * theta.sin(), 0.0);
        }

        let m = self.generators.len();
        let scaled = (turn_frac * (m as f64)).clamp(0.0, (m as f64) - 1e-12);
        let step_idx = (scaled.floor() as usize).min(m - 1);
        let local_t = scaled - (step_idx as f64);

        let mut slot = initial_slot;
        for &g in &self.generators[..step_idx] {
            let left = (g.unsigned_abs() as usize) - 1;
            if slot == left {
                slot = left + 1;
            } else if slot == left + 1 {
                slot = left;
            }
        }

        let g = self.generators[step_idx];
        let left = (g.unsigned_abs() as usize) - 1;
        let smooth = local_t * local_t * local_t * (10.0 + local_t * (-15.0 + 6.0 * local_t));
        let angle = PI * smooth;
        let (u, v) = if slot == left {
            let u_mid = spacing * ((left as f64) + 0.5 - center_shift);
            let r0 = 0.5 * spacing;
            let sign = f64::from(g.signum());
            (u_mid - r0 * angle.cos(), sign * r0 * angle.sin())
        } else if slot == left + 1 {
            let u_mid = spacing * ((left as f64) + 0.5 - center_shift);
            let r0 = 0.5 * spacing;
            let sign = f64::from(g.signum());
            (u_mid + r0 * angle.cos(), -sign * r0 * angle.sin())
        } else {
            (spacing * ((slot as f64) - center_shift), 0.0)
        };
        let rad = major_radius + u;
        Point3::new(rad * theta.cos(), rad * theta.sin(), v)
    }

    /// Formats the braid word using ASCII notation (`"e"` or `"s1 * s2^-1"`).
    #[must_use]
    pub fn format_ascii(&self) -> String {
        if self.generators.is_empty() {
            return "e".to_string();
        }
        self.generators
            .iter()
            .map(|&g| {
                let idx = g.unsigned_abs();
                if g > 0 {
                    format!("s{idx}")
                } else {
                    format!("s{idx}^-1")
                }
            })
            .collect::<Vec<_>>()
            .join(" * ")
    }
}

impl Semigroup for BraidWord {
    #[verified_engine::verified]
    fn operate(a: &Self, b: &Self) -> Self {
        a.try_operate(b)
            .expect("BraidWord::operate failed strand count or MAX_BRAID_WORD_LEN validation")
    }
}

impl fmt::Display for BraidWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format_ascii())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn braid_composition_cancellation_relations_and_closure() {
        let mut b = BraidWord::new(4, vec![1, 3, -3, 2, 1]).unwrap();
        assert_eq!(b.free_reduce_in_place(), 1);
        assert_eq!(b.generators(), &[1, 2, 1]);
        assert!(b.can_apply_artin_relation(0));
        b.apply_artin_relation(0).unwrap();
        assert_eq!(b.generators(), &[2, 1, 2]);

        let mut comm = BraidWord::new(4, vec![1, 3]).unwrap();
        assert!(comm.can_apply_far_commutativity(0));
        comm.apply_far_commutativity(0).unwrap();
        assert_eq!(comm.generators(), &[3, 1]);

        let inv_prod = b.compose(&b.inverse()).unwrap().free_reduce();
        assert!(inv_prod.is_empty());

        let hopf = BraidWord::new(2, vec![1, 1]).unwrap();
        assert_eq!(hopf.component_count(), 2);
        let closed = hopf.alexander_closure_3d(2.0, 0.5, 48);
        assert_eq!(closed.len(), 2);
        assert_eq!(closed[0].len(), 48);
    }
}
