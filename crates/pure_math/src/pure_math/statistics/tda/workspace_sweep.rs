//! Persistence barcode sweep and union-find helpers for [`super::workspace::TdaFiltrationWorkspace`].

use super::persistence::PersistenceInterval;
use super::workspace::{DEFAULT_RADII_SCRATCH_CAP, FiltrationStepSummary, TdaFiltrationWorkspace};
use crate::error::TdaError;

pub(super) fn uf_find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    let mut curr = x;
    while curr != root {
        let next = parent[curr];
        parent[curr] = root;
        curr = next;
    }
    root
}

pub(super) fn uf_union(
    parent: &mut [usize],
    rank: &mut [usize],
    components: &mut usize,
    x: usize,
    y: usize,
) -> bool {
    let rx = uf_find(parent, x);
    let ry = uf_find(parent, y);
    if rx == ry {
        return false;
    }
    if rank[rx] < rank[ry] {
        parent[rx] = ry;
    } else if rank[rx] > rank[ry] {
        parent[ry] = rx;
    } else {
        parent[ry] = rx;
        rank[rx] += 1;
    }
    *components -= 1;
    true
}

impl TdaFiltrationWorkspace {
    /// Sweeps `radii` and writes `PersistenceInterval`s into `out` with zero heap allocations
    /// when `out` has pre-reserved capacity.
    ///
    /// # Errors
    ///
    /// Returns [`TdaError::InvalidRadius`] if `radii` is empty or contains invalid values.
    #[verified_engine::verified]
    pub fn compute_persistence_into(
        &mut self,
        n: usize,
        radii: &[f64],
        out: &mut Vec<PersistenceInterval>,
    ) -> Result<(), TdaError> {
        if radii.is_empty() {
            return Err(TdaError::InvalidRadius { value: 0.0 });
        }
        let mut sorted = true;
        for (idx, &r) in radii.iter().enumerate() {
            if r < 0.0 || !r.is_finite() {
                return Err(TdaError::InvalidRadius { value: r });
            }
            if idx > 0 && r < radii[idx - 1] {
                sorted = false;
            }
        }
        if n == 0 || n > self.max_points {
            return Err(TdaError::EmptyPointCloud);
        }

        out.clear();
        self.comp_births_len = 0;
        self.hole_births_len = 0;
        self.prev_radius = None;

        if sorted {
            self.sweep_monotonic_radii(n, radii, out)
        } else if radii.len() <= DEFAULT_RADII_SCRATCH_CAP {
            let len = radii.len();
            {
                let scratch = &mut self.arena.region_mut(self.radii_scratch_reg)[..len];
                scratch.copy_from_slice(radii);
                scratch.sort_unstable_by(f64::total_cmp);
            }
            self.sweep_scratch_radii(n, len, out)
        } else {
            let mut owned = radii.to_vec();
            owned.sort_unstable_by(f64::total_cmp);
            self.sweep_monotonic_radii(n, &owned, out)
        }
    }

    #[verified_engine::verified]
    fn record_step_births_and_deaths(
        &mut self,
        prev_b0: usize,
        prev_b1: usize,
        step: FiltrationStepSummary,
        r: f64,
        out: &mut Vec<PersistenceInterval>,
    ) -> Result<(), TdaError> {
        update_births_and_deaths(
            self.arena.region_mut(self.comp_births_reg),
            &mut self.comp_births_len,
            prev_b0,
            step.beta0,
            r,
            0,
            out,
        )?;
        update_births_and_deaths(
            self.arena.region_mut(self.hole_births_reg),
            &mut self.hole_births_len,
            prev_b1,
            step.beta1,
            r,
            1,
            out,
        )
    }

    #[verified_engine::verified]
    fn sweep_monotonic_radii(
        &mut self,
        n: usize,
        radii: &[f64],
        out: &mut Vec<PersistenceInterval>,
    ) -> Result<(), TdaError> {
        let mut prev_b0 = 0usize;
        let mut prev_b1 = 0usize;
        for &r in radii {
            let step = self.evaluate_radius(n, r)?;
            self.record_step_births_and_deaths(prev_b0, prev_b1, step, r, out)?;
            prev_b0 = step.beta0;
            prev_b1 = step.beta1;
        }
        let comp = &self.arena.region(self.comp_births_reg)[..self.comp_births_len];
        let hole = &self.arena.region(self.hole_births_reg)[..self.hole_births_len];
        flush_essential_births(comp, hole, radii[radii.len() - 1], out)
    }

    #[verified_engine::verified]
    fn sweep_scratch_radii(
        &mut self,
        n: usize,
        len: usize,
        out: &mut Vec<PersistenceInterval>,
    ) -> Result<(), TdaError> {
        let mut prev_b0 = 0usize;
        let mut prev_b1 = 0usize;
        for idx in 0..len {
            let r = self.arena.region(self.radii_scratch_reg)[idx];
            let step = self.evaluate_radius(n, r)?;
            self.record_step_births_and_deaths(prev_b0, prev_b1, step, r, out)?;
            prev_b0 = step.beta0;
            prev_b1 = step.beta1;
        }
        let final_r = self.arena.region(self.radii_scratch_reg)[len - 1];
        let comp = &self.arena.region(self.comp_births_reg)[..self.comp_births_len];
        let hole = &self.arena.region(self.hole_births_reg)[..self.hole_births_len];
        flush_essential_births(comp, hole, final_r, out)
    }
}

#[verified_engine::verified]
fn flush_essential_births(
    comp_births: &[f64],
    hole_births: &[f64],
    final_r: f64,
    out: &mut Vec<PersistenceInterval>,
) -> Result<(), TdaError> {
    for &birth in comp_births {
        if birth < final_r {
            out.push(PersistenceInterval::new(birth, final_r, 0)?);
        }
    }
    for &birth in hole_births {
        if birth < final_r {
            out.push(PersistenceInterval::new(birth, final_r, 1)?);
        }
    }
    Ok(())
}

fn update_births_and_deaths(
    births: &mut [f64],
    births_len: &mut usize,
    prev_betti: usize,
    curr_betti: usize,
    radius: f64,
    dim: usize,
    out: &mut Vec<PersistenceInterval>,
) -> Result<(), TdaError> {
    if curr_betti > prev_betti {
        for _ in 0..(curr_betti - prev_betti) {
            births[*births_len] = radius;
            *births_len += 1;
        }
    } else if curr_betti < prev_betti {
        for _ in 0..(prev_betti - curr_betti) {
            if *births_len > 0 {
                *births_len -= 1;
                let birth = births[*births_len];
                if birth < radius {
                    out.push(PersistenceInterval::new(birth, radius, dim)?);
                }
            }
        }
    }
    Ok(())
}

// [cite:clinical_trials_statistics]
