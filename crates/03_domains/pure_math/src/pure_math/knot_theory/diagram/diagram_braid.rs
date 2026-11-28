//! Braid-closure planar diagram construction with concentric annular strand routing.

// [cite:knot_braid_topology]

use super::{Crossing, CrossingSign, DiagramSegment, Endpoint, PlanarDiagram};
use std::f64::consts::TAU;

#[verified_engine::verified]
fn strand_radius(slot: usize, n: usize) -> f64 {
    let span = (n.max(2) - 1) as f64;
    0.44 + 0.40 * ((slot as f64) / span)
}

#[verified_engine::verified]
fn crossing_radius(left: usize, n: usize) -> f64 {
    0.5 * (strand_radius(left, n) + strand_radius(left + 1, n))
}

#[verified_engine::verified]
fn annular_track_waypoints(
    r_track: f64,
    step_tail: usize,
    step_head: usize,
    c_count: usize,
) -> Vec<[f64; 2]> {
    let step_angle = TAU / (c_count.max(1) as f64);
    let theta_tail = step_angle * (step_tail as f64);
    let theta_head = step_angle * (step_head as f64);
    let mut span = (theta_head - theta_tail).rem_euclid(TAU);
    if span < 1e-9 {
        span = TAU;
    }
    let collar = (0.25 * span).min(0.32 * step_angle).min(0.22);
    let inner_span = (span - 2.0 * collar).max(0.0);
    let mid_div = ((inner_span / 0.36).ceil() as usize).clamp(1, 14);
    let mut waypoints = Vec::with_capacity(mid_div + 2);
    let a_start = theta_tail + collar;
    waypoints.push([r_track * a_start.cos(), r_track * a_start.sin()]);
    if inner_span > 1e-6 {
        if mid_div == 1 {
            let a_mid = theta_tail + 0.5 * span;
            waypoints.push([r_track * a_mid.cos(), r_track * a_mid.sin()]);
        } else {
            for k in 1..mid_div {
                let a = a_start + inner_span * (k as f64) / (mid_div as f64);
                waypoints.push([r_track * a.cos(), r_track * a.sin()]);
            }
        }
    }
    let a_end = theta_tail + span - collar;
    if (a_end - a_start).abs() > 1e-6 {
        waypoints.push([r_track * a_end.cos(), r_track * a_end.sin()]);
    }
    waypoints
}

#[verified_engine::verified]
pub(super) fn build_braid_closure_diagram(n: usize, gens: &[i32]) -> PlanarDiagram {
    let n = n.max(2);
    if gens.is_empty()
        || gens.iter().any(|&g| {
            let idx = g.unsigned_abs() as usize;
            idx == 0 || idx >= n
        })
    {
        return PlanarDiagram {
            crossings: Vec::new(),
            segments: Vec::new(),
            free_loops: n,
        };
    }

    let c_count = gens.len();
    let mut crossings = Vec::with_capacity(c_count);
    let mut first_in_slot: Vec<Option<Endpoint>> = vec![None; n];
    let mut last_out_at_slot: Vec<Option<Endpoint>> = vec![None; n];
    let mut initial_slot_at_curr: Vec<usize> = (0..n).collect();
    let mut segments: Vec<DiagramSegment> = Vec::with_capacity(2 * c_count);

    for (step, &g) in gens.iter().enumerate() {
        let left = (g.unsigned_abs() as usize) - 1;
        let right = left + 1;
        let sign = if g > 0 {
            CrossingSign::Positive
        } else {
            CrossingSign::Negative
        };
        let theta = TAU * (step as f64) / (c_count as f64);
        let r = crossing_radius(left, n);
        crossings.push(Crossing {
            sign,
            slots: [0; 4],
            position: [r * theta.cos(), r * theta.sin()],
        });

        let (left_in_slot, right_in_slot, left_next_out, right_next_out) = if g > 0 {
            (sign.over_in_slot(), 0_u8, 2_u8, sign.over_out_slot())
        } else {
            (0_u8, sign.over_in_slot(), sign.over_out_slot(), 2_u8)
        };

        for (slot_idx, in_slot) in [(left, left_in_slot), (right, right_in_slot)] {
            let head = Endpoint {
                crossing: step,
                slot: in_slot,
            };
            if let Some(tail) = last_out_at_slot[slot_idx] {
                let waypoints = annular_track_waypoints(
                    strand_radius(slot_idx, n),
                    tail.crossing,
                    step,
                    c_count,
                );
                let seg_id = segments.len();
                segments.push(DiagramSegment {
                    tail,
                    head,
                    waypoints,
                });
                crossings[tail.crossing].slots[tail.slot as usize] = seg_id;
                crossings[head.crossing].slots[head.slot as usize] = seg_id;
            } else {
                first_in_slot[initial_slot_at_curr[slot_idx]] = Some(head);
            }
        }
        last_out_at_slot[left] = Some(Endpoint {
            crossing: step,
            slot: left_next_out,
        });
        last_out_at_slot[right] = Some(Endpoint {
            crossing: step,
            slot: right_next_out,
        });
        initial_slot_at_curr.swap(left, right);
    }

    let free_loops = close_braid_strands(
        n,
        c_count,
        &mut crossings,
        &mut segments,
        &first_in_slot,
        &last_out_at_slot,
        &initial_slot_at_curr,
    );
    PlanarDiagram {
        crossings,
        segments,
        free_loops,
    }
}

#[verified_engine::verified]
fn close_braid_strands(
    n: usize,
    c_count: usize,
    crossings: &mut [Crossing],
    segments: &mut Vec<DiagramSegment>,
    first_in_slot: &[Option<Endpoint>],
    last_out_at_slot: &[Option<Endpoint>],
    initial_slot_at_curr: &[usize],
) -> usize {
    let mut visited_top = vec![false; n];
    let mut free_loops = 0;
    for (top_slot, &tail_opt) in last_out_at_slot.iter().enumerate().take(n) {
        let Some(tail) = tail_opt else {
            continue;
        };
        let mut cur = top_slot;
        let head = loop {
            if let Some(h) = first_in_slot[cur] {
                break h;
            }
            visited_top[cur] = true;
            cur = initial_slot_at_curr
                .iter()
                .position(|&orig| orig == cur)
                .unwrap_or(cur);
        };
        let waypoints = annular_track_waypoints(
            strand_radius(top_slot, n),
            tail.crossing,
            head.crossing,
            c_count,
        );
        let seg_id = segments.len();
        segments.push(DiagramSegment {
            tail,
            head,
            waypoints,
        });
        crossings[tail.crossing].slots[tail.slot as usize] = seg_id;
        crossings[head.crossing].slots[head.slot as usize] = seg_id;
    }
    for s in 0..n {
        if first_in_slot[s].is_none() && last_out_at_slot[s].is_none() && !visited_top[s] {
            free_loops += 1;
        }
    }
    free_loops
}
