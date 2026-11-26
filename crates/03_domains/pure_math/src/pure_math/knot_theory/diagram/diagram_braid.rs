//! Braid-closure planar diagram construction helper.

// [cite:knot_braid_topology]

use super::{Crossing, CrossingSign, DiagramSegment, Endpoint, PlanarDiagram};
use std::f64::consts::TAU;

#[verified_engine::verified]
pub(super) fn build_braid_closure_diagram(n: usize, gens: &[i32]) -> PlanarDiagram {
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
        let r = 0.56
            + 0.22 * (((left as f64) + 0.5 - 0.5 * ((n - 1) as f64)) / ((n.max(2) - 1) as f64));
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
                let seg_id = segments.len();
                segments.push(DiagramSegment {
                    tail,
                    head,
                    waypoints: Vec::new(),
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
        let p_tail = crossings[tail.crossing].position;
        let p_head = crossings[head.crossing].position;
        let ang_t = p_tail[1].atan2(p_tail[0]);
        let mut ang_h = p_head[1].atan2(p_head[0]);
        if ang_h <= ang_t {
            ang_h += TAU;
        }
        let r_arc = 0.84 + 0.04 * (top_slot as f64);
        let waypoints = (1..=3)
            .map(|k| {
                let a = ang_t + (ang_h - ang_t) * (k as f64) / 4.0;
                [r_arc * a.cos(), r_arc * a.sin()]
            })
            .collect();
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
