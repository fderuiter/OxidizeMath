//! Structural, finiteness, and Euler rotation-system planarity validation for [`PlanarDiagram`].

// [cite:knot_braid_topology]

use super::{Crossing, DiagramSegment, MAX_DIAGRAM_CROSSINGS, PlanarDiagram};
use crate::pure_math::knot_theory::KnotError;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPlanarDiagram {
    crossings: Vec<Crossing>,
    segments: Vec<DiagramSegment>,
    #[serde(default)]
    free_loops: usize,
}

impl<'de> serde::Deserialize<'de> for PlanarDiagram {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawPlanarDiagram::deserialize(deserializer)?;
        let diagram = Self {
            crossings: raw.crossings,
            segments: raw.segments,
            free_loops: raw.free_loops,
        };
        diagram.validate().map_err(serde::de::Error::custom)?;
        Ok(diagram)
    }
}

#[verified_engine::verified]
pub(super) fn validate_planar_diagram(diagram: &PlanarDiagram) -> Result<(), KnotError> {
    let c = diagram.crossings.len();
    if c > MAX_DIAGRAM_CROSSINGS || diagram.free_loops > MAX_DIAGRAM_CROSSINGS {
        return Err(KnotError::InvalidDiagram {
            reason: "diagram exceeds maximum crossing or free-loop bound",
        });
    }
    if diagram.segments.len() != 2 * c {
        return Err(KnotError::InvalidDiagram {
            reason: "segment count must equal twice the crossing count",
        });
    }
    if c == 0 && diagram.free_loops == 0 {
        return Err(KnotError::InvalidDiagram {
            reason: "empty diagram must contain at least one free loop",
        });
    }
    for cr in &diagram.crossings {
        if !cr.position[0].is_finite() || !cr.position[1].is_finite() {
            return Err(KnotError::InvalidDiagram {
                reason: "crossing coordinates must be finite",
            });
        }
    }
    let slot_half = validate_segments_and_slots(diagram)?;
    if c > 0 {
        validate_euler_planarity(diagram, &slot_half)?;
    }
    debug_assert_eq!(diagram.segments.len(), 2 * c);
    Ok(())
}

#[verified_engine::verified]
fn validate_segments_and_slots(diagram: &PlanarDiagram) -> Result<Vec<[usize; 4]>, KnotError> {
    let c = diagram.crossings.len();
    let mut slot_half = vec![[usize::MAX; 4]; c];
    for (seg_id, seg) in diagram.segments.iter().enumerate() {
        validate_single_segment(diagram, seg_id, seg, &mut slot_half)?;
    }
    for slots in &slot_half {
        if slots.contains(&usize::MAX) {
            return Err(KnotError::InvalidDiagram {
                reason: "every crossing slot must attach to a unique segment endpoint",
            });
        }
    }
    debug_assert_eq!(slot_half.len(), c);
    Ok(slot_half)
}

#[verified_engine::verified]
fn validate_single_segment(
    diagram: &PlanarDiagram,
    seg_id: usize,
    seg: &DiagramSegment,
    slot_half: &mut [[usize; 4]],
) -> Result<(), KnotError> {
    if seg.waypoints.len() > 32 {
        return Err(KnotError::InvalidDiagram {
            reason: "segment waypoint list exceeds maximum bound",
        });
    }
    if seg
        .waypoints
        .iter()
        .any(|wp| !wp[0].is_finite() || !wp[1].is_finite())
    {
        return Err(KnotError::InvalidDiagram {
            reason: "segment waypoint coordinates must be finite",
        });
    }
    let Some(tail_c) = diagram.crossings.get(seg.tail.crossing) else {
        return Err(KnotError::InvalidDiagram {
            reason: "segment tail crossing index out of bounds",
        });
    };
    let Some(head_c) = diagram.crossings.get(seg.head.crossing) else {
        return Err(KnotError::InvalidDiagram {
            reason: "segment head crossing index out of bounds",
        });
    };
    if seg.tail.slot >= 4 || !tail_c.sign.is_outgoing_slot(seg.tail.slot) {
        return Err(KnotError::InvalidDiagram {
            reason: "segment tail must attach to an outgoing crossing slot",
        });
    }
    if seg.head.slot >= 4 || head_c.sign.is_outgoing_slot(seg.head.slot) {
        return Err(KnotError::InvalidDiagram {
            reason: "segment head must attach to an incoming crossing slot",
        });
    }
    if tail_c.slots[seg.tail.slot as usize] != seg_id
        || head_c.slots[seg.head.slot as usize] != seg_id
    {
        return Err(KnotError::InvalidDiagram {
            reason: "crossing slot table does not match segment endpoints",
        });
    }
    let tail_ref = &mut slot_half[seg.tail.crossing][seg.tail.slot as usize];
    if *tail_ref != usize::MAX {
        return Err(KnotError::InvalidDiagram {
            reason: "duplicate segment attachment at outgoing crossing slot",
        });
    }
    *tail_ref = 2 * seg_id;
    let head_ref = &mut slot_half[seg.head.crossing][seg.head.slot as usize];
    if *head_ref != usize::MAX {
        return Err(KnotError::InvalidDiagram {
            reason: "duplicate segment attachment at incoming crossing slot",
        });
    }
    *head_ref = 2 * seg_id + 1;
    debug_assert_ne!(2 * seg_id, 2 * seg_id + 1);
    Ok(())
}

#[verified_engine::verified]
fn validate_euler_planarity(
    diagram: &PlanarDiagram,
    slot_half: &[[usize; 4]],
) -> Result<(), KnotError> {
    let c = diagram.crossings.len();
    let (comp_count, comp_of_crossing, comp_vertices) = crossing_components(diagram);
    let total_half = 4 * c;
    let mut half_loc = vec![(0_usize, 0_u8); total_half];
    for (ci, slots) in slot_half.iter().enumerate() {
        for s in 0..4_u8 {
            half_loc[slots[s as usize]] = (ci, s);
        }
    }

    let mut visited_half = vec![false; total_half];
    let mut comp_faces = vec![0_usize; comp_count];
    for start_h in 0..total_half {
        if visited_half[start_h] {
            continue;
        }
        let comp_id = comp_of_crossing[half_loc[start_h].0];
        comp_faces[comp_id] += 1;
        let mut cur = start_h;
        while !visited_half[cur] {
            visited_half[cur] = true;
            let opp = cur ^ 1;
            let (v, slot) = half_loc[opp];
            cur = slot_half[v][((slot + 1) % 4) as usize];
        }
    }

    for comp_id in 0..comp_count {
        let v_k = comp_vertices[comp_id];
        let e_k = 2 * v_k;
        let f_k = comp_faces[comp_id];
        if v_k + f_k != e_k + 2 {
            return Err(KnotError::InvalidDiagram {
                reason: "non-planar virtual diagram violates Euler characteristic V - E + F = 2",
            });
        }
    }
    debug_assert!(visited_half.iter().all(|&vis| vis));
    Ok(())
}

#[verified_engine::verified]
fn crossing_components(diagram: &PlanarDiagram) -> (usize, Vec<usize>, Vec<usize>) {
    let c = diagram.crossings.len();
    let mut comp_of = vec![usize::MAX; c];
    let mut comp_sizes = Vec::new();
    let mut stack = Vec::new();
    for start in 0..c {
        if comp_of[start] != usize::MAX {
            continue;
        }
        let comp_id = comp_sizes.len();
        comp_of[start] = comp_id;
        stack.push(start);
        let mut v_count = 0_usize;
        while let Some(v) = stack.pop() {
            v_count += 1;
            for &seg_id in &diagram.crossings[v].slots {
                let seg = &diagram.segments[seg_id];
                for nbr in [seg.tail.crossing, seg.head.crossing] {
                    if comp_of[nbr] == usize::MAX {
                        comp_of[nbr] = comp_id;
                        stack.push(nbr);
                    }
                }
            }
        }
        comp_sizes.push(v_count);
    }
    debug_assert!(!comp_of.contains(&usize::MAX));
    (comp_sizes.len(), comp_of, comp_sizes)
}
