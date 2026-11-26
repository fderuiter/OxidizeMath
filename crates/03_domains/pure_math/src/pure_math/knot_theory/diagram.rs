//! Oriented planar knot and link diagrams (`PlanarDiagram`) with PD slot invariants.

// [cite:knot_braid_topology]

use super::KnotError;
use crate::pure_math::algebra::BraidWord;
use math_commons::math_kernel::types::flatten_2d_index;

/// Right-hand rule orientation sign of a transverse planar crossing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CrossingSign {
    /// Positive right-handed crossing ($\epsilon = +1$, $\det(T_{\text{over}}, T_{\text{under}}) > 0$).
    Positive,
    /// Negative left-handed crossing ($\epsilon = -1$, $\det(T_{\text{over}}, T_{\text{under}}) < 0$).
    Negative,
}

impl CrossingSign {
    /// Algebraic sign $\pm 1$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn value(self) -> i64 {
        match self {
            Self::Positive => 1,
            Self::Negative => -1,
        }
    }

    /// Mirror crossing sign (`Positive <-> Negative`).
    #[must_use]
    #[verified_engine::verified]
    pub const fn flipped(self) -> Self {
        match self {
            Self::Positive => Self::Negative,
            Self::Negative => Self::Positive,
        }
    }

    /// Crossing slot (`1` or `3`) where the over-strand enters.
    #[must_use]
    #[verified_engine::verified]
    pub const fn over_in_slot(self) -> u8 {
        match self {
            Self::Positive => 3,
            Self::Negative => 1,
        }
    }

    /// Crossing slot (`1` or `3`) where the over-strand exits.
    #[must_use]
    #[verified_engine::verified]
    pub const fn over_out_slot(self) -> u8 {
        match self {
            Self::Positive => 1,
            Self::Negative => 3,
        }
    }

    /// Returns `true` if `slot` (`0..4`) is an outgoing slot for `self`.
    #[must_use]
    #[verified_engine::verified]
    pub const fn is_outgoing_slot(self, slot: u8) -> bool {
        slot == 2 || slot == self.over_out_slot()
    }
}

/// Endpoint attachment `(crossing, slot)` of a directed [`DiagramSegment`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    /// Target crossing index in `PlanarDiagram::crossings`.
    pub crossing: usize,
    /// Counter-clockwise slot `0..4` (`0 = under_in`, `2 = under_out`, `{1, 3} = over`).
    pub slot: u8,
}

/// Directed segment connecting an outgoing crossing slot (`tail`) to an incoming slot (`head`).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagramSegment {
    /// Outgoing endpoint where this segment leaves a crossing.
    pub tail: Endpoint,
    /// Incoming endpoint where this segment enters a crossing.
    pub head: Endpoint,
    /// Optional 2D layout waypoints guiding planar routing and 3D lifting.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub waypoints: Vec<[f64; 2]>,
}

/// Transverse 4-valent planar crossing with counter-clockwise incident segment slots.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Crossing {
    /// Right-hand crossing sign.
    pub sign: CrossingSign,
    /// Segment indices `[under_in, east_over, under_out, west_over]` in CCW planar order.
    pub slots: [usize; 4],
    /// Planar 2D coordinates `[x, y]` for interactive diagram rendering and 3D lifting.
    pub position: [f64; 2],
}

impl Crossing {
    /// Incoming under-strand segment index (`slots[0]`).
    #[must_use]
    #[verified_engine::verified]
    pub const fn under_in(&self) -> usize {
        self.slots[0]
    }

    /// Outgoing under-strand segment index (`slots[2]`).
    #[must_use]
    #[verified_engine::verified]
    pub const fn under_out(&self) -> usize {
        self.slots[2]
    }

    /// Incoming over-strand segment index (`slots[3]` if positive, `slots[1]` if negative).
    #[must_use]
    #[verified_engine::verified]
    pub const fn over_in(&self) -> usize {
        self.slots[self.sign.over_in_slot() as usize]
    }

    /// Outgoing over-strand segment index (`slots[1]` if positive, `slots[3]` if negative).
    #[must_use]
    #[verified_engine::verified]
    pub const fn over_out(&self) -> usize {
        self.slots[self.sign.over_out_slot() as usize]
    }
}

/// Maximum crossings permitted in a serialized [`PlanarDiagram`].
pub const MAX_DIAGRAM_CROSSINGS: usize = 128;

/// Oriented planar knot or multi-component link diagram.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanarDiagram {
    /// Ordered crossings of the diagram (`segments.len() == 2 * crossings.len()`).
    pub crossings: Vec<Crossing>,
    /// Directed arcs between crossing slots.
    pub segments: Vec<DiagramSegment>,
    /// Number of detached 0-crossing unknot components.
    #[serde(default)]
    pub free_loops: usize,
}

impl Default for PlanarDiagram {
    fn default() -> Self {
        Self::unknot()
    }
}

impl PlanarDiagram {
    /// Single 0-crossing unknot diagram.
    #[must_use]
    #[verified_engine::verified]
    pub fn unknot() -> Self {
        let d = Self {
            crossings: Vec::new(),
            segments: Vec::new(),
            free_loops: 1,
        };
        debug_assert_eq!(d.free_loops, 1);
        d
    }

    /// `count`-component crossing-free unlink diagram.
    ///
    /// # Errors
    /// Returns [`KnotError::InvalidDiagram`] if `count == 0` or `count > MAX_DIAGRAM_CROSSINGS`.
    #[verified_engine::verified]
    pub fn unlink(count: usize) -> Result<Self, KnotError> {
        if count == 0 || count > MAX_DIAGRAM_CROSSINGS {
            return Err(KnotError::InvalidDiagram {
                reason: "unlink component count must lie in 1..=128",
            });
        }
        let d = Self {
            crossings: Vec::new(),
            segments: Vec::new(),
            free_loops: count,
        };
        assert_eq!(d.free_loops, count);
        Ok(d)
    }

    /// Verifies 4-valent slot reciprocity, orientation consistency, and size bounds.
    ///
    /// # Errors
    /// Returns [`KnotError::InvalidDiagram`] if any structural invariant is violated.
    #[verified_engine::verified]
    pub fn validate(&self) -> Result<(), KnotError> {
        let c = self.crossings.len();
        if c > MAX_DIAGRAM_CROSSINGS {
            return Err(KnotError::InvalidDiagram {
                reason: "diagram exceeds maximum crossing bound",
            });
        }
        if self.segments.len() != 2 * c {
            return Err(KnotError::InvalidDiagram {
                reason: "segment count must equal twice the crossing count",
            });
        }
        if c == 0 && self.free_loops == 0 {
            return Err(KnotError::InvalidDiagram {
                reason: "empty diagram must contain at least one free loop",
            });
        }
        for (seg_id, seg) in self.segments.iter().enumerate() {
            if seg.waypoints.len() > 32 {
                return Err(KnotError::InvalidDiagram {
                    reason: "segment waypoint list exceeds maximum bound",
                });
            }
            let Some(tail_c) = self.crossings.get(seg.tail.crossing) else {
                return Err(KnotError::InvalidDiagram {
                    reason: "segment tail crossing index out of bounds",
                });
            };
            let Some(head_c) = self.crossings.get(seg.head.crossing) else {
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
        }
        assert_eq!(self.segments.len(), 2 * c);
        Ok(())
    }

    /// Number of crossings $c$ in the planar projection.
    #[must_use]
    #[verified_engine::verified]
    pub fn crossing_count(&self) -> usize {
        self.crossings.len()
    }

    /// Algebraic diagram writhe $w(D) = \sum_c \epsilon(c)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn writhe(&self) -> i64 {
        self.crossings.iter().map(|c| c.sign.value()).sum()
    }

    /// Mirror image diagram obtained by flipping every crossing over/under (negating signs).
    #[must_use]
    #[verified_engine::verified]
    pub fn mirror(&self) -> Self {
        let mut out = self.clone();
        for c in &mut out.crossings {
            // Swapping over <-> under swaps slots (0 <-> 3, 1 <-> 2) or simpler: flip sign and swap slots 1 <-> 3?
            // Wait: if we flip over <-> under in 3D while keeping segment directions fixed,
            // old over_in becomes under_in (0), old over_out becomes under_out (2), etc.
            let old_ui = c.under_in();
            let old_uo = c.under_out();
            let old_oi = c.over_in();
            let old_oo = c.over_out();
            let next_sign = c.sign.flipped();
            let mut next_slots = [0; 4];
            next_slots[0] = old_oi;
            next_slots[2] = old_oo;
            next_slots[next_sign.over_in_slot() as usize] = old_ui;
            next_slots[next_sign.over_out_slot() as usize] = old_uo;
            c.sign = next_sign;
            c.slots = next_slots;
        }
        for (seg_id, seg) in out.segments.iter_mut().enumerate() {
            let tc = &out.crossings[seg.tail.crossing];
            let hc = &out.crossings[seg.head.crossing];
            for s in 0..4_u8 {
                if tc.slots[s as usize] == seg_id && tc.sign.is_outgoing_slot(s) {
                    seg.tail.slot = s;
                }
                if hc.slots[s as usize] == seg_id && !hc.sign.is_outgoing_slot(s) {
                    seg.head.slot = s;
                }
            }
        }
        out
    }

    /// Maps every segment to its connected link component index `0..component_count`,
    /// returning `(total_components, component_of_segment)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn segment_components(&self) -> (usize, Vec<usize>) {
        let m = self.segments.len();
        let mut comp_of = vec![usize::MAX; m];
        let mut comp_idx = 0;
        for start in 0..m {
            if comp_of[start] != usize::MAX {
                continue;
            }
            let mut cur = start;
            let mut guard = 0;
            while comp_of[cur] == usize::MAX && guard <= m {
                comp_of[cur] = comp_idx;
                let head = self.segments[cur].head;
                let c = &self.crossings[head.crossing];
                cur = if head.slot == 0 {
                    c.under_out()
                } else {
                    c.over_out()
                };
                guard += 1;
            }
            comp_idx += 1;
        }
        (comp_idx + self.free_loops, comp_of)
    }

    /// Total number of closed link components in the diagram.
    #[must_use]
    #[verified_engine::verified]
    pub fn component_count(&self) -> usize {
        self.segment_components().0
    }

    /// Exact combinatorial pairwise linking matrix $\operatorname{Lk}(C_i, C_j) = \frac{1}{2} \sum_{c \in C_i \cap C_j} \epsilon(c)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn combinatorial_linking_matrix(&self) -> Vec<Vec<i64>> {
        let (k, comp_of) = self.segment_components();
        let mut double_lk = vec![0_i64; k * k];
        for c in &self.crossings {
            let c_over = comp_of[c.over_in()];
            let c_under = comp_of[c.under_in()];
            if c_over != c_under {
                let s = c.sign.value();
                double_lk[flatten_2d_index(c_over, c_under, k)] += s;
                double_lk[flatten_2d_index(c_under, c_over, k)] += s;
            }
        }
        let mut rows = Vec::with_capacity(k);
        for i in 0..k {
            let mut row = Vec::with_capacity(k);
            for j in 0..k {
                row.push(double_lk[flatten_2d_index(i, j, k)] / 2);
            }
            rows.push(row);
        }
        rows
    }

    /// Constructs the exact oriented planar closure diagram of an Artin [`BraidWord`].
    #[must_use]
    #[verified_engine::verified]
    pub fn from_braid(braid: &BraidWord) -> Self {
        let n = braid.strands();
        let gens = braid.generators();
        if gens.is_empty() {
            return Self {
                crossings: Vec::new(),
                segments: Vec::new(),
                free_loops: n,
            };
        }
        diagram_braid::build_braid_closure_diagram(n, gens)
    }
}

mod diagram_braid;
