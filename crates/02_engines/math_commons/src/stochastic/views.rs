//! Borrowed `#![no_std]` slice views and [`ArenaLayout`] region descriptors for `stochastic`.
// [cite:math_commons]

use math_memory::{ArenaError, ByteCursor, FlagSlice};
#[cfg(feature = "alloc")]
use math_memory::{ArenaLayout, ArenaRegion, FlagRegion, RawByteCursor};

/// Borrowed `#![no_std]` scratch views for $n$-state Tarjan SCC and Householder QR solvers.
pub(super) struct StochasticViews<'a> {
    pub(super) tarjan_index: &'a mut [usize],
    pub(super) tarjan_lowlink: &'a mut [usize],
    pub(super) tarjan_on_stack: FlagSlice<'a>,
    pub(super) tarjan_stack: &'a mut [usize],
    pub(super) call_node: &'a mut [usize],
    pub(super) call_next_edge: &'a mut [usize],
    pub(super) component_id: &'a mut [usize],
    pub(super) component_is_sink: FlagSlice<'a>,
    pub(super) sink_members: &'a mut [usize],
    pub(super) row_scales: &'a mut [f64],
    pub(super) qr_matrix: &'a mut [f64],
    pub(super) qr_rhs: &'a mut [f64],
    pub(super) qr_sol: &'a mut [f64],
}

impl<'a> StochasticViews<'a> {
    /// Carves all $n$-state Tarjan SCC and Householder QR buffers from a [`ByteCursor`].
    pub(super) fn carve(cur: &mut ByteCursor<'a>, n: usize) -> Result<Self, ArenaError> {
        let np1 = n.checked_add(1).ok_or(ArenaError::ArithmeticOverflow)?;
        let qr_len = np1.checked_mul(n).ok_or(ArenaError::ArithmeticOverflow)?;
        Ok(Self {
            tarjan_index: cur.alloc_slice::<usize>(n)?,
            tarjan_lowlink: cur.alloc_slice::<usize>(n)?,
            tarjan_on_stack: cur.alloc_flags(n)?,
            tarjan_stack: cur.alloc_slice::<usize>(n)?,
            call_node: cur.alloc_slice::<usize>(n)?,
            call_next_edge: cur.alloc_slice::<usize>(n)?,
            component_id: cur.alloc_slice::<usize>(n)?,
            component_is_sink: cur.alloc_flags(n)?,
            sink_members: cur.alloc_slice::<usize>(n)?,
            row_scales: cur.alloc_slice::<f64>(n)?,
            qr_matrix: cur.alloc_slice::<f64>(qr_len)?,
            qr_rhs: cur.alloc_slice::<f64>(np1)?,
            qr_sol: cur.alloc_slice::<f64>(n)?,
        })
    }
}

/// Persistent single-arena region handles for [`super::StochasticWorkspace`].
#[cfg(feature = "alloc")]
#[derive(Copy, Clone, Debug, Default)]
pub(super) struct StochasticRegions {
    pub(super) operator_buf: ArenaRegion<f64>,
    pub(super) solution_buf: ArenaRegion<f64>,
    pub(super) tarjan_index: ArenaRegion<usize>,
    pub(super) tarjan_lowlink: ArenaRegion<usize>,
    pub(super) tarjan_on_stack: FlagRegion,
    pub(super) tarjan_stack: ArenaRegion<usize>,
    pub(super) call_node: ArenaRegion<usize>,
    pub(super) call_next_edge: ArenaRegion<usize>,
    pub(super) component_id: ArenaRegion<usize>,
    pub(super) component_is_sink: FlagRegion,
    pub(super) sink_members: ArenaRegion<usize>,
    pub(super) row_scales: ArenaRegion<f64>,
    pub(super) qr_matrix: ArenaRegion<f64>,
    pub(super) qr_rhs: ArenaRegion<f64>,
    pub(super) qr_sol: ArenaRegion<f64>,
    pub(super) sample_scratch: ArenaRegion<f64>,
}

#[cfg(feature = "alloc")]
impl StochasticRegions {
    /// Reserves all state and sample regions in deterministic order on `layout`.
    pub(super) fn reserve(
        layout: &mut ArenaLayout,
        n: usize,
        samples: usize,
    ) -> Result<Self, ArenaError> {
        let nn = n.checked_mul(n).ok_or(ArenaError::ArithmeticOverflow)?;
        let np1 = if n == 0 {
            0
        } else {
            n.checked_add(1).ok_or(ArenaError::ArithmeticOverflow)?
        };
        let qr_len = np1.checked_mul(n).ok_or(ArenaError::ArithmeticOverflow)?;
        Ok(Self {
            operator_buf: layout.reserve::<f64>(nn)?,
            solution_buf: layout.reserve::<f64>(n)?,
            tarjan_index: layout.reserve::<usize>(n)?,
            tarjan_lowlink: layout.reserve::<usize>(n)?,
            tarjan_on_stack: layout.reserve_flags(n)?,
            tarjan_stack: layout.reserve::<usize>(n)?,
            call_node: layout.reserve::<usize>(n)?,
            call_next_edge: layout.reserve::<usize>(n)?,
            component_id: layout.reserve::<usize>(n)?,
            component_is_sink: layout.reserve_flags(n)?,
            sink_members: layout.reserve::<usize>(n)?,
            row_scales: layout.reserve::<f64>(n)?,
            qr_matrix: layout.reserve::<f64>(qr_len)?,
            qr_rhs: layout.reserve::<f64>(np1)?,
            qr_sol: layout.reserve::<f64>(n)?,
            sample_scratch: layout.reserve::<f64>(samples)?,
        })
    }

    /// Splits `operator_buf`, `solution_buf`, and [`StochasticViews`] simultaneously from `raw`.
    pub(super) fn split_all<'a>(
        &self,
        raw: &mut RawByteCursor<'a>,
        n: usize,
    ) -> (&'a mut [f64], &'a mut [f64], StochasticViews<'a>) {
        let nn = n * n;
        let np1 = n + 1;
        let qr_len = np1 * n;
        let op = &mut raw.region_mut(self.operator_buf)[..nn];
        let sol = &mut raw.region_mut(self.solution_buf)[..n];
        let views = StochasticViews {
            tarjan_index: &mut raw.region_mut(self.tarjan_index)[..n],
            tarjan_lowlink: &mut raw.region_mut(self.tarjan_lowlink)[..n],
            tarjan_on_stack: raw.flags_mut(self.tarjan_on_stack).into_prefix(n),
            tarjan_stack: &mut raw.region_mut(self.tarjan_stack)[..n],
            call_node: &mut raw.region_mut(self.call_node)[..n],
            call_next_edge: &mut raw.region_mut(self.call_next_edge)[..n],
            component_id: &mut raw.region_mut(self.component_id)[..n],
            component_is_sink: raw.flags_mut(self.component_is_sink).into_prefix(n),
            sink_members: &mut raw.region_mut(self.sink_members)[..n],
            row_scales: &mut raw.region_mut(self.row_scales)[..n],
            qr_matrix: &mut raw.region_mut(self.qr_matrix)[..qr_len],
            qr_rhs: &mut raw.region_mut(self.qr_rhs)[..np1],
            qr_sol: &mut raw.region_mut(self.qr_sol)[..n],
        };
        (op, sol, views)
    }
}
