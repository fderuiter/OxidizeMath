//! Iterated elimination of strictly (`IESDS`) and weakly (`IEWDS`) dominated pure strategies.

use super::BimatrixGameView;
use crate::error::GameTheoryError;
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::registry::TOLERANCE_HIGH;

/// Dominance criterion applied during iterated pure-strategy elimination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DominanceMode {
    /// Eliminate strategies strictly dominated by another surviving pure strategy.
    Strict,
    /// Eliminate strategies weakly dominated by another surviving pure strategy.
    Weak,
}

/// Summary of iterated pure-strategy dominance elimination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DominanceResult {
    /// Number of surviving row pure strategies (`>= 1`).
    pub surviving_rows: usize,
    /// Number of surviving column pure strategies (`>= 1`).
    pub surviving_cols: usize,
    /// Number of elimination rounds executed before reaching a fixed point.
    pub rounds: usize,
}

/// Preallocated scratch flags for [`iterated_dominance`].
#[derive(Clone, Debug)]
pub struct DominanceWorkspace {
    row_alive: Vec<bool>,
    col_alive: Vec<bool>,
}

impl DominanceWorkspace {
    /// Allocates survival bitflags supporting up to `max_rows x max_cols` games.
    #[must_use]
    pub fn new(max_rows: usize, max_cols: usize) -> Self {
        Self {
            row_alive: vec![true; max_rows],
            col_alive: vec![true; max_cols],
        }
    }

    /// Expands scratch storage if needed outside verified zero-allocation regions.
    pub fn ensure_capacity(&mut self, rows: usize, cols: usize) {
        if self.row_alive.len() < rows {
            self.row_alive.resize(rows, true);
        }
        if self.col_alive.len() < cols {
            self.col_alive.resize(cols, true);
        }
    }
}

#[verified_engine::verified]
fn row_dominates(
    game: BimatrixGameView<'_>,
    dominator: usize,
    candidate: usize,
    col_alive: &[bool],
    mode: DominanceMode,
) -> bool {
    let mut any_strict = false;
    for (c, &alive) in col_alive.iter().enumerate() {
        if !alive {
            continue;
        }
        let u_dom = game.row_payoffs[flatten_2d_index(c, dominator, game.cols)];
        let u_cand = game.row_payoffs[flatten_2d_index(c, candidate, game.cols)];
        match mode {
            DominanceMode::Strict => {
                if u_dom <= u_cand + TOLERANCE_HIGH {
                    return false;
                }
                any_strict = true;
            }
            DominanceMode::Weak => {
                if u_dom + TOLERANCE_HIGH < u_cand {
                    return false;
                }
                if u_dom > u_cand + TOLERANCE_HIGH {
                    any_strict = true;
                }
            }
        }
    }
    any_strict
}

#[verified_engine::verified]
fn col_dominates(
    game: BimatrixGameView<'_>,
    dominator: usize,
    candidate: usize,
    row_alive: &[bool],
    mode: DominanceMode,
) -> bool {
    let mut any_strict = false;
    for (r, &alive) in row_alive.iter().enumerate() {
        if !alive {
            continue;
        }
        let v_dom = game.col_payoffs[flatten_2d_index(dominator, r, game.cols)];
        let v_cand = game.col_payoffs[flatten_2d_index(candidate, r, game.cols)];
        match mode {
            DominanceMode::Strict => {
                if v_dom <= v_cand + TOLERANCE_HIGH {
                    return false;
                }
                any_strict = true;
            }
            DominanceMode::Weak => {
                if v_dom + TOLERANCE_HIGH < v_cand {
                    return false;
                }
                if v_dom > v_cand + TOLERANCE_HIGH {
                    any_strict = true;
                }
            }
        }
    }
    any_strict
}

#[verified_engine::verified]
fn sweep_rows(
    game: BimatrixGameView<'_>,
    mode: DominanceMode,
    row_alive: &mut [bool],
    col_alive: &[bool],
    surviving_rows: &mut usize,
) -> bool {
    let mut changed = false;
    for r in 0..game.rows {
        if !row_alive[r] || *surviving_rows <= 1 {
            continue;
        }
        for other in 0..game.rows {
            if other != r && row_alive[other] && row_dominates(game, other, r, col_alive, mode) {
                row_alive[r] = false;
                *surviving_rows -= 1;
                changed = true;
                break;
            }
        }
    }
    changed
}

#[verified_engine::verified]
fn sweep_cols(
    game: BimatrixGameView<'_>,
    mode: DominanceMode,
    row_alive: &[bool],
    col_alive: &mut [bool],
    surviving_cols: &mut usize,
) -> bool {
    let mut changed = false;
    for c in 0..game.cols {
        if !col_alive[c] || *surviving_cols <= 1 {
            continue;
        }
        for other in 0..game.cols {
            if other != c && col_alive[other] && col_dominates(game, other, c, row_alive, mode) {
                col_alive[c] = false;
                *surviving_cols -= 1;
                changed = true;
                break;
            }
        }
    }
    changed
}

/// Runs iterated pure-strategy dominance elimination into caller-supplied survival masks.
#[verified_engine::verified]
pub fn iterated_dominance(
    game: BimatrixGameView<'_>,
    mode: DominanceMode,
    workspace: &mut DominanceWorkspace,
    row_alive_out: &mut [bool],
    col_alive_out: &mut [bool],
) -> Result<DominanceResult, GameTheoryError> {
    if workspace.row_alive.len() < game.rows {
        return Err(GameTheoryError::WorkspaceTooSmall {
            required: game.rows,
            actual: workspace.row_alive.len(),
        });
    }
    if workspace.col_alive.len() < game.cols {
        return Err(GameTheoryError::WorkspaceTooSmall {
            required: game.cols,
            actual: workspace.col_alive.len(),
        });
    }
    if row_alive_out.len() < game.rows {
        return Err(GameTheoryError::WorkspaceTooSmall {
            required: game.rows,
            actual: row_alive_out.len(),
        });
    }
    if col_alive_out.len() < game.cols {
        return Err(GameTheoryError::WorkspaceTooSmall {
            required: game.cols,
            actual: col_alive_out.len(),
        });
    }

    let r_alive = &mut workspace.row_alive[..game.rows];
    let c_alive = &mut workspace.col_alive[..game.cols];
    r_alive.fill(true);
    c_alive.fill(true);

    let mut surviving_rows = game.rows;
    let mut surviving_cols = game.cols;
    let mut rounds = 0usize;
    let max_rounds = game.rows + game.cols;

    for _ in 0..max_rounds {
        let r_changed = sweep_rows(game, mode, r_alive, c_alive, &mut surviving_rows);
        let c_changed = sweep_cols(game, mode, r_alive, c_alive, &mut surviving_cols);
        if !r_changed && !c_changed {
            break;
        }
        rounds += 1;
    }

    row_alive_out[..game.rows].copy_from_slice(r_alive);
    col_alive_out[..game.cols].copy_from_slice(c_alive);
    Ok(DominanceResult {
        surviving_rows,
        surviving_cols,
        rounds,
    })
}

// [cite:graph_parameters_rust]
