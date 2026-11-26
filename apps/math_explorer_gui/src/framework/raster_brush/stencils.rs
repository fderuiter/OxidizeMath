#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::bresenham::bresenham_line;

/// Preset geometric brush and obstacle stencils for 2D raster sculpting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BrushStencil {
    /// Continuous disk stroke interpolated along Bresenham pointer segments.
    #[default]
    Freehand,
    /// Discrete filled circle stamp of configurable radius.
    Circle,
    /// Two-endpoint straight Bresenham line segment of configurable thickness.
    Line,
    /// Symmetric NACA-profile aerodynamic teardrop stamp.
    Airfoil,
    /// Converging-diverging Venturi channel constriction walls.
    VenturiNozzle,
}

impl BrushStencil {
    /// All selectable brush stencils in UI display order.
    pub const ALL: [Self; 5] = [
        Self::Freehand,
        Self::Circle,
        Self::Line,
        Self::Airfoil,
        Self::VenturiNozzle,
    ];

    /// User-facing label for combo boxes and live-region announcements.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Freehand => "Freehand",
            Self::Circle => "Circle",
            Self::Line => "Line",
            Self::Airfoil => "Airfoil",
            Self::VenturiNozzle => "Venturi Nozzle",
        }
    }

    /// Whether dragging continuously emits strokes along the path.
    #[must_use]
    pub const fn is_continuous_drag(self) -> bool {
        matches!(self, Self::Freehand | Self::Circle)
    }
}

/// Spatial symmetry transformations applied to every stamped raster coordinate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SymmetryMode {
    /// Single-point stamp with no mirror or rotational copies.
    #[default]
    None,
    /// Mirror across the vertical midline (`x <-> width - 1 - x`).
    BilateralX,
    /// Mirror across the horizontal midline (`y <-> height - 1 - y`).
    BilateralY,
    /// Four-fold rotational and quadrant symmetry around the grid center.
    Radial,
}

impl SymmetryMode {
    /// All selectable symmetry modes in UI display order.
    pub const ALL: [Self; 4] = [Self::None, Self::BilateralX, Self::BilateralY, Self::Radial];

    /// User-facing label for combo boxes and live-region announcements.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::BilateralX => "Bilateral (X)",
            Self::BilateralY => "Bilateral (Y)",
            Self::Radial => "Radial (4x)",
        }
    }
}

#[derive(Clone, Copy)]
struct StampDomain {
    symmetry: SymmetryMode,
    grid_w: usize,
    grid_h: usize,
}

fn push_in_bounds(
    x: i32,
    y: i32,
    grid_w: usize,
    grid_h: usize,
    seen: &mut [bool],
    out: &mut Vec<(usize, usize)>,
) {
    if x >= 0 && y >= 0 && (x as usize) < grid_w && (y as usize) < grid_h {
        let ux = x as usize;
        let uy = y as usize;
        let idx = math_commons::math_kernel::types::flatten_2d_index(ux, uy, grid_w);
        if !seen[idx] {
            seen[idx] = true;
            out.push((ux, uy));
        }
    }
}

fn push_with_symmetry(
    x: i32,
    y: i32,
    dom: StampDomain,
    seen: &mut [bool],
    out: &mut Vec<(usize, usize)>,
) {
    push_in_bounds(x, y, dom.grid_w, dom.grid_h, seen, out);
    let max_x = dom.grid_w as i32 - 1;
    let max_y = dom.grid_h as i32 - 1;
    match dom.symmetry {
        SymmetryMode::None => {}
        SymmetryMode::BilateralX => {
            push_in_bounds(max_x - x, y, dom.grid_w, dom.grid_h, seen, out);
        }
        SymmetryMode::BilateralY => {
            push_in_bounds(x, max_y - y, dom.grid_w, dom.grid_h, seen, out);
        }
        SymmetryMode::Radial => {
            push_in_bounds(max_x - x, max_y - y, dom.grid_w, dom.grid_h, seen, out);
            let dx2 = 2 * x - max_x;
            let dy2 = 2 * y - max_y;
            let rx1 = (max_x - dy2).div_euclid(2);
            let ry1 = (max_y + dx2).div_euclid(2);
            let rx2 = (max_x + dy2).div_euclid(2);
            let ry2 = (max_y - dx2).div_euclid(2);
            push_in_bounds(rx1, ry1, dom.grid_w, dom.grid_h, seen, out);
            push_in_bounds(rx2, ry2, dom.grid_w, dom.grid_h, seen, out);
        }
    }
}

fn stamp_disk(
    cx: i32,
    cy: i32,
    r: i32,
    dom: StampDomain,
    seen: &mut [bool],
    out: &mut Vec<(usize, usize)>,
) {
    let r_clamped = r.max(0);
    let r2 = r_clamped * r_clamped;
    for dy in -r_clamped..=r_clamped {
        for dx in -r_clamped..=r_clamped {
            if dx * dx + dy * dy <= r2 {
                push_with_symmetry(cx + dx, cy + dy, dom, seen, out);
            }
        }
    }
}

fn stamp_airfoil(
    cx: i32,
    cy: i32,
    radius: usize,
    dom: StampDomain,
    seen: &mut [bool],
    out: &mut Vec<(usize, usize)>,
) {
    let chord = (radius.max(2) * 5).clamp(6, dom.grid_w.max(6)) as i32;
    let half_chord = chord / 2;
    let thickness_ratio = 0.24_f64;
    for i in 0..=chord {
        let s = f64::from(i) / f64::from(chord);
        let profile = 5.0
            * thickness_ratio
            * f64::from(chord)
            * (0.2969 * s.sqrt() - 0.1260 * s - 0.3516 * s.powi(2) + 0.2843 * s.powi(3)
                - 0.1015 * s.powi(4));
        let half_t = profile.round().max(0.0) as i32;
        let px = cx - half_chord + i;
        for dy in -half_t..=half_t {
            push_with_symmetry(px, cy + dy, dom, seen, out);
        }
    }
}

fn stamp_venturi(
    cx: i32,
    cy: i32,
    radius: usize,
    dom: StampDomain,
    seen: &mut [bool],
    out: &mut Vec<(usize, usize)>,
) {
    let r_eff = radius.max(2) as i32;
    let half_len = (r_eff * 4).clamp(6, (dom.grid_w.max(12) / 2) as i32);
    let throat_half_gap = (r_eff + 1).max(2);
    let flare = (r_eff * 2).max(3);
    let wall_thickness = (1 + radius / 2).max(2) as i32;

    for dx in -half_len..=half_len {
        let norm = f64::from(dx.abs()) / f64::from(half_len);
        let bell = 0.5 * (1.0 - (std::f64::consts::PI * (1.0 - norm)).cos());
        let half_gap = throat_half_gap + (f64::from(flare) * bell).round() as i32;
        for t in 0..wall_thickness {
            push_with_symmetry(cx + dx, cy + half_gap + t, dom, seen, out);
            push_with_symmetry(cx + dx, cy - half_gap - t, dom, seen, out);
        }
    }
}

/// Expands a brush stroke or shape stamp into unique in-bounds `(x, y)` grid coordinates.
#[must_use]
pub fn generate_brush_cells(
    stencil: BrushStencil,
    symmetry: SymmetryMode,
    from: (usize, usize),
    to: (usize, usize),
    radius: usize,
    grid_w: usize,
    grid_h: usize,
) -> Vec<(usize, usize)> {
    if grid_w == 0 || grid_h == 0 {
        return Vec::new();
    }
    let total = grid_w.saturating_mul(grid_h);
    let mut seen = vec![false; total];
    let mut out = Vec::new();
    let r = radius.min(64) as i32;
    let dom = StampDomain {
        symmetry,
        grid_w,
        grid_h,
    };

    match stencil {
        BrushStencil::Freehand | BrushStencil::Line => {
            for (px, py) in bresenham_line(from.0 as i32, from.1 as i32, to.0 as i32, to.1 as i32) {
                stamp_disk(px, py, r, dom, &mut seen, &mut out);
            }
        }
        BrushStencil::Circle => {
            stamp_disk(to.0 as i32, to.1 as i32, r, dom, &mut seen, &mut out);
        }
        BrushStencil::Airfoil => {
            stamp_airfoil(to.0 as i32, to.1 as i32, radius, dom, &mut seen, &mut out);
        }
        BrushStencil::VenturiNozzle => {
            stamp_venturi(to.0 as i32, to.1 as i32, radius, dom, &mut seen, &mut out);
        }
    }
    out
}
