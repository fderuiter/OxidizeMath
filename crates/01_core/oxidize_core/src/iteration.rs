/// Provides high-performance iteration patterns that separate interior updates
/// from boundary processing.
pub struct IterationPattern;

impl IterationPattern {
    /// Iterates over all interior points (excluding the 1-cell thick boundary).
    #[inline(always)]
    pub fn for_each_interior<F>(width: usize, height: usize, mut f: F)
    where
        F: FnMut(usize, usize),
    {
        if width > 2 && height > 2 {
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    f(x, y);
                }
            }
        }
    }

    /// Iterates over all boundary points (top, bottom, left, right).
    #[inline(always)]
    pub fn for_each_boundary<F>(width: usize, height: usize, mut f: F)
    where
        F: FnMut(usize, usize),
    {
        if width == 0 || height == 0 {
            return;
        }

        // Top and Bottom
        for y in [0, height.saturating_sub(1)] {
            if y == 0 || height > 1 {
                for x in 0..width {
                    f(x, y);
                }
            }
        }

        // Left and Right (excluding corners)
        if height > 2 {
            for y in 1..height - 1 {
                for x in [0, width.saturating_sub(1)] {
                    if x == 0 || width > 1 {
                        f(x, y);
                    }
                }
            }
        }
    }
}
