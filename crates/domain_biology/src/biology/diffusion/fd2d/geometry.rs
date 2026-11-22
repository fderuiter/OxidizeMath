use math_commons::math_kernel::types::{Dimension, StepSize};

#[allow(missing_docs)]
pub trait GeometryStrategy {
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn width(&self) -> Dimension;
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn height(&self) -> Dimension;
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn dx(&self) -> StepSize;
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn dy(&self) -> StepSize;
    #[allow(missing_docs)]
    #[verified_engine::verified]
    fn size(&self) -> usize {
        *self.width() * *self.height()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct Cartesian2D {
    #[allow(missing_docs)]
    pub width: Dimension,
    #[allow(missing_docs)]
    pub height: Dimension,
    #[allow(missing_docs)]
    pub dx: StepSize,
    #[allow(missing_docs)]
    pub dy: StepSize,
}

impl GeometryStrategy for Cartesian2D {
    #[verified_engine::verified]
    fn width(&self) -> Dimension {
        self.width
    }

    #[verified_engine::verified]
    fn height(&self) -> Dimension {
        self.height
    }

    #[verified_engine::verified]
    fn dx(&self) -> StepSize {
        self.dx
    }

    #[verified_engine::verified]
    fn dy(&self) -> StepSize {
        self.dy
    }
}
