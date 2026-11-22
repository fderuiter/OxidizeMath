pub use oxidize_core::mesh::{Mesh, Point3D};

/// A 3D grid of scalar values, used as input for isosurface extraction.
///
/// The grid data is stored in a flat vector, indexed by `z * width * height + y * width + x`.
#[derive(Debug, Clone)]
pub struct VoxelGrid {
    /// Number of voxels along the X axis.
    width: usize,
    /// Number of voxels along the Y axis.
    height: usize,
    /// Number of voxels along the Z axis.
    depth: usize,
    /// Flat vector of scalar values (e.g., signed distance or density).
    /// Indexing is `z` (slowest) -> `y` -> `x` (fastest).
    data: Vec<f32>,
    /// The physical size of each voxel (dx, dy, dz).
    voxel_size: Point3D,
    /// The physical coordinate of the grid's origin (0,0,0 index).
    origin: Point3D,
    /// Optional minimum scalar value enforced on all six exterior grid faces to seal clipped solids.
    clamp_exterior: Option<f32>,
}

#[derive(Debug)]
#[allow(missing_docs)]
pub enum VoxelGridError {
    #[allow(missing_docs)]
    InvalidDimensions,
    #[allow(missing_docs)]
    DataSizeMismatch,
    #[allow(missing_docs)]
    InvalidVoxelSize,
}

impl std::fmt::Display for VoxelGridError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions => write!(f, "Grid dimensions must be greater than zero"),
            Self::DataSizeMismatch => {
                write!(f, "Data vector size does not match width * height * depth")
            }
            Self::InvalidVoxelSize => write!(f, "Voxel size components must be strictly positive"),
        }
    }
}

impl std::error::Error for VoxelGridError {}

#[allow(missing_docs)]
pub struct VoxelGridBuilder {
    width: Option<usize>,
    height: Option<usize>,
    depth: Option<usize>,
    data: Option<Vec<f32>>,
    voxel_size: Point3D,
    origin: Point3D,
    clamp_exterior: Option<f32>,
}

impl Default for VoxelGridBuilder {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

impl VoxelGridBuilder {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self {
            width: None,
            height: None,
            depth: None,
            data: None,
            voxel_size: Point3D::new(1.0, 1.0, 1.0),
            origin: Point3D::new(0.0, 0.0, 0.0),
            clamp_exterior: None,
        }
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn dimensions(mut self, width: usize, height: usize, depth: usize) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self.depth = Some(depth);
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn data(mut self, data: Vec<f32>) -> Self {
        self.data = Some(data);
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn voxel_size(mut self, size: Point3D) -> Self {
        self.voxel_size = size;
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn origin(mut self, origin: Point3D) -> Self {
        self.origin = origin;
        self
    }

    /// Configures optional minimum scalar clamping across all six exterior boundary faces.
    #[verified_engine::verified]
    pub fn clamp_exterior(mut self, clamp_min: Option<f32>) -> Self {
        self.clamp_exterior = clamp_min;
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn build(self) -> Result<VoxelGrid, VoxelGridError> {
        let width = self.width.ok_or(VoxelGridError::InvalidDimensions)?;
        let height = self.height.ok_or(VoxelGridError::InvalidDimensions)?;
        let depth = self.depth.ok_or(VoxelGridError::InvalidDimensions)?;
        let data = self.data.ok_or(VoxelGridError::DataSizeMismatch)?;

        if width == 0 || height == 0 || depth == 0 {
            return Err(VoxelGridError::InvalidDimensions);
        }

        if data.len() != width * height * depth {
            return Err(VoxelGridError::DataSizeMismatch);
        }

        if self.voxel_size.x <= 0.0 || self.voxel_size.y <= 0.0 || self.voxel_size.z <= 0.0 {
            return Err(VoxelGridError::InvalidVoxelSize);
        }

        let mut grid = VoxelGrid {
            width,
            height,
            depth,
            data,
            voxel_size: self.voxel_size,
            origin: self.origin,
            clamp_exterior: self.clamp_exterior,
        };
        grid.apply_exterior_clamp();
        Ok(grid)
    }
}

impl VoxelGrid {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn builder() -> VoxelGridBuilder {
        VoxelGridBuilder::new()
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn depth(&self) -> usize {
        self.depth
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    /// Returns a mutable slice into the underlying scalar buffer for zero-allocation resampling.
    #[verified_engine::verified]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn voxel_size(&self) -> &Point3D {
        &self.voxel_size
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn origin(&self) -> &Point3D {
        &self.origin
    }
    /// Returns the optional exterior boundary clamp floor.
    #[verified_engine::verified]
    pub fn clamp_exterior(&self) -> Option<f32> {
        self.clamp_exterior
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn set_voxel_size(&mut self, size: Point3D) -> Result<(), VoxelGridError> {
        if size.x <= 0.0 || size.y <= 0.0 || size.z <= 0.0 {
            return Err(VoxelGridError::InvalidVoxelSize);
        }
        self.voxel_size = size;
        Ok(())
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn set_origin(&mut self, origin: Point3D) {
        self.origin = origin;
    }

    /// Updates the optional exterior boundary clamp floor and applies it when `Some`.
    #[verified_engine::verified]
    pub fn set_clamp_exterior(&mut self, clamp_min: Option<f32>) {
        self.clamp_exterior = clamp_min;
        self.apply_exterior_clamp();
    }

    /// Resizes the grid dimensions in place, reusing existing buffer capacity when large enough.
    #[verified_engine::verified]
    pub fn resize_dimensions(
        &mut self,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<(), VoxelGridError> {
        if width == 0 || height == 0 || depth == 0 {
            return Err(VoxelGridError::InvalidDimensions);
        }
        let total = width
            .checked_mul(height)
            .and_then(|wh| wh.checked_mul(depth))
            .ok_or(VoxelGridError::InvalidDimensions)?;
        self.width = width;
        self.height = height;
        self.depth = depth;
        self.data.resize(total, 0.0);
        Ok(())
    }

    /// Clamps all six exterior faces of the scalar buffer so `v >= min_val` when `clamp_exterior` is set.
    #[verified_engine::verified]
    pub fn apply_exterior_clamp(&mut self) {
        let Some(min_val) = self.clamp_exterior else {
            return;
        };
        let (w, h, d) = (self.width, self.height, self.depth);
        if w == 0 || h == 0 || d == 0 || self.data.len() < w * h * d {
            return;
        }
        for z in 0..d {
            let z_face = z == 0 || z + 1 == d;
            for y in 0..h {
                let yz_face = z_face || y == 0 || y + 1 == h;
                let row_start = (z * h + y) * w;
                if yz_face {
                    for slot in &mut self.data[row_start..row_start + w] {
                        *slot = clamp_sample(*slot, min_val);
                    }
                } else {
                    self.data[row_start] = clamp_sample(self.data[row_start], min_val);
                    let last = row_start + w - 1;
                    self.data[last] = clamp_sample(self.data[last], min_val);
                }
            }
        }
    }

    /// Retrieves the value at the given grid index (x, y, z).
    ///
    /// Returns the exterior clamp floor (or `0.0`) if the coordinates are out of bounds.
    #[inline]
    #[verified_engine::verified]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        let oob_default = self.clamp_exterior.unwrap_or(0.0);
        if x >= self.width || y >= self.height || z >= self.depth {
            return oob_default; // Boundary condition
        }
        // Sentinel: Prevent panic if data buffer is malformed (smaller than dimensions imply)
        let idx = z
            .checked_mul(self.width)
            .and_then(|val| val.checked_mul(self.height))
            .and_then(|val| val.checked_add(y.checked_mul(self.width)?))
            .and_then(|val| val.checked_add(x));

        let raw = match idx {
            Some(i) => self.data.get(i).copied().unwrap_or(oob_default),
            None => oob_default,
        };
        if let Some(min_val) = self.clamp_exterior
            && (x == 0
                || x + 1 == self.width
                || y == 0
                || y + 1 == self.height
                || z == 0
                || z + 1 == self.depth)
        {
            return clamp_sample(raw, min_val);
        }
        raw
    }
}

#[inline]
#[verified_engine::verified]
fn clamp_sample(val: f32, min_val: f32) -> f32 {
    if val.is_finite() {
        val.max(min_val)
    } else {
        min_val + 1.0
    }
}
