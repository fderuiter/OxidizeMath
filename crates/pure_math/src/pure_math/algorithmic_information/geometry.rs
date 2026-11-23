#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DyadicRational {
    #[allow(missing_docs)]
    pub m: i64,
    #[allow(missing_docs)]
    pub r: u32,
}

impl DyadicRational {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(m: i64, r: u32) -> Self {
        Self { m, r }
    }
}
