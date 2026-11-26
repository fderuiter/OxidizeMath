use crate::error::PharmacokineticsError;

/// Validated pharmacokinetic parameters for a one-compartment model.
///
/// Ensures all physical constants are valid (e.g., positive rates, non-negative dose).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PKParameters {
    /// Bioavailability (fraction, 0.0 - 1.0).
    f: f64,
    /// Dose (amount, must be non-negative).
    d: f64,
    /// Absorption rate constant (1/time, must be positive).
    ka: f64,
    /// Elimination rate constant (1/time, must be positive).
    ke: f64,
    /// Apparent volume of distribution (volume, must be positive).
    v: f64,
}

impl PKParameters {
    /// Creates a new `PKParameters` struct with validation.
    ///
    /// # Arguments
    /// * `f` - Bioavailability (0.0 to 1.0).
    /// * `d` - Dose amount (>= 0).
    /// * `ka` - Absorption rate constant (> 0).
    /// * `ke` - Elimination rate constant (> 0).
    /// * `v` - Volume of distribution (> 0).
    #[verified_engine::verified]
    pub fn new(f: f64, d: f64, ka: f64, ke: f64, v: f64) -> Result<Self, PharmacokineticsError> {
        if !(0.0..=1.0).contains(&f) {
            return Err(PharmacokineticsError::InvalidParameter(format!(
                "Bioavailability f={} must be between 0.0 and 1.0",
                f
            )));
        }
        if d < 0.0 {
            return Err(PharmacokineticsError::InvalidParameter(format!(
                "Dose d={} must be non-negative",
                d
            )));
        }
        if ka <= 0.0 {
            return Err(PharmacokineticsError::InvalidParameter(format!(
                "Absorption rate ka={} must be positive",
                ka
            )));
        }
        if ke <= 0.0 {
            return Err(PharmacokineticsError::InvalidParameter(format!(
                "Elimination rate ke={} must be positive",
                ke
            )));
        }
        if v <= 0.0 {
            return Err(PharmacokineticsError::InvalidParameter(format!(
                "Volume of distribution v={} must be positive",
                v
            )));
        }

        Ok(Self { f, d, ka, ke, v })
    }

    /// Returns the bioavailability fraction (f).
    #[verified_engine::verified]
    pub fn f(&self) -> f64 {
        self.f
    }

    /// Returns the dose amount (d).
    #[verified_engine::verified]
    pub fn d(&self) -> f64 {
        self.d
    }

    /// Returns the absorption rate constant (ka).
    #[verified_engine::verified]
    pub fn ka(&self) -> f64 {
        self.ka
    }

    /// Returns the elimination rate constant (ke).
    #[verified_engine::verified]
    pub fn ke(&self) -> f64 {
        self.ke
    }

    /// Returns the volume of distribution (v).
    #[verified_engine::verified]
    pub fn v(&self) -> f64 {
        self.v
    }

    /// Returns a new `PKParameters` with the dose updated.
    #[verified_engine::verified]
    pub fn with_dose(&self, d: f64) -> Result<Self, PharmacokineticsError> {
        Self::new(self.f, d, self.ka, self.ke, self.v)
    }
}

/// Builder for `PKParameters`.
#[derive(Debug, Default)]
pub struct PKParametersBuilder {
    f: Option<f64>,
    d: Option<f64>,
    ka: Option<f64>,
    ke: Option<f64>,
    v: Option<f64>,
}

impl PKParametersBuilder {
    /// Creates a new builder.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets bioavailability (f).
    #[verified_engine::verified]
    pub fn bioavailability(mut self, f: f64) -> Self {
        self.f = Some(f);
        self
    }

    /// Sets dose (d).
    #[verified_engine::verified]
    pub fn dose(mut self, d: f64) -> Self {
        self.d = Some(d);
        self
    }

    /// Sets absorption rate constant (ka).
    #[verified_engine::verified]
    pub fn absorption_rate(mut self, ka: f64) -> Self {
        self.ka = Some(ka);
        self
    }

    /// Sets elimination rate constant (ke).
    #[verified_engine::verified]
    pub fn elimination_rate(mut self, ke: f64) -> Self {
        self.ke = Some(ke);
        self
    }

    /// Sets volume of distribution (v).
    #[verified_engine::verified]
    pub fn volume(mut self, v: f64) -> Self {
        self.v = Some(v);
        self
    }

    /// Builds the `PKParameters` struct, validating all fields.
    #[verified_engine::verified]
    pub fn build(self) -> Result<PKParameters, PharmacokineticsError> {
        let f = self.f.ok_or_else(|| {
            PharmacokineticsError::InvalidParameter("Bioavailability (f) is required".to_string())
        })?;
        let d = self.d.ok_or_else(|| {
            PharmacokineticsError::InvalidParameter("Dose (d) is required".to_string())
        })?;
        let ka = self.ka.ok_or_else(|| {
            PharmacokineticsError::InvalidParameter("Absorption rate (ka) is required".to_string())
        })?;
        let ke = self.ke.ok_or_else(|| {
            PharmacokineticsError::InvalidParameter("Elimination rate (ke) is required".to_string())
        })?;
        let v = self.v.ok_or_else(|| {
            PharmacokineticsError::InvalidParameter("Volume (v) is required".to_string())
        })?;

        PKParameters::new(f, d, ka, ke, v)
    }
}
