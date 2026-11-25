//! Unit-system frame tags isolating SI, nuclear `MeV`-`fm`, galactic `kpc`, CAD `mm`, and
//! normalized viewport coordinate scales at compile time.
// [cite:math_scalars]

use core::fmt::Debug;

/// Compile-time unit-system frame tag attached to a [`crate::Quantity`].
///
/// Arithmetic across distinct unit frames (for example adding or dividing an [`SiFrame`] speed in
/// $\text{m/s}$ with a [`NuclearMeVFmFrame`] speed in $\text{fm/s}$, or passing a [`ViewportFrame`]
/// offset directly to a [`GalacticKpcFrame`] lensing kernel) is rejected by the type checker.
pub trait UnitFrame:
    Copy + Clone + Debug + Default + PartialEq + Eq + Send + Sync + 'static
{
    /// Short human-readable identifier for diagnostics.
    const NAME: &'static str;
}

/// Coherent International System of Units (`m`, `kg`, `s`, `A`, `K`, `mol`, `cd`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct SiFrame;

impl UnitFrame for SiFrame {
    const NAME: &'static str = "SI";
}

/// Nuclear femtometer–mega-electronvolt–second scale (`fm`, `MeV`, `s`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct NuclearMeVFmFrame;

impl UnitFrame for NuclearMeVFmFrame {
    const NAME: &'static str = "Nuclear(MeV-fm-s)";
}

/// Astrophysical kiloparsec length frame (`kpc`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GalacticKpcFrame;

impl UnitFrame for GalacticKpcFrame {
    const NAME: &'static str = "Galactic(kpc)";
}

/// Interactive 2D/3D normalized viewport coordinate frame.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ViewportFrame;

impl UnitFrame for ViewportFrame {
    const NAME: &'static str = "Viewport";
}

/// Calibrated desktop-slicer millimeter frame (`mm`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MillimeterFrame;

impl UnitFrame for MillimeterFrame {
    const NAME: &'static str = "Millimeter(mm)";
}
