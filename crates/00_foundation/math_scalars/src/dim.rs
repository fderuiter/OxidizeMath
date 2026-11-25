//! Seven-exponent SI dimension vector [`Dim<L, M, T, I, Theta, N, J>`] and open algebra traits
//! [`AddDim`], [`SubDim`], [`NegDim`], and [`HalfDim`].
// [cite:math_scalars]

use core::fmt::Debug;
use core::marker::PhantomData;

use crate::int::{AddExp, HalfExp, N1, N2, N3, NegExp, P1, P2, P3, SubExp, TypeInt, Z0};

/// Compile-time specification of the seven SI base exponents `[L, M, T, I, Theta, N, J]`.
pub trait DimSpec: Copy + Clone + Debug + Default + PartialEq + Eq + Send + Sync + 'static {
    /// Length exponent $L$ (meter, $\text{m}$).
    const L: i8;
    /// Mass exponent $M$ (kilogram, $\text{kg}$).
    const M: i8;
    /// Time exponent $T$ (second, $\text{s}$).
    const T: i8;
    /// Electric current exponent $I$ (ampere, $\text{A}$).
    const I: i8;
    /// Thermodynamic temperature exponent $\Theta$ (kelvin, $\text{K}$).
    const THETA: i8;
    /// Amount of substance exponent $N$ (mole, $\text{mol}$).
    const N: i8;
    /// Luminous intensity exponent $J$ (candela, $\text{cd}$).
    const J: i8;

    /// Returns all seven SI base exponents `[L, M, T, I, Theta, N, J]`.
    #[inline(always)]
    fn exponents() -> [i8; 7] {
        [
            Self::L,
            Self::M,
            Self::T,
            Self::I,
            Self::THETA,
            Self::N,
            Self::J,
        ]
    }
}

/// Zero-sized SI dimensional exponent vector $(L, M, T, I, \Theta, N, J)$.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Dim<
    L: TypeInt,
    M: TypeInt,
    T: TypeInt,
    I: TypeInt,
    Theta: TypeInt,
    N: TypeInt,
    J: TypeInt,
>(PhantomData<(L, M, T, I, Theta, N, J)>);

impl<L: TypeInt, M: TypeInt, T: TypeInt, I: TypeInt, Theta: TypeInt, N: TypeInt, J: TypeInt> DimSpec
    for Dim<L, M, T, I, Theta, N, J>
{
    const L: i8 = L::EXPONENT;
    const M: i8 = M::EXPONENT;
    const T: i8 = T::EXPONENT;
    const I: i8 = I::EXPONENT;
    const THETA: i8 = Theta::EXPONENT;
    const N: i8 = N::EXPONENT;
    const J: i8 = J::EXPONENT;
}

/// Elementwise exponent addition deriving product dimensions `D1 * D2`.
pub trait AddDim<Rhs: DimSpec>: DimSpec {
    /// Product dimension with summed SI exponents.
    type Output: DimSpec;
}

impl<L1, M1, T1, I1, Th1, N1Exp, J1, L2, M2, T2, I2, Th2, N2Exp, J2>
    AddDim<Dim<L2, M2, T2, I2, Th2, N2Exp, J2>> for Dim<L1, M1, T1, I1, Th1, N1Exp, J1>
where
    L1: AddExp<L2>,
    M1: AddExp<M2>,
    T1: AddExp<T2>,
    I1: AddExp<I2>,
    Th1: AddExp<Th2>,
    N1Exp: AddExp<N2Exp>,
    J1: AddExp<J2>,
    L2: TypeInt,
    M2: TypeInt,
    T2: TypeInt,
    I2: TypeInt,
    Th2: TypeInt,
    N2Exp: TypeInt,
    J2: TypeInt,
{
    type Output =
        Dim<L1::Output, M1::Output, T1::Output, I1::Output, Th1::Output, N1Exp::Output, J1::Output>;
}

/// Elementwise exponent subtraction deriving quotient dimensions `D1 / D2`.
pub trait SubDim<Rhs: DimSpec>: DimSpec {
    /// Quotient dimension with subtracted SI exponents.
    type Output: DimSpec;
}

impl<L1, M1, T1, I1, Th1, N1Exp, J1, L2, M2, T2, I2, Th2, N2Exp, J2>
    SubDim<Dim<L2, M2, T2, I2, Th2, N2Exp, J2>> for Dim<L1, M1, T1, I1, Th1, N1Exp, J1>
where
    L1: SubExp<L2>,
    M1: SubExp<M2>,
    T1: SubExp<T2>,
    I1: SubExp<I2>,
    Th1: SubExp<Th2>,
    N1Exp: SubExp<N2Exp>,
    J1: SubExp<J2>,
    L2: TypeInt,
    M2: TypeInt,
    T2: TypeInt,
    I2: TypeInt,
    Th2: TypeInt,
    N2Exp: TypeInt,
    J2: TypeInt,
{
    type Output =
        Dim<L1::Output, M1::Output, T1::Output, I1::Output, Th1::Output, N1Exp::Output, J1::Output>;
}

/// Elementwise exponent negation deriving reciprocal dimensions `1 / D`.
pub trait NegDim: DimSpec {
    /// Reciprocal dimension with negated SI exponents.
    type Output: DimSpec;
}

impl<L, M, T, I, Th, NExp, J> NegDim for Dim<L, M, T, I, Th, NExp, J>
where
    L: NegExp,
    M: NegExp,
    T: NegExp,
    I: NegExp,
    Th: NegExp,
    NExp: NegExp,
    J: NegExp,
{
    type Output =
        Dim<L::Output, M::Output, T::Output, I::Output, Th::Output, NExp::Output, J::Output>;
}

/// Elementwise exact exponent halving deriving square-root dimensions $\sqrt{D}$ when all seven
/// SI base exponents are even.
pub trait HalfDim: DimSpec {
    /// Square-root dimension with halved SI exponents.
    type Output: DimSpec;
}

impl<L, M, T, I, Th, NExp, J> HalfDim for Dim<L, M, T, I, Th, NExp, J>
where
    L: HalfExp,
    M: HalfExp,
    T: HalfExp,
    I: HalfExp,
    Th: HalfExp,
    NExp: HalfExp,
    J: HalfExp,
{
    type Output =
        Dim<L::Output, M::Output, T::Output, I::Output, Th::Output, NExp::Output, J::Output>;
}

/// Dimensionless ratio ($L^0 M^0 T^0 I^0 \Theta^0 N^0 J^0$).
pub type Dimensionless = Dim<Z0, Z0, Z0, Z0, Z0, Z0, Z0>;
/// Length ($L^1$).
pub type DimLength = Dim<P1, Z0, Z0, Z0, Z0, Z0, Z0>;
/// Area ($L^2$).
pub type DimArea = Dim<P2, Z0, Z0, Z0, Z0, Z0, Z0>;
/// Volume ($L^3$).
pub type DimVolume = Dim<P3, Z0, Z0, Z0, Z0, Z0, Z0>;
/// Mass ($M^1$).
pub type DimMass = Dim<Z0, P1, Z0, Z0, Z0, Z0, Z0>;
/// Time ($T^1$).
pub type DimTime = Dim<Z0, Z0, P1, Z0, Z0, Z0, Z0>;
/// Frequency / rate ($T^{-1}$).
pub type DimFrequency = Dim<Z0, Z0, N1, Z0, Z0, Z0, Z0>;
/// Electric current ($I^1$).
pub type DimCurrent = Dim<Z0, Z0, Z0, P1, Z0, Z0, Z0>;
/// Thermodynamic temperature ($\Theta^1$).
pub type DimTemperature = Dim<Z0, Z0, Z0, Z0, P1, Z0, Z0>;
/// Amount of substance ($N^1$).
pub type DimAmount = Dim<Z0, Z0, Z0, Z0, Z0, P1, Z0>;
/// Luminous intensity ($J^1$).
pub type DimLuminousIntensity = Dim<Z0, Z0, Z0, Z0, Z0, Z0, P1>;
/// Velocity ($L^1 T^{-1}$).
pub type DimVelocity = Dim<P1, Z0, N1, Z0, Z0, Z0, Z0>;
/// Acceleration ($L^1 T^{-2}$).
pub type DimAcceleration = Dim<P1, Z0, N2, Z0, Z0, Z0, Z0>;
/// Force ($L^1 M^1 T^{-2}$).
pub type DimForce = Dim<P1, P1, N2, Z0, Z0, Z0, Z0>;
/// Pressure / volumetric energy density ($L^{-1} M^1 T^{-2}$).
pub type DimPressure = Dim<N1, P1, N2, Z0, Z0, Z0, Z0>;
/// Volumetric energy density ($L^{-1} M^1 T^{-2}$).
pub type DimEnergyDensity = DimPressure;
/// Energy ($L^2 M^1 T^{-2}$).
pub type DimEnergy = Dim<P2, P1, N2, Z0, Z0, Z0, Z0>;
/// Power ($L^2 M^1 T^{-3}$).
pub type DimPower = Dim<P2, P1, N3, Z0, Z0, Z0, Z0>;
/// Action / angular momentum ($L^2 M^1 T^{-1}$).
pub type DimAction = Dim<P2, P1, N1, Z0, Z0, Z0, Z0>;
/// Energy times length ($L^3 M^1 T^{-2}$, e.g. $\hbar c$ or $e^2$ in nuclear units).
pub type DimEnergyLength = Dim<P3, P1, N2, Z0, Z0, Z0, Z0>;
/// Magnetic flux density ($M^1 T^{-2} I^{-1}$, Tesla).
pub type DimMagneticFluxDensity = Dim<Z0, P1, N2, N1, Z0, Z0, Z0>;
/// Gyromagnetic ratio ($M^{-1} T^1 I^1$, $\text{rad}\cdot\text{s}^{-1}\cdot\text{T}^{-1}$).
pub type DimGyromagneticRatio = Dim<Z0, N1, P1, P1, Z0, Z0, Z0>;
/// Entropy / Boltzmann heat capacity ($L^2 M^1 T^{-2} \Theta^{-1}$, $\text{J}\cdot\text{K}^{-1}$).
pub type DimHeatCapacity = Dim<P2, P1, N2, Z0, N1, Z0, Z0>;
/// Molar energy ($L^2 M^1 T^{-2} N^{-1}$, $\text{J}\cdot\text{mol}^{-1}$).
pub type DimMolarEnergy = Dim<P2, P1, N2, Z0, Z0, N1, Z0>;
/// Molar heat capacity / universal gas constant ($L^2 M^1 T^{-2} \Theta^{-1} N^{-1}$, $\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}$).
pub type DimMolarHeatCapacity = Dim<P2, P1, N2, Z0, N1, N1, Z0>;
/// Newtonian gravitational constant ($L^3 M^{-1} T^{-2}$).
pub type DimGravitationalConstant = Dim<P3, N1, N2, Z0, Z0, Z0, Z0>;
