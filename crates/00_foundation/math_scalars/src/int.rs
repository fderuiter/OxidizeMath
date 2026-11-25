//! Open canonical signed type-level integer exponents on stable Rust without macros or nightly
//! `generic_const_exprs`.
// [cite:math_scalars]

use core::fmt::Debug;
use core::marker::PhantomData;

/// Strictly positive Peano natural number (`1, 2, 3, ...`) used inside [`PInt`] and [`NInt`].
pub trait NonZeroNat:
    Copy + Clone + Debug + Default + PartialEq + Eq + Send + Sync + 'static
{
    /// Positive integer magnitude (`>= 1`).
    const VALUE: i8;
}

/// Type-level positive natural `1`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct U1;

impl NonZeroNat for U1 {
    const VALUE: i8 = 1;
}

/// Successor `N + 1` of a strictly positive natural `N`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Succ<N: NonZeroNat>(PhantomData<N>);

impl<N: NonZeroNat> NonZeroNat for Succ<N> {
    const VALUE: i8 = N::VALUE + 1;
}

/// Canonical signed type-level integer (`..., -2, -1, 0, +1, +2, ...`).
///
/// Every integer has a unique normal form ([`Z0`], [`PInt<N>`], or [`NInt<N>`]), guaranteeing
/// canonical type equality after arbitrary sequences of additions, subtractions, and square roots.
pub trait TypeInt: Copy + Clone + Debug + Default + PartialEq + Eq + Send + Sync + 'static {
    /// Runtime `i8` value of this type-level integer exponent.
    const EXPONENT: i8;
}

/// Type-level integer `0`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Z0;

impl TypeInt for Z0 {
    const EXPONENT: i8 = 0;
}

/// Strictly positive type-level integer `+N`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PInt<N: NonZeroNat>(PhantomData<N>);

impl<N: NonZeroNat> TypeInt for PInt<N> {
    const EXPONENT: i8 = N::VALUE;
}

/// Strictly negative type-level integer `-N`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct NInt<N: NonZeroNat>(PhantomData<N>);

impl<N: NonZeroNat> TypeInt for NInt<N> {
    const EXPONENT: i8 = -N::VALUE;
}

/// Type-level positive natural `2`.
pub type Nat2 = Succ<U1>;
/// Type-level positive natural `3`.
pub type Nat3 = Succ<Nat2>;
/// Type-level positive natural `4`.
pub type Nat4 = Succ<Nat3>;
/// Type-level positive natural `5`.
pub type Nat5 = Succ<Nat4>;
/// Type-level positive natural `6`.
pub type Nat6 = Succ<Nat5>;
/// Type-level positive natural `7`.
pub type Nat7 = Succ<Nat6>;
/// Type-level positive natural `8`.
pub type Nat8 = Succ<Nat7>;

/// Type-level integer `+1`.
pub type P1 = PInt<U1>;
/// Type-level integer `+2`.
pub type P2 = PInt<Nat2>;
/// Type-level integer `+3`.
pub type P3 = PInt<Nat3>;
/// Type-level integer `+4`.
pub type P4 = PInt<Nat4>;
/// Type-level integer `+5`.
pub type P5 = PInt<Nat5>;
/// Type-level integer `+6`.
pub type P6 = PInt<Nat6>;
/// Type-level integer `+7`.
pub type P7 = PInt<Nat7>;
/// Type-level integer `+8`.
pub type P8 = PInt<Nat8>;

/// Type-level integer `-1`.
pub type N1 = NInt<U1>;
/// Type-level integer `-2`.
pub type N2 = NInt<Nat2>;
/// Type-level integer `-3`.
pub type N3 = NInt<Nat3>;
/// Type-level integer `-4`.
pub type N4 = NInt<Nat4>;
/// Type-level integer `-5`.
pub type N5 = NInt<Nat5>;
/// Type-level integer `-6`.
pub type N6 = NInt<Nat6>;
/// Type-level integer `-7`.
pub type N7 = NInt<Nat7>;
/// Type-level integer `-8`.
pub type N8 = NInt<Nat8>;

/// Addition of two strictly positive Peano naturals.
pub trait AddNat<Rhs: NonZeroNat>: NonZeroNat {
    /// Sum `Self + Rhs`.
    type Output: NonZeroNat;
}

impl<B: NonZeroNat> AddNat<B> for U1 {
    type Output = Succ<B>;
}

impl<A: NonZeroNat + AddNat<B>, B: NonZeroNat> AddNat<B> for Succ<A> {
    type Output = Succ<<A as AddNat<B>>::Output>;
}

/// Difference `Self - Rhs` of two strictly positive naturals normalized to a [`TypeInt`].
pub trait SubNat<Rhs: NonZeroNat>: NonZeroNat {
    /// Canonical signed integer result of `Self - Rhs`.
    type Output: TypeInt;
}

impl SubNat<U1> for U1 {
    type Output = Z0;
}

impl<A: NonZeroNat> SubNat<U1> for Succ<A> {
    type Output = PInt<A>;
}

impl<B: NonZeroNat> SubNat<Succ<B>> for U1 {
    type Output = NInt<B>;
}

impl<A: NonZeroNat + SubNat<B>, B: NonZeroNat> SubNat<Succ<B>> for Succ<A> {
    type Output = <A as SubNat<B>>::Output;
}

/// Open type-level integer addition `Self + Rhs`.
pub trait AddExp<Rhs: TypeInt>: TypeInt {
    /// Canonical sum exponent.
    type Output: TypeInt;
}

impl<R: TypeInt> AddExp<R> for Z0 {
    type Output = R;
}

impl<A: NonZeroNat> AddExp<Z0> for PInt<A> {
    type Output = PInt<A>;
}

impl<A: NonZeroNat> AddExp<Z0> for NInt<A> {
    type Output = NInt<A>;
}

impl<A: NonZeroNat + AddNat<B>, B: NonZeroNat> AddExp<PInt<B>> for PInt<A> {
    type Output = PInt<<A as AddNat<B>>::Output>;
}

impl<A: NonZeroNat + AddNat<B>, B: NonZeroNat> AddExp<NInt<B>> for NInt<A> {
    type Output = NInt<<A as AddNat<B>>::Output>;
}

impl<A: NonZeroNat + SubNat<B>, B: NonZeroNat> AddExp<NInt<B>> for PInt<A> {
    type Output = <A as SubNat<B>>::Output;
}

impl<A: NonZeroNat, B: NonZeroNat + SubNat<A>> AddExp<PInt<B>> for NInt<A> {
    type Output = <B as SubNat<A>>::Output;
}

/// Open type-level integer negation `-Self`.
pub trait NegExp: TypeInt {
    /// Canonical negated exponent.
    type Output: TypeInt;
}

impl NegExp for Z0 {
    type Output = Z0;
}

impl<A: NonZeroNat> NegExp for PInt<A> {
    type Output = NInt<A>;
}

impl<A: NonZeroNat> NegExp for NInt<A> {
    type Output = PInt<A>;
}

/// Open type-level integer subtraction `Self - Rhs`.
pub trait SubExp<Rhs: TypeInt>: TypeInt {
    /// Canonical difference exponent.
    type Output: TypeInt;
}

impl<L, R> SubExp<R> for L
where
    L: TypeInt + AddExp<<R as NegExp>::Output>,
    R: NegExp,
{
    type Output = <L as AddExp<<R as NegExp>::Output>>::Output;
}

/// Exact halving of even positive Peano naturals (`2 -> 1`, `4 -> 2`, ...).
///
/// Odd naturals (`1`, `3`, `5`, ...) intentionally omit `HalfNat` so odd-exponent square roots
/// fail to compile.
pub trait HalfNat: NonZeroNat {
    /// Exact half `Self / 2`.
    type Output: NonZeroNat;
}

impl HalfNat for Succ<U1> {
    type Output = U1;
}

impl<A: HalfNat> HalfNat for Succ<Succ<A>> {
    type Output = Succ<A::Output>;
}

/// Open exact halving `Self / 2` for even type-level integer exponents (`Z0`, `P2`, `N2`, `P4`, ...).
pub trait HalfExp: TypeInt {
    /// Exact halved exponent.
    type Output: TypeInt;
}

impl HalfExp for Z0 {
    type Output = Z0;
}

impl<A: HalfNat> HalfExp for PInt<A> {
    type Output = PInt<A::Output>;
}

impl<A: HalfNat> HalfExp for NInt<A> {
    type Output = NInt<A::Output>;
}
