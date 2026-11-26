//! Integration tests for IEEE-754 outward-rounded interval enclosures in `math_commons`.

use math_commons::primitives::{Interval, NonNegativeFloat, PositiveFloat, UnitInterval};
use std::f64::consts::PI;

#[test]
#[verified_engine::verified]
fn test_outward_rounding_encloses_non_representable_tenths() {
    let one = Interval::point(1.0).unwrap();
    let ten = Interval::point(10.0).unwrap();
    let tenth = one.checked_div(ten).unwrap();
    assert!(tenth.lo() < 0.1_f64 || tenth.hi() > 0.1_f64);
    assert!(tenth.width() > 0.0);
    assert!((tenth * ten).contains(1.0));
}

#[test]
#[verified_engine::verified]
fn test_transcendental_and_domain_wrappers_preserve_soundness() {
    let cos_iv = Interval::new(2.5, 4.0).unwrap().cos();
    assert_eq!(cos_iv.lo(), -1.0);
    assert!(cos_iv.contains(PI.cos()));
    let comp = UnitInterval::new(0.25).unwrap().to_interval().complement();
    assert!(comp.interval().contains(0.75));
    let sq = PositiveFloat::new(2.0)
        .unwrap()
        .to_interval()
        .interval()
        .sqrt()
        .unwrap()
        .sqr();
    assert!(sq.contains(2.0));
    assert!(
        NonNegativeFloat::new(0.0)
            .unwrap()
            .to_interval()
            .interval()
            .contains(0.0)
    );
}
