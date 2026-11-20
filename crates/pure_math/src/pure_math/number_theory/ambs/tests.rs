use super::*;

#[test]
fn test_tonelli_shanks_branches_and_refinement() {
    let p7 = Integer::from(7);
    assert_eq!(
        AmbsDsp::tonelli_shanks(&Integer::from(0), &p7).unwrap(),
        vec![Integer::from(0)]
    );
    assert!(
        AmbsDsp::tonelli_shanks(&Integer::from(3), &p7)
            .unwrap()
            .is_empty()
    );
    let mut r7 = AmbsDsp::tonelli_shanks(&Integer::from(2), &p7).unwrap();
    r7.sort();
    assert_eq!(r7, vec![Integer::from(3), Integer::from(4)]);

    // p = 41 (p - 1 = 5 * 2^3) exercises multi-step Tonelli-Shanks refinement
    let p41 = Integer::from(41);
    let mut r41 = AmbsDsp::tonelli_shanks(&Integer::from(2), &p41).unwrap();
    r41.sort();
    assert_eq!(r41, vec![Integer::from(17), Integer::from(24)]);

    // p = 17 (p - 1 = 2^4) exercises high 2-adic valuation
    let p17 = Integer::from(17);
    let mut r17 = AmbsDsp::tonelli_shanks(&Integer::from(2), &p17).unwrap();
    r17.sort();
    assert_eq!(r17, vec![Integer::from(6), Integer::from(11)]);
}
