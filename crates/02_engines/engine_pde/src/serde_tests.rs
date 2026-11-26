// [cite:engine_pde]
#[test]
fn pde_serde_json_dev_dep_linked() {
    let v: usize = serde_json::from_str("8").expect("json scalar");
    assert_eq!(v, 8);
}

#[cfg(feature = "serde")]
#[test]
fn pde_boundary_noise_and_stencils_roundtrip_no_std() {
    use crate::{BoundaryCondition, SpdeNoiseKind, StencilNeighbors3D, StencilOperators};
    use math_commons::math_kernel::types::StepSize;

    for bc in [
        BoundaryCondition::Periodic,
        BoundaryCondition::Neumann,
        BoundaryCondition::SymmetricNeumann,
        BoundaryCondition::Dirichlet(-2.5),
    ] {
        let json = serde_json::to_string(&bc).unwrap();
        assert_eq!(
            serde_json::from_str::<BoundaryCondition>(&json).unwrap(),
            bc
        );
    }

    for noise in [
        SpdeNoiseKind::None,
        SpdeNoiseKind::Additive { intensity: 0.05 },
        SpdeNoiseKind::ConservedFlux {
            mobility_temp: 0.015,
        },
    ] {
        let json = serde_json::to_string(&noise).unwrap();
        assert_eq!(serde_json::from_str::<SpdeNoiseKind>(&json).unwrap(), noise);
    }

    let neighbors = StencilNeighbors3D::<2> {
        xm: [1.0, 2.0],
        xp: [3.0, 4.0],
        ym: [5.0, 6.0],
        yp: [7.0, 8.0],
        zm: [9.0, 10.0],
        zp: [11.0, 12.0],
    };
    let n_json = serde_json::to_string(&neighbors).unwrap();
    assert_eq!(
        serde_json::from_str::<StencilNeighbors3D<2>>(&n_json).unwrap(),
        neighbors
    );

    let neighbors33 = StencilNeighbors3D::<33> {
        xm: [0.5; 33],
        xp: [1.5; 33],
        ym: [2.5; 33],
        yp: [3.5; 33],
        zm: [4.5; 33],
        zp: [5.5; 33],
    };
    let n33_json = serde_json::to_string(&neighbors33).unwrap();
    assert_eq!(
        serde_json::from_str::<StencilNeighbors3D<33>>(&n33_json).unwrap(),
        neighbors33
    );

    let ops = StencilOperators::new_3d(StepSize(0.25), StepSize(0.5), StepSize(1.0));
    let ops_json = serde_json::to_string(&ops).unwrap();
    let decoded_ops: StencilOperators = serde_json::from_str(&ops_json).unwrap();
    assert_eq!(decoded_ops, ops);
    assert_eq!(decoded_ops.inv_dx, 4.0);
    assert_eq!(decoded_ops.inv_dy, 2.0);
}

#[cfg(feature = "serde")]
#[test]
fn pde_steppers_roundtrip_no_std() {
    use crate::{
        BoundaryCondition, ExplicitSpdeStencilStepper, FusedStencilStepper,
        SemiImplicitFourierStepper, SpdeNoiseKind, SplitStepSpdeStencilStepper, StencilOperators,
    };
    use math_commons::math_kernel::types::StepSize;

    let ops = StencilOperators::new_3d(StepSize(0.25), StepSize(0.5), StepSize(1.0));

    let fused = FusedStencilStepper { ops };
    let fused_json = serde_json::to_string(&fused).unwrap();
    assert_eq!(
        serde_json::from_str::<FusedStencilStepper>(&fused_json).unwrap(),
        fused
    );

    let fourier = SemiImplicitFourierStepper::new_2d(0.5, 0.25);
    let fourier_json = serde_json::to_string(&fourier).unwrap();
    assert_eq!(
        serde_json::from_str::<SemiImplicitFourierStepper>(&fourier_json).unwrap(),
        fourier
    );

    let explicit = ExplicitSpdeStencilStepper::new(
        ops,
        BoundaryCondition::Neumann,
        SpdeNoiseKind::Additive { intensity: 0.1 },
    );
    let explicit_json = serde_json::to_string(&explicit).unwrap();
    assert_eq!(
        serde_json::from_str::<ExplicitSpdeStencilStepper>(&explicit_json).unwrap(),
        explicit
    );

    let split = SplitStepSpdeStencilStepper::new(
        ops,
        BoundaryCondition::Periodic,
        SpdeNoiseKind::ConservedFlux {
            mobility_temp: 0.02,
        },
        4,
    );
    let split_json = serde_json::to_string(&split).unwrap();
    assert_eq!(
        serde_json::from_str::<SplitStepSpdeStencilStepper>(&split_json).unwrap(),
        split
    );
}

#[cfg(feature = "serde")]
#[test]
fn pde_error_roundtrips_no_std() {
    use crate::PdeError;
    use math_commons::error::MathError;

    let samples = [
        PdeError::InvalidGridDimension {
            axis: "width",
            size: 0,
        },
        PdeError::NonRadix2Dimension { axis: "n", size: 6 },
        PdeError::IndexOutOfBounds {
            axis: "y",
            index: 9,
            len: 4,
        },
        PdeError::BufferTooSmall {
            buffer: "phi",
            required: 16,
            actual: 8,
        },
        PdeError::InvalidParameter { parameter: "dx" },
        PdeError::WorkspaceAllocationFailed { capacity: 256 },
        PdeError::Math(MathError::WorkspaceTooSmall {
            required: 32,
            available: 12,
        }),
    ];
    for sample in samples {
        let json = serde_json::to_string(&sample).expect("serialize PdeError");
        let decoded: PdeError = serde_json::from_str(&json).expect("deserialize PdeError");
        assert_eq!(decoded, sample);
    }
}
