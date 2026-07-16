// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0.
// You may obtain a copy of this license in the LICENSE.txt file in
// the root directory of this source tree or at
// http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::*;
use crate::circuit::{Instruction, StandardGate};
use crate::compile::device_planning::{DeviceGateState, NativePlanCatalog};
use crate::device::{Device, InstructionProp, PhysicalQubit, QubitProp};
use smallvec::smallvec;

#[test]
fn robust_error_key_orders_coverage_before_numeric_loss() {
    let known = RobustErrorKey {
        unavailable_count: 0,
        imputed_count: 0,
        log_error: 100.0,
    };
    let imputed = RobustErrorKey {
        unavailable_count: 0,
        imputed_count: 1,
        log_error: 0.0,
    };
    let unavailable = RobustErrorKey {
        unavailable_count: 1,
        imputed_count: 0,
        log_error: 0.0,
    };

    assert_eq!(known.compare(imputed), Ordering::Less);
    assert_eq!(imputed.compare(unavailable), Ordering::Less);
    assert_eq!(unavailable.compare(known), Ordering::Greater);
}

#[test]
fn robust_error_and_duration_keys_accumulate_every_component() {
    let error = RobustErrorKey {
        unavailable_count: 1,
        imputed_count: 2,
        log_error: 0.25,
    }
    .combine(RobustErrorKey {
        unavailable_count: 3,
        imputed_count: 4,
        log_error: 0.5,
    });
    let duration = RobustDurationKey {
        unavailable_count: 2,
        imputed_count: 1,
        duration_work: 10.0,
    }
    .combine(RobustDurationKey {
        unavailable_count: 4,
        imputed_count: 3,
        duration_work: 7.5,
    });

    assert_eq!(
        error,
        RobustErrorKey {
            unavailable_count: 4,
            imputed_count: 6,
            log_error: 0.75,
        }
    );
    assert_eq!(
        duration,
        RobustDurationKey {
            unavailable_count: 6,
            imputed_count: 4,
            duration_work: 17.5,
        }
    );
}

#[test]
fn estimator_imputes_missing_calibration_from_the_same_gate() {
    let p0 = PhysicalQubit::new(0);
    let p1 = PhysicalQubit::new(1);
    let h = Instruction::Standard(StandardGate::H);
    let mut device = Device::line("calibrated-h", 2)
        .unwrap()
        .with_native_gates(vec![h.clone()])
        .unwrap();
    device
        .add_qubit_properties(
            p0,
            QubitProp::new(0.0)
                .with_native_instruction(InstructionProp::new(h, 0.02).with_length(12.0))
                .unwrap(),
        )
        .unwrap();
    let known = DeviceGateState::standard(StandardGate::H, smallvec![p0]);
    let missing = DeviceGateState::standard(StandardGate::H, smallvec![p1]);
    let catalog = NativePlanCatalog::build(&device, [known, missing.clone()]).unwrap();
    let estimator = CalibrationEstimator::from_catalog(&catalog);

    let cost = estimator.cost(catalog.summary(&missing).unwrap());
    let error = cost.error.expect("observed H error enables error scoring");
    let duration = cost
        .duration
        .expect("observed H duration enables duration scoring");

    assert_eq!(error.unavailable_count, 0);
    assert_eq!(error.imputed_count, 1);
    assert!((error.log_error - negative_log_success(0.02)).abs() < 1e-12);
    assert_eq!(duration.unavailable_count, 0);
    assert_eq!(duration.imputed_count, 1);
    assert_eq!(duration.duration_work, 12.0);
}

#[test]
fn device_estimator_uses_calibration_outside_the_prepared_catalog_roots() {
    let p0 = PhysicalQubit::new(0);
    let p1 = PhysicalQubit::new(1);
    let h = Instruction::Standard(StandardGate::H);
    let mut device = Device::line("device-wide-h", 2)
        .unwrap()
        .with_native_gates(vec![h.clone()])
        .unwrap();
    device
        .add_qubit_properties(
            p0,
            QubitProp::new(0.0)
                .with_native_instruction(InstructionProp::new(h, 0.03).with_length(18.0))
                .unwrap(),
        )
        .unwrap();
    let missing = DeviceGateState::standard(StandardGate::H, smallvec![p1]);
    let catalog = NativePlanCatalog::build(&device, [missing.clone()]).unwrap();

    let catalog_estimator = CalibrationEstimator::from_catalog(&catalog);
    let device_estimator = CalibrationEstimator::from_device(&device, &[p0, p1]);
    let summary = catalog.summary(&missing).unwrap();

    assert_eq!(catalog_estimator.cost(summary).error, None);
    let cost = device_estimator.cost(summary);
    assert_eq!(cost.error.unwrap().imputed_count, 1);
    assert_eq!(cost.duration.unwrap().duration_work, 18.0);
}

#[test]
fn estimator_disables_a_metric_when_the_device_has_no_samples() {
    let p0 = PhysicalQubit::new(0);
    let device = Device::line("uncalibrated-h", 1)
        .unwrap()
        .with_native_gates(vec![Instruction::Standard(StandardGate::H)])
        .unwrap();
    let root = DeviceGateState::standard(StandardGate::H, smallvec![p0]);
    let catalog = NativePlanCatalog::build(&device, [root.clone()]).unwrap();
    let estimator = CalibrationEstimator::from_catalog(&catalog);

    let cost = estimator.cost(catalog.summary(&root).unwrap());

    assert_eq!(cost.error, None);
    assert_eq!(cost.duration, None);
}

#[test]
fn p90_uses_the_conservative_observed_sample() {
    let estimates = quantiles(HashMap::from([("cx", vec![0.01, 0.02, 0.03, 0.04])]));
    assert_eq!(estimates["cx"], 0.04);
}

#[test]
fn p90_handles_single_and_ten_sample_boundaries() {
    let estimates = quantiles(HashMap::from([
        ("single", vec![0.125]),
        (
            "ten",
            (1..=10).map(|value| f64::from(value) / 100.0).collect(),
        ),
    ]));

    assert_eq!(estimates["single"], 0.125);
    assert_eq!(estimates["ten"], 0.09);
}

#[test]
fn unit_error_has_infinite_log_loss() {
    assert_eq!(negative_log_success(1.0), f64::INFINITY);
    assert_eq!(negative_log_success(0.0), 0.0);
    assert!(negative_log_success(0.25).is_finite());
}
