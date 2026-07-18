// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use cqlib_core::circuit::{Circuit, Qubit, StandardGate, UnitaryGate};
use cqlib_core::compile::transform::decompose::unitary::{
    UnitaryDecomposeConfig, decompose_unitaries_with_rule_stats,
};
use criterion::{Criterion, criterion_group, criterion_main};

fn repeated_two_qubit_unitaries(repetitions: usize) -> Circuit {
    let matrix = StandardGate::FSIM
        .matrix(&[0.23, -0.31])
        .expect("fSim matrix")
        .into_owned();
    let mut circuit = Circuit::new(2);
    for _ in 0..repetitions {
        let gate = UnitaryGate::new("repeated", 2, 0)
            .with_matrix(matrix.clone())
            .expect("valid two-qubit unitary gate");
        circuit
            .unitary(gate, vec![Qubit::new(0), Qubit::new(1)])
            .expect("valid unitary operation");
    }
    circuit
}

fn benchmark_repeated_two_qubit_synthesis_cache(criterion: &mut Criterion) {
    let circuit = repeated_two_qubit_unitaries(128);
    criterion.bench_function(
        "2q_unitary_decomposition/repeated_cache_warm_path",
        |bencher| {
            bencher.iter(|| {
                let (_, stats) = decompose_unitaries_with_rule_stats(
                    &circuit,
                    UnitaryDecomposeConfig::default(),
                )
                .expect("repeated 2q unitary decomposition");
                assert_eq!(stats.misses, 1);
                assert_eq!(stats.inserts, 1);
                assert_eq!(stats.hits, 127);
            });
        },
    );
}

criterion_group!(benches, benchmark_repeated_two_qubit_synthesis_cache);
criterion_main!(benches);
