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

use cqlib_core::circuit::{Circuit, Instruction, Qubit, StandardGate};
use cqlib_core::compile::transform::KnowledgeRewriter;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

fn cancellation_circuit(operation_count: usize) -> Circuit {
    let mut circuit = Circuit::new(1);
    let qubit = Qubit::new(0);
    for _ in 0..operation_count / 2 {
        circuit.h(qubit).unwrap();
        circuit.h(qubit).unwrap();
    }
    circuit
}

fn sparse_two_round_circuit(operation_count: usize) -> Circuit {
    let mut circuit = Circuit::new(1);
    let qubit = Qubit::new(0);
    let tail_start = operation_count / 2;
    for position in 0..operation_count {
        let gate = match position.checked_sub(tail_start) {
            Some(0 | 3) => StandardGate::X,
            Some(1 | 2) => StandardGate::H,
            _ => {
                let stable_gate = if position == 0 {
                    StandardGate::X
                } else {
                    StandardGate::H
                };
                circuit
                    .append(
                        Instruction::Standard(stable_gate),
                        [qubit],
                        std::iter::empty(),
                        Some("stable"),
                    )
                    .unwrap();
                continue;
            }
        };
        circuit
            .append(
                Instruction::Standard(gate),
                [qubit],
                std::iter::empty(),
                None,
            )
            .unwrap();
    }
    circuit
}

fn benchmark_knowledge_rewrite(c: &mut Criterion) {
    let mut group = c.benchmark_group("knowledge_rewrite_incremental");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));
    for operation_count in [4_200, 70_000] {
        let circuit = cancellation_circuit(operation_count);
        group.bench_with_input(
            BenchmarkId::new("cancel_h_pairs", operation_count),
            &circuit,
            |b, circuit| {
                b.iter(|| {
                    black_box(
                        KnowledgeRewriter::production()
                            .run(black_box(circuit))
                            .unwrap(),
                    )
                });
            },
        );
    }
    let sparse = sparse_two_round_circuit(70_000);
    group.bench_with_input(
        BenchmarkId::new("sparse_two_round_tail", 70_000),
        &sparse,
        |b, circuit| {
            b.iter(|| {
                black_box(
                    KnowledgeRewriter::production()
                        .run(black_box(circuit))
                        .unwrap(),
                )
            });
        },
    );
    group.finish();
}

criterion_group!(benches, benchmark_knowledge_rewrite);
criterion_main!(benches);
