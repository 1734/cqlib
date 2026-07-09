// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Numeric two-qubit unitary synthesis.
//!
//! This module converts a concrete 4x4 unitary matrix into standard-gate
//! operations over two supplied qubits plus a scalar global phase. The
//! circuit-agnostic `kak_decompose` primitive first factors the matrix into
//! local single-qubit matrices and a Cartan interaction. This layer then emits
//! the factors in the output basis selected by
//! [`TwoQubitUnitaryDecomposeBasis`].
//!
//! Local factors are lowered through the one-qubit synthesizer and emitted as
//! [`StandardGate::U`] operations. Numerically trivial local gates and
//! interaction rotations are omitted using `ANGLE_EPS`, while their scalar
//! phases remain accumulated in the returned phase.
//!
//! The `PauliRotations` backend emits the Cartan core directly as
//! `RXX`/`RYY`/`RZZ`. The `Cx`/`Cy`/`Cz` backends deterministically select
//! among templates containing zero through three native entanglers using an
//! average-fidelity score. The `Rzz` backend emits the Cartan core with at most
//! three `RZZ` interactions, using local basis changes for the `XX` and `YY`
//! axes. For template backends, a higher-count template is selected only when
//! its score improves by more than `1e-12`, so near-boundary inputs may
//! intentionally keep a lower-count approximation.

use super::two_qubit_kak::{KakDecomposition, kak_decompose};
use super::unitary_1q::{OneQubitUnitaryDecomposition, synthesize_numeric_1q_unitary};
use crate::circuit::gate::gate_matrix::rz_gate;
use crate::circuit::{ParameterValue, Qubit, StandardGate, ValueOperation};
use crate::compile::CompilerError;
use crate::util::matrix::{c, dagger, mat2};
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4, PI};

const ANGLE_EPS: f64 = 1e-12;
const FIDELITY_IMPROVEMENT_EPS: f64 = 1e-12;

/// Output basis used for two-qubit unitary synthesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TwoQubitUnitaryDecomposeBasis {
    /// Emit local `U` gates plus `RXX/RYY/RZZ` for the Cartan core.
    PauliRotations,
    /// Emit local `U` gates plus optimized `CX` templates.
    Cx,
    /// Emit local `U` gates plus optimized `CY` templates.
    Cy,
    /// Emit local `U` gates plus optimized `CZ` templates.
    Cz,
    /// Emit local `U` gates plus `RZZ` interactions for the Cartan core.
    Rzz,
}

/// Numeric synthesis result for a two-qubit unitary matrix.
#[derive(Clone, Debug)]
pub struct TwoQubitUnitarySynthesisResult {
    /// Self-contained standard-gate operations implementing the unitary up to
    /// [`global_phase`](Self::global_phase).
    pub operations: Vec<ValueOperation>,
    /// Scalar phase multiplying the emitted operation sequence.
    pub global_phase: f64,
}

/// Synthesizes a finite 4x4 unitary matrix into the selected interaction basis.
///
/// # Errors
///
/// Returns [`CompilerError`] when `matrix` is not a finite 4x4 unitary, the
/// qubits are not distinct, or the numeric decomposition cannot be validated.
pub fn synthesize_numeric_2q_unitary(
    matrix: &Array2<Complex64>,
    qubits: [Qubit; 2],
    basis: TwoQubitUnitaryDecomposeBasis,
) -> Result<TwoQubitUnitarySynthesisResult, CompilerError> {
    if qubits[0] == qubits[1] {
        return Err(CompilerError::InvalidInput(format!(
            "2q unitary synthesis requires distinct qubits, both are {}",
            qubits[0]
        )));
    }
    let decomp = kak_decompose(matrix)?;
    let mut builder = OperationBuilder::default();
    match basis {
        TwoQubitUnitaryDecomposeBasis::PauliRotations => {
            emit_pauli_rotations(&mut builder, &decomp, qubits[0], qubits[1])?
        }
        TwoQubitUnitaryDecomposeBasis::Cx => emit_cx(&mut builder, &decomp, qubits[0], qubits[1])?,
        TwoQubitUnitaryDecomposeBasis::Cy => emit_cy(&mut builder, &decomp, qubits[0], qubits[1])?,
        TwoQubitUnitaryDecomposeBasis::Cz => emit_cz(&mut builder, &decomp, qubits[0], qubits[1])?,
        TwoQubitUnitaryDecomposeBasis::Rzz => {
            emit_rzz_only(&mut builder, &decomp, qubits[0], qubits[1])?
        }
    }
    Ok(TwoQubitUnitarySynthesisResult {
        operations: builder.operations,
        global_phase: builder.global_phase,
    })
}

#[derive(Default)]
struct OperationBuilder {
    operations: Vec<ValueOperation>,
    global_phase: f64,
}

impl OperationBuilder {
    fn push_local_u(
        &mut self,
        qubit: Qubit,
        matrix: &Array2<Complex64>,
    ) -> Result<(), CompilerError> {
        let OneQubitUnitaryDecomposition {
            theta,
            phi,
            lambda,
            global_phase,
        } = synthesize_numeric_1q_unitary(matrix)?;
        self.global_phase += global_phase;
        if theta.abs() <= ANGLE_EPS && phi.abs() <= ANGLE_EPS && lambda.abs() <= ANGLE_EPS {
            return Ok(());
        }

        self.operations.push(ValueOperation::from_standard(
            StandardGate::U,
            [qubit],
            [
                ParameterValue::Fixed(theta),
                ParameterValue::Fixed(phi),
                ParameterValue::Fixed(lambda),
            ],
        ));
        Ok(())
    }

    fn push_rotation(&mut self, gate: StandardGate, first: Qubit, second: Qubit, theta: f64) {
        if theta.abs() <= ANGLE_EPS {
            return;
        }

        self.operations.push(ValueOperation::from_standard(
            gate,
            [first, second],
            [ParameterValue::Fixed(theta)],
        ));
    }

    fn push_1q_gate(&mut self, gate: StandardGate, qubit: Qubit) {
        self.operations
            .push(ValueOperation::from_standard(gate, [qubit], []));
    }

    fn push_1q_rotation(&mut self, gate: StandardGate, qubit: Qubit, theta: f64) {
        if theta.abs() <= ANGLE_EPS {
            return;
        }

        self.operations.push(ValueOperation::from_standard(
            gate,
            [qubit],
            [ParameterValue::Fixed(theta)],
        ));
    }

    fn push_cx(&mut self, control: Qubit, target: Qubit) {
        self.operations.push(ValueOperation::from_standard(
            StandardGate::CX,
            [control, target],
            [],
        ));
    }

    fn push_cy(&mut self, control: Qubit, target: Qubit) {
        self.operations.push(ValueOperation::from_standard(
            StandardGate::CY,
            [control, target],
            [],
        ));
    }

    fn push_cz(&mut self, first: Qubit, second: Qubit) {
        self.operations.push(ValueOperation::from_standard(
            StandardGate::CZ,
            [first, second],
            [],
        ));
    }
}

fn emit_pauli_rotations(
    builder: &mut OperationBuilder,
    decomp: &KakDecomposition,
    first: Qubit,
    second: Qubit,
) -> Result<(), CompilerError> {
    builder.global_phase += decomp.global_phase;
    builder.push_local_u(first, &decomp.k2l)?;
    builder.push_local_u(second, &decomp.k2r)?;
    builder.push_rotation(StandardGate::RXX, first, second, -2.0 * decomp.a);
    builder.push_rotation(StandardGate::RYY, first, second, -2.0 * decomp.b);
    builder.push_rotation(StandardGate::RZZ, first, second, -2.0 * decomp.c);
    builder.push_local_u(first, &decomp.k1l)?;
    builder.push_local_u(second, &decomp.k1r)?;
    Ok(())
}

fn emit_rzz_only(
    builder: &mut OperationBuilder,
    decomp: &KakDecomposition,
    first: Qubit,
    second: Qubit,
) -> Result<(), CompilerError> {
    builder.global_phase += decomp.global_phase;
    builder.push_local_u(first, &decomp.k2l)?;
    builder.push_local_u(second, &decomp.k2r)?;
    emit_rxx_as_rzz(builder, first, second, -2.0 * decomp.a);
    emit_ryy_as_rzz(builder, first, second, -2.0 * decomp.b);
    builder.push_rotation(StandardGate::RZZ, first, second, -2.0 * decomp.c);
    builder.push_local_u(first, &decomp.k1l)?;
    builder.push_local_u(second, &decomp.k1r)?;
    Ok(())
}

fn emit_rxx_as_rzz(builder: &mut OperationBuilder, first: Qubit, second: Qubit, theta: f64) {
    if theta.abs() <= ANGLE_EPS {
        return;
    }

    builder.push_1q_gate(StandardGate::H, first);
    builder.push_1q_gate(StandardGate::H, second);
    builder.push_rotation(StandardGate::RZZ, first, second, theta);
    builder.push_1q_gate(StandardGate::H, second);
    builder.push_1q_gate(StandardGate::H, first);
}

fn emit_ryy_as_rzz(builder: &mut OperationBuilder, first: Qubit, second: Qubit, theta: f64) {
    if theta.abs() <= ANGLE_EPS {
        return;
    }

    builder.push_1q_rotation(StandardGate::RX, first, FRAC_PI_2);
    builder.push_1q_rotation(StandardGate::RX, second, FRAC_PI_2);
    builder.push_rotation(StandardGate::RZZ, first, second, theta);
    builder.push_1q_rotation(StandardGate::RX, second, -FRAC_PI_2);
    builder.push_1q_rotation(StandardGate::RX, first, -FRAC_PI_2);
}

fn emit_cx(
    builder: &mut OperationBuilder,
    target: &KakDecomposition,
    first: Qubit,
    second: Qubit,
) -> Result<(), CompilerError> {
    let basis = CxBasisData::new()?;
    let num_cx = basis.num_basis_gates(target);
    let locals = basis.local_decomposition(target, num_cx);

    builder.global_phase += target.global_phase - num_cx as f64 * basis.global_phase;
    if num_cx == 2 {
        builder.global_phase += PI;
    }

    for i in 0..num_cx {
        builder.push_local_u(first, &locals[2 * i + 1])?;
        builder.push_local_u(second, &locals[2 * i])?;
        builder.push_cx(first, second);
    }
    builder.push_local_u(first, &locals[2 * num_cx + 1])?;
    builder.push_local_u(second, &locals[2 * num_cx])?;
    Ok(())
}

fn emit_cy(
    builder: &mut OperationBuilder,
    target: &KakDecomposition,
    first: Qubit,
    second: Qubit,
) -> Result<(), CompilerError> {
    let basis = CxBasisData::new()?;
    let num_cy = basis.num_basis_gates(target);
    let mut locals = basis.local_decomposition(target, num_cy);
    let s = StandardGate::S
        .matrix(&[])
        .map_err(|e| CompilerError::InvalidInput(e.to_string()))?
        .into_owned();
    let sdg = StandardGate::SDG
        .matrix(&[])
        .map_err(|e| CompilerError::InvalidInput(e.to_string()))?
        .into_owned();
    absorb_cx_replacement_locals(&mut locals, num_cy, &s, &sdg);

    builder.global_phase += target.global_phase - num_cy as f64 * basis.global_phase;
    if num_cy == 2 {
        builder.global_phase += PI;
    }

    for i in 0..num_cy {
        builder.push_local_u(first, &locals[2 * i + 1])?;
        builder.push_local_u(second, &locals[2 * i])?;
        builder.push_cy(first, second);
    }
    builder.push_local_u(first, &locals[2 * num_cy + 1])?;
    builder.push_local_u(second, &locals[2 * num_cy])?;
    Ok(())
}

fn emit_cz(
    builder: &mut OperationBuilder,
    target: &KakDecomposition,
    first: Qubit,
    second: Qubit,
) -> Result<(), CompilerError> {
    let basis = CxBasisData::new()?;
    let num_cz = basis.num_basis_gates(target);
    let mut locals = basis.local_decomposition(target, num_cz);
    let h = StandardGate::H
        .matrix(&[])
        .map_err(|e| CompilerError::InvalidInput(e.to_string()))?
        .into_owned();
    absorb_cx_replacement_locals(&mut locals, num_cz, &h, &h);

    builder.global_phase += target.global_phase - num_cz as f64 * basis.global_phase;
    if num_cz == 2 {
        builder.global_phase += PI;
    }

    for i in 0..num_cz {
        builder.push_local_u(first, &locals[2 * i + 1])?;
        builder.push_local_u(second, &locals[2 * i])?;
        builder.push_cz(first, second);
    }
    builder.push_local_u(first, &locals[2 * num_cz + 1])?;
    builder.push_local_u(second, &locals[2 * num_cz])?;
    Ok(())
}

fn absorb_cx_replacement_locals(
    locals: &mut [Array2<Complex64>],
    entangler_count: usize,
    pre: &Array2<Complex64>,
    post: &Array2<Complex64>,
) {
    if entangler_count == 0 {
        return;
    }

    for local_index in 0..=entangler_count {
        let right_index = 2 * local_index;
        locals[right_index] = match local_index {
            0 => pre.dot(&locals[right_index]),
            index if index == entangler_count => locals[right_index].dot(post),
            _ => pre.dot(&locals[right_index].dot(post)),
        };
    }
}

struct CxBasisData {
    basis: KakDecomposition,
    u0l: Array2<Complex64>,
    u0r: Array2<Complex64>,
    u1l: Array2<Complex64>,
    u1ra: Array2<Complex64>,
    u1rb: Array2<Complex64>,
    u2la: Array2<Complex64>,
    u2lb: Array2<Complex64>,
    u2ra: Array2<Complex64>,
    u2rb: Array2<Complex64>,
    u3l: Array2<Complex64>,
    u3r: Array2<Complex64>,
    q0l: Array2<Complex64>,
    q0r: Array2<Complex64>,
    q1la: Array2<Complex64>,
    q1lb: Array2<Complex64>,
    q1ra: Array2<Complex64>,
    q1rb: Array2<Complex64>,
    q2l: Array2<Complex64>,
    q2r: Array2<Complex64>,
    global_phase: f64,
}

impl CxBasisData {
    fn new() -> Result<Self, CompilerError> {
        let cx = StandardGate::CX
            .matrix(&[])
            .map_err(|e| CompilerError::InvalidInput(e.to_string()))?;
        let basis = kak_decompose(cx.as_ref())?;
        let b = basis.b;

        // Closed-form local-equivalence templates for realizing a target KAK
        // point with 0, 1, 2, or 3 CX basis gates. The matrices below are the
        // fixed local corrections around the CX basis KAK coordinates; the
        // target-specific angles are injected later in `local_decomposition`.
        let k12r = mat2(
            c(0.0, FRAC_1_SQRT_2),
            c(FRAC_1_SQRT_2, 0.0),
            c(-FRAC_1_SQRT_2, 0.0),
            c(0.0, -FRAC_1_SQRT_2),
        );
        let k12r_dg = dagger(&k12r);
        let k12l = mat2(c(0.5, 0.5), c(0.5, 0.5), c(-0.5, 0.5), c(0.5, -0.5));
        let k12l_dg = dagger(&k12l);
        let k22l = mat2(
            c(FRAC_1_SQRT_2, 0.0),
            c(-FRAC_1_SQRT_2, 0.0),
            c(FRAC_1_SQRT_2, 0.0),
            c(FRAC_1_SQRT_2, 0.0),
        );
        let k22r = mat2(c(0.0, 0.0), c(1.0, 0.0), c(-1.0, 0.0), c(0.0, 0.0));
        let ipz = mat2(c(0.0, 1.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, -1.0));

        let exp_pos_b = Complex64::from_polar(1.0, b);
        let exp_neg_b = Complex64::from_polar(1.0, -b);
        let exp_pos_2b = Complex64::from_polar(1.0, 2.0 * b);
        let exp_neg_2b = Complex64::from_polar(1.0, -2.0 * b);
        let i = c(0.0, 1.0);
        let minus_i = c(0.0, -1.0);

        let k11l = mat2(
            c(0.5, -0.5) * minus_i * exp_neg_b,
            c(0.5, -0.5) * exp_neg_b,
            c(0.5, -0.5) * minus_i * exp_pos_b,
            c(0.5, -0.5) * -exp_pos_b,
        );
        let k11r = mat2(
            c(FRAC_1_SQRT_2, 0.0) * i * exp_neg_b,
            c(FRAC_1_SQRT_2, 0.0) * -exp_neg_b,
            c(FRAC_1_SQRT_2, 0.0) * exp_pos_b,
            c(FRAC_1_SQRT_2, 0.0) * minus_i * exp_pos_b,
        );
        let k32l_k21l = mat2(
            c(FRAC_1_SQRT_2, 0.0) * c(1.0, (2.0 * b).cos()),
            c(FRAC_1_SQRT_2, 0.0) * i * c((2.0 * b).sin(), 0.0),
            c(FRAC_1_SQRT_2, 0.0) * i * c((2.0 * b).sin(), 0.0),
            c(FRAC_1_SQRT_2, 0.0) * c(1.0, -(2.0 * b).cos()),
        );
        let k21r = mat2(
            c(0.5, 0.5) * minus_i * exp_neg_2b,
            c(0.5, 0.5) * exp_neg_2b,
            c(0.5, 0.5) * i * exp_pos_2b,
            c(0.5, 0.5) * exp_pos_2b,
        );
        let k31l = mat2(
            c(FRAC_1_SQRT_2, 0.0) * exp_neg_b,
            c(FRAC_1_SQRT_2, 0.0) * exp_neg_b,
            c(FRAC_1_SQRT_2, 0.0) * -exp_pos_b,
            c(FRAC_1_SQRT_2, 0.0) * exp_pos_b,
        );
        let k31r = mat2(i * exp_pos_b, c(0.0, 0.0), c(0.0, 0.0), minus_i * exp_neg_b);
        let k32r = mat2(
            c(0.5, 0.5) * exp_pos_b,
            c(0.5, 0.5) * -exp_neg_b,
            c(0.5, 0.5) * minus_i * exp_pos_b,
            c(0.5, 0.5) * minus_i * exp_neg_b,
        );

        let k1ld = dagger(&basis.k1l);
        let k1rd = dagger(&basis.k1r);
        let k2ld = dagger(&basis.k2l);
        let k2rd = dagger(&basis.k2r);

        let u0l = k31l.dot(&k1ld);
        let u0r = k31r.dot(&k1rd);
        let u1l = k2ld.dot(&k32l_k21l.dot(&k1ld));
        let u1ra = k2rd.dot(&k32r);
        let u1rb = k21r.dot(&k1rd);
        let u2la = k2ld.dot(&k22l);
        let u2lb = k11l.dot(&k1ld);
        let u2ra = k2rd.dot(&k22r);
        let u2rb = k11r.dot(&k1rd);
        let u3l = k2ld.dot(&k12l);
        let u3r = k2rd.dot(&k12r);
        let q0l = k12l_dg.dot(&k1ld);
        let q0r = k12r_dg.dot(&ipz.dot(&k1rd));
        let q1la = k2ld.dot(&dagger(&k11l));
        let q1lb = k11l.dot(&k1ld);
        let q1ra = k2rd.dot(&ipz.dot(&dagger(&k11r)));
        let q1rb = k11r.dot(&k1rd);
        let q2l = k2ld.dot(&k12l);
        let q2r = k2rd.dot(&k12r);
        let global_phase = basis.global_phase;

        Ok(Self {
            basis,
            u0l,
            u0r,
            u1l,
            u1ra,
            u1rb,
            u2la,
            u2lb,
            u2ra,
            u2rb,
            u3l,
            u3r,
            q0l,
            q0r,
            q1la,
            q1lb,
            q1ra,
            q1rb,
            q2l,
            q2r,
            global_phase,
        })
    }

    fn num_basis_gates(&self, target: &KakDecomposition) -> usize {
        let traces = [
            c(
                4.0 * target.a.cos() * target.b.cos() * target.c.cos(),
                4.0 * target.a.sin() * target.b.sin() * target.c.sin(),
            ),
            c(
                4.0 * (FRAC_PI_4 - target.a).cos()
                    * (self.basis.b - target.b).cos()
                    * target.c.cos(),
                4.0 * (FRAC_PI_4 - target.a).sin()
                    * (self.basis.b - target.b).sin()
                    * target.c.sin(),
            ),
            c(4.0 * target.c.cos(), 0.0),
            c(4.0, 0.0),
        ];

        let mut best_index = 0usize;
        let mut best_fidelity = (4.0 + traces[0].norm_sqr()) / 20.0;
        for (index, trace) in traces.iter().enumerate().skip(1) {
            let fidelity = (4.0 + trace.norm_sqr()) / 20.0;
            if fidelity > best_fidelity + FIDELITY_IMPROVEMENT_EPS {
                best_index = index;
                best_fidelity = fidelity;
            }
        }
        best_index
    }

    fn local_decomposition(
        &self,
        target: &KakDecomposition,
        num_cx: usize,
    ) -> Vec<Array2<Complex64>> {
        match num_cx {
            0 => vec![target.k1r.dot(&target.k2r), target.k1l.dot(&target.k2l)],
            1 => vec![
                dagger(&self.basis.k2r).dot(&target.k2r),
                dagger(&self.basis.k2l).dot(&target.k2l),
                target.k1r.dot(&dagger(&self.basis.k1r)),
                target.k1l.dot(&dagger(&self.basis.k1l)),
            ],
            2 => vec![
                self.q2r.dot(&target.k2r),
                self.q2l.dot(&target.k2l),
                self.q1ra.dot(&rz_gate(2.0 * target.b).dot(&self.q1rb)),
                self.q1la.dot(&rz_gate(-2.0 * target.a).dot(&self.q1lb)),
                target.k1r.dot(&self.q0r),
                target.k1l.dot(&self.q0l),
            ],
            3 => vec![
                self.u3r.dot(&target.k2r),
                self.u3l.dot(&target.k2l),
                self.u2ra.dot(&rz_gate(2.0 * target.b).dot(&self.u2rb)),
                self.u2la.dot(&rz_gate(-2.0 * target.a).dot(&self.u2lb)),
                self.u1ra.dot(&rz_gate(-2.0 * target.c).dot(&self.u1rb)),
                self.u1l.clone(),
                target.k1r.dot(&self.u0r),
                target.k1l.dot(&self.u0l),
            ],
            _ => unreachable!("CX decomposer supports at most 3 basis gates"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{Circuit, Instruction, Parameter, UnitaryGate, circuit_to_matrix};
    use approx::assert_abs_diff_eq;
    use ndarray::linalg::kron;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn synthesized_output(
        matrix: &Array2<Complex64>,
        basis: TwoQubitUnitaryDecomposeBasis,
    ) -> (Circuit, Array2<Complex64>, Array2<Complex64>) {
        let gate = UnitaryGate::new("source_2q", 2, 0)
            .with_matrix(matrix.clone())
            .unwrap();
        let mut source = Circuit::new(2);
        source
            .unitary(gate, vec![Qubit::new(0), Qubit::new(1)])
            .unwrap();
        let expected = circuit_to_matrix(&source, None).unwrap();

        let synthesis =
            synthesize_numeric_2q_unitary(matrix, [Qubit::new(0), Qubit::new(1)], basis).unwrap();
        let mut circuit = Circuit::from_operations(
            vec![Qubit::new(0), Qubit::new(1)],
            synthesis.operations,
            None,
            None,
        )
        .unwrap();
        circuit.set_global_phase(Parameter::from(synthesis.global_phase));
        let matrix = if circuit.operations().is_empty() {
            let phase = Complex64::from_polar(1.0, synthesis.global_phase);
            Array2::eye(4).mapv(|value: Complex64| phase * value)
        } else {
            circuit_to_matrix(&circuit, None).unwrap()
        };
        (circuit, expected, matrix)
    }

    fn count_gate(circuit: &Circuit, gate: StandardGate) -> usize {
        circuit
            .operations()
            .iter()
            .filter(|operation| matches!(operation.instruction, Instruction::Standard(actual) if actual == gate))
            .count()
    }

    fn seeded_random_unitary_4(rng: &mut StdRng) -> Array2<Complex64> {
        let mut columns = Vec::<[Complex64; 4]>::with_capacity(4);
        for _ in 0..4 {
            let mut column = random_complex_column(rng);
            for previous in &columns {
                let projection = column_inner(previous, &column);
                for row in 0..4 {
                    column[row] -= projection * previous[row];
                }
            }
            let norm = column
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt();
            assert!(norm > 1e-10, "seeded random basis column is degenerate");
            for value in &mut column {
                *value /= norm;
            }
            columns.push(column);
        }

        Array2::from_shape_fn((4, 4), |(row, col)| columns[col][row])
    }

    fn random_complex_column(rng: &mut StdRng) -> [Complex64; 4] {
        std::array::from_fn(|_| {
            Complex64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
        })
    }

    fn column_inner(lhs: &[Complex64; 4], rhs: &[Complex64; 4]) -> Complex64 {
        lhs.iter()
            .zip(rhs)
            .map(|(left, right)| left.conj() * right)
            .sum()
    }

    #[test]
    fn pauli_backend_reconstructs_common_2q_unitaries() {
        let phase = Complex64::from_polar(1.0, 0.37);
        let cases = [
            StandardGate::CX.matrix(&[]).unwrap().into_owned(),
            StandardGate::CZ.matrix(&[]).unwrap().into_owned(),
            StandardGate::SWAP.matrix(&[]).unwrap().into_owned(),
            StandardGate::FSIM
                .matrix(&[0.2, -0.3])
                .unwrap()
                .into_owned(),
            StandardGate::CX
                .matrix(&[])
                .unwrap()
                .into_owned()
                .mapv(|value| phase * value),
        ];

        for matrix in cases {
            let (decomposed, before, after) =
                synthesized_output(&matrix, TwoQubitUnitaryDecomposeBasis::PauliRotations);

            assert!(decomposed.operations().iter().all(|operation| matches!(
                operation.instruction,
                Instruction::Standard(StandardGate::U)
                    | Instruction::Standard(StandardGate::RXX)
                    | Instruction::Standard(StandardGate::RYY)
                    | Instruction::Standard(StandardGate::RZZ)
            )));
            assert_abs_diff_eq!(before, after, epsilon = 1e-8);
        }
    }

    #[test]
    fn cx_backend_uses_expected_exact_cx_counts() {
        let rxx = StandardGate::RXX.matrix(&[0.7]).unwrap().into_owned();
        let ryy = StandardGate::RYY.matrix(&[-0.4]).unwrap().into_owned();
        let two_cx_matrix = rxx.dot(&ryy);
        let cases = [
            (Array2::eye(4), 0usize),
            (StandardGate::CX.matrix(&[]).unwrap().into_owned(), 1usize),
            (two_cx_matrix, 2usize),
            (StandardGate::SWAP.matrix(&[]).unwrap().into_owned(), 3usize),
        ];

        for (matrix, expected_cx) in cases {
            let (decomposed, before, after) =
                synthesized_output(&matrix, TwoQubitUnitaryDecomposeBasis::Cx);

            assert_eq!(count_gate(&decomposed, StandardGate::CX), expected_cx);
            assert!(decomposed.operations().iter().all(|operation| matches!(
                operation.instruction,
                Instruction::Standard(StandardGate::U) | Instruction::Standard(StandardGate::CX)
            )));
            assert_abs_diff_eq!(before, after, epsilon = 1e-8);
        }
    }

    #[test]
    fn cy_backend_uses_expected_exact_cy_counts() {
        let rxx = StandardGate::RXX.matrix(&[0.7]).unwrap().into_owned();
        let ryy = StandardGate::RYY.matrix(&[-0.4]).unwrap().into_owned();
        let two_cy_matrix = rxx.dot(&ryy);
        let cases = [
            (Array2::eye(4), 0usize),
            (StandardGate::CY.matrix(&[]).unwrap().into_owned(), 1usize),
            (two_cy_matrix, 2usize),
            (StandardGate::SWAP.matrix(&[]).unwrap().into_owned(), 3usize),
        ];

        for (matrix, expected_cy) in cases {
            let (decomposed, before, after) =
                synthesized_output(&matrix, TwoQubitUnitaryDecomposeBasis::Cy);

            assert_eq!(count_gate(&decomposed, StandardGate::CY), expected_cy);
            assert_eq!(count_gate(&decomposed, StandardGate::CX), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::CZ), 0);
            assert!(decomposed.operations().iter().all(|operation| matches!(
                operation.instruction,
                Instruction::Standard(StandardGate::U) | Instruction::Standard(StandardGate::CY)
            )));
            assert_abs_diff_eq!(before, after, epsilon = 1e-8);
        }
    }

    #[test]
    fn cz_backend_uses_expected_exact_cz_counts() {
        let rxx = StandardGate::RXX.matrix(&[0.7]).unwrap().into_owned();
        let ryy = StandardGate::RYY.matrix(&[-0.4]).unwrap().into_owned();
        let two_cz_matrix = rxx.dot(&ryy);
        let cases = [
            (Array2::eye(4), 0usize),
            (StandardGate::CZ.matrix(&[]).unwrap().into_owned(), 1usize),
            (two_cz_matrix, 2usize),
            (StandardGate::SWAP.matrix(&[]).unwrap().into_owned(), 3usize),
        ];

        for (matrix, expected_cz) in cases {
            let (decomposed, before, after) =
                synthesized_output(&matrix, TwoQubitUnitaryDecomposeBasis::Cz);

            assert_eq!(count_gate(&decomposed, StandardGate::CZ), expected_cz);
            assert_eq!(count_gate(&decomposed, StandardGate::CX), 0);
            assert!(decomposed.operations().iter().all(|operation| matches!(
                operation.instruction,
                Instruction::Standard(StandardGate::U) | Instruction::Standard(StandardGate::CZ)
            )));
            assert_abs_diff_eq!(before, after, epsilon = 1e-8);
        }
    }

    #[test]
    fn rzz_backend_uses_rzz_only_for_cartan_core() {
        let rxx = StandardGate::RXX.matrix(&[0.7]).unwrap().into_owned();
        let ryy = StandardGate::RYY.matrix(&[-0.4]).unwrap().into_owned();
        let cases = [
            (Array2::eye(4), 0usize),
            (
                StandardGate::RZZ.matrix(&[0.5]).unwrap().into_owned(),
                1usize,
            ),
            (rxx.dot(&ryy), 2usize),
            (StandardGate::SWAP.matrix(&[]).unwrap().into_owned(), 3usize),
        ];

        for (matrix, expected_rzz) in cases {
            let (decomposed, before, after) =
                synthesized_output(&matrix, TwoQubitUnitaryDecomposeBasis::Rzz);

            assert_eq!(count_gate(&decomposed, StandardGate::RZZ), expected_rzz);
            assert_eq!(count_gate(&decomposed, StandardGate::CX), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::CY), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::CZ), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::RXX), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::RYY), 0);
            assert!(decomposed.operations().iter().all(|operation| matches!(
                operation.instruction,
                Instruction::Standard(StandardGate::U)
                    | Instruction::Standard(StandardGate::H)
                    | Instruction::Standard(StandardGate::RX)
                    | Instruction::Standard(StandardGate::RZZ)
            )));
            assert_abs_diff_eq!(before, after, epsilon = 1e-8);
        }
    }

    #[test]
    fn backends_handle_identity_without_entangling_operations() {
        for basis in [
            TwoQubitUnitaryDecomposeBasis::PauliRotations,
            TwoQubitUnitaryDecomposeBasis::Cx,
            TwoQubitUnitaryDecomposeBasis::Cy,
            TwoQubitUnitaryDecomposeBasis::Cz,
            TwoQubitUnitaryDecomposeBasis::Rzz,
        ] {
            let (decomposed, before, after) = synthesized_output(&Array2::eye(4), basis);

            assert_eq!(count_gate(&decomposed, StandardGate::CX), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::CY), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::CZ), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::RXX), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::RYY), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::RZZ), 0);
            assert_abs_diff_eq!(before, after, epsilon = 1e-8);
        }
    }

    #[test]
    fn cx_backend_preserves_near_zero_entangling_rotation() {
        let matrix = StandardGate::RXX.matrix(&[-1.0e-5]).unwrap().into_owned();
        let (decomposed, before, after) =
            synthesized_output(&matrix, TwoQubitUnitaryDecomposeBasis::Cx);

        assert_abs_diff_eq!(before, after, epsilon = 1e-8);
        assert_eq!(count_gate(&decomposed, StandardGate::CX), 2);
    }

    #[test]
    fn backends_preserve_asymmetric_local_product_without_entanglers() {
        let left = StandardGate::U.matrix(&[0.3, -0.4, 0.5]).unwrap();
        let right = StandardGate::U.matrix(&[0.7, 0.2, -0.6]).unwrap();
        let matrix = kron(left.as_ref(), right.as_ref());

        for basis in [
            TwoQubitUnitaryDecomposeBasis::PauliRotations,
            TwoQubitUnitaryDecomposeBasis::Cx,
            TwoQubitUnitaryDecomposeBasis::Cy,
            TwoQubitUnitaryDecomposeBasis::Cz,
            TwoQubitUnitaryDecomposeBasis::Rzz,
        ] {
            let (decomposed, before, after) = synthesized_output(&matrix, basis);

            assert_eq!(count_gate(&decomposed, StandardGate::CX), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::CY), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::CZ), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::RXX), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::RYY), 0);
            assert_eq!(count_gate(&decomposed, StandardGate::RZZ), 0);
            assert_abs_diff_eq!(before, after, epsilon = 1e-8);
        }
    }

    #[test]
    fn backends_preserve_asymmetric_locals_around_cartan_core() {
        let k1l = StandardGate::U.matrix(&[0.2, -0.4, 0.9]).unwrap();
        let k1r = StandardGate::U.matrix(&[1.0, 0.8, -0.7]).unwrap();
        let k2l = StandardGate::U.matrix(&[0.7, -0.5, 0.1]).unwrap();
        let k2r = StandardGate::U.matrix(&[0.3, 0.6, -0.2]).unwrap();
        let rxx = StandardGate::RXX.matrix(&[-0.62]).unwrap();
        let ryy = StandardGate::RYY.matrix(&[-0.34]).unwrap();
        let rzz = StandardGate::RZZ.matrix(&[0.16]).unwrap();
        let matrix = kron(k1l.as_ref(), k1r.as_ref())
            .dot(&rxx.dot(&ryy.dot(&rzz.dot(&kron(k2l.as_ref(), k2r.as_ref())))));

        for basis in [
            TwoQubitUnitaryDecomposeBasis::PauliRotations,
            TwoQubitUnitaryDecomposeBasis::Cx,
            TwoQubitUnitaryDecomposeBasis::Cy,
            TwoQubitUnitaryDecomposeBasis::Cz,
            TwoQubitUnitaryDecomposeBasis::Rzz,
        ] {
            let (_, before, after) = synthesized_output(&matrix, basis);
            assert_abs_diff_eq!(before, after, epsilon = 1e-8);
        }
    }

    #[test]
    fn backends_reconstruct_controlled_and_cartan_rotation_family() {
        let phase = Complex64::from_polar(1.0, -0.28);
        let cases = [
            StandardGate::CRX.matrix(&[0.31]).unwrap().into_owned(),
            StandardGate::CRY.matrix(&[-0.47]).unwrap().into_owned(),
            StandardGate::CRZ.matrix(&[0.83]).unwrap().into_owned(),
            StandardGate::RXX.matrix(&[0.19]).unwrap().into_owned(),
            StandardGate::RYY.matrix(&[-0.23]).unwrap().into_owned(),
            StandardGate::RZZ
                .matrix(&[0.41])
                .unwrap()
                .into_owned()
                .mapv(|value| phase * value),
        ];

        for matrix in cases {
            for basis in [
                TwoQubitUnitaryDecomposeBasis::PauliRotations,
                TwoQubitUnitaryDecomposeBasis::Cx,
                TwoQubitUnitaryDecomposeBasis::Cy,
                TwoQubitUnitaryDecomposeBasis::Cz,
                TwoQubitUnitaryDecomposeBasis::Rzz,
            ] {
                let (_, before, after) = synthesized_output(&matrix, basis);
                assert_abs_diff_eq!(before, after, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn backends_reconstruct_seeded_random_2q_unitaries() {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        for _ in 0..50 {
            let matrix = seeded_random_unitary_4(&mut rng);
            for basis in [
                TwoQubitUnitaryDecomposeBasis::PauliRotations,
                TwoQubitUnitaryDecomposeBasis::Cx,
                TwoQubitUnitaryDecomposeBasis::Cy,
                TwoQubitUnitaryDecomposeBasis::Cz,
                TwoQubitUnitaryDecomposeBasis::Rzz,
            ] {
                let (decomposed, before, after) = synthesized_output(&matrix, basis);

                assert_abs_diff_eq!(before, after, epsilon = 1e-8);
                assert!(count_gate(&decomposed, StandardGate::CX) <= 3);
                assert!(count_gate(&decomposed, StandardGate::CY) <= 3);
                assert!(count_gate(&decomposed, StandardGate::CZ) <= 3);
                if basis == TwoQubitUnitaryDecomposeBasis::Rzz {
                    assert!(count_gate(&decomposed, StandardGate::RZZ) <= 3);
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_shape_and_non_unitary_matrix() {
        let bad_shape = Array2::<Complex64>::eye(3);
        let err = synthesize_numeric_2q_unitary(
            &bad_shape,
            [Qubit::new(0), Qubit::new(1)],
            TwoQubitUnitaryDecomposeBasis::PauliRotations,
        )
        .unwrap_err();
        assert!(err.to_string().contains("4x4"));

        let non_unitary = ndarray::array![
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0)
            ]
        ];
        let err = synthesize_numeric_2q_unitary(
            &non_unitary,
            [Qubit::new(0), Qubit::new(1)],
            TwoQubitUnitaryDecomposeBasis::Cx,
        )
        .unwrap_err();
        assert!(err.to_string().contains("not unitary"));

        let mut non_finite = Array2::<Complex64>::eye(4);
        non_finite[[1, 2]] = Complex64::new(f64::INFINITY, 0.0);
        let err = synthesize_numeric_2q_unitary(
            &non_finite,
            [Qubit::new(0), Qubit::new(1)],
            TwoQubitUnitaryDecomposeBasis::PauliRotations,
        )
        .unwrap_err();
        assert!(err.to_string().contains("non-finite"));

        let err = synthesize_numeric_2q_unitary(
            &Array2::eye(4),
            [Qubit::new(0), Qubit::new(0)],
            TwoQubitUnitaryDecomposeBasis::PauliRotations,
        )
        .unwrap_err();
        assert!(err.to_string().contains("distinct qubits"));
    }
}
