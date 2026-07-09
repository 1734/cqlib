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

//! Numeric block resynthesis helpers.
//!
//! This module currently contains internal infrastructure for future numeric
//! one- and two-qubit block resynthesis passes. The helpers deliberately stay
//! local to the resynthesis transform boundary until their API has been proven
//! by an implemented collector and synthesizer.

pub(crate) mod commutation;
