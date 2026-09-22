Cqlib 概览
==========

.. toctree::
   :maxdepth: 2
   :caption: 快速开始

   documentation/cn/0_get_started/0_introduction
   documentation/cn/0_get_started/1_installation
   documentation/cn/0_get_started/2_quickstart

.. toctree::
   :maxdepth: 2
   :caption: 量子线路

   documentation/cn/1_cqlib/0_circuit/0_overview
   documentation/cn/1_cqlib/0_circuit/1_gates
   documentation/cn/1_cqlib/0_circuit/2_structures
   documentation/cn/1_cqlib/0_circuit/3_parameters
   documentation/cn/1_cqlib/0_circuit/4_circuit_analysis
   documentation/cn/1_cqlib/0_circuit/5_control_flow

.. toctree::
   :maxdepth: 2
   :caption: 中间表示

   documentation/cn/1_cqlib/1_ir/0_overview
   documentation/cn/1_cqlib/1_ir/1_qcis
   documentation/cn/1_cqlib/1_ir/2_qasm2
   documentation/cn/1_cqlib/1_ir/3_qasm3
   documentation/cn/1_cqlib/1_ir/4_conversion_workflow

.. toctree::
   :maxdepth: 2
   :caption: 设备与后端

   documentation/cn/1_cqlib/2_device/0_overview
   documentation/cn/1_cqlib/2_device/1_topology
   documentation/cn/1_cqlib/2_device/2_device
   documentation/cn/1_cqlib/2_device/3_layout
   documentation/cn/1_cqlib/2_device/4_noise
   documentation/cn/1_cqlib/2_device/5_result

.. toctree::
   :maxdepth: 2
   :caption: 量子信息

   documentation/cn/1_cqlib/3_qis/0_overview
   documentation/cn/1_cqlib/3_qis/1_statevector
   documentation/cn/1_cqlib/3_qis/2_density_matrix
   documentation/cn/1_cqlib/3_qis/3_density_matrix_noise
   documentation/cn/1_cqlib/3_qis/4_stabilizer
   documentation/cn/1_cqlib/3_qis/5_pauli_and_hamiltonian
   documentation/cn/1_cqlib/3_qis/6_metrics_entropy

.. toctree::
   :maxdepth: 2
   :caption: 编译优化

   documentation/cn/1_cqlib/4_compiler/0_overview
   documentation/cn/1_cqlib/4_compiler/1_layout
   documentation/cn/1_cqlib/4_compiler/2_sabre_mapping
   documentation/cn/1_cqlib/4_compiler/3_template_optimization
   documentation/cn/1_cqlib/4_compiler/4_commutative_and_clifford

.. toctree::
   :maxdepth: 2
   :caption: 可视化

   documentation/cn/1_cqlib/5_visualization/0_overview
   documentation/cn/1_cqlib/5_visualization/1_draw_text
   documentation/cn/1_cqlib/5_visualization/2_draw_figure
   documentation/cn/1_cqlib/5_visualization/3_notebook_and_docs
   documentation/cn/1_cqlib/5_visualization/4_visualization_practices
   documentation/cn/1_cqlib/5_visualization/5_control_flow_and_special
   documentation/cn/1_cqlib/5_visualization/6_result_visualization
   documentation/cn/1_cqlib/5_visualization/7_state_visualization

.. toctree::
   :maxdepth: 2
   :caption: 错误缓解

   documentation/cn/1_cqlib/6_error_mitigation/0_overview
   documentation/cn/1_cqlib/6_error_mitigation/1_zne
   documentation/cn/1_cqlib/6_error_mitigation/2_virtual_distillation
   documentation/cn/1_cqlib/6_error_mitigation/3_unified_api

.. toctree::
   :maxdepth: 2
   :caption: 天衍平台

   documentation/cn/1_cqlib/7_tianyan/0_overview
   documentation/cn/1_cqlib/7_tianyan/1_auth_config
   documentation/cn/1_cqlib/7_tianyan/2_backend_device
   documentation/cn/1_cqlib/7_tianyan/3_task_result
   documentation/cn/1_cqlib/7_tianyan/4_qcis_ir_workflow
   documentation/cn/1_cqlib/7_tianyan/5_readout_mitigation

.. toctree::
   :maxdepth: 1
   :caption: Python API 参考

   api/cn/python/0_overview
   api/cn/python/0_circuit/0_overview
   api/cn/python/0_circuit/1_circuit
   api/cn/python/0_circuit/2_qubit
   api/cn/python/0_circuit/3_parameter
   api/cn/python/0_circuit/4_operation_instruction
   api/cn/python/0_circuit/5_gates_standard
   api/cn/python/0_circuit/6_gates_unitary
   api/cn/python/0_circuit/7_gates_mc_gate
   api/cn/python/0_circuit/8_gates_circuit_gate
   api/cn/python/0_circuit/9_classical_control_flow
   api/cn/python/0_circuit/10_symbolic_matrix
   api/cn/python/0_circuit/11_ansatz
   api/cn/python/0_circuit/12_circuit_to_matrix
   api/cn/python/0_circuit/13_circuit_dag
   api/cn/python/1_ir/0_overview
   api/cn/python/1_ir/1_qcis
   api/cn/python/1_ir/2_qasm2
   api/cn/python/1_ir/3_qasm3
   api/cn/python/2_device/0_overview
   api/cn/python/2_device/1_topology
   api/cn/python/2_device/2_properties_device
   api/cn/python/2_device/3_layout
   api/cn/python/2_device/4_noise
   api/cn/python/2_device/5_result
   api/cn/python/2_device/6_qubits
   api/cn/python/3_qis/0_overview
   api/cn/python/3_qis/1_statevector
   api/cn/python/3_qis/2_density_matrix
   api/cn/python/3_qis/3_density_matrix_noise
   api/cn/python/3_qis/4_stabilizer
   api/cn/python/3_qis/5_classical_state
   api/cn/python/3_qis/6_pauli
   api/cn/python/3_qis/7_hamiltonian
   api/cn/python/3_qis/8_evolution
   api/cn/python/3_qis/9_metrics_entropy
   api/cn/python/4_compile/0_overview
   api/cn/python/4_compile/1_compiler
   api/cn/python/4_compile/2_transform
   api/cn/python/4_compile/3_layout
   api/cn/python/4_compile/4_routing
   api/cn/python/4_compile/5_sabre
   api/cn/python/4_compile/6_decompose_resynthesis
   api/cn/python/4_compile/7_knowledge
   api/cn/python/4_compile/8_resource
   api/cn/python/4_compile/9_commutation
   api/cn/python/5_visualization/0_overview
   api/cn/python/5_visualization/1_draw_text
   api/cn/python/5_visualization/2_draw_figure
   api/cn/python/5_visualization/3_state_plots
   api/cn/python/5_visualization/4_result_plots
   api/cn/python/5_visualization/5_render_to_file
   api/cn/python/6_error_mitigation/0_overview
   api/cn/python/6_error_mitigation/1_zne
   api/cn/python/6_error_mitigation/2_virtual_distillation
   api/cn/python/6_error_mitigation/3_unified

.. toctree::
   :maxdepth: 1
   :caption: Rust API 参考

   api/cn/rust/0_overview
   api/cn/rust/0_circuit/0_overview
   api/cn/rust/0_circuit/1_circuit
   api/cn/rust/0_circuit/2_qubit
   api/cn/rust/0_circuit/3_parameter
   api/cn/rust/0_circuit/4_operation_instruction
   api/cn/rust/0_circuit/5_gate_standard
   api/cn/rust/0_circuit/6_gate_unitary
   api/cn/rust/0_circuit/7_gate_mc_gate
   api/cn/rust/0_circuit/8_gate_circuit_gate
   api/cn/rust/0_circuit/9_classical_control_flow
   api/cn/rust/0_circuit/10_symbolic_matrix
   api/cn/rust/0_circuit/11_ansatz
   api/cn/rust/0_circuit/12_cfg
   api/cn/rust/0_circuit/13_circuit_to_matrix
   api/cn/rust/0_circuit/14_circuit_dag
   api/cn/rust/1_ir/0_overview
   api/cn/rust/1_ir/1_qcis
   api/cn/rust/1_ir/2_qasm2
   api/cn/rust/1_ir/3_qasm3
   api/cn/rust/2_device/0_overview
   api/cn/rust/2_device/1_topology
   api/cn/rust/2_device/2_properties_device
   api/cn/rust/2_device/3_layout
   api/cn/rust/2_device/4_noise
   api/cn/rust/2_device/5_result
   api/cn/rust/2_device/6_qubits
   api/cn/rust/3_qis/0_overview
   api/cn/rust/3_qis/1_statevector
   api/cn/rust/3_qis/2_density_matrix
   api/cn/rust/3_qis/3_density_matrix_noise
   api/cn/rust/3_qis/4_stabilizer
   api/cn/rust/3_qis/5_classical_state
   api/cn/rust/3_qis/6_pauli
   api/cn/rust/3_qis/7_hamiltonian
   api/cn/rust/3_qis/8_evolution
   api/cn/rust/3_qis/9_metrics_entropy
   api/cn/rust/4_compile/0_overview
   api/cn/rust/4_compile/1_compiler
   api/cn/rust/4_compile/2_transform
   api/cn/rust/4_compile/3_layout
   api/cn/rust/4_compile/4_routing
   api/cn/rust/4_compile/5_sabre
   api/cn/rust/4_compile/6_decompose_resynthesis
   api/cn/rust/4_compile/7_knowledge
   api/cn/rust/4_compile/8_resource
   api/cn/rust/4_compile/9_commutation
   api/cn/rust/5_visualization/0_overview
   api/cn/rust/5_visualization/1_draw_text
   api/cn/rust/5_visualization/2_draw_figure
   api/cn/rust/5_visualization/3_state_plots
   api/cn/rust/5_visualization/4_result_plots
   api/cn/rust/5_visualization/5_render_to_file
   api/cn/rust/5_visualization/6_visual_ir
   api/cn/rust/6_error_mitigation/0_overview
   api/cn/rust/6_error_mitigation/1_zne
   api/cn/rust/6_error_mitigation/2_virtual_distillation
   api/cn/rust/6_error_mitigation/3_unified

.. toctree::
   :maxdepth: 1
   :caption: C API 参考

   api/cn/c/0_overview
   api/cn/c/0_circuit/0_overview
   api/cn/c/0_circuit/1_circuit
   api/cn/c/0_circuit/2_parameter
   api/cn/c/0_circuit/3_circuit_to_matrix
   api/cn/c/1_ir/1_qcis
   api/cn/c/1_ir/2_qasm2
   api/cn/c/1_ir/3_qasm3
   api/cn/c/2_device/1_topology
   api/cn/c/2_device/2_properties_device
   api/cn/c/2_device/3_layout
   api/cn/c/2_device/4_noise
   api/cn/c/2_device/5_result
   api/cn/c/3_qis/0_overview
   api/cn/c/3_qis/1_statevector
   api/cn/c/3_qis/2_density_matrix
   api/cn/c/3_qis/3_stabilizer_state
   api/cn/c/3_qis/4_pauli_string
   api/cn/c/3_qis/5_hamiltonian
   api/cn/c/4_compile/1_compile
   api/cn/c/5_visualization/1_text_drawer
   api/cn/c/5_visualization/2_figure_drawer
   api/cn/c/6_error_mitigation/1_error_mitigation
   api/cn/c/6_error_mitigation/2_zne_mitigation
   api/cn/c/6_error_mitigation/3_virtual_distillation


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   documentation/en/0_get_started/0_introduction
   documentation/en/0_get_started/1_installation
   documentation/en/0_get_started/2_quickstart

.. toctree::
   :maxdepth: 2
   :caption: Quantum Circuit

   documentation/en/1_cqlib/0_circuit/0_overview
   documentation/en/1_cqlib/0_circuit/1_gates
   documentation/en/1_cqlib/0_circuit/2_structures
   documentation/en/1_cqlib/0_circuit/3_parameters
   documentation/en/1_cqlib/0_circuit/4_circuit_analysis
   documentation/en/1_cqlib/0_circuit/5_control_flow

.. toctree::
   :maxdepth: 2
   :caption: Intermediate Representation

   documentation/en/1_cqlib/1_ir/0_overview
   documentation/en/1_cqlib/1_ir/1_qcis
   documentation/en/1_cqlib/1_ir/2_qasm2
   documentation/en/1_cqlib/1_ir/3_qasm3
   documentation/en/1_cqlib/1_ir/4_conversion_workflow

.. toctree::
   :maxdepth: 2
   :caption: Devices and Backends

   documentation/en/1_cqlib/2_device/0_overview
   documentation/en/1_cqlib/2_device/1_topology
   documentation/en/1_cqlib/2_device/2_device
   documentation/en/1_cqlib/2_device/3_layout
   documentation/en/1_cqlib/2_device/4_noise
   documentation/en/1_cqlib/2_device/5_result

.. toctree::
   :maxdepth: 2
   :caption: Quantum Information

   documentation/en/1_cqlib/3_qis/0_overview
   documentation/en/1_cqlib/3_qis/1_statevector
   documentation/en/1_cqlib/3_qis/2_density_matrix
   documentation/en/1_cqlib/3_qis/3_density_matrix_noise
   documentation/en/1_cqlib/3_qis/4_stabilizer
   documentation/en/1_cqlib/3_qis/5_pauli_and_hamiltonian
   documentation/en/1_cqlib/3_qis/6_metrics_entropy

.. toctree::
   :maxdepth: 2
   :caption: Compilation

   documentation/en/1_cqlib/4_compiler/0_overview
   documentation/en/1_cqlib/4_compiler/1_layout
   documentation/en/1_cqlib/4_compiler/2_sabre_mapping
   documentation/en/1_cqlib/4_compiler/3_template_optimization
   documentation/en/1_cqlib/4_compiler/4_commutative_and_clifford

.. toctree::
   :maxdepth: 2
   :caption: Visualization

   documentation/en/1_cqlib/5_visualization/0_overview
   documentation/en/1_cqlib/5_visualization/1_draw_text
   documentation/en/1_cqlib/5_visualization/2_draw_figure
   documentation/en/1_cqlib/5_visualization/3_notebook_and_docs
   documentation/en/1_cqlib/5_visualization/4_visualization_practices
   documentation/en/1_cqlib/5_visualization/5_control_flow_and_special
   documentation/en/1_cqlib/5_visualization/6_result_visualization
   documentation/en/1_cqlib/5_visualization/7_state_visualization

.. toctree::
   :maxdepth: 2
   :caption: Error Mitigation

   documentation/en/1_cqlib/6_error_mitigation/0_overview
   documentation/en/1_cqlib/6_error_mitigation/1_zne
   documentation/en/1_cqlib/6_error_mitigation/2_virtual_distillation
   documentation/en/1_cqlib/6_error_mitigation/3_unified_api

.. toctree::
   :maxdepth: 2
   :caption: Tianyan Platform

   documentation/en/1_cqlib/7_tianyan/0_overview
   documentation/en/1_cqlib/7_tianyan/1_auth_config
   documentation/en/1_cqlib/7_tianyan/2_backend_device
   documentation/en/1_cqlib/7_tianyan/3_task_result
   documentation/en/1_cqlib/7_tianyan/4_qcis_ir_workflow
   documentation/en/1_cqlib/7_tianyan/5_readout_mitigation

.. toctree::
   :maxdepth: 2
   :caption: Python API Reference

   api/en/python/0_overview
   api/en/python/6_error_mitigation/0_overview
   api/en/python/2_device/0_overview
   api/en/python/5_visualization/0_overview
   api/en/python/3_qis/0_overview
   api/en/python/4_compile/0_overview
   api/en/python/1_ir/0_overview
   api/en/python/0_circuit/0_overview
   api/en/python/0_circuit/1_circuit
   api/en/python/4_compile/1_compiler
   api/en/python/5_visualization/1_draw_text
   api/en/python/1_ir/1_qcis
   api/en/python/3_qis/1_statevector
   api/en/python/2_device/1_topology
   api/en/python/6_error_mitigation/1_zne
   api/en/python/3_qis/2_density_matrix
   api/en/python/5_visualization/2_draw_figure
   api/en/python/2_device/2_properties_device
   api/en/python/1_ir/2_qasm2
   api/en/python/0_circuit/2_qubit
   api/en/python/4_compile/2_transform
   api/en/python/6_error_mitigation/2_virtual_distillation
   api/en/python/3_qis/3_density_matrix_noise
   api/en/python/2_device/3_layout
   api/en/python/4_compile/3_layout
   api/en/python/0_circuit/3_parameter
   api/en/python/1_ir/3_qasm3
   api/en/python/5_visualization/3_state_plots
   api/en/python/6_error_mitigation/3_unified
   api/en/python/2_device/4_noise
   api/en/python/0_circuit/4_operation_instruction
   api/en/python/5_visualization/4_result_plots
   api/en/python/4_compile/4_routing
   api/en/python/3_qis/4_stabilizer
   api/en/python/3_qis/5_classical_state
   api/en/python/0_circuit/5_gates_standard
   api/en/python/5_visualization/5_render_to_file
   api/en/python/2_device/5_result
   api/en/python/4_compile/5_sabre
   api/en/python/4_compile/6_decompose_resynthesis
   api/en/python/0_circuit/6_gates_unitary
   api/en/python/3_qis/6_pauli
   api/en/python/2_device/6_qubits
   api/en/python/0_circuit/7_gates_mc_gate
   api/en/python/3_qis/7_hamiltonian
   api/en/python/4_compile/7_knowledge
   api/en/python/3_qis/8_evolution
   api/en/python/0_circuit/8_gates_circuit_gate
   api/en/python/4_compile/8_resource
   api/en/python/0_circuit/9_classical_control_flow
   api/en/python/4_compile/9_commutation
   api/en/python/3_qis/9_metrics_entropy
   api/en/python/0_circuit/10_symbolic_matrix
   api/en/python/0_circuit/11_ansatz
   api/en/python/0_circuit/12_circuit_to_matrix
   api/en/python/0_circuit/13_circuit_dag

.. toctree::
   :maxdepth: 2
   :caption: Rust API Reference

   api/en/rust/0_overview
   api/en/rust/6_error_mitigation/0_overview
   api/en/rust/2_device/0_overview
   api/en/rust/5_visualization/0_overview
   api/en/rust/3_qis/0_overview
   api/en/rust/4_compile/0_overview
   api/en/rust/1_ir/0_overview
   api/en/rust/0_circuit/0_overview
   api/en/rust/0_circuit/1_circuit
   api/en/rust/4_compile/1_compiler
   api/en/rust/5_visualization/1_draw_text
   api/en/rust/1_ir/1_qcis
   api/en/rust/3_qis/1_statevector
   api/en/rust/2_device/1_topology
   api/en/rust/6_error_mitigation/1_zne
   api/en/rust/3_qis/2_density_matrix
   api/en/rust/5_visualization/2_draw_figure
   api/en/rust/2_device/2_properties_device
   api/en/rust/1_ir/2_qasm2
   api/en/rust/0_circuit/2_qubit
   api/en/rust/4_compile/2_transform
   api/en/rust/6_error_mitigation/2_virtual_distillation
   api/en/rust/3_qis/3_density_matrix_noise
   api/en/rust/2_device/3_layout
   api/en/rust/4_compile/3_layout
   api/en/rust/0_circuit/3_parameter
   api/en/rust/1_ir/3_qasm3
   api/en/rust/5_visualization/3_state_plots
   api/en/rust/6_error_mitigation/3_unified
   api/en/rust/2_device/4_noise
   api/en/rust/0_circuit/4_operation_instruction
   api/en/rust/5_visualization/4_result_plots
   api/en/rust/4_compile/4_routing
   api/en/rust/3_qis/4_stabilizer
   api/en/rust/3_qis/5_classical_state
   api/en/rust/0_circuit/5_gate_standard
   api/en/rust/5_visualization/5_render_to_file
   api/en/rust/2_device/5_result
   api/en/rust/4_compile/5_sabre
   api/en/rust/4_compile/6_decompose_resynthesis
   api/en/rust/0_circuit/6_gate_unitary
   api/en/rust/3_qis/6_pauli
   api/en/rust/2_device/6_qubits
   api/en/rust/5_visualization/6_visual_ir
   api/en/rust/0_circuit/7_gate_mc_gate
   api/en/rust/3_qis/7_hamiltonian
   api/en/rust/4_compile/7_knowledge
   api/en/rust/3_qis/8_evolution
   api/en/rust/0_circuit/8_gate_circuit_gate
   api/en/rust/4_compile/8_resource
   api/en/rust/0_circuit/9_classical_control_flow
   api/en/rust/4_compile/9_commutation
   api/en/rust/3_qis/9_metrics_entropy
   api/en/rust/0_circuit/10_symbolic_matrix
   api/en/rust/0_circuit/11_ansatz
   api/en/rust/0_circuit/12_cfg
   api/en/rust/0_circuit/13_circuit_to_matrix
   api/en/rust/0_circuit/14_circuit_dag

.. toctree::
   :maxdepth: 1
   :caption: C API Reference

   api/en/c/0_overview
   api/en/c/0_circuit/0_overview
   api/en/c/0_circuit/1_circuit
   api/en/c/0_circuit/2_parameter
   api/en/c/0_circuit/3_circuit_to_matrix
   api/en/c/1_ir/1_qcis
   api/en/c/1_ir/2_qasm2
   api/en/c/1_ir/3_qasm3
   api/en/c/2_device/1_topology
   api/en/c/2_device/2_properties_device
   api/en/c/2_device/3_layout
   api/en/c/2_device/4_noise
   api/en/c/2_device/5_result
   api/en/c/3_qis/0_overview
   api/en/c/3_qis/1_statevector
   api/en/c/3_qis/2_density_matrix
   api/en/c/3_qis/3_stabilizer_state
   api/en/c/3_qis/4_pauli_string
   api/en/c/3_qis/5_hamiltonian
   api/en/c/4_compile/1_compile
   api/en/c/5_visualization/1_text_drawer
   api/en/c/5_visualization/2_figure_drawer
   api/en/c/6_error_mitigation/1_error_mitigation
   api/en/c/6_error_mitigation/2_zne_mitigation
   api/en/c/6_error_mitigation/3_virtual_distillation

