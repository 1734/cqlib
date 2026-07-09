use super::*;
use crate::compile::transform::decompose::unitary::TwoQubitUnitaryDecomposeBasis;

#[test]
fn normal_config_uses_bounded_default_search_budget() {
    let config = TwoQubitBlockResynthesisConfig::normal(TwoQubitUnitaryDecomposeBasis::Cx);

    assert_eq!(config.two_qubit_basis, TwoQubitUnitaryDecomposeBasis::Cx);
    assert_eq!(config.max_block_ops, 16);
    assert_eq!(config.max_crossed_ops, 4);
    assert_eq!(config.max_scan_span, 32);
    assert!(config.skip_labeled_ops);
    assert!(config.recurse_control_flow);
    assert!(config.commutation.enable_rule_oracle);
    assert!(!config.commutation.enable_matrix_fallback);
}

#[test]
fn enhanced_config_increases_only_search_budgets() {
    let normal = TwoQubitBlockResynthesisConfig::normal(TwoQubitUnitaryDecomposeBasis::Cz);
    let enhanced = TwoQubitBlockResynthesisConfig::enhanced(TwoQubitUnitaryDecomposeBasis::Cz);

    assert_eq!(enhanced.two_qubit_basis, normal.two_qubit_basis);
    assert!(enhanced.max_block_ops > normal.max_block_ops);
    assert!(enhanced.max_crossed_ops > normal.max_crossed_ops);
    assert!(enhanced.max_scan_span > normal.max_scan_span);
    assert_eq!(enhanced.skip_labeled_ops, normal.skip_labeled_ops);
    assert_eq!(enhanced.recurse_control_flow, normal.recurse_control_flow);
    assert_eq!(enhanced.commutation, normal.commutation);
}
