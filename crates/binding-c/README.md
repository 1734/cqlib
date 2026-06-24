# C Binding for Cqlib

This crate exposes a small C ABI for building `cqlib-core` quantum circuits.
The current surface is intentionally limited to circuit construction and
symbolic gate parameters.

## Build

```bash
cargo build -p binding-c --release
```

The C header is generated at `crates/binding-c/include/cqlib_c.h`.

## Example

```bash
gcc crates/binding-c/examples/main.c \
    -I crates/binding-c/include \
    -L target/release \
    -lbinding_c \
    -lm \
    -o target/release/cqlib_c_example

./target/release/cqlib_c_example
```

## API

### Circuit

| Function | Description |
| --- | --- |
| `circuit_new(size_t num_qubits)` | Create a circuit. |
| `circuit_free(CircuitWrapper*)` | Free a circuit. |
| `circuit_num_qubits(const CircuitWrapper*)` | Return qubit count. |
| `circuit_num_operations(const CircuitWrapper*)` | Return operation count. |
| `circuit_num_parameters(const CircuitWrapper*)` | Return interned symbolic parameter count. |
| `circuit_validate(const CircuitWrapper*)` | Validate circuit consistency. |

### Gates

| Function | Description |
| --- | --- |
| `circuit_h/x/y/z(CircuitWrapper*, uint32_t qubit)` | Single-qubit fixed gates. |
| `circuit_rx/ry/rz(CircuitWrapper*, uint32_t qubit, double theta)` | Numeric rotations. |
| `circuit_cx/cz(CircuitWrapper*, uint32_t control, uint32_t target)` | Two-qubit gates. |
| `circuit_measure(CircuitWrapper*, uint32_t qubit)` | Measure a qubit. |
| `circuit_reset(CircuitWrapper*, uint32_t qubit)` | Reset a qubit. |

### Parameters

Bindings use `name:value,name2:value2` format.

| Function | Description |
| --- | --- |
| `param_parse(const char*)` | Parse a symbolic parameter expression. |
| `param_free(ParameterWrapper*)` | Free a parameter. |
| `param_evaluate(const ParameterWrapper*, const char* bindings)` | Evaluate a parameter. |
| `circuit_rx_param/ry_param/rz_param(CircuitWrapper*, uint32_t qubit, const ParameterWrapper*)` | Symbolic rotations. |
| `circuit_assign_params(const CircuitWrapper*, const char* bindings)` | Return a new circuit with assigned parameters. |

## Return Values

- `int32_t`: `0` on success, negative on error.
- `-1`: null pointer or invalid C string.
- `-2`: qubit index out of bounds.
- `-3`: core circuit or parameter error.
- Pointer return values use `NULL` for errors.
- `param_evaluate` returns `0.0` on error.

## Tests

```bash
cargo test -p binding-c
```
