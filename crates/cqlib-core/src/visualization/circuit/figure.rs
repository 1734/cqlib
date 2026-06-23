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

//! Figure visualization backend.
//!
//! This module renders circuits with a Rust-native SVG-first pipeline and optional PNG
//! rasterization (via `resvg`).
//!
//! # Core Features
//!
//! - **SVG-first rendering**: directly generates scalable vector output.
//! - **Optional PNG export**: rasterizes SVG through `resvg` when bitmap output is needed.
//! - **Shared IR pipeline**: consumes [`VisualCircuit`] built by the common visualization builder.
//! - **Style-map driven drawing**: gate colors/fonts/line styles come from `styles/default.json`
//!   with optional overrides.
//!
//! # Typical Workflow
//!
//! 1. Build visualization IR from a circuit.
//! 2. Convert IR into SVG output.
//! 3. Save SVG directly or rasterize to PNG.
//!
//! # Examples
//!
//! ```rust
//! use cqlib_core::circuit::{Circuit, Qubit};
//! use cqlib_core::visualization::{FigureDrawerOptions, circuit_to_figure};
//!
//! let mut circuit = Circuit::new(2);
//! circuit.h(Qubit::new(0)).unwrap();
//! circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
//!
//! let script = circuit_to_figure(&circuit, &FigureDrawerOptions::default()).unwrap();
//! assert!(script.contains("<svg"));
//! ```

use crate::circuit::Circuit;
use crate::visualization::VisualizationError;
use crate::visualization::circuit::ir::{
    VisualCircuit, VisualControlFlowKind, VisualOpStyle, VisualOperation,
    flatten_control_flow_visual, reverse_visual_lanes,
};
use crate::visualization::circuit::layout::{VisualBuildOptions, build_visual_circuit};
use crate::visualization::circuit::params::ParameterFormatOptions;
use crate::visualization::circuit::style::{GateStyle, StyleBook};
use crate::visualization::svg::render_svg_to_file;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
struct FigurePalette {
    wire_color: &'static str,
    connector_color: &'static str,
    text_color: &'static str,
    gate_edge_color: &'static str,
    gate_fill_color: Option<&'static str>,
    barrier_color: &'static str,
    wire_linewidth: f64,
    connector_linewidth: f64,
    gate_linewidth: f64,
    gate_fontsize: u8,
}

/// Minimum logical width reserved for an empty folded row.
const MIN_ROW_LOGICAL_WIDTH: f64 = 2.0;
/// Logical Y pitch between adjacent qubit wires.
const WIRE_PITCH: f64 = 2.0;
/// Left X bound of the drawing area (keeps qubit labels visible).
const CANVAS_MIN_X: f64 = -1.4;
/// Extra padding on the right side of the drawing area.
const CANVAS_RIGHT_PADDING: f64 = 0.3;
/// Symmetric Y padding around the drawing area.
const CANVAS_Y_PADDING: f64 = 1.0;
/// Base pixel scale per logical unit (further scaled by options).
const LOGICAL_UNIT_TO_PX: f64 = 80.0;
/// Inner text padding for gate labels (in pixels).
const LABEL_INNER_PADDING_PX: f64 = 10.0;
/// Minimum usable text area per direction (in pixels).
const LABEL_MIN_INNER_PX: f64 = 4.0;
/// Relative parameter font size against the gate label font.
const PARAM_FONT_SCALE: f64 = 0.78;
/// Approximate width factor for gate-name fitting.
const NAME_WIDTH_FACTOR: f64 = 0.60;
/// Approximate width factor for parameter-line fitting.
const PARAM_WIDTH_FACTOR: f64 = 0.56;
/// Relative vertical gap between gate name and parameter line.
const LABEL_LINE_GAP_SCALE: f64 = 0.22;
/// Maximum fitting iterations for label down-scaling.
const LABEL_FIT_MAX_ITERS: usize = 24;
/// Upper/lower clamp for each fitting step scale factor.
const LABEL_FIT_MAX_STEP: f64 = 0.95;
const LABEL_FIT_MIN_STEP: f64 = 0.10;
/// Module/generic span-gate label width estimator tuning.
const MODULE_LABEL_WIDTH_DIVISOR: f64 = 4.0;
const MODULE_LABEL_PADDING_THRESHOLD: f64 = 6.0;
const MODULE_LABEL_PADDING_CHARS: f64 = 1.0;
/// Extra headroom when packing columns into folded rows.
const FOLD_TARGET_SLACK: f64 = 1.12;
/// SVG canvas background color.
const CANVAS_BACKGROUND_COLOR: &str = "#ffffff";
/// Default rasterization DPI used as the 1.0 PNG scale baseline.
const DEFAULT_FIGURE_DPI: u32 = 160;

/// Figure rendering theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FigureDrawStyle {
    /// Cqlib default style loaded from `styles/default.json`.
    Cqlib,
}

/// Options for figure drawing.
///
/// Figure rendering consumes [`crate::visualization::VisualCircuit`] IR and supports SVG-first
/// output with optional PNG rasterization through [`crate::visualization::render_figure_to_file`].
#[derive(Debug, Clone)]
pub struct FigureDrawerOptions {
    /// Whether to append parameter text to gate labels.
    pub show_params: bool,
    /// Whether to decompose circuit-gates before drawing.
    pub decompose_circuit_gates: bool,
    /// Parameter display format used by visualization IR builder.
    pub parameter_format: ParameterFormatOptions,
    /// Figure width scale per logical column.
    pub width_per_column: f64,
    /// Figure height scale per qubit.
    pub height_per_qubit: f64,
    /// Rasterization DPI when exporting PNG.
    ///
    /// The default value (`160`) maps to the native SVG pixel size. Larger values
    /// scale PNG dimensions proportionally while leaving SVG output unchanged.
    pub dpi: u32,
    /// Base gate width (data units), also used as minimum column width.
    pub gate_width: f64,
    /// Base gate height (data units).
    pub gate_height: f64,
    /// Horizontal spacing between adjacent columns.
    pub moment_spacing: f64,
    /// Vertical spacing between folded rows.
    pub connect_height: f64,
    /// Maximum columns per row (`-1` disables folding).
    pub fold: i32,
    /// Plot style preset.
    pub style: FigureDrawStyle,
    /// Optional per-gate style overrides (merged over base style map).
    pub gate_styles: HashMap<String, GateStyle>,
    /// Whether to show `|0>` in qubit labels.
    pub initial_state: bool,
    /// Whether to reverse display order of qubits.
    pub reverse_bits: bool,
}

impl Default for FigureDrawerOptions {
    fn default() -> Self {
        Self {
            show_params: true,
            decompose_circuit_gates: false,
            parameter_format: ParameterFormatOptions::default(),
            width_per_column: 1.2,
            height_per_qubit: 0.9,
            dpi: DEFAULT_FIGURE_DPI,
            gate_width: 1.1,
            gate_height: 1.5,
            moment_spacing: 0.3,
            connect_height: 2.0,
            fold: 18,
            style: FigureDrawStyle::Cqlib,
            gate_styles: HashMap::new(),
            initial_state: false,
            reverse_bits: false,
        }
    }
}

fn figure_palette(style: FigureDrawStyle) -> FigurePalette {
    match style {
        FigureDrawStyle::Cqlib => FigurePalette {
            wire_color: "black",
            connector_color: "black",
            text_color: "black",
            gate_edge_color: "black",
            gate_fill_color: Some("white"),
            barrier_color: "gray",
            wire_linewidth: 1.1,
            connector_linewidth: 1.0,
            gate_linewidth: 1.1,
            gate_fontsize: 9,
        },
    }
}

/// Generate SVG markup for a circuit.
///
/// # Arguments
///
/// * `circuit` - Input circuit to render.
/// * `options` - Figure rendering options.
///
/// # Returns
///
/// SVG markup as a UTF-8 string.
///
/// # Errors
///
/// Returns [`VisualizationError`] when IR build fails (for example, unknown qubit references).
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::{Circuit, Qubit};
/// use cqlib_core::visualization::{FigureDrawerOptions, circuit_to_figure};
///
/// let mut circuit = Circuit::new(1);
/// circuit.h(Qubit::new(0)).unwrap();
///
/// let script = circuit_to_figure(&circuit, &FigureDrawerOptions::default()).unwrap();
/// assert!(script.contains("<svg"));
/// ```
pub fn circuit_to_figure(
    circuit: &Circuit,
    options: &FigureDrawerOptions,
) -> Result<String, VisualizationError> {
    let visual_options = VisualBuildOptions {
        decompose_circuit_gates: options.decompose_circuit_gates,
        parameter_format: options.parameter_format,
        ..VisualBuildOptions::default()
    };
    let visual = build_visual_circuit(circuit, &visual_options)?;
    Ok(draw_figure_svg_from_visual(&visual, options))
}

/// Render a circuit directly to an output file (`.svg` or `.png`).
///
/// # Arguments
///
/// * `circuit` - Input circuit to render.
/// * `output_path` - Target file path. `.svg` writes vector output, `.png` writes raster output.
/// * `options` - Figure rendering options.
///
/// # Errors
///
/// Returns [`VisualizationError`] when IR build, file writing, or PNG rasterization fails.
///
/// # Examples
///
/// ```no_run
/// use cqlib_core::circuit::{Circuit, Qubit};
/// use cqlib_core::visualization::{FigureDrawerOptions, render_figure_to_file};
///
/// let mut circuit = Circuit::new(1);
/// circuit.h(Qubit::new(0)).unwrap();
///
/// render_figure_to_file(&circuit, "circuit.png", &FigureDrawerOptions::default()).unwrap();
/// ```
pub fn render_figure_to_file(
    circuit: &Circuit,
    output_path: &str,
    options: &FigureDrawerOptions,
) -> Result<(), VisualizationError> {
    let visual_options = VisualBuildOptions {
        decompose_circuit_gates: options.decompose_circuit_gates,
        parameter_format: options.parameter_format,
        ..VisualBuildOptions::default()
    };
    let visual = build_visual_circuit(circuit, &visual_options)?;
    let svg = draw_figure_svg_from_visual(&visual, options);
    render_svg_to_file(&svg, output_path, figure_png_scale(options)?)
}

/// Converts figure DPI into the PNG rasterization scale.
///
/// The default DPI maps to scale `1.0` so existing visual references stay stable.
/// Larger DPI values produce proportionally larger PNG dimensions.
fn figure_png_scale(options: &FigureDrawerOptions) -> Result<f64, VisualizationError> {
    if options.dpi == 0 {
        return Err(VisualizationError::InvalidInput(
            "figure dpi must be greater than zero".to_string(),
        ));
    }
    Ok(options.dpi as f64 / DEFAULT_FIGURE_DPI as f64)
}

fn draw_figure_svg_from_visual(visual: &VisualCircuit, options: &FigureDrawerOptions) -> String {
    let mut visual_data = flatten_control_flow_visual(visual);
    if options.reverse_bits {
        visual_data = reverse_visual_lanes(visual_data);
    }

    let num_qubits = visual_data.num_qubits();
    let num_columns = visual_data.num_columns.max(1);
    let palette = figure_palette(options.style);
    let style_book = StyleBook::new("default", &options.gate_styles);

    let mut cols_ops: Vec<Vec<&_>> = vec![Vec::new(); num_columns];
    for op in &visual_data.operations {
        if op.column < num_columns {
            cols_ops[op.column].push(op);
        }
    }
    // Keep a uniform base slot width. Selected operations reserve extra width:
    // - module/unitary span gates with long labels,
    // - control-flow markers whose labels should not shrink.
    let mut col_widths = vec![options.gate_width; num_columns];
    for col in 0..num_columns {
        for op in &cols_ops[col] {
            if is_module_span_gate(op) {
                col_widths[col] = col_widths[col].max(module_span_column_width(
                    op,
                    options.show_params,
                    options.gate_width,
                ));
            } else if is_control_flow_box(op) {
                col_widths[col] =
                    col_widths[col].max(control_flow_column_width(op, options.gate_width));
            }
        }
    }

    let row_columns = split_columns_by_fold(
        &col_widths,
        options.fold,
        options.moment_spacing,
        options.gate_width,
    );
    let mut row_widths = Vec::with_capacity(row_columns.len());
    for row in &row_columns {
        if row.is_empty() {
            row_widths.push(MIN_ROW_LOGICAL_WIDTH);
            continue;
        }
        let mut width = options.moment_spacing;
        for (i, col) in row.iter().enumerate() {
            width += col_widths[*col];
            if i + 1 < row.len() {
                width += options.moment_spacing;
            }
        }
        width += options.moment_spacing;
        row_widths.push(width.max(MIN_ROW_LOGICAL_WIDTH));
    }
    let x_max = row_widths
        .iter()
        .fold(0.0f64, |acc, w| acc.max(*w))
        .max(MIN_ROW_LOGICAL_WIDTH);
    let qubits_height = if num_qubits == 0 {
        WIRE_PITCH
    } else {
        (num_qubits as f64 - 1.0) * WIRE_PITCH
    };
    let total_height = if row_columns.is_empty() {
        WIRE_PITCH
    } else {
        (row_columns.len() as f64 - 1.0) * (qubits_height + options.connect_height) + qubits_height
    };

    let min_x = CANVAS_MIN_X;
    let max_x = x_max + CANVAS_RIGHT_PADDING;
    let min_y = -CANVAS_Y_PADDING;
    let max_y = total_height + CANVAS_Y_PADDING;
    let sx = LOGICAL_UNIT_TO_PX * options.width_per_column;
    let sy = LOGICAL_UNIT_TO_PX * options.height_per_qubit;
    let canvas_w = ((max_x - min_x) * sx).max(1.0);
    let canvas_h = ((max_y - min_y) * sy).max(1.0);
    let px = |x: f64| (x - min_x) * sx;
    let py = |y: f64| (y - min_y) * sy;

    let mut elements = Vec::new();
    elements.push(format!(
        "<rect x=\"0\" y=\"0\" width=\"{:.3}\" height=\"{:.3}\" fill=\"{}\"/>",
        canvas_w, canvas_h, CANVAS_BACKGROUND_COLOR
    ));

    let wire_color = style_book
        .get("default")
        .line_color
        .as_deref()
        .unwrap_or(palette.wire_color);
    let default_text_color = style_book
        .get("default")
        .text_color
        .as_deref()
        .unwrap_or(palette.text_color);
    let global_text_fs = style_book
        .get("default")
        .font_size
        .unwrap_or(palette.gate_fontsize as f64)
        .clamp(8.0, 48.0);

    for (row_idx, row_cols) in row_columns.iter().enumerate() {
        // Keep all folded rows at a consistent visual width.
        let row_x_max = x_max;
        let y_base = row_idx as f64 * (qubits_height + options.connect_height);

        for (lane, qubit) in visual_data.qubits.iter().enumerate() {
            let y = lane_to_y(lane, y_base);
            let q_label = if options.initial_state {
                format!("q{} |0>", qubit.id())
            } else {
                format!("q{}", qubit.id())
            };
            elements.push(svg_line(
                px(0.0),
                py(y),
                px(row_x_max),
                py(y),
                wire_color,
                palette.wire_linewidth,
                None,
            ));
            elements.push(svg_text(
                px(-0.08),
                py(y),
                &q_label,
                global_text_fs,
                default_text_color,
                "end",
            ));
        }
        if row_idx > 0 && num_qubits > 0 {
            elements.push(svg_line(
                px(0.0),
                py(y_base),
                px(0.0),
                py(y_base + qubits_height),
                wire_color,
                palette.wire_linewidth * 1.3,
                None,
            ));
        }
        if row_idx + 1 < row_columns.len() && num_qubits > 0 {
            elements.push(svg_line(
                px(row_x_max),
                py(y_base),
                px(row_x_max),
                py(y_base + qubits_height),
                wire_color,
                palette.wire_linewidth * 1.3,
                None,
            ));
        }

        let mut x = options.moment_spacing;
        for (i, col) in row_cols.iter().enumerate() {
            let col_w = col_widths[*col];
            x += col_w / 2.0;
            let x_center = x;

            for op in &cols_ops[*col] {
                // Keep regular gate boxes fixed-size. Module and control-flow boxes can expand
                // according to column width to preserve readable labels.
                let op_w = if is_module_span_gate(op) || is_control_flow_box(op) {
                    col_w
                } else {
                    options.gate_width
                };
                let label = compose_label(&op.label, &op.params, options.show_params);
                let gate_style = style_book.get(op_style_key(op));
                let min_lane = op.covered_lanes.iter().copied().min();
                let max_lane = op.covered_lanes.iter().copied().max();
                let connector_color = gate_style
                    .line_color
                    .as_deref()
                    .unwrap_or(palette.connector_color);
                let connector_lw = gate_style.line_width.unwrap_or(palette.connector_linewidth);

                match &op.style {
                    VisualOpStyle::Gate => {
                        if op.label == "FSIM" && op.lanes.len() >= 2 {
                            if let (Some(min_l), Some(max_l)) = (
                                op.lanes.iter().copied().min(),
                                op.lanes.iter().copied().max(),
                            ) {
                                if max_l > min_l {
                                    let y0 = lane_to_y(min_l, y_base);
                                    let y1 = lane_to_y(max_l, y_base);
                                    elements.push(svg_line(
                                        px(x_center),
                                        py(y0.min(y1)),
                                        px(x_center),
                                        py(y0.max(y1)),
                                        connector_color,
                                        connector_lw,
                                        None,
                                    ));
                                }
                            }
                            let r = (options.gate_width * 0.35 * sx.min(sy)).clamp(14.0, 28.0);
                            let circle_face = normalized_fill_color(gate_style, &palette)
                                .unwrap_or_else(|| "white".to_string());
                            let circle_edge = normalized_edge_color(gate_style, &palette)
                                .unwrap_or_else(|| connector_color.to_string());
                            let text_color = gate_style
                                .text_color
                                .as_deref()
                                .unwrap_or(palette.text_color);
                            // Keep FSIM text inside the circular marker.
                            let fsim_font = (r * 0.62).clamp(7.0, (global_text_fs * 0.9).max(7.0));
                            for lane in &op.lanes {
                                let y = lane_to_y(*lane, y_base);
                                elements.push(svg_circle(
                                    px(x_center),
                                    py(y),
                                    r,
                                    Some(&circle_face),
                                    Some(&circle_edge),
                                    connector_lw,
                                ));
                                elements.push(svg_text(
                                    px(x_center),
                                    py(y),
                                    "FSIM",
                                    fsim_font,
                                    text_color,
                                    "middle",
                                ));
                            }
                            continue;
                        }

                        if op.lanes.len() > 1 {
                            let gate_box_w = op_w;
                            let show_markers = show_span_lane_markers(op);
                            let marker_gutter = if show_markers {
                                (gate_box_w * 0.28).clamp(0.24, 0.42)
                            } else {
                                0.0
                            };
                            let start_lane = op.lanes.iter().copied().min().unwrap_or(0);
                            let end_lane = op.lanes.iter().copied().max().unwrap_or(start_lane);
                            let y0 = lane_to_y(start_lane, y_base);
                            let y1 = lane_to_y(end_lane, y_base);
                            let y_min = y0.min(y1) - options.gate_height / 2.0;
                            let box_h = (y0.max(y1) - y0.min(y1) + options.gate_height)
                                .max(options.gate_height);
                            elements.extend(draw_span_box_svg(
                                // Keep span-gate center strictly aligned to the moment center.
                                x_center,
                                y_min,
                                box_h,
                                &label,
                                &palette,
                                gate_style,
                                global_text_fs,
                                gate_box_w,
                                sx,
                                sy,
                                &px,
                                &py,
                            ));
                            if show_markers {
                                // Keep lane markers inside the box with a small left inset.
                                let marker_x = x_center - gate_box_w / 2.0 + marker_gutter * 0.2;
                                let marker_font_size = global_text_fs;
                                for (idx, lane) in op.lanes.iter().enumerate() {
                                    let y = lane_to_y(*lane, y_base);
                                    elements.push(svg_text(
                                        px(marker_x),
                                        py(y),
                                        &idx.to_string(),
                                        marker_font_size,
                                        gate_style
                                            .text_color
                                            .as_deref()
                                            .unwrap_or(palette.text_color),
                                        "start",
                                    ));
                                }
                            }
                        } else {
                            let anchor = op.lanes.iter().copied().min().unwrap_or(0);
                            let y = lane_to_y(anchor, y_base);
                            elements.extend(draw_box_svg(
                                x_center,
                                y,
                                &label,
                                &palette,
                                gate_style,
                                global_text_fs,
                                op_w,
                                options.gate_height,
                                sx,
                                sy,
                                &px,
                                &py,
                            ));
                        }
                    }
                    VisualOpStyle::Controlled { num_controls } => {
                        if let (Some(min_l), Some(max_l)) = (
                            op.lanes.iter().copied().min(),
                            op.lanes.iter().copied().max(),
                        ) {
                            if max_l > min_l {
                                let y0 = lane_to_y(min_l, y_base);
                                let y1 = lane_to_y(max_l, y_base);
                                elements.push(svg_line(
                                    px(x_center),
                                    py(y0.min(y1)),
                                    px(x_center),
                                    py(y0.max(y1)),
                                    connector_color,
                                    connector_lw,
                                    None,
                                ));
                            }
                        }
                        for lane in op.lanes.iter().take(*num_controls) {
                            let y = lane_to_y(*lane, y_base);
                            elements.push(svg_circle(
                                px(x_center),
                                py(y),
                                (0.07 * sx.min(sy)).max(4.0),
                                Some(connector_color),
                                Some(connector_color),
                                connector_lw,
                            ));
                        }
                        for lane in op.lanes.iter().skip(*num_controls) {
                            let y = lane_to_y(*lane, y_base);
                            elements.extend(draw_box_svg(
                                x_center,
                                y,
                                &label,
                                &palette,
                                gate_style,
                                global_text_fs,
                                op_w,
                                options.gate_height,
                                sx,
                                sy,
                                &px,
                                &py,
                            ));
                        }
                    }
                    VisualOpStyle::Cz => {
                        if let (Some(min_l), Some(max_l)) = (
                            op.lanes.iter().copied().min(),
                            op.lanes.iter().copied().max(),
                        ) {
                            if max_l > min_l {
                                let y0 = lane_to_y(min_l, y_base);
                                let y1 = lane_to_y(max_l, y_base);
                                elements.push(svg_line(
                                    px(x_center),
                                    py(y0.min(y1)),
                                    px(x_center),
                                    py(y0.max(y1)),
                                    connector_color,
                                    connector_lw,
                                    None,
                                ));
                            }
                        }
                        for lane in &op.lanes {
                            let y = lane_to_y(*lane, y_base);
                            elements.push(svg_circle(
                                px(x_center),
                                py(y),
                                (0.07 * sx.min(sy)).max(4.0),
                                Some(connector_color),
                                Some(connector_color),
                                connector_lw,
                            ));
                        }
                    }
                    VisualOpStyle::Swap => {
                        let swap_lw = connector_lw * 1.8;
                        if let (Some(min_l), Some(max_l)) = (
                            op.lanes.iter().copied().min(),
                            op.lanes.iter().copied().max(),
                        ) {
                            if max_l > min_l {
                                let y0 = lane_to_y(min_l, y_base);
                                let y1 = lane_to_y(max_l, y_base);
                                elements.push(svg_line(
                                    px(x_center),
                                    py(y0.min(y1)),
                                    px(x_center),
                                    py(y0.max(y1)),
                                    connector_color,
                                    swap_lw,
                                    None,
                                ));
                            }
                        }
                        for lane in &op.lanes {
                            let y = lane_to_y(*lane, y_base);
                            elements.push(svg_line(
                                px(x_center - 0.15),
                                py(y - 0.15),
                                px(x_center + 0.15),
                                py(y + 0.15),
                                connector_color,
                                swap_lw,
                                None,
                            ));
                            elements.push(svg_line(
                                px(x_center - 0.15),
                                py(y + 0.15),
                                px(x_center + 0.15),
                                py(y - 0.15),
                                connector_color,
                                swap_lw,
                                None,
                            ));
                        }
                    }
                    VisualOpStyle::Barrier => {
                        let barrier_lw =
                            gate_style.line_width.unwrap_or(palette.gate_linewidth) * 1.8;
                        let (start_lane, end_lane) =
                            if let (Some(min_l), Some(max_l)) = (min_lane, max_lane) {
                                (min_l, max_l)
                            } else if num_qubits > 0 {
                                (0, num_qubits - 1)
                            } else {
                                (0, 0)
                            };
                        let y0 = lane_to_y(start_lane, y_base);
                        let y1 = lane_to_y(end_lane, y_base);
                        elements.push(svg_line(
                            px(x_center),
                            py(y0.min(y1) - options.gate_height / 2.0),
                            px(x_center),
                            py(y0.max(y1) + options.gate_height / 2.0),
                            gate_style
                                .line_color
                                .as_deref()
                                .unwrap_or(palette.barrier_color),
                            barrier_lw,
                            Some("6,4"),
                        ));
                    }
                    VisualOpStyle::Measure => {
                        if op.lanes.is_empty() {
                            elements.extend(draw_measure_svg(
                                x_center,
                                lane_to_y(0, y_base),
                                &palette,
                                gate_style,
                                global_text_fs,
                                op_w,
                                options.gate_height,
                                sx,
                                sy,
                                &px,
                                &py,
                            ));
                        } else {
                            for lane in &op.lanes {
                                elements.extend(draw_measure_svg(
                                    x_center,
                                    lane_to_y(*lane, y_base),
                                    &palette,
                                    gate_style,
                                    global_text_fs,
                                    op_w,
                                    options.gate_height,
                                    sx,
                                    sy,
                                    &px,
                                    &py,
                                ));
                            }
                        }
                    }
                    VisualOpStyle::Reset | VisualOpStyle::Delay => {
                        if op.lanes.is_empty() {
                            elements.extend(draw_box_svg(
                                x_center,
                                lane_to_y(0, y_base),
                                &label,
                                &palette,
                                gate_style,
                                global_text_fs,
                                op_w,
                                options.gate_height,
                                sx,
                                sy,
                                &px,
                                &py,
                            ));
                        } else {
                            for lane in &op.lanes {
                                elements.extend(draw_box_svg(
                                    x_center,
                                    lane_to_y(*lane, y_base),
                                    &label,
                                    &palette,
                                    gate_style,
                                    global_text_fs,
                                    op_w,
                                    options.gate_height,
                                    sx,
                                    sy,
                                    &px,
                                    &py,
                                ));
                            }
                        }
                    }
                    VisualOpStyle::ControlFlow { kind } => {
                        let start_lane = min_lane
                            .or_else(|| op.lanes.iter().copied().min())
                            .unwrap_or(0);
                        let end_lane = max_lane
                            .or_else(|| op.lanes.iter().copied().max())
                            .unwrap_or(start_lane);
                        let y0 = lane_to_y(start_lane, y_base);
                        let y1 = lane_to_y(end_lane, y_base);
                        let y_min = y0.min(y1) - options.gate_height / 2.0;
                        let box_h = (y0.max(y1) - y0.min(y1) + options.gate_height)
                            .max(options.gate_height);
                        elements.extend(draw_flow_box_svg(
                            x_center,
                            y_min,
                            box_h,
                            &label,
                            &palette,
                            gate_style,
                            global_text_fs,
                            op_w,
                            sx,
                            sy,
                            &px,
                            &py,
                        ));
                        if matches!(
                            kind,
                            VisualControlFlowKind::IfElseBlock {
                                has_false_branch: true,
                                ..
                            }
                        ) {
                            elements.push(svg_text(
                                px(x_center + op_w / 2.0 + 0.08),
                                py(y_min + box_h - 0.12),
                                "else",
                                global_text_fs,
                                gate_style
                                    .text_color
                                    .as_deref()
                                    .unwrap_or(palette.text_color),
                                "start",
                            ));
                        }
                    }
                }
            }

            x += col_w / 2.0;
            if i + 1 < row_cols.len() {
                x += options.moment_spacing;
            }
        }
    }

    let mut out = String::new();
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.0}\" height=\"{:.0}\" viewBox=\"0 0 {:.3} {:.3}\">",
        canvas_w, canvas_h, canvas_w, canvas_h
    ));
    for e in elements {
        out.push_str(&e);
    }
    out.push_str("</svg>");
    out
}

/// Generate SVG markup from pre-built visualization IR.
///
/// Use this API when you want to cache or transform [`VisualCircuit`] once and render it
/// multiple times with different figure options.
///
/// # Arguments
///
/// * `visual` - Pre-built visualization IR produced by [`build_visual_circuit`].
/// * `options` - Figure rendering options.
/// * `_output_path` - Reserved for API compatibility; currently ignored.
///
/// # Returns
///
/// SVG markup as a UTF-8 string.
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::{Circuit, Qubit};
/// use cqlib_core::visualization::{
///     FigureDrawerOptions, VisualBuildOptions, build_visual_circuit, draw_figure_from_visual,
/// };
///
/// let mut circuit = Circuit::new(1);
/// circuit.h(Qubit::new(0)).unwrap();
/// let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
/// let svg = draw_figure_from_visual(&visual, &FigureDrawerOptions::default(), None);
/// assert!(svg.contains("<svg"));
/// ```
pub fn draw_figure_from_visual(
    visual: &VisualCircuit,
    options: &FigureDrawerOptions,
    _output_path: Option<&str>,
) -> String {
    draw_figure_svg_from_visual(visual, options)
}

fn svg_line(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    stroke: &str,
    stroke_width: f64,
    dash: Option<&str>,
) -> String {
    let dash_attr = dash
        .map(|d| format!(" stroke-dasharray=\"{}\"", d))
        .unwrap_or_default();
    format!(
        "<line x1=\"{:.3}\" y1=\"{:.3}\" x2=\"{:.3}\" y2=\"{:.3}\" stroke=\"{}\" stroke-width=\"{:.3}\"{} />",
        x1, y1, x2, y2, stroke, stroke_width, dash_attr
    )
}

#[allow(clippy::too_many_arguments)]
fn svg_rect(
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    fill: &str,
    stroke: &str,
    lw: f64,
    dash: Option<&str>,
) -> String {
    let dash_attr = dash
        .map(|d| format!(" stroke-dasharray=\"{}\"", d))
        .unwrap_or_default();
    format!(
        "<rect x=\"{:.3}\" y=\"{:.3}\" width=\"{:.3}\" height=\"{:.3}\" fill=\"{}\" stroke=\"{}\" stroke-width=\"{:.3}\"{} />",
        x, y, w, h, fill, stroke, lw, dash_attr
    )
}

fn svg_circle(
    cx: f64,
    cy: f64,
    r: f64,
    fill: Option<&str>,
    stroke: Option<&str>,
    lw: f64,
) -> String {
    format!(
        "<circle cx=\"{:.3}\" cy=\"{:.3}\" r=\"{:.3}\" fill=\"{}\" stroke=\"{}\" stroke-width=\"{:.3}\" />",
        cx,
        cy,
        r,
        fill.unwrap_or("none"),
        stroke.unwrap_or("none"),
        lw
    )
}

fn svg_text(x: f64, y: f64, text: &str, font_size: f64, color: &str, anchor: &str) -> String {
    format!(
        "<text x=\"{:.3}\" y=\"{:.3}\" fill=\"{}\" font-size=\"{:.3}\" font-family=\"DejaVu Sans, Arial, sans-serif\" text-anchor=\"{}\" dominant-baseline=\"middle\">{}</text>",
        x,
        y,
        color,
        font_size,
        anchor,
        escape_xml(text)
    )
}

#[allow(clippy::too_many_arguments)]
fn draw_gate_label_svg(
    x_px: f64,
    y_px: f64,
    label: &str,
    text_color: &str,
    box_w_px: f64,
    box_h_px: f64,
    base_name_fs: f64,
    allow_shrink: bool,
) -> Vec<String> {
    let mut parts = label.splitn(2, '\n');
    let name = parts.next().unwrap_or_default();
    let param = parts.next().filter(|s| !s.is_empty());
    let avail_w = (box_w_px - LABEL_INNER_PADDING_PX).max(LABEL_MIN_INNER_PX);
    let avail_h = (box_h_px - LABEL_INNER_PADDING_PX).max(LABEL_MIN_INNER_PX);
    // Keep a unified global base size (from style default), and only shrink if this gate overflows.
    let mut nfs = base_name_fs.max(1.0);
    let mut pfs = (nfs * PARAM_FONT_SCALE).max(1.0);
    if allow_shrink {
        for _ in 0..LABEL_FIT_MAX_ITERS {
            // Use conservative width factors so measured text is less likely to overflow the box.
            let name_w = name.chars().count() as f64 * nfs * NAME_WIDTH_FACTOR;
            let (need_w, need_h) = if let Some(p) = param {
                let param_w = p.chars().count() as f64 * pfs * PARAM_WIDTH_FACTOR;
                let gap = nfs * LABEL_LINE_GAP_SCALE;
                (name_w.max(param_w), nfs + gap + pfs)
            } else {
                (name_w, nfs)
            };
            if need_w <= avail_w && need_h <= avail_h {
                break;
            }
            let s = (avail_w / need_w)
                .min(avail_h / need_h)
                .clamp(LABEL_FIT_MIN_STEP, LABEL_FIT_MAX_STEP);
            nfs = (nfs * s).max(1.0);
            pfs = (nfs * PARAM_FONT_SCALE).max(1.0);
        }
    }
    let mut out = Vec::new();
    if let Some(p) = param {
        let gap = nfs * LABEL_LINE_GAP_SCALE;
        let total_h = nfs + gap + pfs;
        let top = y_px - total_h / 2.0;
        let name_y = top + nfs / 2.0;
        let param_y = name_y + (nfs / 2.0 + gap + pfs / 2.0);
        out.push(svg_text(x_px, name_y, name, nfs, text_color, "middle"));
        out.push(svg_text(x_px, param_y, p, pfs, text_color, "middle"));
    } else {
        out.push(svg_text(x_px, y_px, name, nfs, text_color, "middle"));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn draw_box_svg(
    x: f64,
    y: f64,
    label: &str,
    palette: &FigurePalette,
    style: &GateStyle,
    base_font_size: f64,
    width: f64,
    height: f64,
    sx: f64,
    sy: f64,
    px: &impl Fn(f64) -> f64,
    py: &impl Fn(f64) -> f64,
) -> Vec<String> {
    draw_labeled_rect_svg(
        x,
        y,
        width,
        height,
        label,
        palette,
        style,
        base_font_size,
        sx,
        sy,
        px,
        py,
        None,
        None,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn draw_span_box_svg(
    x: f64,
    y_min: f64,
    height: f64,
    label: &str,
    palette: &FigurePalette,
    style: &GateStyle,
    base_font_size: f64,
    width: f64,
    sx: f64,
    sy: f64,
    px: &impl Fn(f64) -> f64,
    py: &impl Fn(f64) -> f64,
) -> Vec<String> {
    draw_labeled_rect_svg(
        x,
        y_min + height / 2.0,
        width,
        height,
        label,
        palette,
        style,
        base_font_size,
        sx,
        sy,
        px,
        py,
        None,
        None,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn draw_flow_box_svg(
    x: f64,
    y_min: f64,
    height: f64,
    label: &str,
    palette: &FigurePalette,
    style: &GateStyle,
    base_font_size: f64,
    width: f64,
    sx: f64,
    sy: f64,
    px: &impl Fn(f64) -> f64,
    py: &impl Fn(f64) -> f64,
) -> Vec<String> {
    draw_labeled_rect_svg(
        x,
        y_min + height / 2.0,
        width,
        height,
        label,
        palette,
        style,
        base_font_size,
        sx,
        sy,
        px,
        py,
        Some("6,4"),
        Some(palette.gate_edge_color),
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn draw_labeled_rect_svg(
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    label: &str,
    palette: &FigurePalette,
    style: &GateStyle,
    base_font_size: f64,
    sx: f64,
    sy: f64,
    px: &impl Fn(f64) -> f64,
    py: &impl Fn(f64) -> f64,
    dash: Option<&str>,
    edge_fallback: Option<&str>,
    allow_shrink: bool,
) -> Vec<String> {
    let fill = normalized_fill_color(style, palette).unwrap_or_else(|| "none".to_string());
    // Border color defaults to the fill color so the gate box outline matches its background.
    let edge = normalized_edge_color(style, palette)
        .or_else(|| edge_fallback.map(str::to_string))
        .unwrap_or_else(|| fill.clone());
    let lw = style_line_width(style, palette);
    let text_color = str_color(style.text_color.as_deref(), palette.text_color);
    let rx = px(x - width / 2.0);
    let ry = py(y - height / 2.0);
    let rw = width * sx;
    let rh = height * sy;
    let mut out = vec![svg_rect(rx, ry, rw, rh, &fill, &edge, lw, dash)];
    out.extend(draw_gate_label_svg(
        px(x),
        py(y),
        label,
        &text_color,
        rw,
        rh,
        base_font_size,
        allow_shrink,
    ));
    out
}

#[allow(clippy::too_many_arguments)]
fn draw_measure_svg(
    x: f64,
    y: f64,
    palette: &FigurePalette,
    style: &GateStyle,
    base_font_size: f64,
    width: f64,
    height: f64,
    sx: f64,
    sy: f64,
    px: &impl Fn(f64) -> f64,
    py: &impl Fn(f64) -> f64,
) -> Vec<String> {
    let mut out = draw_box_svg(
        x,
        y,
        "",
        palette,
        style,
        base_font_size,
        width,
        height,
        sx,
        sy,
        px,
        py,
    );
    let lw = style_line_width(style, palette);
    let line_color = str_color(style.line_color.as_deref(), palette.text_color);

    // Gauge arc: larger semicircle in the lower half of the box
    let cx = px(x);
    let cy = py(y + 0.18 * height);
    let rx = 0.35 * width * sx;
    let ry = 0.35 * height * sy;
    let arc_lw = (lw * 3.0).max(3.0);
    let arc_sx = cx - rx;
    let arc_ex = cx + rx;
    out.push(format!(
        "<path d=\"M {:.3} {:.3} A {:.3} {:.3} 0 0 1 {:.3} {:.3}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{:.3}\" stroke-linecap=\"round\" />",
        arc_sx, cy, rx, ry, arc_ex, cy, line_color, arc_lw
    ));

    // Small circle at the bottom center (pivot point)
    let dot_r = (lw * 3.0).max(3.0);
    out.push(svg_circle(cx, cy, dot_r, Some(&line_color), None, 0.0));

    // Arrow from pivot toward upper-right (45° direction)
    let arrow_len_x = rx * 0.9;
    let arrow_len_y = ry * 0.9;
    let ax1 = cx + arrow_len_x;
    let ay1 = cy - arrow_len_y;
    out.push(svg_line(cx, cy, ax1, ay1, &line_color, lw, None));

    // Arrowhead
    let head = (lw * 3.5).max(6.0);
    let angle: f64 = std::f64::consts::FRAC_PI_4; // 45°
    let perp = angle + std::f64::consts::FRAC_PI_2;
    let p1x = ax1 - head * angle.cos() + (head * 0.4) * perp.cos();
    let p1y = ay1 + head * angle.sin() - (head * 0.4) * perp.sin();
    let p2x = ax1 - head * angle.cos() - (head * 0.4) * perp.cos();
    let p2y = ay1 + head * angle.sin() + (head * 0.4) * perp.sin();
    out.push(format!(
        "<polygon points=\"{:.3},{:.3} {:.3},{:.3} {:.3},{:.3}\" fill=\"{}\"/>",
        ax1, ay1, p1x, p1y, p2x, p2y, line_color
    ));

    // "0" label at upper-left, "1" label at upper-right
    let label_fs = (base_font_size * 0.85).max(6.0);
    let label_y = py(y - 0.30 * height);
    let label_x0 = cx - rx * 0.75;
    let label_x1 = cx + rx * 0.75;
    out.push(svg_text(
        label_x0,
        label_y,
        "0",
        label_fs,
        &line_color,
        "middle",
    ));
    out.push(svg_text(
        label_x1,
        label_y,
        "1",
        label_fs,
        &line_color,
        "middle",
    ));

    out
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('\"', "&quot;")
}

/// Compose a multi-line gate label with optional formatted parameters.
fn compose_label(label: &str, params: &[String], show_params: bool) -> String {
    if show_params && !params.is_empty() {
        format!("{label}\n{}", params.join(","))
    } else {
        label.to_string()
    }
}

fn show_span_lane_markers(op: &VisualOperation) -> bool {
    matches!(op.label.as_str(), "RXX" | "RYY" | "RZX" | "RZZ" | "UNITARY")
}

fn is_module_span_gate(op: &VisualOperation) -> bool {
    matches!(op.style, VisualOpStyle::Gate) && op.span_box
}

fn is_control_flow_box(op: &VisualOperation) -> bool {
    matches!(op.style, VisualOpStyle::ControlFlow { .. })
}

fn module_span_column_width(op: &VisualOperation, show_params: bool, gate_width: f64) -> f64 {
    let label = compose_label(&op.label, &op.params, show_params);
    let text_len = label
        .lines()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0) as f64;
    if text_len <= 0.0 {
        return gate_width;
    }
    let padded_len = if text_len > MODULE_LABEL_PADDING_THRESHOLD {
        text_len + MODULE_LABEL_PADDING_CHARS
    } else {
        text_len
    };
    // Reserve extra horizontal space for module/unitary labels to prevent overflow.
    gate_width.max((padded_len / MODULE_LABEL_WIDTH_DIVISOR) * gate_width)
}

fn control_flow_column_width(op: &VisualOperation, gate_width: f64) -> f64 {
    let label_len = op.label.chars().count() as f64;
    if label_len <= 0.0 {
        return gate_width;
    }
    // Keep control-flow labels at base font-size and expand box width instead of shrinking text.
    const CONTROL_FLOW_LABEL_DIVISOR: f64 = 2.9;
    const CONTROL_FLOW_LABEL_PADDING_CHARS: f64 = 0.0;
    gate_width.max(
        ((label_len + CONTROL_FLOW_LABEL_PADDING_CHARS) / CONTROL_FLOW_LABEL_DIVISOR) * gate_width,
    )
}

/// Split columns into folded rows using an order-preserving greedy strategy.
///
/// The algorithm keeps each row as full as possible under the computed width budget.
fn split_columns_by_fold(
    col_widths: &[f64],
    fold: i32,
    moment_spacing: f64,
    gate_width: f64,
) -> Vec<Vec<usize>> {
    if col_widths.is_empty() {
        return vec![Vec::new()];
    }
    if fold < 0 {
        return vec![(0..col_widths.len()).collect()];
    }
    let target_cols = usize::try_from(fold)
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or(col_widths.len());

    // Width budget derived from actual average column width (not only gate_width),
    // with a small slack to keep rows as long/compact as possible.
    let avg_col_width =
        col_widths.iter().copied().sum::<f64>() / (col_widths.len() as f64).max(1.0);
    let effective_col_width = avg_col_width.max(gate_width);
    let target_width = (2.0 * moment_spacing
        + target_cols as f64 * effective_col_width
        + target_cols.saturating_sub(1) as f64 * moment_spacing)
        * FOLD_TARGET_SLACK;

    let mut rows = Vec::new();
    let mut start = 0usize;
    while start < col_widths.len() {
        let mut row = vec![start];
        let mut width = 2.0 * moment_spacing + col_widths[start];
        let mut next = start + 1;
        while next < col_widths.len() {
            let candidate = width + moment_spacing + col_widths[next];
            if candidate <= target_width {
                row.push(next);
                width = candidate;
                next += 1;
            } else {
                break;
            }
        }
        rows.push(row);
        start = next;
    }
    rows
}

fn lane_to_y(lane: usize, y_base: f64) -> f64 {
    y_base + lane as f64 * WIRE_PITCH
}

fn str_color(candidate: Option<&str>, fallback: &str) -> String {
    candidate.unwrap_or(fallback).to_string()
}

fn normalize_style_color(candidate: Option<&str>) -> Option<String> {
    match candidate {
        Some(value)
            if value.eq_ignore_ascii_case("none") || value.eq_ignore_ascii_case("transparent") =>
        {
            None
        }
        Some(value) => Some(value.to_string()),
        None => None,
    }
}

fn normalized_fill_color(style: &GateStyle, palette: &FigurePalette) -> Option<String> {
    normalize_style_color(style.background_color.as_deref())
        .or_else(|| normalize_style_color(palette.gate_fill_color))
}

fn normalized_edge_color(style: &GateStyle, palette: &FigurePalette) -> Option<String> {
    normalize_style_color(style.border_color.as_deref())
        .or_else(|| normalize_style_color(Some(palette.gate_edge_color)))
}

fn style_line_width(style: &GateStyle, palette: &FigurePalette) -> f64 {
    style.line_width.unwrap_or(palette.gate_linewidth)
}

/// Resolve style key used to query `StyleBook`.
///
/// Priority:
/// 1. Primitive style categories (`M/R/D/B/CZ/SWAP`);
/// 2. Control-flow families (`IF/ELSE/END/WHILE/FOR/SWITCH`);
/// 3. Span-box module category (`MODULE`);
/// 4. Gate label fallback.
fn op_style_key(op: &VisualOperation) -> &str {
    match &op.style {
        VisualOpStyle::Measure => "M",
        VisualOpStyle::Reset => "R",
        VisualOpStyle::Delay => "D",
        VisualOpStyle::Barrier => "B",
        VisualOpStyle::Cz => "CZ",
        VisualOpStyle::Swap => "SWAP",
        VisualOpStyle::ControlFlow { kind } => control_flow_style_key(kind),
        VisualOpStyle::Gate if op.span_box => "MODULE",
        _ => op.label.as_str(),
    }
}

fn control_flow_style_key(kind: &VisualControlFlowKind) -> &'static str {
    match kind {
        VisualControlFlowKind::IfElseBlock { .. } | VisualControlFlowKind::IfStart => "IF",
        VisualControlFlowKind::ElseStart => "ELSE",
        VisualControlFlowKind::WhileBlock { .. } | VisualControlFlowKind::WhileStart => "WHILE",
        VisualControlFlowKind::ForBlock { .. } | VisualControlFlowKind::ForStart => "FOR",
        VisualControlFlowKind::SwitchBlock { .. } | VisualControlFlowKind::SwitchStart => "SWITCH",
        VisualControlFlowKind::CaseStart => "CASE",
        VisualControlFlowKind::DefaultStart => "DEFAULT",
        VisualControlFlowKind::Break => "BREAK",
        VisualControlFlowKind::Continue => "CONTINUE",
        VisualControlFlowKind::End => "END",
    }
}
