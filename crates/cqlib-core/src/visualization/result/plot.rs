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

//! SVG rendering and file output for result/statistics plots.
//!
//! This module converts prepared bar-chart data into SVG markup and writes `.svg` or
//! rasterized `.png` output files.

use super::data::PreparedResultPlot;
use super::options::{DEFAULT_COLORS, ResultPlotKind, ResultPlotOptions};
use crate::visualization::VisualizationError;
use crate::visualization::svg::{escape_attr, escape_text, render_svg_to_file};

struct BarChartLayout {
    /// Outer SVG width in pixels.
    width: f64,
    /// Outer SVG height in pixels.
    height: f64,
    /// Left margin reserved for y-axis labels.
    margin_left: f64,
    /// Top margin reserved for the optional title.
    margin_top: f64,
    /// Width of the drawable plotting region.
    plot_w: f64,
    /// Height of the drawable plotting region.
    plot_h: f64,
    /// Minimum y-axis value after padding.
    min_y: f64,
    /// Maximum y-axis value after padding.
    max_y: f64,
    /// Pixel y-coordinate corresponding to value `0`.
    zero_y: f64,
    /// Width allocated to one x-axis label group.
    group_w: f64,
    /// Width allocated to one dataset bar inside a group.
    bar_w: f64,
}

impl BarChartLayout {
    /// Map a data value into SVG y-coordinate space.
    fn y_for_value(&self, value: f64) -> f64 {
        self.margin_top + self.plot_h - ((value - self.min_y) / self.span()) * self.plot_h
    }

    /// Numerically safe y-axis span used by coordinate conversion.
    fn span(&self) -> f64 {
        (self.max_y - self.min_y).max(1e-9)
    }
}

/// Write result-plot SVG markup to an output file.
///
/// # Arguments
///
/// * `svg` - SVG markup produced by [`crate::visualization::plot_histogram`] or
///   [`crate::visualization::plot_distribution`].
/// * `output_path` - Target file path. `.svg` writes vector output, `.png` writes raster output.
///
/// # Errors
///
/// Returns [`VisualizationError::Io`] or [`VisualizationError::SvgRenderFailed`] when file
/// output or PNG rasterization fails.
///
/// # Examples
///
/// ```no_run
/// use cqlib_core::visualization::{
///     ResultPlotOptions, plot_histogram, render_result_plot_to_file,
/// };
///
/// # fn demo(result: &cqlib_core::device::ExecutionResult) {
/// let svg = plot_histogram(result, &ResultPlotOptions::default()).unwrap();
/// render_result_plot_to_file(&svg, "hist.svg").unwrap();
/// # }
/// ```
pub fn render_result_plot_to_file(svg: &str, output_path: &str) -> Result<(), VisualizationError> {
    render_svg_to_file(svg, output_path, 1.0)
}

/// Render prepared grouped bar-chart data as SVG.
///
/// This function assumes labels and per-dataset values are already aligned by
/// [`crate::visualization::result::data::prepare_result_plot`].
pub(crate) fn render_bar_svg(plot: &PreparedResultPlot, options: &ResultPlotOptions) -> String {
    let layout = bar_chart_layout(plot, options);
    let colors = resolved_colors(options);

    let mut out = String::new();
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.0}\" height=\"{:.0}\" viewBox=\"0 0 {:.3} {:.3}\">",
        layout.width, layout.height, layout.width, layout.height
    ));
    out.push_str("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>");
    if let Some(title) = &options.title {
        out.push_str(&svg_text(
            layout.width / 2.0,
            28.0,
            title,
            18,
            "middle",
            "#202124",
        ));
    }

    for tick in 0..=5 {
        let t = tick as f64 / 5.0;
        let y = layout.margin_top + layout.plot_h * (1.0 - t);
        let value = layout.min_y + layout.span() * t;
        out.push_str(&svg_line(
            layout.margin_left,
            y,
            layout.margin_left + layout.plot_w,
            y,
            "#dfe3eb",
            1.0,
        ));
        out.push_str(&svg_text(
            layout.margin_left - 10.0,
            y + 4.0,
            &format_tick(value, plot.kind),
            11,
            "end",
            "#5f6673",
        ));
    }
    out.push_str(&svg_line(
        layout.margin_left,
        layout.margin_top,
        layout.margin_left,
        layout.margin_top + layout.plot_h,
        "#303642",
        1.2,
    ));
    out.push_str(&svg_line(
        layout.margin_left,
        layout.zero_y,
        layout.margin_left + layout.plot_w,
        layout.zero_y,
        "#303642",
        1.2,
    ));

    for (set_idx, set_values) in plot.values.iter().enumerate() {
        for (label_idx, value) in set_values.iter().enumerate() {
            if value.abs() < 1e-12 {
                continue;
            }
            let center =
                layout.margin_left + label_idx as f64 * layout.group_w + layout.group_w / 2.0;
            let x = center
                + (set_idx as f64 - (plot.values.len() as f64 - 1.0) / 2.0) * layout.bar_w
                - layout.bar_w / 2.0;
            let y_value = layout.y_for_value(*value);
            let y = layout.zero_y.min(y_value);
            let h = (layout.zero_y - y_value).abs().max(1.0);
            let color = &colors[set_idx % colors.len()];
            out.push_str(&format!(
                "<rect x=\"{x:.3}\" y=\"{y:.3}\" width=\"{:.3}\" height=\"{h:.3}\" rx=\"2\" fill=\"{}\"/>",
                layout.bar_w,
                escape_attr(color)
            ));
            if options.bar_labels {
                let label_y = if *value >= 0.0 { y - 5.0 } else { y + h + 13.0 };
                out.push_str(&svg_text(
                    x + layout.bar_w / 2.0,
                    label_y,
                    &format_bar_value(*value, plot.kind),
                    10,
                    "middle",
                    "#303642",
                ));
            }
        }
    }

    for (idx, label) in plot.labels.iter().enumerate() {
        let x = layout.margin_left + idx as f64 * layout.group_w + layout.group_w / 2.0;
        let y = layout.margin_top + layout.plot_h + 18.0;
        out.push_str(&format!(
            "<text x=\"{x:.3}\" y=\"{y:.3}\" font-family=\"Arial, sans-serif\" font-size=\"11\" fill=\"#303642\" text-anchor=\"end\" transform=\"rotate(-55 {x:.3} {y:.3})\">{}</text>",
            escape_text(label)
        ));
    }
    let y_label = if plot.kind == ResultPlotKind::Histogram {
        "Count"
    } else {
        "Probability"
    };
    out.push_str(&format!(
        "<text x=\"22\" y=\"{:.3}\" font-family=\"Arial, sans-serif\" font-size=\"13\" fill=\"#303642\" text-anchor=\"middle\" transform=\"rotate(-90 22 {:.3})\">{}</text>",
        layout.margin_top + layout.plot_h / 2.0,
        layout.margin_top + layout.plot_h / 2.0,
        y_label
    ));

    if let Some(legend) = &options.legend {
        let x = layout.margin_left + layout.plot_w + 24.0;
        let mut y = layout.margin_top + 10.0;
        for (idx, item) in legend.iter().enumerate() {
            out.push_str(&format!(
                "<rect x=\"{x:.3}\" y=\"{:.3}\" width=\"13\" height=\"13\" rx=\"2\" fill=\"{}\"/>",
                y - 10.0,
                escape_attr(&colors[idx % colors.len()])
            ));
            out.push_str(&svg_text(x + 20.0, y + 1.0, item, 12, "start", "#303642"));
            y += 22.0;
        }
    }
    out.push_str("</svg>");
    out
}

/// Compute chart geometry from data range and user figure options.
///
/// Histograms anchor the y-axis at zero. Distribution plots keep a small negative range
/// only when negative values are present in the generic numeric input.
fn bar_chart_layout(plot: &PreparedResultPlot, options: &ResultPlotOptions) -> BarChartLayout {
    let (width, height) = options
        .figsize
        .map(|(w, h)| (w.max(2.0) * 100.0, h.max(2.0) * 100.0))
        .unwrap_or((760.0, 480.0));
    let margin_left = 72.0;
    let margin_right = if options.legend.is_some() {
        150.0
    } else {
        34.0
    };
    let margin_top = if options.title.is_some() { 58.0 } else { 32.0 };
    let margin_bottom = 96.0;
    let plot_w = (width - margin_left - margin_right).max(40.0);
    let plot_h = (height - margin_top - margin_bottom).max(40.0);

    let zeroish = plot
        .values
        .iter()
        .flat_map(|values| values.iter())
        .all(|value| value.abs() < 1e-12);
    let raw_min = plot
        .values
        .iter()
        .flat_map(|values| values.iter().copied())
        .fold(0.0_f64, f64::min);
    let raw_max = plot
        .values
        .iter()
        .flat_map(|values| values.iter().copied())
        .fold(0.0_f64, f64::max);

    let min_y = if plot.kind == ResultPlotKind::Distribution {
        (raw_min * 1.12).min(0.0)
    } else {
        0.0
    };
    let max_y = if zeroish {
        1.0
    } else {
        (raw_max * 1.12).max(1e-3)
    };
    let span = (max_y - min_y).max(1e-9);
    let zero_y = margin_top + plot_h - ((0.0 - min_y) / span) * plot_h;
    let n_labels = plot.labels.len().max(1);
    let n_sets = plot.values.len().max(1);
    let group_w = plot_w / n_labels as f64;
    let bar_w = (group_w / (n_sets as f64 + 1.0)).clamp(2.0, 44.0);

    BarChartLayout {
        width,
        height,
        margin_left,
        margin_top,
        plot_w,
        plot_h,
        min_y,
        max_y,
        zero_y,
        group_w,
        bar_w,
    }
}

/// Resolve user colors or fall back to Cqlib's built-in qualitative palette.
fn resolved_colors(options: &ResultPlotOptions) -> Vec<String> {
    if options.color.is_empty() {
        DEFAULT_COLORS
            .iter()
            .map(|color| color.to_string())
            .collect()
    } else {
        options.color.clone()
    }
}

/// Format y-axis tick values for the selected plot family.
fn format_tick(value: f64, kind: ResultPlotKind) -> String {
    if kind == ResultPlotKind::Histogram && value.abs() >= 10.0 {
        format!("{value:.0}")
    } else if value.abs() >= 1.0 {
        format!("{value:.2}")
    } else {
        format!("{value:.3}")
    }
}

/// Format bar-top labels for the selected plot family.
fn format_bar_value(value: f64, kind: ResultPlotKind) -> String {
    if kind == ResultPlotKind::Histogram {
        format!("{value:.0}")
    } else {
        format!("{value:.3}")
    }
}

/// SVG line element with attribute escaping for color.
fn svg_line(x1: f64, y1: f64, x2: f64, y2: f64, color: &str, width: f64) -> String {
    format!(
        "<line x1=\"{x1:.3}\" y1=\"{y1:.3}\" x2=\"{x2:.3}\" y2=\"{y2:.3}\" stroke=\"{}\" stroke-width=\"{width:.3}\"/>",
        escape_attr(color)
    )
}

/// SVG text element with escaped text and attributes.
fn svg_text(x: f64, y: f64, text: &str, size: u8, anchor: &str, color: &str) -> String {
    format!(
        "<text x=\"{x:.3}\" y=\"{y:.3}\" font-family=\"Arial, sans-serif\" font-size=\"{size}\" fill=\"{}\" text-anchor=\"{}\">{}</text>",
        escape_attr(color),
        escape_attr(anchor),
        escape_text(text)
    )
}
