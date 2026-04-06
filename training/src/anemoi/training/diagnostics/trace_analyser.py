"""Trace analyser for PyTorch Lightning Profiler trace files.

Provides functions to parse, analyse, and report on training performance
from JSON trace files produced by PyTorch's profiler, including kernel
duration breakdowns, data loading stalls, memory usage, and per-component
runtime analysis.
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
LOGGER = logging.getLogger(__name__)
console = Console(record=True, width=200)


def trace_to_dataframe(trace_file: str, cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Read a PyTorch trace JSON file and convert it into a pandas DataFrame.

    Parameters
    ----------
    trace_file : str
        Path to the JSON trace file.
    cols : list[str], optional
        Columns to extract from each event. Defaults to ["cat", "name", "ts", "dur"].

    Returns
    -------
    pd.DataFrame
        DataFrame containing the trace events with a 'rank' column.
    """
    with open(trace_file, "r") as f:
        trace_data = json.load(f)

    rank = None
    if "distributedInfo" in trace_data:
        rank = trace_data["distributedInfo"].get("rank")

    if "traceEvents" not in trace_data:
        error_msg = "The JSON file does not contain 'traceEvents'."
        raise ValueError(error_msg)

    if cols is None:
        cols = ["cat", "name", "ts", "dur"]

    events = []
    for event in trace_data["traceEvents"]:
        events.append({k: event[k] for k in cols if k in event})

    df = pd.DataFrame(events)
    # Drop rows where 'dur' is missing or NaN to avoid propagating NaN into end_time
    if "dur" in df.columns:
        df = df.dropna(subset=["dur"])
    df["rank"] = rank
    return df


def merge_overlapping_intervals_df(
    df: pd.DataFrame,
    start_col: str = "ts",
    end_col: str = "end_time",
    cat_col: Optional[str] = None,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Merge overlapping time intervals into non-overlapping intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with start and end time columns.
    start_col : str
        Name of the start time column.
    end_col : str
        Name of the end time column.
    cat_col : str, optional
        Column to group by before merging.
    threshold : float
        Minimum gap to still merge intervals.

    Returns
    -------
    pd.DataFrame
        DataFrame with merged intervals.
    """

    def _merge(intervals):
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            last_start, last_end = merged[-1]
            cur_start, cur_end = current
            if cur_start <= last_end or cur_start - last_end < threshold:
                merged[-1] = (last_start, max(last_end, cur_end))
            else:
                merged.append(current)
        return merged

    merged_results = []
    if cat_col:
        for cat, group in df.groupby(cat_col):
            intervals = list(zip(group[start_col], group[end_col]))
            merged = _merge(intervals)
            merged_results.extend([{cat_col: cat, start_col: s, end_col: e} for s, e in merged])
    else:
        intervals = list(zip(df[start_col], df[end_col]))
        merged = _merge(intervals)
        merged_results = [{start_col: s, end_col: e} for s, e in merged]

    return pd.DataFrame(merged_results)


def get_runtime_breakdown(
    df: pd.DataFrame,
    categories: Optional[list[str]] = None,
    names_list: Optional[list[list[str]]] = None,
    no_names: Optional[list[str]] = None,
    regex: bool = True,
) -> tuple:
    """Compute runtime breakdown for specified event categories and names.

    Parameters
    ----------
    df : pd.DataFrame
        Trace DataFrame with 'cat', 'name', 'ts', 'end_time' columns.
    categories : list[str], optional
        Event categories to filter on.
    names_list : list[list[str]], optional
        List of name-pattern groups; each group is analysed separately.
    no_names : list[str], optional
        Name patterns to exclude.
    regex : bool
        Whether to use regex matching.

    Returns
    -------
    tuple
        (total_time, filtered_times, rest_times, total_filtered_time, total_rest_time, df_results)
    """
    if categories is None:
        categories = []
    if names_list is None:
        names_list = []
    if no_names is None:
        no_names = []

    start_time = df["ts"].min()
    end_time = df["end_time"].max()
    total_time = end_time - start_time
    filtered_times = []
    rest_times = []
    df_result = []

    for names in names_list:
        df_out = df.copy()
        if len(categories) != 0:
            regex_pattern_cat = "|".join(categories)
            df_out = df_out[df_out["cat"].str.contains(regex_pattern_cat, regex=regex, na=False)].sort_values(by="ts")
        if len(names) != 0:
            regex_pattern_name = "|".join(names)
            df_out = df_out[df_out["name"].str.contains(regex_pattern_name, regex=True, na=False)].sort_values(by="ts")
        if len(no_names) != 0:
            regex_pattern_no_name = "|".join(no_names)
            df_out = df_out[~df_out["name"].str.contains(regex_pattern_no_name, regex=True, na=False)].sort_values(
                by="ts"
            )

        # Only pass ts and end_time to merge (cat column is not needed as cat_col)
        df_out = merge_overlapping_intervals_df(df_out[["ts", "end_time"]])
        df_out["batch_dur"] = df_out["end_time"] - df_out["ts"]
        filtered_time = df_out["batch_dur"].sum()
        idle_time = total_time - filtered_time
        filtered_times.append(filtered_time)
        rest_times.append(idle_time)
        df_result.append(df_out)

    if not df_result:
        return total_time, [], [], 0.0, total_time, []

    merged_df = pd.concat(df_result, ignore_index=True)
    merged_df = merge_overlapping_intervals_df(merged_df[["ts", "end_time"]])
    merged_df["batch_dur"] = merged_df["end_time"] - merged_df["ts"]
    total_filtered_time = merged_df["batch_dur"].sum()
    total_rest_time = total_time - total_filtered_time

    return total_time, filtered_times, rest_times, total_filtered_time, total_rest_time, df_result


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge a list of (start, end) intervals, combining overlaps.

    Parameters
    ----------
    intervals : list[tuple[float, float]]
        List of (start, end) tuples.

    Returns
    -------
    list[tuple[float, float]]
        Merged intervals.
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for cur_start, cur_end in intervals[1:]:
        last_start, last_end = merged[-1]
        if cur_start <= last_end:
            merged[-1] = (last_start, max(last_end, cur_end))
        else:
            merged.append((cur_start, cur_end))
    return merged


def runtime_analysis(
    df: pd.DataFrame,
    start_col: str = "ts",
    end_col: str = "end_time",
    name_col: str = "name",
) -> pd.DataFrame:
    """Compute runtime metrics per named group, accounting for overlapping intervals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with start/end time columns and a name column.
    start_col, end_col, name_col : str
        Column names.

    Returns
    -------
    pd.DataFrame
        Per-name runtime statistics.
    """
    results = []
    for name, group in df.groupby(name_col):
        intervals = list(zip(group[start_col], group[end_col]))
        runtimes = group[end_col] - group[start_col]
        ncalls = len(runtimes)
        ttime = sum(runtimes)
        avgtime = ttime / ncalls
        maxtime = max(runtimes)
        mintime = min(runtimes)
        merged = merge_intervals(intervals)
        total_runtime = sum(e - s for s, e in merged)
        overlap = 100 * (ttime - total_runtime) / ttime if ttime > 0 else 0.0
        results.append(
            {
                name_col: name,
                "total time us": total_runtime,
                "ncalls": ncalls,
                "avg per call us": avgtime,
                "max per call us": maxtime,
                "min per call us": mintime,
                "overlap %": overlap,
            }
        )

    return pd.DataFrame(results)


def trim_chain(s: str) -> str:
    """Trim a user-annotation chain string to a shorter representation.

    Keeps the first two dash-separated prefix parts and the last three
    dot-separated suffix parts.
    """
    if not isinstance(s, str):
        return s
    parts = s.split(".")
    new_suffix = ".".join(parts[-3:])
    prefix = parts[0].split("-")
    new_prefix = prefix[0] + "-" + prefix[1] if len(prefix) > 1 else prefix[0]
    return "...".join([new_prefix, new_suffix])


def _build_breakdown_table(
    df_display: pd.DataFrame,
    title: str,
    section: str,
    ttime_exc_overlap: float,
    ttime_exc_overlap_p: float,
) -> Table:
    """Build a Rich Table for a subset of the breakdown dataframe."""

    def _trim_annotation_name(name: str) -> str:
        """Shorten a PyTorch annotation name.

        Input:  'anemoi-GraphTransformerConv-model.model.decoder.proc.conv.forward'
        Output: 'GraphTransformerConv | proc.conv | fwd'
        """
        try:
            _, class_name, full_path = name.split("-", 2)
            
            # Strip call direction from the end first
            if full_path.endswith(".forward"):
                call = "fwd"
                full_path = full_path[: -len(".forward")]
            elif full_path.endswith(".backward"):
                call = "bwd"
                full_path = full_path[: -len(".backward")]
            else:
                call = ""

            # Build prefix from the full_path structure directly:
            # full_path is always "model.model.{section}.{subpath}"
            # e.g. "model.model.decoder.proc.conv" with section="model.decoder"
            # The fixed outer wrapper is always "model.model." + section + "."
            # But section may itself be "model.decoder" or just "decoder",
            # so derive the prefix by just finding the section string in the path.
            section_leaf = section.split(".")[-1]  # e.g. "decoder" from "model.decoder"
            marker = f".{section_leaf}."
            idx = full_path.find(marker)
            if idx != -1:
                subpath = full_path[idx + len(marker):]
            else:
                subpath = full_path

            parts = [class_name]
            if subpath:
                parts.append(subpath)
            if call:
                parts.append(call)
            out = " | ".join(parts)
            return out

        except ValueError:
            return name
        
    timing_cols = ["total time sec", "avg per call us", "max per call us", "min per call us", "overlap %"]
    df_display = df_display.copy()  # Avoid SettingWithCopyWarning: df_display may be a slice from .head() or boolean indexing
    df_display[timing_cols] = df_display[timing_cols].round(3)
    df_display["method"] = df_display["name"].apply(_trim_annotation_name)

    table = Table(
        title=f"[bold]{title}[/bold]",
        title_justify="left",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        show_footer=True,
    )
    table.add_column("Method",    style="bold",    footer="[dim]Total (excl. overlap)[/dim]")
    table.add_column("Total (s)", justify="right", footer=f"[dim]{ttime_exc_overlap / 1e6:.3f}[/dim]")
    table.add_column("# Calls",   justify="right", footer="")
    table.add_column("Avg (µs)",  justify="right", footer="")
    table.add_column("Max (µs)",  justify="right", footer="")
    table.add_column("Min (µs)",  justify="right", footer="")
    table.add_column("Overlap %", justify="right", footer=f"[dim]{ttime_exc_overlap_p:.2f}%[/dim]")

    for _, row in df_display.iterrows():
        table.add_row(
            row["method"],
            f"{row['total time sec']:.3f}",
            str(int(row["ncalls"])),
            f"{row['avg per call us']:.3f}",
            f"{row['max per call us']:.3f}",
            f"{row['min per call us']:.3f}",
            f"{row['overlap %']:.3f}",
        )
    return table


def _plot_time_breakdowns(
    fwd_bwd_data: Optional[tuple],
    enc_dec_data: Optional[tuple],
    nbatches: int,
    ttotal_recorded: float,
    savepath: str = "",
) -> None:
    """
    Plot publication-quality pie charts for time breakdowns.

    Parameters
    ----------
    fwd_bwd_data : tuple or None
        (labels, values) for the forward/backward/data-loader breakdown.
    enc_dec_data : tuple or None
        (labels, values) for the encoder/processor/decoder breakdown.
    nbatches : int
        Number of batches processed.
    ttotal_recorded : float
        Total recorded wall time in microseconds.
    savepath : str
        If non-empty, the figure is saved to this path (PNG/PDF/SVG inferred
        from extension). DPI is 300 for raster formats.
    """
   
    # ── Publication style overrides ────────────────────────────────────────
    pub_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,   # embeds fonts in PDF (required by many journals)
        "ps.fonttype": 42,
        "text.usetex": False, # set True if LaTeX is available
    }

    # ── Colorblind-safe, print-friendly palettes ───────────────────────────
    # Based on Paul Tol's "muted" qualitative scheme — distinguishable in
    # greyscale and by dichromats.
    PALETTES = [
        # chart 1: forward / backward / data-loader / other
        ["#4878CF", "#D65F5F", "#6ACC65", "#B8B8B8"],
        # chart 2: encoder / processor / decoder / other
        ["#7B4173", "#A9B800", "#E49444", "#B8B8B8"],
    ]
    EDGE_COLOR = "white"
    TEXT_COLOR = "#1A1A1A"

    # ── Data assembly ──────────────────────────────────────────────────────
    charts = [d for d in [fwd_bwd_data, enc_dec_data] if d is not None]
    if not charts:
        return

    panel_labels = ["(a)", "(b)"]
   # subtitles = [
   #     "Forward · Backward · Data loader",
   #     "Encoder · Processor · Decoder",
   # ]
   # subtitles = subtitles[-len(charts):]
    panel_labels = panel_labels[-len(charts):]

    # ── Layout ────────────────────────────────────────────────────────────
    col_width_in = 3.3           # ~1 journal column each
    fig_w = col_width_in * len(charts) + 0.4
    fig_h = 3.8

    with mpl.rc_context(pub_rc):
        fig, axes = plt.subplots(
            1, len(charts),
            figsize=(fig_w, fig_h),
            facecolor="white",
        )
        if len(charts) == 1:
            axes = [axes]

        # ── Suptitle ──────────────────────────────────────────────────────
        throughput = nbatches / (ttotal_recorded / 1e6) if ttotal_recorded > 0 else 0
        total_s = ttotal_recorded / 1e6
        fig.suptitle(
            f"Batches: {nbatches}  ·  Total: {total_s:.3f} s  ·  Throughput: {throughput:.2f} it/s",
            fontsize=10,
            fontweight="bold",
            color=TEXT_COLOR,
            y=1.01,
        )

        for idx, (ax, (labels, values), panel) in enumerate(
            zip(axes, charts, panel_labels)
        ):
            colors = PALETTES[idx % len(PALETTES)]

            # Filter zero-value slices
            triples = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
            if not triples:
                ax.set_visible(False)
                continue
            f_labels, f_values, f_colors = zip(*triples)
            total_val = sum(f_values)

            # ── Pie chart ─────────────────────────────────────────────────
            explode = [0.025] * len(f_labels)
            wedges, _, autotexts = ax.pie(
                f_values,
                labels=None,
                autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
                colors=f_colors,
                startangle=90,
                pctdistance=1.18,        # push text outside the wedge
                explode=explode,
                wedgeprops={
                    "linewidth": 0.8,
                    "edgecolor": EDGE_COLOR,
                    "antialiased": True,
                },
                textprops={"fontsize": 7.5, "color": TEXT_COLOR},
            )
            for at in autotexts:
                at.set_fontsize(7.5)
                at.set_color(TEXT_COLOR)
                at.set_fontweight("bold")

            # ── Panel label (a), (b) top-left ─────────────────────────────
            ax.text(
                -1.35, 1.15, panel,
                transform=ax.transData,
                fontsize=9, fontweight="bold",
                color=TEXT_COLOR, va="top",
            )

            # ── Subtitle below panel label ─────────────────────────────────
           # ax.set_title(subtitle, fontsize=9, color=TEXT_COLOR, pad=8, loc="center")

            # ── Legend: coloured square + label + time + % ─────────────────
            legend_entries = [
                rf"{l}  {v / 1e6:.3f} s"
                if total_val > 0 else l
                for l, v in zip(f_labels, f_values)
            ]
            leg = ax.legend(
                wedges,
                legend_entries,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.34),
                ncol=1,
                frameon=True,
                framealpha=0.0,
                edgecolor="none",
                labelcolor=TEXT_COLOR,
                fontsize=7.5,
                handlelength=1.2,
                handleheight=1.0,
                handletextpad=0.5,
                borderpad=0.3,
                labelspacing=0.35,
            )
            # Square handles instead of pie wedges
            for handle in leg.legend_handles:
                handle.set_linewidth(0)

            ax.set_facecolor("white")

        # ── Shared thin frame around the whole figure ──────────────────────
        for spine in ["top", "right", "bottom", "left"]:
            for ax in axes:
                ax.spines[spine].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if savepath:
            ext = savepath.rsplit(".", 1)[-1].lower()
            dpi = 300 if ext in ("png", "tiff", "jpg", "jpeg") else None
            fig.savefig(
                savepath,
                bbox_inches="tight",
                dpi=dpi,
                facecolor="white",
            )

        plt.show()


def total_time_breakdown(tracefile_or_df: Union[str, pd.DataFrame], plot: bool = False) -> pd.DataFrame:
    """Print a coarse-grained time breakdown (forward, backward, dataloader, encoder, processor, decoder).
    Parameters
    ----------
    tracefile_or_df : str or pd.DataFrame
        Path to the trace JSON file, or an already-loaded trace DataFrame.
    plot : bool, optional
        If True, display pie charts for both breakdowns. Default is False.
    Returns
    -------
    pd.DataFrame
        Full trace DataFrame with 'end_time' column added.
    """
    if isinstance(tracefile_or_df, pd.DataFrame):
        df = tracefile_or_df
    else:
        df = trace_to_dataframe(tracefile_or_df)
    if "end_time" not in df.columns:
        df["end_time"] = df["ts"] + df["dur"]

    nbatches = len(df[df["name"].str.contains("run_training_batch", regex=True, na=False)])

    console.print()
    console.rule(f"TIME BREAKDOWN — {nbatches} recorded batches")

    # ── Forward / Backward / Dataloader ───────────────────────────────────────
    allowed_names = [["DDPGroupStrategy.training_step"], ["DDPGroupStrategy.backward"], ["train_dataloader_next"]]
    ttotal, tselected_l, _tidl_l, _tselected, tidle, _df_list = get_runtime_breakdown(df, names_list=allowed_names)
    ttotal_recorded = ttotal  # save before overwritten by encoder/decoder call

    throughput = nbatches / (ttotal / 1e6) if ttotal > 0 else 0
    console.print(
        f"\n[bold]Total:[/bold] [yellow]{ttotal / 1e6:.2f} s[/yellow]  "
        f"[bold]Throughput:[/bold] [yellow]{throughput:.2f} it/s[/yellow]\n"
    )

    fwd_bwd_data = None
    if ttotal > 0:
        fwd_bwd_data = (["Forward", "Backward", "Data Loader", "Elsewhere"],
                        [float(t) for t in tselected_l] + [float(tidle)])
        table = Table(title="[bold]Forward · Backward · Data Loader[/bold]", title_justify="left",
                      box=box.SIMPLE_HEAD, header_style="bold cyan", show_footer=True)
        table.add_column("Section",      style="bold",    footer="[dim]Elsewhere[/dim]")
        table.add_column("Duration (s)", justify="right", footer=f"[dim]{tidle / 1e6:.3f}[/dim]")
        table.add_column("% of Total",   justify="right", footer=f"[dim]{100 * tidle / ttotal:.2f}%[/dim]")
        for label, t in zip(["Forward", "Backward", "Data Loader"], tselected_l):
            table.add_row(label, f"{float(t) / 1e6:.3f}", f"{100 * float(t) / ttotal:.2f}%")
        console.print(table)
    else:
        console.print("[yellow]⚠  Total recorded time is zero — skipping forward/backward/dataloader breakdown.[/yellow]")

    # ── Encoder / Processor / Decoder ─────────────────────────────────────────
    allowed_names = [["model.encoder"], ["model.processor"], ["model.decoder"]]
    ttotal, tselected_l, _tidl_l, _tselected, tidle, _df_list = get_runtime_breakdown(df, names_list=allowed_names)

    enc_dec_data = None
    if ttotal > 0:
        enc_dec_data = (["Encoder", "Processor", "Decoder", "Elsewhere"],
                        [float(t) for t in tselected_l] + [float(tidle)])
        table2 = Table(title="[bold]Encoder · Processor · Decoder[/bold]", title_justify="left",
                       box=box.SIMPLE_HEAD, header_style="bold cyan", show_footer=True)
        table2.add_column("Section",      style="bold",    footer="[dim]Elsewhere[/dim]")
        table2.add_column("Duration (s)", justify="right", footer=f"[dim]{tidle / 1e6:.3f}[/dim]")
        table2.add_column("% of Total",   justify="right", footer=f"[dim]{100 * tidle / ttotal:.2f}%[/dim]")
        for label, t in zip(["Encoder", "Processor", "Decoder"], tselected_l):
            table2.add_row(label, f"{float(t) / 1e6:.3f}", f"{100 * float(t) / ttotal:.2f}%")
        console.print(table2)
    else:
        console.print("[yellow]⚠  Total recorded time is zero — skipping encoder/processor/decoder breakdown.[/yellow]")

    console.print("[dim]ℹ  Note: not all decoder/encoder/processor sections are instrumented yet.[/dim]")
    console.print()

    if plot and (fwd_bwd_data is not None or enc_dec_data is not None):
        _plot_time_breakdowns(fwd_bwd_data, enc_dec_data, nbatches, ttotal_recorded)

    return df


def get_detailed_breakdown(df: pd.DataFrame, section: str = "model.decoder", gpu: bool = True) -> pd.DataFrame:
    """Compute a detailed runtime breakdown for a given model section.
    Parameters
    ----------
    df : pd.DataFrame
        Trace DataFrame with 'ts', 'dur', 'end_time', 'cat', 'name' columns.
    section : str
        Name pattern to filter annotations (e.g. "model.encoder").
    gpu : bool
        If True, filter for GPU annotations; otherwise CPU annotations.
    Returns
    -------
    pd.DataFrame
        Per-method runtime statistics for the section.
    """
    if "end_time" not in df.columns:
        df = df.copy()
        df["end_time"] = df["ts"] + df["dur"]

    df_annotations = df[df["name"].str.contains(section, regex=True, na=False)].sort_values(by="ts")
    if df_annotations.empty:
        console.print(f"[yellow]⚠  No annotations found for section '{section}'[/yellow]")
        return pd.DataFrame()

    # Total wall time for the section
    df_annotations_merged = merge_overlapping_intervals_df(df_annotations[["name", "ts", "end_time"]])
    df_annotations_merged["batch_dur"] = df_annotations_merged["end_time"] - df_annotations_merged["ts"]
    total_wall_time = df_annotations_merged["batch_dur"].sum()

    # Filter for GPU or CPU annotations
    label = "GPU" if gpu else "CPU"
    cat_filter = "gpu_user_annotation" if gpu else "user_annotation"
    df_annotations = df_annotations[df_annotations["cat"] == cat_filter].sort_values(by="ts")
    if df_annotations.empty:
        console.print(f"[yellow]⚠  No {label} annotations found for section '{section}'[/yellow]")
        return pd.DataFrame()

    # Calculate total time excluding overlap between methods
    df_annotations_merged = merge_overlapping_intervals_df(df_annotations[["name", "ts", "end_time"]])
    df_annotations_merged["batch_dur"] = df_annotations_merged["end_time"] - df_annotations_merged["ts"]
    ttime_exc_overlap = df_annotations_merged["batch_dur"].sum()
    ttime_exc_overlap_p = 100 * ttime_exc_overlap / total_wall_time if total_wall_time > 0 else 0.0

    df_annotations = runtime_analysis(df_annotations).sort_values(
        ["total time us", "name"], ascending=[False, True]
    )
    df_annotations["total time sec"] = df_annotations["total time us"] / 1e6

    console.print()
    console.rule(f"DETAILED {label} BREAKDOWN — {section}")
    console.print(
        f"\n[bold]Total wall time:[/bold] [yellow]{total_wall_time / 1e6:.3f} s[/yellow]   "
        f"[bold]Total {label} time (excl. overlap):[/bold] [yellow]{ttime_exc_overlap / 1e6:.3f} s[/yellow]  "
        f"[yellow]{ttime_exc_overlap_p:.2f}%[/yellow] of wall time\n"
    )

    # ── Forward pass ──────────────────────────────────────────────────────────
    df_fwd = df_annotations[df_annotations["name"].str.endswith(".forward")].head(10)
    if not df_fwd.empty:
        df_fwd_merged = merge_overlapping_intervals_df(
            df[df["name"].isin(df_fwd["name"])][["name", "ts", "end_time"]]
        )
        df_fwd_merged["batch_dur"] = df_fwd_merged["end_time"] - df_fwd_merged["ts"]
        ttime_fwd = df_fwd_merged["batch_dur"].sum()
        ttime_fwd_p = 100 * ttime_fwd / total_wall_time if total_wall_time > 0 else 0.0
        console.print(_build_breakdown_table(df_fwd, "Top 10 Forward Pass Contributors", section, ttime_fwd, ttime_fwd_p))
    else:
        console.print("[dim]No forward pass annotations found.[/dim]")

    # ── Backward pass ─────────────────────────────────────────────────────────
    df_bwd = df_annotations[df_annotations["name"].str.endswith(".backward")].head(10)
    if not df_bwd.empty:
        df_bwd_merged = merge_overlapping_intervals_df(
            df[df["name"].isin(df_bwd["name"])][["name", "ts", "end_time"]]
        )
        df_bwd_merged["batch_dur"] = df_bwd_merged["end_time"] - df_bwd_merged["ts"]
        ttime_bwd = df_bwd_merged["batch_dur"].sum()
        ttime_bwd_p = 100 * ttime_bwd / total_wall_time if total_wall_time > 0 else 0.0
        console.print(_build_breakdown_table(df_bwd, "Top 10 Backward Pass Contributors", section, ttime_bwd, ttime_bwd_p))
    else:
        console.print("[dim]No backward pass annotations found.[/dim]")

    console.print()
    return df_annotations


def _plot_gpu_time_breakdown(
    labels: list[str],
    times_us: list[float],
    ttotal: float,
    nbatches: int,
    savepath: str = "",
) -> None:
    """Plot a publication-quality grouped bar chart for the GPU time breakdown.

    Parameters
    ----------
    labels : list[str]
        Section names (Computation, Communication, Memory Ops, Data Loader, GPU Idle).
    times_us : list[float]
        Duration per section in microseconds.
    ttotal : float
        Total recorded time in microseconds.
    nbatches : int
        Number of batches.
    savepath : str
        Optional path to save the figure.
    """
    pub_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
    }

    # Colorblind-safe palette (Paul Tol muted + grey for idle)
    COLORS = ["#4878CF", "#D65F5F", "#6ACC65", "#E49444", "#B8B8B8"]
    TEXT_COLOR = "#1A1A1A"
    EDGE_COLOR = "white"

    values_s = [t / 1e6 for t in times_us]

    with mpl.rc_context(pub_rc):
        fig, ax = plt.subplots(figsize=(6, 3.8), facecolor="white")

        throughput = nbatches / (ttotal / 1e6) if ttotal > 0 else 0
        fig.suptitle(
            f"Batches: {nbatches}  ·  Total: {ttotal / 1e6:.3f} s  ·  Throughput: {throughput:.2f} it/s",
            fontsize=10,
            fontweight="bold",
            color=TEXT_COLOR,
            y=1.02,
        )

        x = np.arange(len(labels))
        colors = COLORS[: len(labels)]

        bars = ax.bar(
            x, values_s,
            color=colors,
            edgecolor=EDGE_COLOR,
            linewidth=0.8,
        )

        # Percentage labels on each bar
        max_val = max(values_s) if values_s else 1
        for rect, val in zip(bars, values_s):
            pct = 100 * val / (ttotal / 1e6) if ttotal > 0 else 0
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.015 * max_val,
                f"{pct:.1f}%",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=TEXT_COLOR,
            )
            # Duration below percentage
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.07 * max_val,
                f"{val:.3f} s",
                ha="center", va="bottom",
                fontsize=6.5, color=TEXT_COLOR,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, fontweight="bold")
        ax.set_ylabel("Duration (s)")
        ax.set_title(
            "GPU Time Breakdown (sections may overlap)",
            loc="left", fontsize=9, fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("white")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if savepath:
            ext = savepath.rsplit(".", 1)[-1].lower()
            dpi = 300 if ext in ("png", "tiff", "jpg", "jpeg") else None
            fig.savefig(savepath, bbox_inches="tight", dpi=dpi, facecolor="white")

        plt.show()

def gpu_time_breakdown(
    tracefile_or_df: Union[str, pd.DataFrame],
    plot: bool = False,
    savepath: str = "",
) -> pd.DataFrame:
    """Print a GPU-centric time breakdown (computation, communication, memory ops, data loader).

    These categories overlap across GPU streams, so percentages may sum to
    more than 100%.  An optional grouped bar chart is provided instead of
    a pie chart.

    Parameters
    ----------
    tracefile_or_df : str or pd.DataFrame
        Path to the trace JSON file, or an already-loaded trace DataFrame.
    plot : bool, optional
        If True, display a grouped bar chart of the breakdown.
    savepath : str, optional
        If non-empty and *plot* is True, save the figure to this path.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with columns
        ``['Section', 'Duration (s)', '% of Total', 'Intervals']``.
    """
    if isinstance(tracefile_or_df, pd.DataFrame):
        df = tracefile_or_df
    else:
        df = trace_to_dataframe(tracefile_or_df)
    if "end_time" not in df.columns:
        df["end_time"] = df["ts"] + df["dur"]

    start_time = df["ts"].min()
    end_time_max = df["end_time"].max()
    ttotal = end_time_max - start_time

    nbatches = len(df[df["name"].str.contains("run_training_batch", regex=True, na=False)])

    console.print()
    console.rule(f"GPU TIME BREAKDOWN — {nbatches} recorded batches")

    throughput = nbatches / (ttotal / 1e6) if ttotal > 0 else 0
    console.print(
        f"\n[bold]Total recorded time:[/bold] [yellow]{ttotal / 1e6:.3f} s[/yellow]  "
        f"[bold]Throughput:[/bold] [yellow]{throughput:.2f} it/s[/yellow]\n"
    )

    # ── Section definitions ───────────────────────────────────────────────────
    # (label, cat_patterns, name_patterns, name_excludes)
    sections = [
        ("Computation",   ["kernel"],                    [],                         ["nccl"]),
        ("Communication", [],                            ["nccl"],                   []),
        ("Memory Ops",    ["gpu_memcpy", "gpu_memset"],  [],                         []),
        ("Data Loader",   [],                            ["train_dataloader_next"],  []),
    ]

    section_labels: list[str] = []
    section_times: list[float] = []
    section_intervals: list[int] = []
    gpu_stream_dfs: list[pd.DataFrame] = []          # first 3 for GPU active

    for label, cats, names, no_names in sections:
        df_sec = df.copy()
        if cats:
            pat = "|".join(cats)
            df_sec = df_sec[df_sec["cat"].str.contains(pat, regex=True, na=False)]
        if names:
            pat = "|".join(names)
            df_sec = df_sec[df_sec["name"].str.contains(pat, regex=True, na=False)]
        if no_names:
            pat = "|".join(no_names)
            df_sec = df_sec[~df_sec["name"].str.contains(pat, regex=True, na=False)]

        df_sec = df_sec.sort_values(by="ts")
        df_merged = merge_overlapping_intervals_df(df_sec[["ts", "end_time"]])
        df_merged["batch_dur"] = df_merged["end_time"] - df_merged["ts"]
        t = float(df_merged["batch_dur"].sum())

        section_labels.append(label)
        section_times.append(t)
        section_intervals.append(len(df_merged))
        gpu_stream_dfs.append(df_merged)

    # ── Combined GPU activity (computation + communication + memory) ──────────
    gpu_only = [gpu_stream_dfs[i] for i in range(3) if not gpu_stream_dfs[i].empty]
    if gpu_only:
        combined = pd.concat(gpu_only, ignore_index=True)
        combined = merge_overlapping_intervals_df(combined[["ts", "end_time"]])
        combined["batch_dur"] = combined["end_time"] - combined["ts"]
        gpu_active_time = float(combined["batch_dur"].sum())
    else:
        gpu_active_time = 0.0

    gpu_idle_time = float(ttotal) - gpu_active_time

    # ── Rich table ────────────────────────────────────────────────────────────
    table = Table(
        title="[bold]GPU Time Breakdown (sections may overlap)[/bold]",
        title_justify="left",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        show_footer=True,
    )
    table.add_column("Section", style="bold", footer="[dim]GPU Active (merged)[/dim]")
    table.add_column(
        "Duration (s)", justify="right",
        footer=f"[dim]{gpu_active_time / 1e6:.3f}[/dim]",
    )
    table.add_column(
        "% of Total", justify="right",
        footer=f"[dim]{100 * gpu_active_time / ttotal:.2f}%[/dim]" if ttotal > 0 else "[dim]—[/dim]",
    )
    table.add_column("Intervals", justify="right", footer="")

    for label, t, n_int in zip(section_labels, section_times, section_intervals):
        pct = f"{100 * t / ttotal:.2f}%" if ttotal > 0 else "—"
        table.add_row(label, f"{t / 1e6:.3f}", pct, str(n_int))

    console.print(table)

    if ttotal > 0:
        console.print(
            f"[dim]GPU Idle (not on any GPU stream): {gpu_idle_time / 1e6:.3f} s "
            f"({100 * gpu_idle_time / ttotal:.2f}%)[/dim]"
        )
    console.print("[dim]ℹ  Sections overlap across streams — percentages may sum to >100%.[/dim]")
    console.print()

    # ── Summary DataFrame ─────────────────────────────────────────────────────
    summary_rows = []
    for label, t, n_int in zip(section_labels, section_times, section_intervals):
        summary_rows.append({
            "Section": label,
            "Duration (s)": t / 1e6,
            "% of Total": 100 * t / ttotal if ttotal > 0 else 0.0,
            "Intervals": n_int,
        })
    summary_rows.append({
        "Section": "GPU Active (merged)",
        "Duration (s)": gpu_active_time / 1e6,
        "% of Total": 100 * gpu_active_time / ttotal if ttotal > 0 else 0.0,
        "Intervals": 0,
    })
    summary_rows.append({
        "Section": "GPU Idle",
        "Duration (s)": gpu_idle_time / 1e6,
        "% of Total": 100 * gpu_idle_time / ttotal if ttotal > 0 else 0.0,
        "Intervals": 0,
    })
    summary_df = pd.DataFrame(summary_rows)

    # ── Optional plot ─────────────────────────────────────────────────────────
    if plot and ttotal > 0:
        all_labels = section_labels + ["GPU Idle"]
        all_times = section_times + [gpu_idle_time]
        _plot_gpu_time_breakdown(all_labels, all_times, ttotal, nbatches, savepath=savepath)

    return summary_df

def find_first_trace_file(dirpath: Union[str, Path]) -> Optional[str]:
    """Find the first ``*.pt.trace.json`` file in a directory.

    Parameters
    ----------
    dirpath : str or Path
        Directory to search (non-recursively).

    Returns
    -------
    str or None
        Full path to the first matching file, or None if not found.
    """
    for filename in sorted(os.listdir(dirpath)):
        if filename.endswith(".pt.trace.json"):
            return os.path.join(dirpath, filename)
    return None


def analyze_anemoi_durations(json_file_path: str) -> dict:
    """Compute total and average durations for anemoi encoder/decoder/processor GPU annotations.

    Parameters
    ----------
    json_file_path : str
        Path to the trace JSON file.

    Returns
    -------
    dict
        Summary with total_duration, average_duration, and count per component.
    """
    allowed_names = {"anemoi-encoder", "anemoi-decoder", "anemoi-processor"}
    durations = defaultdict(list)

    with open(json_file_path, "r") as f:
        data = json.load(f)
        events = data.get("traceEvents", data)
        for event in events:
            if event.get("cat") == "gpu_user_annotation" and event.get("name") in allowed_names:
                durations[event["name"]].append(event.get("dur", 0))

    summary = {}
    for name, dur_list in durations.items():
        total = sum(dur_list)
        avg = total / len(dur_list) if dur_list else 0
        summary[name] = {"total_duration": total, "average_duration": avg, "count": len(dur_list)}

    return summary


def classify_kernel(name: str) -> str:
    """Classify a kernel name into 'memory', 'comms', or 'compute'."""
    name_lower = name.lower()
    if "memcpy" in name_lower or "memset" in name_lower:
        return "memory"
    if "nccl" in name_lower:
        return "comms"
    return "compute"


def sum_kernel_durations(kernel_durations: dict) -> list[tuple[str, float]]:
    """Sort kernel durations by total duration (descending).

    Parameters
    ----------
    kernel_durations : dict
        Mapping of kernel name to total duration.

    Returns
    -------
    list[tuple[str, float]]
        Sorted list of (kernel_name, total_duration).
    """
    return sorted(kernel_durations.items(), key=lambda x: x[1], reverse=True)


def print_kernel_table(
    data: list[tuple[str, float]],
    kernel_counts: dict,
    kernel_weighted_occupancies: dict,
    top_n: int = 10,
    num_iterations: int = 20,
) -> None:
    """Print a formatted table of kernel durations and occupancy.

    Parameters
    ----------
    data : list[tuple[str, float]]
        Sorted list of (kernel_name, total_duration_us).
    kernel_counts : dict
        Mapping of kernel name to call count.
    kernel_weighted_occupancies : dict
        Mapping of kernel name to duration-weighted occupancy.
    top_n : int
        Number of top kernels to show individually.
    num_iterations : int
        Number of training iterations to average over.
    """
    if num_iterations <= 0:
        LOGGER.warning("num_iterations is %d, skipping kernel table", num_iterations)
        return

    total_duration_us = sum(duration for _, duration in data) / num_iterations

    data_sorted = sorted(data, key=lambda x: x[1], reverse=True)
    top_kernels = data_sorted[:top_n]
    remaining = data_sorted[top_n:]

    rows = []
    total_count = 0
    for name, duration_us in top_kernels:
        count = kernel_counts[name] / num_iterations
        total_count += count
        occupancy = kernel_weighted_occupancies[name] / duration_us * 100 if duration_us > 0 else 0.0
        category = classify_kernel(name)
        percent = ((duration_us / num_iterations) / total_duration_us) * 100 if total_duration_us > 0 else 0.0
        rows.append((name, category, duration_us / 1e6 / num_iterations, percent, count, occupancy))

    if remaining:
        other_duration_us = sum(d for _, d in remaining) / num_iterations
        other_percent = (other_duration_us / total_duration_us) * 100 if total_duration_us > 0 else 0.0
        other_count = sum(kernel_counts[name] for name, _ in remaining) / num_iterations
        total_count += other_count
        rows.append(("other", "-", other_duration_us / 1e6, other_percent, other_count, 0.0))

    rows.append(
        ("total (kernels on different streams can overlap)", "-", total_duration_us / 1e6, 100.0, total_count, 0.0)
    )

    console.print(
        f"{'Kernel':60} {'Category':10} {'Duration (s)':>15} {'% Time':>10} {'Count':>10} {'% Occupancy':>10}"
    )
    console.print("-" * 120)
    for name, category, duration, percent, count, occupancy in rows:
        console.print(
            f"{name[:58]:60} {category:10} {duration:15.2f} {percent:10.2f} {count:10.0f} {occupancy:10.2f}"
        )


def count_iterations(json_file_path: str) -> int:
    """Count the number of training iterations in a trace file.

    Parameters
    ----------
    json_file_path : str
        Path to the trace JSON file.

    Returns
    -------
    int
        Number of iterations detected.
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)
        events = data.get("traceEvents", data)

    iteration_count = 0
    for event in events:
        if event.get("cat") == "user_annotation" and "transfer_batch_to_device" in event.get("name", ""):
            iteration_count += 1
    return iteration_count


def compute_av_time_per_iter_and_dl_stalls(
    iteration_durations_us: list[float],
    dataloading_stall_durations_us: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute average iteration time and data-loading stall statistics.

    Parameters
    ----------
    iteration_durations_us : list[float]
        Per-iteration durations in microseconds.
    dataloading_stall_durations_us : list[float]
        Per-iteration data-loading stall durations in microseconds.

    Returns
    -------
    tuple
        (iteration_durations_s, dataloading_stall_durations_s, dataloading_stall_percentages)
    """
    if not iteration_durations_us:
        LOGGER.warning("No iteration durations found; skipping iteration/stall analysis")
        return np.array([]), np.array([]), np.array([])

    dataloading_stall_durations_s = np.array(dataloading_stall_durations_us) / 1e6
    iteration_durations_s = np.array(iteration_durations_us) / 1e6

    # Align lengths (take the minimum) and guard against zero-duration iterations
    min_len = min(len(iteration_durations_s), len(dataloading_stall_durations_s))
    if min_len > 0:
        iter_slice = iteration_durations_s[:min_len]
        stall_slice = dataloading_stall_durations_s[:min_len]
        # Avoid division by zero for individual iterations
        with np.errstate(divide="ignore", invalid="ignore"):
            dataloading_stall_percentages = np.where(
                iter_slice > 0, stall_slice / iter_slice * 100, 0.0
            )
    else:
        dataloading_stall_percentages = np.array([])

    av_iteration_duration_s = np.median(iteration_durations_s)
    av_throughput = 1 / av_iteration_duration_s if av_iteration_duration_s > 0 else 0
    av_dataloading_stall_duration_s = (
        np.median(dataloading_stall_durations_s) if len(dataloading_stall_durations_s) > 0 else 0
    )
    av_dataloading_stall_percentage = (
        np.median(dataloading_stall_percentages) if len(dataloading_stall_percentages) > 0 else 0
    )

    console.print(
        f"Each training iteration took an average of {av_iteration_duration_s:.2f}s "
        f"({av_throughput:.2f} iterations per second)"
    )
    console.print(
        f"An average of {av_dataloading_stall_duration_s:.2f}s ({av_dataloading_stall_percentage:.2f}%) "
        f"of each iteration was spent idling while loading data"
    )
    if av_dataloading_stall_percentage > 5.0:
        console.print(
            "Warning! Dataloading stall times are high. Try increasing the number of dataloader workers. "
            "If CPU memory is limited, try decreasing prefetch factor to 1 to allow more workers."
        )
    return iteration_durations_s, dataloading_stall_durations_s, dataloading_stall_percentages


def analyse_HtoD_memcpy(
    batch_sizes_GB: list[float],
    batch_transfer_bw_GBs: list[float],
    batch_transfer_durations_us: list[float],
) -> tuple[float, float, float]:
    """Analyse Host-to-Device memory copy performance.

    Parameters
    ----------
    batch_sizes_GB : list[float]
        Batch sizes in GB.
    batch_transfer_bw_GBs : list[float]
        Transfer bandwidths in GB/s.
    batch_transfer_durations_us : list[float]
        Transfer durations in microseconds.

    Returns
    -------
    tuple
        (av_batch_size_GB, av_batch_transfer_bw_GBs, av_batch_transfer_durations_s)
    """
    if not batch_sizes_GB:
        LOGGER.warning("No HtoD memcpy events found")
        return 0.0, 0.0, 0.0

    av_batch_size_GB = np.mean(batch_sizes_GB)
    av_batch_transfer_bw_GBs = np.mean(batch_transfer_bw_GBs)
    av_batch_transfer_durations_s = np.mean(batch_transfer_durations_us) / 1e6
    console.print(
        f"av_batch_size_GB={av_batch_size_GB:.2f}GB, "
        f"av_batch_transfer_durations_s={av_batch_transfer_durations_s:.2f}s, "
        f"(av_batch_transfer_bw_GBs={av_batch_transfer_bw_GBs:.2f}GB/s)"
    )
    return av_batch_size_GB, av_batch_transfer_bw_GBs, av_batch_transfer_durations_s


def parse_json_trace_file(json_file_path: str) -> tuple:
    """Parse a JSON trace file and extract all key performance metrics in a single pass.

    Parameters
    ----------
    json_file_path : str
        Path to the trace JSON file.

    Returns
    -------
    tuple
        (batch_sizes_GB, batch_transfer_bw_GBs, batch_transfer_durations_us,
         dataloading_stall_durations_us, iteration_durations_us,
         kernel_durations, kernel_counts, kernel_weighted_occupancies, iteration_count,
         df)
        Where df is the full trace DataFrame (reusable by downstream functions).
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)
        events = data.get("traceEvents", data)

    # HtoD memcpy analysis
    batch_sizes_GB = []
    batch_transfer_bw_GBs = []
    batch_transfer_durations_us = []

    # Iteration time and dataloading stall analysis
    dataloading_stall_durations_us = []
    iteration_durations_us = []

    # Kernel analysis
    kernel_durations = defaultdict(float)
    kernel_counts = defaultdict(int)
    kernel_weighted_occupancies = defaultdict(float)

    # Collect kernel events on the main stream for sorted idle-time calculation
    main_stream = 7  # assumption
    main_stream_kernel_times = []
    iteration_count = 0

    for event in events:
        cat = event.get("cat", "")
        name = event.get("name", "")
        dur = event.get("dur")
        args = event.get("args", {})

        # Skip events without a valid duration
        if dur is None:
            continue

        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            # Normalise kernel name
            kernel_name = name
            if kernel_name.startswith("void "):
                kernel_name = kernel_name[len("void "):]
            kernel_name = kernel_name.split("<")[0]

            kernel_occupancy_pct = args.get("est. achieved occupancy %", 0) or 0
            kernel_occupancy_pct /= 100

            kernel_durations[kernel_name] += dur
            kernel_counts[kernel_name] += 1
            kernel_weighted_occupancies[kernel_name] += kernel_occupancy_pct * dur

        # Iteration time and dataloading stall time
        if cat == "user_annotation" and "train_dataloader_next" in name:
            dataloading_stall_durations_us.append(dur)
        if cat == "user_annotation" and "run_training_batch" in name:
            iteration_durations_us.append(dur)
        if cat == "user_annotation" and "transfer_batch_to_device" in name:
            iteration_count += 1

        # HtoD memcpy
        if cat == "gpu_memcpy" and "Memcpy HtoD (Pinned" in name:
            batch_transfer_durations_us.append(dur)
            batch_sizes_GB.append(args.get("bytes", 0) / 1e9)
            batch_transfer_bw_GBs.append(args.get("memory bandwidth (GB/s)", 0))

        # Collect kernel events on main stream for idle time (sort later)
        if cat == "kernel" and args.get("stream") == main_stream:
            ts = event.get("ts", 0)
            main_stream_kernel_times.append((ts, ts + dur))

    # Sort by start time before computing idle gaps
    main_stream_kernel_times.sort(key=lambda x: x[0])
    gpu_idle_time = 0
    prev_kernel_end_time = 0
    for kernel_start_time, kernel_end_time in main_stream_kernel_times:
        if prev_kernel_end_time != 0:
            diff = kernel_start_time - prev_kernel_end_time
            if diff > 0:
                gpu_idle_time += diff
        prev_kernel_end_time = kernel_end_time

    console.print(f"gpu_idle_time = {gpu_idle_time / 1e6}s")
    if iteration_count > 0:
        console.print(f"gpu_idle_time per iteration = {gpu_idle_time / 1e6 / iteration_count}s")
    else:
        console.print("gpu_idle_time per iteration = N/A (no iterations detected)")

    # Build a DataFrame from the raw events so downstream functions don't re-parse
    cols = ["cat", "name", "ts", "dur"]
    df_events = []
    for event in events:
        row = {k: event.get(k) for k in cols}
        if row.get("dur") is not None:
            df_events.append(row)
    df = pd.DataFrame(df_events)
    if not df.empty:
        df["end_time"] = df["ts"] + df["dur"]
        if "distributedInfo" in data:
            df["rank"] = data["distributedInfo"].get("rank")
        else:
            df["rank"] = None

    return (
        batch_sizes_GB,
        batch_transfer_bw_GBs,
        batch_transfer_durations_us,
        dataloading_stall_durations_us,
        iteration_durations_us,
        kernel_durations,
        kernel_counts,
        kernel_weighted_occupancies,
        iteration_count,
        df,
    )


def analyse_gpu_memory_usage(device: int = 0) -> None:
    """Analyse and report GPU memory usage for the given device.

    Parameters
    ----------
    device : int
        CUDA device index.
    """
    try:
        import torch
    except ImportError:
        LOGGER.warning("PyTorch not available; skipping GPU memory analysis")
        return

    if not torch.cuda.is_available():
        LOGGER.warning("CUDA not available; skipping GPU memory analysis")
        return

    props = torch.cuda.get_device_properties(device)
    max_available_memory_GB = props.total_memory / 1024 / 1024 / 1024
    max_reserved_memory_GB = torch.cuda.max_memory_reserved(device) / 1024 / 1024 / 1024
    max_allocated_memory_GB = torch.cuda.max_memory_allocated(device) / 1024 / 1024 / 1024
    console.print(
        f"max_available_memory_GB={max_available_memory_GB:.2f}, "
        f"max_reserved_memory_GB={max_reserved_memory_GB:.2f}, "
        f"max_allocated_memory_GB={max_allocated_memory_GB:.2f}"
    )

    max_reserved_but_unused_memory_GB = max_reserved_memory_GB - max_allocated_memory_GB
    if max_reserved_but_unused_memory_GB > 2:
        console.print(
            f"Warning! You have {max_reserved_but_unused_memory_GB:.2f}GB of memory reserved by PyTorch but not "
            f"actively allocated. This memory fragmentation can result in avoidable Out-Of-Memory errors. "
            f"Try 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' to reduce fragmentation."
        )

    max_allocated_memory_percentage = (
        max_allocated_memory_GB / max_available_memory_GB * 100 if max_available_memory_GB > 0 else 0
    )
    console.print(f"Peak (allocated) memory usage was {max_allocated_memory_percentage:.2f}% of total device memory")
    if max_allocated_memory_percentage < 50.0:
        console.print(
            "Warning! Your peak device memory usage is low. "
            "You could try increasing the batch size or reducing the number of GPUs."
        )


def analyse_trace(dirpath: Union[str, Path], device: int = 0) -> None:
    """Run the full trace analysis pipeline on the first trace file in a directory.

    Parameters
    ----------
    dirpath : str or Path
        Directory containing ``*.pt.trace.json`` files.
    device : int
        CUDA device index for memory analysis.
    """
    filename = find_first_trace_file(dirpath)
    if filename is None:
        LOGGER.warning("No trace file found in %s", dirpath)
        return

    console.print(f"Analysing {filename}")
   # (
        #batch_sizes_GB,
        #batch_transfer_bw_GBs,
        #batch_transfer_durations_us,
        #dataloading_stall_durations_us,
        #iteration_durations_us,
        #kernel_durations,
        #kernel_counts,
        #kernel_weighted_occupancies,
        #iteration_count,
        #df,
    #) = parse_json_trace_file(filename)

    #kernel_durations_sorted = sum_kernel_durations(kernel_durations)
    #print_kernel_table(
        #kernel_durations_sorted, kernel_counts, kernel_weighted_occupancies, top_n=10, num_iterations=iteration_count
    #)
    #console.print("\n")
    #compute_av_time_per_iter_and_dl_stalls(iteration_durations_us, dataloading_stall_durations_us)
    #console.print("\n")
    #analyse_HtoD_memcpy(batch_sizes_GB, batch_transfer_bw_GBs, batch_transfer_durations_us)
    #console.print("\n")
    #analyse_gpu_memory_usage(device=device)

    df = trace_to_dataframe(filename)
    console.print("\n")
    total_time_breakdown(df)
    console.print("\n")
    gpu_time_breakdown(df)
    console.print("\n")
    get_detailed_breakdown(df, section="model.encoder")
    get_detailed_breakdown(df, section="model.encoder", gpu=False)
    get_detailed_breakdown(df, section="model.processor")
    get_detailed_breakdown(df, section="model.processor", gpu=False)
    get_detailed_breakdown(df, section="model.decoder")
    get_detailed_breakdown(df, section="model.decoder", gpu=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        console.print("Usage: python trace_analyser.py <trace_file_or_directory>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        analyse_trace(target)
    else:
        # Single file: wrap its parent directory logic or analyse directly
        dirpath = os.path.dirname(target) or "."
        analyse_trace(dirpath)
