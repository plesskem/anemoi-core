"""Trace analyser for PyTorch Lightning Profiler trace files.

Provides functions to parse, analyse, and report on training performance
from JSON trace files produced by PyTorch's profiler, including kernel
duration breakdowns, data loading stalls, memory usage, and per-component
runtime analysis.
"""

import argparse
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


def _parse_device_properties(device_list: list) -> dict:
    """Parse device properties from trace metadata.

    Parameters
    ----------
    device_list : list
        List of device property dictionaries from the trace file.

    Returns
    -------
    dict
        Dictionary mapping device ID to device properties, including:
        - name: Device name
        - totalGlobalMem: Total global memory in bytes
        - totalGlobalMemGB: Total global memory in GB
        - computeMajor, computeMinor: Compute capability version
        - maxThreadsPerBlock, maxThreadsPerMultiprocessor: Thread limits
        - regsPerBlock, regsPerMultiprocessor: Register limits
        - warpSize: Warp size
        - sharedMemPerBlock, sharedMemPerBlockOptIn, sharedMemPerMultiprocessor: Shared memory sizes
        - numSms: Number of streaming multiprocessors
    """
    devices = {}
    for device in device_list:
        device_id = device.get("id")
        if device_id is not None:
            total_mem = device.get("totalGlobalMem")
            devices[device_id] = {
                "name": device.get("name"),
                "totalGlobalMem": total_mem,
                "totalGlobalMemGB": total_mem / (1024**3) if total_mem else None,
                "computeMajor": device.get("computeMajor"),
                "computeMinor": device.get("computeMinor"),
                "maxThreadsPerBlock": device.get("maxThreadsPerBlock"),
                "maxThreadsPerMultiprocessor": device.get("maxThreadsPerMultiprocessor"),
                "regsPerBlock": device.get("regsPerBlock"),
                "warpSize": device.get("warpSize"),
                "sharedMemPerBlock": device.get("sharedMemPerBlock"),
                "sharedMemPerBlockOptIn": device.get("sharedMemPerBlockOptin"),
                "numSms": device.get("numSms"),
                "regsPerMultiprocessor": device.get("regsPerMultiprocessor"),
                "sharedMemPerMultiprocessor": device.get("sharedMemPerMultiprocessor"),
            }
    return devices if devices else None


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
        Additional metadata is attached as DataFrame attributes:
        - device_properties: dict mapping device ID to device info
        - cuda_runtime_version: CUDA runtime version (int)
        - cuda_driver_version: CUDA driver version (int)
        - cupti_version: CUPTI version (int)
        - framework: Framework name (str)
        - trace_id: Trace ID (str)
        - distributed_info: Distributed training configuration (dict)
    """
    with open(trace_file, "r") as f:
        trace_data = json.load(f)

    # Extract device properties
    device_properties = None
    if "deviceProperties" in trace_data:
        device_properties = _parse_device_properties(trace_data["deviceProperties"])

    # Extract CUDA and framework metadata
    cuda_runtime_version = trace_data.get("cuda_runtime_version")
    cuda_driver_version = trace_data.get("cuda_driver_version")
    cupti_version = trace_data.get("cupti_version")
    framework = trace_data.get("Framework")
    trace_id = trace_data.get("trace_id")
    distributed_info = trace_data.get("distributedInfo")

    rank = None
    if distributed_info:
        rank = distributed_info.get("rank")

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

    # Attach metadata via pandas attrs (safe/custom metadata channel).
    df.attrs["device_properties"] = device_properties
    df.attrs["cuda_runtime_version"] = cuda_runtime_version
    df.attrs["cuda_driver_version"] = cuda_driver_version
    df.attrs["cupti_version"] = cupti_version
    df.attrs["framework"] = framework
    df.attrs["trace_id"] = trace_id
    df.attrs["distributed_info"] = distributed_info

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
    ttime_active: float,
    ttime_active_p: float,
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
    table.add_column("Method",        style="bold",    footer=f"[dim]Top-10 active total ({ttime_active_p:.2f}%)[/dim]")
    table.add_column("Total (s)",     justify="right", footer=f"[dim]{ttime_active / 1e6:.3f}")
    table.add_column("# Calls",       justify="right", footer="")
    table.add_column("Avg (µs)",      justify="right", footer="")
    table.add_column("Max (µs)",      justify="right", footer="")
    table.add_column("Min (µs)",      justify="right", footer="")
    table.add_column("Overlap %", justify="right", footer="")

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

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Shared constants ───────────────────────────────────────────────────────────
_PUB_RC = {
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
_PALETTES = [
    ["#4878CF", "#D65F5F", "#6ACC65", "#B8B8B8"],
    ["#7B4173", "#A9B800", "#E49444", "#B8B8B8"],
    ["#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#B8B8B8"],
]
_TEXT_COLOR = "#1A1A1A"


# ── Shared helpers ─────────────────────────────────────────────────────────────
def _load_df(tracefile_or_df: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Load trace into a DataFrame and ensure end_time column exists."""
    df = tracefile_or_df if isinstance(tracefile_or_df, pd.DataFrame) else trace_to_dataframe(tracefile_or_df)
    if "end_time" not in df.columns:
        df = df.copy()
        df["end_time"] = df["ts"] + df["dur"]
    return df


def _merged_duration(df: pd.DataFrame) -> float:
    """Total duration of merged (non-overlapping) intervals in a DataFrame."""
    if df.empty:
        return 0.0
    merged = merge_overlapping_intervals_df(df[["ts", "end_time"]])
    return float((merged["end_time"] - merged["ts"]).sum())


def _batch_stats(df: pd.DataFrame) -> tuple[int, float, float, float]:
    """Return batch count, average batch duration (us), throughput (it/s), total batch duration (us)."""
    batch_df = df[df["name"].str.contains("run_training_batch", na=False)]
    if "dur" in batch_df.columns:
        batch_df = batch_df.dropna(subset=["dur"])

    nbatches = int(len(batch_df))
    if nbatches == 0:
        return 0, 0.0, 0.0, 0.0

    mean_batch_us = float(batch_df["dur"].mean())
    throughput = (1e6 / mean_batch_us) if mean_batch_us > 0 else 0.0
    total_batch_us = float(batch_df["dur"].sum())
    return nbatches, mean_batch_us, throughput, total_batch_us


def _savefig(fig, savepath: str) -> None:
    """Save figure with format-appropriate DPI."""
    if savepath:
        ext = savepath.rsplit(".", 1)[-1].lower()
        fig.savefig(savepath, bbox_inches="tight",
                    dpi=300 if ext in ("png", "tiff", "jpg", "jpeg") else None,
                    facecolor="white")


def _suptitle(fig, nbatches: int, ttotal_us: float, mean_batch_us: float = 0.0) -> None:
    """Attach a standard throughput suptitle to a figure."""
    throughput = (1e6 / mean_batch_us) if mean_batch_us > 0 else (nbatches / (ttotal_us / 1e6) if ttotal_us > 0 else 0)
    fig.suptitle(
        f"Batches: {nbatches}  ·  Total: {ttotal_us / 1e6:.3f} s  ·  "
        f"Throughput: {throughput:.2f} it/s",
        fontsize=10, fontweight="bold", color=_TEXT_COLOR, y=1.02,
    )


def _build_summary_table(
    title: str,
    rows: list[dict],           # each: {Section, Duration_us, pct
    ttotal_us: float,
    nbatches: int,
    footer_label: str = "",
    footer_dur_us: float = 0.0,
    show_intervals: bool = False,
) -> Table:
    """
    Build a unified Rich summary table used by both breakdown functions.

    Columns: Section | Duration (s) | Per batch (ms) | % of Total
    """
    footer_pct = f"{100 * footer_dur_us / ttotal_us:.2f}%" if ttotal_us > 0 else "—"
    footer_ms  = f"{footer_dur_us / nbatches / 1e3:.2f}" if nbatches > 0 else "—"

    t = Table(title=f"[bold]{title}[/bold]", title_justify="left",
              box=box.SIMPLE_HEAD, header_style="bold cyan", show_footer=True)
    t.add_column("Section",        style="bold",    footer=f"[dim]{footer_label}[/dim]")
    t.add_column("Duration (s)",   justify="right", footer=f"[dim]{footer_dur_us / 1e6:.3f}[/dim]")
    t.add_column("Per batch (ms)", justify="right", footer=f"[dim]{footer_ms}[/dim]")
    t.add_column("% of Total",     justify="right", footer=f"[dim]{footer_pct}[/dim]")

    for r in rows:
        pct = f"{100 * r['dur_us'] / ttotal_us:.2f}%" if ttotal_us > 0 else "—"
        ms  = f"{r['dur_us'] / nbatches / 1e3:.2f}" if nbatches > 0 else "—"
        row_vals = [r["label"], f"{r['dur_us'] / 1e6:.3f}", ms, pct]
        if show_intervals:
            row_vals.append(str(r.get("intervals", "")))
        t.add_row(*row_vals)
    return t


def _interval_overlap_duration(intervals_a: pd.DataFrame, intervals_b: pd.DataFrame) -> float:
    """Compute total overlap duration between two merged interval DataFrames."""
    if intervals_a.empty or intervals_b.empty:
        return 0.0

    a = intervals_a[["ts", "end_time"]].sort_values("ts").to_numpy()
    b = intervals_b[["ts", "end_time"]].sort_values("ts").to_numpy()

    i = 0
    j = 0
    overlap = 0.0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            overlap += float(end - start)

        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1

    return overlap


def _format_cuda_version(version: Optional[int]) -> str:
    """Format CUDA version encoded as an integer (e.g. 12080 -> 12.8)."""
    if not version:
        return "-"
    major = version // 1000
    minor = (version % 1000) // 10
    return f"{major}.{minor}"


def print_trace_metadata(df: pd.DataFrame) -> None:
    """Print a concise summary of trace metadata and detected devices."""
    attrs = getattr(df, "attrs", {})

    distributed_info = attrs.get("distributed_info")
    if distributed_info is None:
        distributed_info = getattr(df, "distributed_info", None)

    framework = attrs.get("framework", getattr(df, "framework", None))
    trace_id = attrs.get("trace_id", getattr(df, "trace_id", None))
    cupti_version = attrs.get("cupti_version", getattr(df, "cupti_version", None))
    cuda_runtime_version = attrs.get("cuda_runtime_version", getattr(df, "cuda_runtime_version", None))
    cuda_driver_version = attrs.get("cuda_driver_version", getattr(df, "cuda_driver_version", None))
    device_properties = attrs.get("device_properties", getattr(df, "device_properties", None))

    table = Table(
        title="Trace metadata",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Framework", framework or "-")
    table.add_row("Trace ID", trace_id or "-")
    table.add_row("CUPTI", str(cupti_version) if cupti_version is not None else "-")
    table.add_row("CUDA runtime", _format_cuda_version(cuda_runtime_version))
    table.add_row("CUDA driver", _format_cuda_version(cuda_driver_version))

    if isinstance(distributed_info, dict):
        rank = distributed_info.get("rank", "-")
        backend = distributed_info.get("backend", "-")
        table.add_row("Distributed", f"rank {rank}, backend={backend}")
    else:
        table.add_row("Distributed", "-")

    if isinstance(device_properties, dict) and device_properties:
        nd = len(device_properties)
        names = sorted({d.get("name", "Unknown") for d in device_properties.values()})
        table.add_row("Devices", f"{nd} ({', '.join(names)})")
    else:
        table.add_row("Devices", "-")

    console.print()
    console.print(table)

    if isinstance(device_properties, dict) and device_properties:
        dev_table = Table(
            title="Device details",
            box=box.SIMPLE,
            header_style="bold magenta",
            show_lines=False,
        )
        dev_table.add_column("ID", justify="right")
        dev_table.add_column("Name")
        dev_table.add_column("Mem (GB)", justify="right")
        dev_table.add_column("CC", justify="center")
        dev_table.add_column("SMs", justify="right")
        dev_table.add_column("Warp", justify="right")

        for dev_id in sorted(device_properties):
            info = device_properties[dev_id]
            mem_gb = info.get("totalGlobalMemGB")
            cc_major = info.get("computeMajor")
            cc_minor = info.get("computeMinor")
            cc = f"{cc_major}.{cc_minor}" if cc_major is not None and cc_minor is not None else "-"
            dev_table.add_row(
                str(dev_id),
                str(info.get("name", "Unknown")),
                f"{mem_gb:.1f}" if isinstance(mem_gb, (int, float)) else "-",
                cc,
                str(info.get("numSms", "-")),
                str(info.get("warpSize", "-")),
            )
        console.print(dev_table)


def _build_section_comm_table(
    title: str,
    rows: list[dict],
    ttotal_us: float,
    nbatches: int,
    total_batch_us: float,
    footer_label: str,
    footer_dur_us: float,
) -> Table:
    """Build section breakdown table with NCCL communication overlap columns."""
    footer_pct_total = f"{100 * footer_dur_us / ttotal_us:.2f}%" if ttotal_us > 0 else "—"
    footer_ms = f"{footer_dur_us / nbatches / 1e3:.2f}" if nbatches > 0 else "—"

    t = Table(
        title=f"[bold]{title}[/bold]",
        title_justify="left",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        show_footer=True,
    )
    t.add_column("Section", style="bold", footer=f"[dim]{footer_label}[/dim]")
    t.add_column("Duration (s)", justify="right", footer=f"[dim]{footer_dur_us / 1e6:.3f}[/dim]")
    t.add_column("Per batch (ms)", justify="right", footer=f"[dim]{footer_ms}[/dim]")
    t.add_column("% of Total", justify="right", footer=f"[dim]{footer_pct_total}[/dim]")
    t.add_column("% of Batch", justify="right", footer="")
    t.add_column("Eq. it/s", justify="right", footer="")
    t.add_column("Comm (s)", justify="right", footer="")
    t.add_column("Comm % Section", justify="right", footer="")
    t.add_column("Comm % Total", justify="right", footer="")

    for r in rows:
        dur_us = float(r["dur_us"])
        comm_us = float(r.get("comm_us", 0.0))
        pct_total = f"{100 * dur_us / ttotal_us:.2f}%" if ttotal_us > 0 else "—"
        per_batch_ms = f"{dur_us / nbatches / 1e3:.2f}" if nbatches > 0 else "—"
        pct_batch = f"{100 * dur_us / total_batch_us:.2f}%" if total_batch_us > 0 else "—"
        eq_it_s = f"{1e6 / (dur_us / nbatches):.2f}" if nbatches > 0 and dur_us > 0 else "—"
        comm_pct_section = f"{100 * comm_us / dur_us:.2f}%" if dur_us > 0 else "—"
        comm_pct_total = f"{100 * comm_us / ttotal_us:.2f}%" if ttotal_us > 0 else "—"
        t.add_row(
            r["label"],
            f"{dur_us / 1e6:.3f}",
            per_batch_ms,
            pct_total,
            pct_batch,
            eq_it_s,
            f"{comm_us / 1e6:.3f}",
            comm_pct_section,
            comm_pct_total,
        )
    return t


# ── Plot helpers ───────────────────────────────────────────────────────────────
def _plot_time_breakdowns(
    fwd_bwd_data: Optional[tuple],
    enc_dec_data: Optional[tuple],
    nbatches: int,
    ttotal_recorded: float,
    mean_batch_us: float = 0.0,
    savepath: str = "",
) -> None:
    """Plot publication-quality pie charts for time breakdowns."""
    charts = [d for d in [fwd_bwd_data, enc_dec_data] if d is not None]
    if not charts:
        return

    panel_labels = ["(a)", "(b)"][-len(charts):]

    with mpl.rc_context(_PUB_RC):
        fig, axes = plt.subplots(1, len(charts),
                                 figsize=(3.3 * len(charts) + 0.4, 3.8),
                                 facecolor="white")
        if len(charts) == 1:
            axes = [axes]

        _suptitle(fig, nbatches, ttotal_recorded, mean_batch_us)

        for idx, (ax, chart, panel) in enumerate(zip(axes, charts, panel_labels)):
            if len(chart) == 3:
                labels, values, comm_pcts = chart
            else:
                labels, values = chart
                comm_pcts = [None] * len(labels)

            colors = _PALETTES[idx % len(_PALETTES)]
            triples = [(l, v, c, cp) for l, v, c, cp in zip(labels, values, colors, comm_pcts) if v > 0]
            if not triples:
                ax.set_visible(False)
                continue

            f_labels, f_values, f_colors, f_comm_pcts = zip(*triples)
            total_val = sum(f_values)

            wedges, _, autotexts = ax.pie(
                f_values, labels=None,
                autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
                colors=f_colors, startangle=90, pctdistance=1.18,
                explode=[0.025] * len(f_labels),
                wedgeprops={"linewidth": 0.8, "edgecolor": "white", "antialiased": True},
                textprops={"fontsize": 7.5, "color": _TEXT_COLOR},
            )
            for at in autotexts:
                at.set_fontsize(7.5)
                at.set_color(_TEXT_COLOR)
                at.set_fontweight("bold")

            ax.text(-1.35, 1.15, panel, transform=ax.transData,
                    fontsize=9, fontweight="bold", color=_TEXT_COLOR, va="top")

            legend_entries = []
            for l, v, cp in zip(f_labels, f_values, f_comm_pcts):
                comm_txt = f" (incl. {cp:.1f}% comm.)" if cp is not None else ""
                legend_entries.append(
                    f"{l} {v / 1e6:.3f} s{comm_txt}"
                )
            leg = ax.legend(wedges, legend_entries, loc="lower center",
                            bbox_to_anchor=(0.5, -0.34), ncol=1, frameon=False,
                            labelcolor=_TEXT_COLOR, fontsize=7.5,
                            handlelength=1.2, handleheight=1.0,
                            handletextpad=0.5, borderpad=0.3, labelspacing=0.35)
            for h in leg.legend_handles:
                h.set_linewidth(0)
            ax.set_facecolor("white")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        _savefig(fig, savepath)
        plt.show()


def _plot_gpu_time_breakdown(
    labels: list[str],
    times_us: list[float],
    ttotal: float,
    nbatches: int,
    mean_batch_us: float = 0.0,
    savepath: str = "",
) -> None:
    """Plot a publication-quality bar chart for the GPU time breakdown."""
    colors = _PALETTES[-1 % len(_PALETTES)]
    values_s = [t / 1e6 for t in times_us]
    max_val = max(values_s, default=1)

    with mpl.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(6, 3.8), facecolor="white")
        _suptitle(fig, nbatches, ttotal, mean_batch_us)

        bars = ax.bar(range(len(labels)), values_s,
                      color=colors[:len(labels)], edgecolor="white", linewidth=0.8)

        for rect, val in zip(bars, values_s):
            cx = rect.get_x() + rect.get_width() / 2
            pct = 100 * val / (ttotal / 1e6) if ttotal > 0 else 0
            ax.text(cx, rect.get_height() + 0.07 * max_val,
                    f"{pct:.1f}%", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=_TEXT_COLOR)
            ax.text(cx, rect.get_height() + 0.015 * max_val,
                    f"{val:.3f} s", ha="center", va="bottom",
                    fontsize=6.5, color=_TEXT_COLOR)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, fontweight="bold")
        ax.set_ylabel("Duration (s)")
        ax.set_title("GPU time breakdown (sections may overlap)",
                 loc="left", fontsize=9, fontweight="bold", pad=20)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("white")

        plt.tight_layout()
        _savefig(fig, savepath)
        plt.show()


# ── Public analysis functions ──────────────────────────────────────────────────
def total_time_breakdown(
    tracefile_or_df: Union[str, pd.DataFrame], plot: bool = False, savepath: str = "",
) -> pd.DataFrame:
    """Print a coarse-grained time breakdown and optionally plot pie charts."""
    df = _load_df(tracefile_or_df)
    nbatches, mean_batch_us, throughput, total_batch_us = _batch_stats(df)

    console.print()
    console.rule(f"TIME BREAKDOWN — {nbatches} recorded batches")

    fwd_bwd_data = enc_dec_data = None
    ttotal_recorded = float(df["end_time"].max() - df["ts"].min()) if not df.empty else 0.0
    comm_events = df[df["name"].str.contains("nccl", case=False, na=False)].sort_values("ts")
    batch_events = df[df["name"].str.contains("run_training_batch", na=False)].sort_values("ts")
    batch_intervals = (
        merge_overlapping_intervals_df(batch_events[["ts", "end_time"]])
        if not batch_events.empty
        else pd.DataFrame(columns=["ts", "end_time"])
    )
    comm_intervals = (
        merge_overlapping_intervals_df(comm_events[["ts", "end_time"]])
        if not comm_events.empty
        else pd.DataFrame(columns=["ts", "end_time"])
    )

    for chart_idx, (section_title, names_list, chart_labels, section_patterns) in enumerate([
        ("Forward · Backward · Data loader",
         [["DDPGroupStrategy.training_step"], ["DDPGroupStrategy.backward"], ["train_dataloader_next"]],
         ["Forward", "Backward", "Data Loader"],
         {
             "Forward": "DDPGroupStrategy.training_step",
             "Backward": "DDPGroupStrategy.backward",
             "Data Loader": "train_dataloader_next",
         }),
        ("Encoder · Processor · Decoder",
         [["model.encoder"], ["model.processor"], ["model.decoder"]],
         ["Encoder", "Processor", "Decoder"],
         {
             "Encoder": "model.encoder",
             "Processor": "model.processor",
             "Decoder": "model.decoder",
         }),
    ]):
        ttotal, tselected_l, _, _, tidle, _ = get_runtime_breakdown(df, names_list=names_list)
        if ttotal <= 0:
            console.print(f"[yellow]⚠  Total recorded time is zero — skipping {section_title}.[/yellow]")
            continue

        if chart_idx == 0:
            console.print(
                f"\n[bold]Total:[/bold] [yellow]{ttotal / 1e6:.2f} s[/yellow]  "
                f"[bold]Throughput:[/bold] [yellow]{throughput:.2f} it/s[/yellow]  "
                f"[dim](avg run_training_batch = {mean_batch_us / 1e6:.3f} s)[/dim]\n"
            )

        rows = [{"label": l, "dur_us": float(t)} for l, t in zip(chart_labels, tselected_l)]
        rows_with_comm = []
        for row in rows:
            pattern = section_patterns.get(row["label"], "")
            section_events = df[df["name"].str.contains(pattern, regex=True, na=False)].sort_values("ts")
            section_intervals = (
                merge_overlapping_intervals_df(section_events[["ts", "end_time"]])
                if not section_events.empty
                else pd.DataFrame(columns=["ts", "end_time"])
            )
            # Restrict section time to iteration windows for per-batch realism.
            if not batch_intervals.empty:
                row["dur_us"] = _interval_overlap_duration(section_intervals, batch_intervals)
            row["comm_us"] = _interval_overlap_duration(section_intervals, comm_intervals)
            rows_with_comm.append(row)

        table = _build_section_comm_table(
            title=section_title,
            rows=rows_with_comm,
            ttotal_us=ttotal,
            nbatches=nbatches,
            total_batch_us=total_batch_us,
            footer_label="Elsewhere",
            footer_dur_us=float(tidle),
        )
        console.print(table)

        comm_pcts = [
            (100 * float(r["comm_us"]) / float(r["dur_us"])) if float(r["dur_us"]) > 0 else 0.0
            for r in rows_with_comm
        ]
        data_tuple = (
            chart_labels + ["Elsewhere"],
            [float(t) for t in tselected_l] + [float(tidle)],
            comm_pcts + [None],
        )
        if chart_idx == 0:
            fwd_bwd_data = data_tuple
        else:
            enc_dec_data = data_tuple

    console.print("[dim]ℹ  Note: not all decoder/encoder/processor sections are instrumented yet.[/dim]\n")

    if plot and ttotal_recorded > 0 and (fwd_bwd_data or enc_dec_data):
        _plot_time_breakdowns(
            fwd_bwd_data,
            enc_dec_data,
            nbatches,
            ttotal_recorded,
            mean_batch_us=mean_batch_us,
            savepath=savepath,
        )

    return df


def gpu_time_breakdown(
    tracefile_or_df: Union[str, pd.DataFrame], plot: bool = False, savepath: str = "",
) -> pd.DataFrame:
    """Print a GPU-centric time breakdown and optionally plot a bar chart."""
    df = _load_df(tracefile_or_df)
    ttotal = float(df["end_time"].max() - df["ts"].min())
    nbatches, mean_batch_us, throughput, _ = _batch_stats(df)

    console.print()
    console.rule(f"GPU TIME BREAKDOWN — {nbatches} recorded batches")
    console.print(
        f"\n[bold]Total:[/bold] [yellow]{ttotal / 1e6:.3f} s[/yellow]  "
        f"[bold]Throughput:[/bold] [yellow]{throughput:.2f} it/s[/yellow]  "
        f"[dim](avg run_training_batch = {mean_batch_us / 1e6:.3f} s)[/dim]\n"
    )

    sections = [
        ("Computation",   {"cat": ["kernel"],                   "name": [],                        "excl": ["nccl"]}),
        ("Communication", {"cat": [],                           "name": ["nccl"],                  "excl": []}),
        ("Memory Ops",    {"cat": ["gpu_memcpy", "gpu_memset"], "name": [],                        "excl": []}),
        ("Data Loader",   {"cat": [],                           "name": ["train_dataloader_next"], "excl": []}),
    ]

    section_labels, section_times, gpu_dfs = [], [], []
    for label, filt in sections:
        d = df.copy()
        if filt["cat"]:
            d = d[d["cat"].str.contains("|".join(filt["cat"]), na=False)]
        if filt["name"]:
            d = d[d["name"].str.contains("|".join(filt["name"]), na=False)]
        if filt["excl"]:
            d = d[~d["name"].str.contains("|".join(filt["excl"]), na=False)]
        dur = _merged_duration(d.sort_values("ts"))
        merged_df = merge_overlapping_intervals_df(d[["ts", "end_time"]]) if not d.empty else pd.DataFrame()
        section_labels.append(label)
        section_times.append(dur)
        gpu_dfs.append(merged_df)

    gpu_active = _merged_duration(pd.concat([g for g in gpu_dfs[:3] if not g.empty], ignore_index=True))
    gpu_idle = ttotal - gpu_active

    rows = [{"label": l, "dur_us": t}
            for l, t in zip(section_labels, section_times)]
    console.print(_build_summary_table(
        title="GPU time breakdown (sections may overlap)",
        rows=rows,
        ttotal_us=ttotal,
        nbatches=nbatches,
        footer_label="GPU Active (merged)",
        footer_dur_us=gpu_active,
    ))

    if ttotal > 0:
        console.print(
            f"[dim]GPU Idle: {gpu_idle / 1e6:.3f} s ({100 * gpu_idle / ttotal:.2f}%)[/dim]"
        )
    console.print("[dim]ℹ  Sections overlap across streams — percentages may sum to >100%.[/dim]\n")

    summary_df = pd.DataFrame([
        {"Section": l, "Duration (s)": t / 1e6,
         "% of Total": 100 * t / ttotal if ttotal > 0 else 0}
        for l, t in zip(section_labels, section_times)
    ] + [
        {"Section": "GPU Active (merged)", "Duration (s)": gpu_active / 1e6,
         "% of Total": 100 * gpu_active / ttotal if ttotal > 0 else 0},
        {"Section": "GPU Idle", "Duration (s)": gpu_idle / 1e6,
         "% of Total": 100 * gpu_idle / ttotal if ttotal > 0 else 0},
    ])

    if plot and ttotal > 0:
        _plot_gpu_time_breakdown(section_labels + ["GPU Idle"],
                                 section_times + [gpu_idle], ttotal, nbatches, mean_batch_us=mean_batch_us, savepath=savepath)
    return summary_df

def get_detailed_breakdown(
    df: pd.DataFrame,
    section: str = "model.decoder",
    gpu: bool = True,
) -> pd.DataFrame:
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
    df = _load_df(df)
    label = "GPU" if gpu else "CPU"
    cat_filter = "gpu_user_annotation" if gpu else "user_annotation"

    df_section = df[df["name"].str.contains(section, regex=True, na=False)].sort_values("ts")
    if df_section.empty:
        console.print(f"[yellow]⚠  No annotations found for section '{section}'[/yellow]")
        return pd.DataFrame()

    total_wall_time = _merged_duration(df_section)

    df_annotations = df_section[df_section["cat"] == cat_filter]
    if df_annotations.empty:
        console.print(f"[yellow]⚠  No {label} annotations found for section '{section}'[/yellow]")
        return pd.DataFrame()

    # Filter to leaf-only annotations.
    # Annotation names have the form "anemoi-<Class>-<module.path>.forward/.backward".
    # The class name is irrelevant for hierarchy — only the dot-separated module path
    # determines parent/child relationships.  A name is a parent if its path is a
    # strict prefix (followed by ".") of any other path in the set.
    def _module_path(name: str) -> str:
        parts = name.split("-", 2)
        if len(parts) != 3:
            return name
        path = parts[2]
        for sfx in (".forward", ".backward"):
            if path.endswith(sfx):
                return path[: -len(sfx)]
        return path

    all_names = sorted(df_annotations["name"].unique())
    name_to_path = {n: _module_path(n) for n in all_names}
    path_set = set(name_to_path.values())
    leaf_names = [
        n for n in all_names
        if not any(p.startswith(name_to_path[n] + ".") for p in path_set if p != name_to_path[n])
    ]
    df_annotations = df_annotations[df_annotations["name"].isin(leaf_names)]

    df_annotations = runtime_analysis(df_annotations).sort_values(
        ["total time us", "name"], ascending=[False, True]
    )
    df_annotations["total time sec"] = df_annotations["total time us"] / 1e6

    console.print()
    console.rule(f"DETAILED {label} BREAKDOWN — {section}")
    console.print(
        f"\n[bold]Total active wall time:[/bold] [yellow]{total_wall_time / 1e6:.3f} s[/yellow]\n"
    )

    for suffix, title in [(".forward", "Forward"), (".backward", "Backward")]:
        df_pass = df_annotations[df_annotations["name"].str.endswith(suffix)].head(10)
        if df_pass.empty:
            console.print(f"[dim]No {title.lower()} pass annotations found.[/dim]")
            continue
        ttime_active = _merged_duration(df_section[df_section["name"].isin(df_pass["name"]) & (df_section["cat"] == cat_filter)])
        ttime_active_p = 100 * ttime_active / total_wall_time if total_wall_time > 0 else 0.0
        console.print(_build_breakdown_table(
            df_pass, f"Top 10 {title} Pass Contributors", section, ttime_active, ttime_active_p
        ))

    console.print()
    return df_annotations

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


def find_trace_files(dirpath: Union[str, Path]) -> list[str]:
    """Find all ``*.pt.trace.json`` files in a directory (non-recursively)."""
    return [
        os.path.join(dirpath, filename)
        for filename in sorted(os.listdir(dirpath))
        if filename.endswith(".pt.trace.json")
    ]


def _rank_from_df(df: pd.DataFrame) -> Optional[int]:
    """Extract rank from DataFrame metadata/column when available."""
    rank = df.attrs.get("distributed_info", {}).get("rank") if isinstance(df.attrs.get("distributed_info"), dict) else None
    if rank is None and "rank" in df.columns and not df.empty:
        rank_val = df["rank"].iloc[0]
        if pd.notna(rank_val):
            rank = int(rank_val)
    return rank


def _compute_rank_overview(df: pd.DataFrame) -> dict:
    """Compute lightweight per-rank summary metrics for multi-rank overview."""
    dfx = _load_df(df)
    if dfx.empty:
        return {
            "rank": _rank_from_df(dfx),
            "ttotal_s": 0.0,
            "nbatches": 0,
            "throughput": 0.0,
            "gpu_active_s": 0.0,
            "gpu_idle_s": 0.0,
            "gpu_idle_pct": 0.0,
            "comm_s": 0.0,
            "comm_pct": 0.0,
        }

    ttotal = float(dfx["end_time"].max() - dfx["ts"].min())
    nbatches, mean_batch_us, throughput, total_batch_us = _batch_stats(dfx)

    # Mirror section logic from gpu_time_breakdown for consistent definitions.
    sections = [
        ("Computation", {"cat": ["kernel"], "name": [], "excl": ["nccl"]}),
        ("Communication", {"cat": [], "name": ["nccl"], "excl": []}),
        ("Memory Ops", {"cat": ["gpu_memcpy", "gpu_memset"], "name": [], "excl": []}),
    ]

    section_times = {}
    merged_sections = {}
    for label, filt in sections:
        d = dfx.copy()
        if filt["cat"]:
            d = d[d["cat"].str.contains("|".join(filt["cat"]), na=False)]
        if filt["name"]:
            d = d[d["name"].str.contains("|".join(filt["name"]), na=False)]
        if filt["excl"]:
            d = d[~d["name"].str.contains("|".join(filt["excl"]), na=False)]
        section_times[label] = _merged_duration(d.sort_values("ts"))
        merged_sections[label] = merge_overlapping_intervals_df(d[["ts", "end_time"]]) if not d.empty else pd.DataFrame()

    merged_active_parts = [merged_sections[k] for k in ("Computation", "Communication", "Memory Ops") if not merged_sections[k].empty]
    gpu_active = _merged_duration(pd.concat(merged_active_parts, ignore_index=True)) if merged_active_parts else 0.0
    gpu_idle = max(ttotal - gpu_active, 0.0)

    comm = float(section_times["Communication"])
    return {
        "rank": _rank_from_df(dfx),
        "ttotal_s": ttotal / 1e6,
        "nbatches": nbatches,
        "throughput": throughput,
        "mean_batch_s": mean_batch_us / 1e6,
        "total_batch_s": total_batch_us / 1e6,
        "gpu_active_s": gpu_active / 1e6,
        "gpu_idle_s": gpu_idle / 1e6,
        "gpu_idle_pct": 100.0 * gpu_idle / ttotal if ttotal > 0 else 0.0,
        "comm_s": comm / 1e6,
        "comm_pct": 100.0 * comm / ttotal if ttotal > 0 else 0.0,
    }


def _print_multi_rank_overview(rows: list[dict]) -> None:
    """Print per-rank metrics and aggregate distribution statistics."""
    if not rows:
        console.print("[yellow]⚠ No ranks to summarize.[/yellow]")
        return

    rows_sorted = sorted(rows, key=lambda x: (x["rank"] is None, x["rank"]))

    table = Table(
        title="All-rank performance overview",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("Rank", justify="right")
    table.add_column("Total (s)", justify="right")
    table.add_column("Batches", justify="right")
    table.add_column("Throughput (it/s)", justify="right")
    table.add_column("GPU idle (%)", justify="right")
    table.add_column("Comm (%)", justify="right")
    for r in rows_sorted:
        rank_txt = "?" if r["rank"] is None else str(r["rank"])
        table.add_row(
            rank_txt,
            f"{r['ttotal_s']:.3f}",
            str(r["nbatches"]),
            f"{r['throughput']:.2f}",
            f"{r['gpu_idle_pct']:.2f}",
            f"{r['comm_pct']:.2f}",
        )
    console.print()
    console.print(table)

    metric_specs = [
        ("Throughput (it/s)", "throughput"),
        ("GPU idle (%)", "gpu_idle_pct"),
        ("Comm (%)", "comm_pct"),
        ("Total (s)", "ttotal_s"),
    ]

    stat_table = Table(
        title="Rank distribution stats",
        box=box.SIMPLE,
        header_style="bold magenta",
        show_lines=False,
    )
    stat_table.add_column("Metric")
    stat_table.add_column("min", justify="right")
    stat_table.add_column("median", justify="right")
    stat_table.add_column("p90", justify="right")
    stat_table.add_column("max", justify="right")
    stat_table.add_column("imbalance", justify="right")

    for label, key in metric_specs:
        vals = np.array([float(r[key]) for r in rows if r[key] is not None], dtype=float)
        if vals.size == 0:
            stat_table.add_row(label, "-", "-", "-", "-", "-")
            continue
        vmin = float(np.min(vals))
        vmean = float(np.mean(vals))
        vmed = float(np.median(vals))
        vp90 = float(np.percentile(vals, 90))
        vmax = float(np.max(vals))
        # max/mean >= 1.0 in normal cases; higher means worse imbalance.
        imbalance = vmax / vmean if vmean > 0 else float("inf")
        imbalance_txt = f"max/mean={imbalance:.3f}"
        stat_table.add_row(label, f"{vmin:.3f}", f"{vmed:.3f}", f"{vp90:.3f}", f"{vmax:.3f}", imbalance_txt)
    console.print(stat_table)


#def analyze_anemoi_durations(json_file_path: str) -> dict:
    #"""Compute total and average durations for anemoi encoder/decoder/processor GPU annotations.

    #Parameters
    #----------
    #json_file_path : str
        #Path to the trace JSON file.

    #Returns
    #-------
    #dict
        #Summary with total_duration, average_duration, and count per component.
    #"""
    #allowed_names = {"anemoi-encoder", "anemoi-decoder", "anemoi-processor"}
    #durations = defaultdict(list)

    #with open(json_file_path, "r") as f:
        #data = json.load(f)
        #events = data.get("traceEvents", data)
        #for event in events:
            #if event.get("cat") == "gpu_user_annotation" and event.get("name") in allowed_names:
                #durations[event["name"]].append(event.get("dur", 0))

    #summary = {}
    #for name, dur_list in durations.items():
        #total = sum(dur_list)
        #avg = total / len(dur_list) if dur_list else 0
        #summary[name] = {"total_duration": total, "average_duration": avg, "count": len(dur_list)}

    #return summary


#def classify_kernel(name: str) -> str:
    #"""Classify a kernel name into 'memory', 'comms', or 'compute'."""
    #name_lower = name.lower()
    #if "memcpy" in name_lower or "memset" in name_lower:
        #return "memory"
    #if "nccl" in name_lower:
        #return "comms"
    #return "compute"


#def sum_kernel_durations(kernel_durations: dict) -> list[tuple[str, float]]:
    #"""Sort kernel durations by total duration (descending).

    #Parameters
    #----------
    #kernel_durations : dict
        #Mapping of kernel name to total duration.

    #Returns
    #-------
    #list[tuple[str, float]]
        #Sorted list of (kernel_name, total_duration).
    #"""
    #return sorted(kernel_durations.items(), key=lambda x: x[1], reverse=True)


#def print_kernel_table(
    #data: list[tuple[str, float]],
    #kernel_counts: dict,
    #kernel_weighted_occupancies: dict,
    #top_n: int = 10,
    #num_iterations: int = 20,
#) -> None:
    #"""Print a formatted table of kernel durations and occupancy.

    #Parameters
    #----------
    #data : list[tuple[str, float]]
        #Sorted list of (kernel_name, total_duration_us).
    #kernel_counts : dict
        #Mapping of kernel name to call count.
    #kernel_weighted_occupancies : dict
        #Mapping of kernel name to duration-weighted occupancy.
    #top_n : int
        #Number of top kernels to show individually.
    #num_iterations : int
        #Number of training iterations to average over.
    #"""
    #if num_iterations <= 0:
        #LOGGER.warning("num_iterations is %d, skipping kernel table", num_iterations)
        #return

    #total_duration_us = sum(duration for _, duration in data) / num_iterations

    #data_sorted = sorted(data, key=lambda x: x[1], reverse=True)
    #top_kernels = data_sorted[:top_n]
    #remaining = data_sorted[top_n:]

    #rows = []
    #total_count = 0
    #for name, duration_us in top_kernels:
        #count = kernel_counts[name] / num_iterations
        #total_count += count
        #occupancy = kernel_weighted_occupancies[name] / duration_us * 100 if duration_us > 0 else 0.0
        #category = classify_kernel(name)
        #percent = ((duration_us / num_iterations) / total_duration_us) * 100 if total_duration_us > 0 else 0.0
        #rows.append((name, category, duration_us / 1e6 / num_iterations, percent, count, occupancy))

    #if remaining:
        #other_duration_us = sum(d for _, d in remaining) / num_iterations
        #other_percent = (other_duration_us / total_duration_us) * 100 if total_duration_us > 0 else 0.0
        #other_count = sum(kernel_counts[name] for name, _ in remaining) / num_iterations
        #total_count += other_count
        #rows.append(("other", "-", other_duration_us / 1e6, other_percent, other_count, 0.0))

    #rows.append(
        #("total (kernels on different streams can overlap)", "-", total_duration_us / 1e6, 100.0, total_count, 0.0)
    #)

    #console.print(
        #f"{'Kernel':60} {'Category':10} {'Duration (s)':>15} {'% Time':>10} {'Count':>10} {'% Occupancy':>10}"
    #)
    #console.print("-" * 120)
    #for name, category, duration, percent, count, occupancy in rows:
        #console.print(
            #f"{name[:58]:60} {category:10} {duration:15.2f} {percent:10.2f} {count:10.0f} {occupancy:10.2f}"
        #)


#def count_iterations(json_file_path: str) -> int:
    #"""Count the number of training iterations in a trace file.

    #Parameters
    #----------
    #json_file_path : str
        #Path to the trace JSON file.

    #Returns
    #-------
    #int
        #Number of iterations detected.
    #"""
    #with open(json_file_path, "r") as f:
        #data = json.load(f)
        #events = data.get("traceEvents", data)

    #iteration_count = 0
    #for event in events:
        #if event.get("cat") == "user_annotation" and "transfer_batch_to_device" in event.get("name", ""):
            #iteration_count += 1
    #return iteration_count


#def compute_av_time_per_iter_and_dl_stalls(
    #iteration_durations_us: list[float],
    #dataloading_stall_durations_us: list[float],
#) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #"""Compute average iteration time and data-loading stall statistics.

    #Parameters
    #----------
    #iteration_durations_us : list[float]
        #Per-iteration durations in microseconds.
    #dataloading_stall_durations_us : list[float]
        #Per-iteration data-loading stall durations in microseconds.

    #Returns
    #-------
    #tuple
        #(iteration_durations_s, dataloading_stall_durations_s, dataloading_stall_percentages)
    #"""
    #if not iteration_durations_us:
        #LOGGER.warning("No iteration durations found; skipping iteration/stall analysis")
        #return np.array([]), np.array([]), np.array([])

    #dataloading_stall_durations_s = np.array(dataloading_stall_durations_us) / 1e6
    #iteration_durations_s = np.array(iteration_durations_us) / 1e6

    ## Align lengths (take the minimum) and guard against zero-duration iterations
    #min_len = min(len(iteration_durations_s), len(dataloading_stall_durations_s))
    #if min_len > 0:
        #iter_slice = iteration_durations_s[:min_len]
        #stall_slice = dataloading_stall_durations_s[:min_len]
        ## Avoid division by zero for individual iterations
        #with np.errstate(divide="ignore", invalid="ignore"):
            #dataloading_stall_percentages = np.where(
                #iter_slice > 0, stall_slice / iter_slice * 100, 0.0
            #)
    #else:
        #dataloading_stall_percentages = np.array([])

    #av_iteration_duration_s = np.median(iteration_durations_s)
    #av_throughput = 1 / av_iteration_duration_s if av_iteration_duration_s > 0 else 0
    #av_dataloading_stall_duration_s = (
        #np.median(dataloading_stall_durations_s) if len(dataloading_stall_durations_s) > 0 else 0
    #)
    #av_dataloading_stall_percentage = (
        #np.median(dataloading_stall_percentages) if len(dataloading_stall_percentages) > 0 else 0
    #)

    #console.print(
        #f"Each training iteration took an average of {av_iteration_duration_s:.2f}s "
        #f"({av_throughput:.2f} iterations per second)"
    #)
    #console.print(
        #f"An average of {av_dataloading_stall_duration_s:.2f}s ({av_dataloading_stall_percentage:.2f}%) "
        #f"of each iteration was spent idling while loading data"
    #)
    #if av_dataloading_stall_percentage > 5.0:
        #console.print(
            #"Warning! Dataloading stall times are high. Try increasing the number of dataloader workers. "
            #"If CPU memory is limited, try decreasing prefetch factor to 1 to allow more workers."
        #)
    #return iteration_durations_s, dataloading_stall_durations_s, dataloading_stall_percentages


#def analyse_HtoD_memcpy(
    #batch_sizes_GB: list[float],
    #batch_transfer_bw_GBs: list[float],
    #batch_transfer_durations_us: list[float],
#) -> tuple[float, float, float]:
    #"""Analyse Host-to-Device memory copy performance.

    #Parameters
    #----------
    #batch_sizes_GB : list[float]
        #Batch sizes in GB.
    #batch_transfer_bw_GBs : list[float]
        #Transfer bandwidths in GB/s.
    #batch_transfer_durations_us : list[float]
        #Transfer durations in microseconds.

    #Returns
    #-------
    #tuple
        #(av_batch_size_GB, av_batch_transfer_bw_GBs, av_batch_transfer_durations_s)
    #"""
    #if not batch_sizes_GB:
        #LOGGER.warning("No HtoD memcpy events found")
        #return 0.0, 0.0, 0.0

    #av_batch_size_GB = np.mean(batch_sizes_GB)
    #av_batch_transfer_bw_GBs = np.mean(batch_transfer_bw_GBs)
    #av_batch_transfer_durations_s = np.mean(batch_transfer_durations_us) / 1e6
    #console.print(
        #f"av_batch_size_GB={av_batch_size_GB:.2f}GB, "
        #f"av_batch_transfer_durations_s={av_batch_transfer_durations_s:.2f}s, "
        #f"(av_batch_transfer_bw_GBs={av_batch_transfer_bw_GBs:.2f}GB/s)"
    #)
    #return av_batch_size_GB, av_batch_transfer_bw_GBs, av_batch_transfer_durations_s


#def parse_json_trace_file(json_file_path: str) -> tuple:
    #"""Parse a JSON trace file and extract all key performance metrics in a single pass.

    #Parameters
    #----------
    #json_file_path : str
        #Path to the trace JSON file.

    #Returns
    #-------
    #tuple
        #(batch_sizes_GB, batch_transfer_bw_GBs, batch_transfer_durations_us,
         #dataloading_stall_durations_us, iteration_durations_us,
         #kernel_durations, kernel_counts, kernel_weighted_occupancies, iteration_count,
         #df)
        #Where df is the full trace DataFrame (reusable by downstream functions).
    #"""
    #with open(json_file_path, "r") as f:
        #data = json.load(f)
        #events = data.get("traceEvents", data)

    ## HtoD memcpy analysis
    #batch_sizes_GB = []
    #batch_transfer_bw_GBs = []
    #batch_transfer_durations_us = []

    ## Iteration time and dataloading stall analysis
    #dataloading_stall_durations_us = []
    #iteration_durations_us = []

    ## Kernel analysis
    #kernel_durations = defaultdict(float)
    #kernel_counts = defaultdict(int)
    #kernel_weighted_occupancies = defaultdict(float)

    ## Collect kernel events on the main stream for sorted idle-time calculation
    #main_stream = 7  # assumption
    #main_stream_kernel_times = []
    #iteration_count = 0

    #for event in events:
        #cat = event.get("cat", "")
        #name = event.get("name", "")
        #dur = event.get("dur")
        #args = event.get("args", {})

        ## Skip events without a valid duration
        #if dur is None:
            #continue

        #if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            ## Normalise kernel name
            #kernel_name = name
            #if kernel_name.startswith("void "):
                #kernel_name = kernel_name[len("void "):]
            #kernel_name = kernel_name.split("<")[0]

            #kernel_occupancy_pct = args.get("est. achieved occupancy %", 0) or 0
            #kernel_occupancy_pct /= 100

            #kernel_durations[kernel_name] += dur
            #kernel_counts[kernel_name] += 1
            #kernel_weighted_occupancies[kernel_name] += kernel_occupancy_pct * dur

        ## Iteration time and dataloading stall time
        #if cat == "user_annotation" and "train_dataloader_next" in name:
            #dataloading_stall_durations_us.append(dur)
        #if cat == "user_annotation" and "run_training_batch" in name:
            #iteration_durations_us.append(dur)
        #if cat == "user_annotation" and "transfer_batch_to_device" in name:
            #iteration_count += 1

        ## HtoD memcpy
        #if cat == "gpu_memcpy" and "Memcpy HtoD (Pinned" in name:
            #batch_transfer_durations_us.append(dur)
            #batch_sizes_GB.append(args.get("bytes", 0) / 1e9)
            #batch_transfer_bw_GBs.append(args.get("memory bandwidth (GB/s)", 0))

        ## Collect kernel events on main stream for idle time (sort later)
        #if cat == "kernel" and args.get("stream") == main_stream:
            #ts = event.get("ts", 0)
            #main_stream_kernel_times.append((ts, ts + dur))

    ## Sort by start time before computing idle gaps
    #main_stream_kernel_times.sort(key=lambda x: x[0])
    #gpu_idle_time = 0
    #prev_kernel_end_time = 0
    #for kernel_start_time, kernel_end_time in main_stream_kernel_times:
        #if prev_kernel_end_time != 0:
            #diff = kernel_start_time - prev_kernel_end_time
            #if diff > 0:
                #gpu_idle_time += diff
        #prev_kernel_end_time = kernel_end_time

    #console.print(f"gpu_idle_time = {gpu_idle_time / 1e6}s")
    #if iteration_count > 0:
        #console.print(f"gpu_idle_time per iteration = {gpu_idle_time / 1e6 / iteration_count}s")
    #else:
        #console.print("gpu_idle_time per iteration = N/A (no iterations detected)")

    ## Build a DataFrame from the raw events so downstream functions don't re-parse
    #cols = ["cat", "name", "ts", "dur"]
    #df_events = []
    #for event in events:
        #row = {k: event.get(k) for k in cols}
        #if row.get("dur") is not None:
            #df_events.append(row)
    #df = pd.DataFrame(df_events)
    #if not df.empty:
        #df["end_time"] = df["ts"] + df["dur"]
        #if "distributedInfo" in data:
            #df["rank"] = data["distributedInfo"].get("rank")
        #else:
            #df["rank"] = None

    #return (
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
    #)


#def analyse_gpu_memory_usage(device: int = 0) -> None:
    #"""Analyse and report GPU memory usage for the given device.

    #Parameters
    #----------
    #device : int
        #CUDA device index.
    #"""
    #try:
        #import torch
    #except ImportError:
        #LOGGER.warning("PyTorch not available; skipping GPU memory analysis")
        #return

    #if not torch.cuda.is_available():
        #LOGGER.warning("CUDA not available; skipping GPU memory analysis")
        #return

    #props = torch.cuda.get_device_properties(device)
    #max_available_memory_GB = props.total_memory / 1024 / 1024 / 1024
    #max_reserved_memory_GB = torch.cuda.max_memory_reserved(device) / 1024 / 1024 / 1024
    #max_allocated_memory_GB = torch.cuda.max_memory_allocated(device) / 1024 / 1024 / 1024
    #console.print(
        #f"max_available_memory_GB={max_available_memory_GB:.2f}, "
        #f"max_reserved_memory_GB={max_reserved_memory_GB:.2f}, "
        #f"max_allocated_memory_GB={max_allocated_memory_GB:.2f}"
    #)

    #max_reserved_but_unused_memory_GB = max_reserved_memory_GB - max_allocated_memory_GB
    #if max_reserved_but_unused_memory_GB > 2:
        #console.print(
            #f"Warning! You have {max_reserved_but_unused_memory_GB:.2f}GB of memory reserved by PyTorch but not "
            #f"actively allocated. This memory fragmentation can result in avoidable Out-Of-Memory errors. "
            #f"Try 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' to reduce fragmentation."
        #)

    #max_allocated_memory_percentage = (
        #max_allocated_memory_GB / max_available_memory_GB * 100 if max_available_memory_GB > 0 else 0
    #)
    #console.print(f"Peak (allocated) memory usage was {max_allocated_memory_percentage:.2f}% of total device memory")
    #if max_allocated_memory_percentage < 50.0:
        #console.print(
            #"Warning! Your peak device memory usage is low. "
            #"You could try increasing the batch size or reducing the number of GPUs."
        #)


def analyse_trace(
    dirpath: Union[str, Path],
    max_ranks: Optional[int] = 1,
    detailed: bool = False,
    plot=False,
) -> None:
    """Run the trace analysis pipeline over a selected set of trace files.

    Parameters
    ----------
    dirpath : str or Path
        Directory containing ``*.pt.trace.json`` files.
    max_ranks : int or None
        Optional maximum rank index to include, interpreted as an inclusive
        cap over the sorted trace-file list. ``0`` means only the first trace.
    detailed : bool
        If True, print detailed breakdown tables for each selected trace after
        the overview and per-trace summary tables.
    """
    trace_files = find_trace_files(dirpath)
    console.print(f"Analysing Traces in {dirpath}")
    if not trace_files:
        LOGGER.warning("No trace file found in %s", dirpath)
        return

    selected_trace_files = trace_files
    if max_ranks is not None and max_ranks >= 0:
        selected_trace_files = selected_trace_files[: max_ranks]

    summaries = []
    selected_rows = []
    for index, trace_file in enumerate(selected_trace_files):
        dfi = trace_to_dataframe(trace_file)
        summary = _compute_rank_overview(dfi)
        rank = summary.get("rank")
        if rank is None and index == 0:
            rank = 0
            summary["rank"] = 0

        summary["trace_file"] = trace_file
        summaries.append(summary)
        selected_rows.append((rank, trace_file))

    console.print(f"Analysing {len(selected_trace_files)} trace files in the selected trace set")
    _print_multi_rank_overview(summaries)

    for row_index, (selected_rank, selected_file) in enumerate(selected_rows):
        if selected_rank is None:
            selected_rank = row_index

        console.print(f"Analysing trace file from rank {selected_rank} {selected_file}")
        df = trace_to_dataframe(selected_file)
        print_trace_metadata(df)
        console.print("\n")
        total_time_breakdown(df, plot=plot)
        console.print("\n")
        gpu_time_breakdown(df, plot=plot)
        console.print("\n")

    if not detailed:
        console.print("[dim]Skipping detailed analysis.[/dim]")
        return

    for row_index, (selected_rank, selected_file) in enumerate(selected_rows):
        if selected_rank is None:
            selected_rank = row_index
        console.print(f"Analysing detailed trace file from rank {selected_rank} {selected_file}")
        df = trace_to_dataframe(selected_file)

        get_detailed_breakdown(df, section="model.encoder")
        get_detailed_breakdown(df, section="model.encoder", gpu=False)
        get_detailed_breakdown(df, section="model.processor")
        get_detailed_breakdown(df, section="model.processor", gpu=False)
        get_detailed_breakdown(df, section="model.decoder")
        get_detailed_breakdown(df, section="model.decoder", gpu=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse PyTorch profiler trace files.")
    parser.add_argument("target", help="Trace file or directory containing *.pt.trace.json files")
    parser.add_argument(
        "--max-ranks",
        type=int,
        default=None,
        help="Optional maximum rank index to include, interpreted inclusively over the sorted trace-file list.",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed breakdown tables for each selected trace.",
    )
    args = parser.parse_args()

    target = args.target
    if os.path.isdir(target):
        analyse_trace(
            target,
            max_ranks=args.max_ranks,
            detailed=args.detailed,
        )
    else:
        # Single file: wrap its parent directory logic or analyse directly.
        dirpath = os.path.dirname(target) or "."
        analyse_trace(
            dirpath,
            max_ranks=args.max_ranks,
            detailed=args.detailed,
        )

