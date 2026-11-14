"""Headless CLI for the Multi-Site Paint Sample Collector with `rich` progress bars (Sections 1.2, 5.2)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PYTHON_DIR = Path(__file__).resolve().parents[2]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from paint_sim.collector.jobs import CollectorJobManager
from paint_sim.collector.sources import load_collector_settings


def run_cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m paint_sim.collector.cli",
        description="Multi-Site Paint Sample Collector CLI",
    )
    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        required=True,
        help="Page URL or direct image URL (can be passed multiple times)",
    )
    parser.add_argument(
        "--vision",
        choices=("manual", "auto"),
        default="manual",
        help="Vision segmentation mode: 'manual' stops after extraction and prints UI link; 'auto' runs Gemini Vision.",
    )
    parser.add_argument("--model", default=None, help="Gemini model ID for --vision auto")
    parser.add_argument("--brand-id", default="", help="Default brand_id hint")
    parser.add_argument("--solvent-type", default="", help="Default solvent_type hint")
    parser.add_argument("--paint-role", default="", help="Default paint_role hint")
    parser.add_argument("--paint-code", default="", help="Default paint_code")
    parser.add_argument("--paint-name-en", default="", help="Default English paint name")
    parser.add_argument("--paint-name-ko", default="", help="Default Korean paint name")
    parser.add_argument("--port", type=int, default=8765, help="Local UI port for manual review link")
    parser.add_argument("--auto-ingest", action="store_true", help="Automatically ingest non-ambiguous auto samples")
    args = parser.parse_args(argv)

    console = Console()
    cfg = load_collector_settings()
    mgr = CollectorJobManager()

    defaults = {
        "brand_id": args.brand_id,
        "solvent_type": args.solvent_type,
        "paint_role": args.paint_role,
        "paint_code": args.paint_code,
        "paint_name_en": args.paint_name_en,
        "paint_name_ko": args.paint_name_ko,
    }
    sources = [{"url": u, "defaults": defaults} for u in args.urls]

    job = mgr.create_job(sources, vision_mode=args.vision, model=args.model)
    job_id = job["job_id"]
    console.print(f"[bold cyan]Created Collector Job:[/bold cyan] {job_id} (mode={args.vision})")

    def console_listener(_jid: str, ev: Dict[str, Any]) -> None:
        etype = ev.get("type", "")
        if etype == "image_found":
            console.log(f"[green]Found image:[/green] {ev.get('image_id')} ({ev.get('w')}x{ev.get('h')}) <- {ev.get('image_url')}")
        elif etype == "vision_failed":
            console.log(f"[bold red]Vision failed:[/bold red] {ev.get('image_id')}: {ev.get('error')}")
        elif etype == "sample_ready":
            console.log(f"[cyan]Sample ready:[/cyan] {ev.get('sample_tmp_id')}")
        elif etype == "error":
            console.log(f"[yellow]Fetch warning:[/yellow] {ev.get('url')} -> {ev.get('message')}")

    mgr.add_event_listener(console_listener)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        fetch_task = progress.add_task("Fetching sources & extracting candidate images...", total=1)
        job = mgr.fetch_job_sources(job_id)
        progress.update(fetch_task, advance=1)

        if args.vision == "manual":
            ui_link = f"http://127.0.0.1:{args.port}/?job_id={job_id}"
            console.print(
                f"\n[bold green]Extraction complete ({len(job['images'])} images).[/bold green]\n"
                f"Manual click-to-segment mode requested. Open the browser UI to click samples:\n"
                f"  [bold underline]{ui_link}[/bold underline]\n"
                f"(Start server if not running: [cyan]python -m paint_sim.collector.server --port {args.port}[/cyan])"
            )
            return 0

        img_list = job.get("images", [])
        vis_task = progress.add_task("Running Gemini Vision auto-segmentation...", total=max(1, len(img_list)))
        for im in img_list:
            mgr.segment_image_auto(im["image_id"], model_id=args.model or cfg.get("default_vision_model"))
            progress.update(vis_task, advance=1)

    job = mgr.get_job(job_id)
    table = Table(title=f"Job {job_id} Proposed Samples")
    table.add_column("Sample ID")
    table.add_column("Image ID")
    table.add_column("Hex")
    table.add_column("Gloss Index")
    table.add_column("Needs Review")
    table.add_column("Status")

    for smp in job.get("samples", []):
        opt = smp.get("calibrated_optical_params") or {}
        table.add_row(
            str(smp.get("sample_tmp_id")),
            str(smp.get("image_id")),
            str(smp.get("measured_srgb_hex")),
            str(opt.get("gloss_index", "")),
            "YES" if smp.get("needs_review") else "NO",
            str(smp.get("status")),
        )
    console.print(table)

    if args.auto_ingest:
        accepted = [s["sample_tmp_id"] for s in job.get("samples", []) if not s.get("needs_review")]
        for sid in accepted:
            mgr.update_sample(sid, {"status": "accepted"})
        if accepted:
            rows = mgr.ingest_job_samples(job_id)
            console.print(f"[bold green]Ingested {len(rows)} samples into master index.[/bold green]")

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
