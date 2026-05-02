"""
tailorvision.api.cli
=====================
Click-based command-line interface.

Usage
-----
    python -m tailorvision.api.cli measure \\
        --front client_front.jpg \\
        --side  client_side.jpg \\
        --height 175 \\
        --gender male \\
        --garment traditional \\
        --output results/client_001.json

    python -m tailorvision.api.cli measure --help
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from tailorvision.config import PipelineConfig
from tailorvision.exceptions import TailorVisionError
from tailorvision.pipeline import TailorVisionPipeline


@click.group()
@click.version_option("0.1.0", prog_name="tailor-vision")
def cli() -> None:
    """TailorVision — anthropometric body measurement for garment tailoring."""


@cli.command("measure")
@click.option("--front",   required=True, type=click.Path(exists=True), help="Front-view image path.")
@click.option("--side",    required=True, type=click.Path(exists=True), help="Side-view image path.")
@click.option("--height",  default=None, type=float,  help="Client height in cm (improves scale accuracy).")
@click.option("--gender",  default="neutral", type=click.Choice(["male", "female", "neutral"]))
@click.option("--garment", default="traditional",
              type=click.Choice(["traditional", "suit", "shirt", "trousers"]),
              help="Garment type for ease-allowance table.")
@click.option("--output",  default=None, type=click.Path(), help="Output JSON path (default: output/result.json).")
@click.option("--device",  default="cpu", help="PyTorch device ('cpu' or 'cuda').")
@click.option("--no-debug", is_flag=True, default=False, help="Suppress debug artifact generation.")
@click.option("--model-dir", default=None, type=click.Path(), help="Path to SMPL-X model directory.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable DEBUG logging.")
def measure(
    front, side, height, gender, garment,
    output, device, no_debug, model_dir, verbose,
) -> None:
    """Extract anthropometric measurements from a front+side image pair."""

    log_level = "DEBUG" if verbose else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig(
        gender=gender,
        known_height_cm=height,
        garment_type=garment,
        device=device,
        save_debug_artifacts=not no_debug,
        log_level=log_level,
    )

    if model_dir:
        config.model_dir = Path(model_dir)

    if output:
        config.output_dir = Path(output).parent

    click.echo(f"TailorVision measurement pipeline")
    click.echo(f"  Front : {front}")
    click.echo(f"  Side  : {side}")
    click.echo(f"  Gender: {gender}  |  Height: {height or 'unknown'}  |  Garment: {garment}")
    click.echo("")

    try:
        pipeline = TailorVisionPipeline(config)
        result   = pipeline.run(front, side)
    except TailorVisionError as exc:
        click.secho(f"✗ Pipeline error: {exc}", fg="red", err=True)
        sys.exit(1)
    except Exception as exc:
        click.secho(f"✗ Unexpected error: {exc}", fg="red", err=True)
        raise

    # Print measurements table
    click.secho("\n── Body Measurements ──", bold=True)
    for name, val in result.measurements_cm.items():
        conf = result.measurement_confidence.get(name, "?")
        unc  = result.uncertainty_cm.get(name, 0.0)
        label = f"{name:<35} {val:>7.1f} cm  ±{unc:.1f}  [{conf}]"
        color = "green" if str(conf) == "HIGH" else ("yellow" if str(conf) == "MEDIUM" else "red")
        click.secho(label, fg=color)

    click.secho("\n── Tailoring Recommendations ──", bold=True)
    rec = result.tailoring_recommendations
    for field, val in rec.model_dump().items():
        if isinstance(val, (int, float)) and val:
            click.echo(f"  {field:<35} {val:.1f} cm")

    if result.warnings:
        click.secho("\n── Warnings ──", bold=True, fg="yellow")
        for w in result.warnings:
            click.secho(f"  ⚠  {w.value}", fg="yellow")

    click.secho(f"\n── Overall Quality: {result.quality_scores.overall:.2f} ──", bold=True)

    # Save JSON
    out_path = Path(output) if output else config.output_path("result.json")
    result.save_json(out_path)
    click.secho(f"\n✓ Full result saved → {out_path}", fg="green")


def main() -> None:
    """Entry point for ``tailor-vision`` console script."""
    cli()


if __name__ == "__main__":
    main()
