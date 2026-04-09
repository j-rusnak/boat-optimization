from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .analysis import compute_stability_curve, center_of_mass_body, find_waterline_upright
from .hull import HullParams


REPO_ROOT = Path(__file__).resolve().parent.parent
DESIGNS_DIR = REPO_ROOT / "designs"
HISTORY_DIR = DESIGNS_DIR / "history"
ACTIVE_FILE = DESIGNS_DIR / "active.json"
BEST_FILE = DESIGNS_DIR / "best.json"


def ensure_design_dirs() -> None:
    """Create the design storage directories if needed."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def hull_params_to_dict(params: HullParams) -> dict[str, float]:
    """Convert HullParams to a JSON-serializable dictionary."""
    return {key: float(value) for key, value in asdict(params).items()}


def hull_params_from_dict(data: dict[str, Any]) -> HullParams:
    """Build HullParams from a persisted dictionary."""
    return HullParams(**{key: data[key] for key in HullParams.__dataclass_fields__.keys() if key in data})


def load_active_design(default: HullParams | None = None) -> HullParams:
    """Load the currently active design, or return the provided/default HullParams."""
    if ACTIVE_FILE.exists():
        payload = json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))
        return hull_params_from_dict(payload["params"])
    return default if default is not None else HullParams()


def load_best_record() -> dict[str, Any] | None:
    """Load the best-known design record if one exists."""
    if not BEST_FILE.exists():
        return None
    return json.loads(BEST_FILE.read_text(encoding="utf-8"))


def build_design_record(
    params: HullParams,
    source: str,
    objective_value: float | None = None,
    stability=None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Build a persisted record with parameters and derived performance metrics."""
    if stability is None:
        stability = compute_stability_curve(params)

    com = center_of_mass_body(params)
    waterline = find_waterline_upright(params)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")
    version_id = now.strftime("%Y%m%d_%H%M%S_%f")

    return {
        "version_id": version_id,
        "timestamp": timestamp,
        "source": source,
        "objective_value": None if objective_value is None else float(objective_value),
        "notes": notes,
        "params": hull_params_to_dict(params),
        "metrics": {
            "avs_deg": float(stability.avs_deg),
            "meets_avs_requirement": bool(stability.is_stable),
            "peak_righting_moment_nm": float(stability.righting_moments.max()),
            "center_of_mass_z_m": float(com[1]),
            "upright_waterline_z_m": float(waterline),
            "freeboard_m": float(params.depth - waterline),
        },
    }


def save_design_record(record: dict[str, Any]) -> dict[str, Path | bool]:
    """Persist a record to history, update active.json, and update best.json if improved."""
    ensure_design_dirs()

    history_file = HISTORY_DIR / f"design_{record['version_id']}.json"
    history_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    ACTIVE_FILE.write_text(json.dumps(record, indent=2), encoding="utf-8")

    best_updated = False
    best_record = load_best_record()
    new_cost = record.get("objective_value")

    if best_record is None:
        BEST_FILE.write_text(json.dumps(record, indent=2), encoding="utf-8")
        best_updated = True
    else:
        old_cost = best_record.get("objective_value")
        if new_cost is not None and (old_cost is None or new_cost < old_cost):
            BEST_FILE.write_text(json.dumps(record, indent=2), encoding="utf-8")
            best_updated = True

    return {
        "history_file": history_file,
        "active_file": ACTIVE_FILE,
        "best_file": BEST_FILE,
        "best_updated": best_updated,
    }
