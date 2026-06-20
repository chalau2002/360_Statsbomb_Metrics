"""
Line-Breaking Passes Metric
===========================
Identifies passes that:
  1. Advance the ball ≥10% of pitch length (≥12m in X-axis)
  2. Cross a defensive line (pass between two defenders sharing a similar x-coordinate)

Filtering rules applied before detection:
  - Set-pieces excluded (Corner, Free Kick, Throw-in, Kick Off)
  - Non-complete passes excluded: Incomplete, Out, Pass Offside, Unknown, Injury Clearance
  - Crosses excluded (pass_cross == True)
  - Goalkeepers excluded from defender lines (keeper != True in freeze-frame filter; NaN treated as non-keeper)

Line-break geometry rules (all must be satisfied per line):
  - Line is between passer and pass destination (pass_x ≤ line_x ≤ end_x)
  - Pass trajectory y at line_x falls between two adjacent defenders (gap check)
  - Euclidean distance from passer to intersection point ≥ 5 m

Outcome scoring:
  - pass_shot_assist / pass_goal_assist = True → full xG credit (direct key pass boost)
  - Shot/goal within 10s window but no assist flag → xG × SHOT_ASSIST_FACTOR (indirect credit)

Then assigns a composite contextual score:
  score = zone_value + distance_advanced + defenders_bypassed + line_break_bonus + outcome_value
  (each component normalised to [0,1], weights sum to 1.0, final score = score_raw × 10)

Pitch: StatsBomb coordinate system: 120×80, attacking direction = increasing X.

Data format: events.parquet is an EXPLODED freeze frame table.
  - Each event appears once per tracked player in the frame.
  - location_x = [event_x, event_y]  (pass origin, same for all rows of the event)
  - location_y = [player_x, player_y] (the freeze-frame player's position)
  - teammate=True  → player belongs to the possessing team
  - actor=True     → this row is the passer / event performer
  - keeper=True    → this player is a goalkeeper

Usage:
    python scripts/metrics/line_breaking_passes.py
    → saves data/processed/line_breaking_passes.parquet"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
PITCH_LENGTH = 120.0
PITCH_WIDTH  = 80.0

MIN_ADVANCE_X      = 12.0   # ≥10% of 120m pitch
LINE_GAP_THRESHOLD = 4.5    # gap in X > 4.5m → new defensive line
MIN_LINE_SIZE      = 2      # a line needs ≥2 defenders
OUTCOME_WINDOW_SEC = 10.0   # seconds to look ahead for shot/goal
SHOT_ASSIST_FACTOR = 0.6    # multiplier for passes within the window but NOT a direct key pass

# Score component weights (must sum to 1.0)
WEIGHTS = {
    "zone":        0.10,
    "distance":    0.15,
    "defenders":   0.30,
    "line_break":  0.20,
    "outcome":     0.25,
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

import sys
BASE = Path(__file__).resolve().parents[2]  # project root
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from utils.utils import (
    _extract_xy,
    _zone_value,
    _timestamp_to_seconds,
    _cluster_defenders_into_lines,
    _pass_y_at_x,
    _count_lines_broken,
    _count_defenders_bypassed,
)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def compute_line_breaking_passes(events: pd.DataFrame) -> pd.DataFrame:
    """
    Main function. Takes the exploded events DataFrame and returns a
    per-pass DataFrame with scores.
    """

    print("Step 1: Filtering progressive passes ...")

    # ── Extract actor rows (one row per pass event) ──
    # Exclude goalkeeper passers (keeper=True) — e.g. goal kicks & long throws by GKs
    passes_actor = events[
        (events["type"] == "Pass") &
        (events["actor"] == True) &
        (~events["keeper"].eq(True))
    ].copy()

    # Exclude set pieces
    setpiece_types = {"Corner", "Free Kick", "Throw-in", "Kick Off"}
    if "pass_type" in passes_actor.columns:
        passes_actor = passes_actor[
            ~passes_actor["pass_type"].isin(setpiece_types)
        ]

    # Exclude all non-complete pass outcomes
    non_complete = {"Incomplete", "Out", "Pass Offside", "Unknown", "Injury Clearance"}
    if "pass_outcome" in passes_actor.columns:
        passes_actor = passes_actor[
            ~passes_actor["pass_outcome"].isin(non_complete)
        ]

    # Exclude crosses (StatsBomb sets pass_cross=True; NaN/None means not a cross)
    if "pass_cross" in passes_actor.columns:
        passes_actor = passes_actor[
            passes_actor["pass_cross"].isna()
        ]

    # Only include Ground Pass and Low Pass (exclude High Pass)
    if "pass_height" in passes_actor.columns:
        passes_actor = passes_actor[
            passes_actor["pass_height"].isin({"Ground Pass", "Low Pass"})
        ]

    # Extract coordinates
    passes_actor[["pass_x", "pass_y"]] = passes_actor["location_x"].apply(
        lambda v: pd.Series(_extract_xy(v))
    )
    passes_actor[["end_x", "end_y"]] = passes_actor["pass_end_location"].apply(
        lambda v: pd.Series(_extract_xy(v))
    )

    # Drop rows with missing locations
    passes_actor = passes_actor.dropna(subset=["pass_x", "end_x"])

    # Progressive: must advance ≥12m in X (forward)
    passes_actor["distance_advanced"] = passes_actor["end_x"] - passes_actor["pass_x"]
    progressive = passes_actor[
        passes_actor["distance_advanced"] >= MIN_ADVANCE_X
    ].copy()

    print(f"  Total passes:            {len(passes_actor):,}")
    print(f"  Progressive (≥{MIN_ADVANCE_X}m X):   {len(progressive):,}")

    # ── Step 2: Extract defender positions per event ──
    print("Step 2: Extracting defender positions from freeze frames ...")

    defenders_df = events[
        (events["type"] == "Pass") &
        (events["teammate"] == False) &
        (events["actor"] == False) &
        (~events["keeper"].eq(True))
    ][["id", "location_y"]].copy()

    defenders_df[["def_x", "def_y"]] = defenders_df["location_y"].apply(
        lambda v: pd.Series(_extract_xy(v))
    )
    defenders_df = defenders_df.dropna(subset=["def_x", "def_y"])

    # Group into list of (x,y) tuples per event
    def_positions = (
        defenders_df
        .groupby("id")
        .apply(lambda g: list(zip(g["def_x"], g["def_y"])))
        .reset_index()
        .rename(columns={0: "defenders_xy"})
    )

    progressive = progressive.merge(def_positions, on="id", how="left")
    progressive["defenders_xy"] = progressive["defenders_xy"].apply(
        lambda v: v if isinstance(v, list) else []
    )

    print(f"  Passes with freeze-frame defenders: "
          f"{progressive['defenders_xy'].apply(len).gt(0).sum():,}")

    # ── Step 3 & 4: Line-breaking detection + defenders bypassed ──
    print("Step 3 & 4: Detecting line breaks and counting bypassed defenders ...")

    results = progressive.apply(
        lambda row: pd.Series({
            "lines_broken": _count_lines_broken(
                row["pass_x"], row["pass_y"],
                row["end_x"], row["end_y"],
                row["defenders_xy"]
            ),
            "n_defenders_bypassed": _count_defenders_bypassed(
                row["pass_x"], row["end_x"],
                row["defenders_xy"]
            ),
        }),
        axis=1
    )
    progressive = pd.concat([progressive, results], axis=1)

    # Keep only passes that break at least one line
    line_breakers = progressive[progressive["lines_broken"] >= 1].copy()
    print(f"  Passes breaking ≥1 line: {len(line_breakers):,}")

    # ── Step 5: outcome value using StatsBomb shot/goal assist flags ──
    print("Step 5: Computing shot/goal outcomes with assist-flag boost ...")

    # Pre-compute timestamps in seconds for all events (one row per event via actor=True)
    all_ts = events[["id", "match_id", "possession", "type",
                      "timestamp", "shot_statsbomb_xg", "shot_outcome", "actor"]].copy()
    all_ts = all_ts[all_ts["actor"] == True]
    all_ts = all_ts.drop(columns=["actor"])
    all_ts["ts_sec"] = all_ts["timestamp"].apply(_timestamp_to_seconds)

    # Build shot lookup: (match_id, possession) → [(shot_ts, xg, outcome_name), …]
    shots_lookup = (
        all_ts[all_ts["type"] == "Shot"]
        .groupby(["match_id", "possession"])
        .apply(lambda g: list(zip(g["ts_sec"], g["shot_statsbomb_xg"],
                                   g["shot_outcome"])))
        .to_dict()
    )

    # Merge pass timestamps onto line-breakers
    line_breakers = line_breakers.merge(
        all_ts[["id", "ts_sec"]].rename(columns={"ts_sec": "pass_ts_sec"}),
        on="id", how="left"
    )

    def _get_outcome(row):
        key = (row["match_id"], row["possession"])
        shots = shots_lookup.get(key, [])
        pass_ts = row.get("pass_ts_sec", np.nan)
        if np.isnan(pass_ts):
            return 0.0

        # StatsBomb direct-assist flags on the pass row itself
        is_direct = bool(
            row.get("pass_shot_assist", False) or
            row.get("pass_goal_assist", False)
        )

        for shot_ts, xg, outcome_name in shots:
            if 0 <= (shot_ts - pass_ts) <= OUTCOME_WINDOW_SEC:
                if str(outcome_name) == "Goal":
                    return 1.0 if is_direct else 1.0 * SHOT_ASSIST_FACTOR
                xg_val = float(xg) if xg is not None and not (
                    isinstance(xg, float) and np.isnan(xg)) else 0.05
                # Direct key pass → full xG boost; indirect → SHOT_ASSIST_FACTOR credit
                return xg_val if is_direct else xg_val * SHOT_ASSIST_FACTOR
        return 0.0

    line_breakers["outcome_value"] = line_breakers.apply(_get_outcome, axis=1)
    print(f"  Passes leading to shot within {OUTCOME_WINDOW_SEC}s: "
          f"{(line_breakers['outcome_value'] > 0).sum():,}")

    # ── Step 6: Composite score ──
    print("Step 6: Computing composite score ...")

    line_breakers["zone_value"] = line_breakers["pass_x"].apply(_zone_value)

    # Normalise distance advanced (max theoretical = 120 − 0 = 120)
    max_dist = line_breakers["distance_advanced"].max()
    line_breakers["distance_norm"] = line_breakers["distance_advanced"] / max(max_dist, 1.0)

    # Normalise defenders bypassed
    max_def = line_breakers["n_defenders_bypassed"].max()
    line_breakers["defenders_norm"] = line_breakers["n_defenders_bypassed"] / max(max_def, 1.0)

    # Line break bonus (1 per line broken, normalise by 3 lines as reasonable max)
    line_breakers["line_break_norm"] = (line_breakers["lines_broken"] / 3.0).clip(upper=1.0)

    # Outcome value already 0–1 (xG or 1.0 for goal)
    line_breakers["outcome_norm"] = line_breakers["outcome_value"].clip(upper=1.0)

    # Raw composite score
    line_breakers["score"] = (
        WEIGHTS["zone"]       * line_breakers["zone_value"] +
        WEIGHTS["distance"]   * line_breakers["distance_norm"] +
        WEIGHTS["defenders"]  * line_breakers["defenders_norm"] +
        WEIGHTS["line_break"] * line_breakers["line_break_norm"] +
        WEIGHTS["outcome"]    * line_breakers["outcome_norm"]
    )


    # ── Select output columns ──
    output_cols = [
        "id", "match_id", "player", "player_id", "team", "team_id",
        "period", "minute", "second", "timestamp",
        "pass_x", "pass_y", "end_x", "end_y",
        "pass_length", "pass_angle",
        "distance_advanced",
        "n_defenders_bypassed", "lines_broken",
        "zone_value", "distance_norm", "defenders_norm",
        "line_break_norm", "outcome_value", "outcome_norm",
        "score",
        "pass_outcome", "pass_body_part", "pass_recipient"
    ]
    # Keep only those that exist
    output_cols = [c for c in output_cols if c in line_breakers.columns]
    result = line_breakers[output_cols].copy()

    return result


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[2]  # project root
    EVENTS_PATH = BASE / "data" / "processed" / "events.parquet"
    OUTPUT_PATH = BASE / "data" / "processed" / "line_breaking_passes.parquet"

    print("=" * 60)
    print("LINE-BREAKING PASSES METRIC")
    print("=" * 60)
    print(f"Loading events from {EVENTS_PATH} ...")
    events = pd.read_parquet(EVENTS_PATH)
    print(f"  Loaded {len(events):,} rows, {events['id'].nunique():,} unique events")
    print()

    result = compute_line_breaking_passes(events)

    # ── Sanity checks ──
    print()
    print("Sanity checks ...")
    assert (result["distance_advanced"] >= MIN_ADVANCE_X).all(), "distance_advanced below threshold!"
    assert (result["n_defenders_bypassed"] >= 0).all(), "negative defenders bypassed!"
    assert result["score"].between(0, 10).all(), "score out of [0,10] range!"
    assert result["score"].notna().all(), "NaN scores found!"
    print("  ✓ All checks passed")

    # ── Save ──
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result):,} line-breaking passes to: {OUTPUT_PATH}")

    # ── Quick summary ──
    print()
    print("Score distribution:")
    print(result["score"].describe().round(3))
    print()
    print("Top 5 passes by score:")
    top5_cols = ["player", "team", "minute", "pass_x", "end_x",
                 "lines_broken", "n_defenders_bypassed", "outcome_value", "score"]
    top5_cols = [c for c in top5_cols if c in result.columns]
    print(result.nlargest(5, "score")[top5_cols].to_string(index=False))
    print()
    print("Done ✓")
