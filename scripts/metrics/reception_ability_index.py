"""
Reception Ability Index (RAI) Metric
====================================
Evaluates players' ability to receive the ball effectively within spatial and defensive context.
Produces a normalized score between 0 and 1, where higher values represent receptions
performed in spatially advantageous contexts and/or under higher contextual difficulty.

Filtering rules:
  - Event type must be "Ball Receipt*"
  - Player must not be a goalkeeper (keeper != True)
  - Location must be strictly INSIDE the opponent's defensive convex hull (excluding goalkeeper)

Score Components:
  1. Espaço Disponível (Voronoi Area):
     - Clipped Voronoi area for the receiver in the frame (including all players).
     - Normalized to [0,1] (min-max scaled: higher is more space, i.e. advantageous).
  2. Difficulty Context Index (weighted sum of 4 sub-components):
     - Defensive Density (opponents within 3.0m): higher is harder.
     - Nearest Defender Distance: smaller is harder.
     - Hull Area (defensive block size): smaller block is harder.
     - Zone Value (strategic zone: X-progression sigmoid): higher is better.

Final Score:
  RAI = 0.4 * Voronoi_Area_Norm + 0.6 * Difficulty_Context

Usage:
    python scripts/metrics/reception_ability_index.py
    -> saves data/processed/reception_ability_index.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTS & WEIGHTS
# ─────────────────────────────────────────────
PITCH_LENGTH = 120.0
PITCH_WIDTH  = 80.0
DENSITY_RADIUS = 3.0  # radius in meters to count defenders

# Difficulty Context component weights (must sum to 1.0)
WEIGHTS_DIFFICULTY = {
    "density":  0.30,
    "nearest":  0.30,
    "hull":     0.20,
    "zone":     0.20,
}

# Final RAI weights (must sum to 1.0)
WEIGHTS_RAI = {
    "voronoi":    0.40,
    "difficulty": 0.60,
}

# ─────────────────────────────────────────────
# IMPORTS FROM UTILS
# ─────────────────────────────────────────────
import sys
BASE = Path(__file__).resolve().parents[2]  # project root
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from utils.utils import (
    _extract_xy,
    _is_inside_convex_hull,
    _get_convex_hull_area,
    _get_clipped_voronoi_area,
    _get_defensive_density,
    _get_nearest_defender_distance,
    _zone_value,
    _get_short_name,
)

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def compute_reception_ability_index(events: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to compute the Reception Ability Index (RAI) for Ball Receipt events.
    """
    print("Step 1: Filtering Ball Receipt events ...")
    
    # Extract actor rows for Ball Receipt events (one row per event)
    receipts_actor = events[
        (events["type"] == "Ball Receipt*") &
        (events["actor"] == True) &
        (~events["keeper"].eq(True))
    ].copy()
    
    # Extract receiver coordinates (location_x contains the event performer's location)
    receipts_actor[["rec_x", "rec_y"]] = receipts_actor["location_x"].apply(
        lambda v: pd.Series(_extract_xy(v))
    )
    receipts_actor = receipts_actor.dropna(subset=["rec_x", "rec_y"])
    
    print(f"  Total ball receipts found: {len(receipts_actor):,}")
    
    # ── Step 2: Extract player and defender locations from freeze frames ──
    print("Step 2: Grouping player and opponent coordinates per event frame ...")
    
    # Get all player locations in receipt frames
    receipt_players = events[events["type"] == "Ball Receipt*"].copy()
    receipt_players[["x", "y"]] = receipt_players["location_y"].apply(
        lambda v: pd.Series(_extract_xy(v))
    )
    receipt_players = receipt_players.dropna(subset=["x", "y"])
    
    # Group ALL players per event (for Voronoi space calculation)
    all_players_lookup = (
        receipt_players
        .groupby("id")
        .apply(lambda g: list(zip(g["x"], g["y"])))
        .to_dict()
    )
    
    # Group OPPONENTS (defenders) only (excluding goalkeepers)
    defenders_lookup = (
        receipt_players[
            (receipt_players["teammate"] == False) &
            (~receipt_players["keeper"].eq(True))
        ]
        .groupby("id")
        .apply(lambda g: list(zip(g["x"], g["y"])))
        .to_dict()
    )
    
    # Map coordinates to actor rows
    receipts_actor["all_players_xy"] = receipts_actor["id"].map(all_players_lookup).apply(lambda x: x if isinstance(x, list) else [])
    receipts_actor["defenders_xy"] = receipts_actor["id"].map(defenders_lookup).apply(lambda x: x if isinstance(x, list) else [])
    
    # Filter out events with missing spatial frames or too few defenders to make a hull
    receipts_actor = receipts_actor[receipts_actor["defenders_xy"].apply(len) >= 3].copy()
    print(f"  Receipts with valid defensive freeze frames (>=3 defenders): {len(receipts_actor):,}")
    
    # ── Step 3: Check Convex Hull Containment ──
    print("Step 3: Filtering receptions inside the opponent's defensive convex hull ...")
    
    def check_convex_hull(row):
        pt = (row["rec_x"], row["rec_y"])
        defs = row["defenders_xy"]
        is_inside = _is_inside_convex_hull(pt, defs)
        if not is_inside:
            return pd.Series([False, 0.0])
        hull_area = _get_convex_hull_area(defs)
        return pd.Series([True, hull_area])
        
    hull_checks = receipts_actor.apply(check_convex_hull, axis=1)
    receipts_actor[["is_inside_hull", "hull_area"]] = hull_checks
    
    # Keep only receptions within the opponent's defensive convex hull
    receipts_filtered = receipts_actor[receipts_actor["is_inside_hull"] == True].copy()
    print(f"  Receptions inside opponent's defensive convex hull: {len(receipts_filtered):,} ({(len(receipts_filtered)/max(len(receipts_actor), 1))*100:.1f}%)")
    
    if len(receipts_filtered) == 0:
        print("  WARNING: No receipts found inside the defensive convex hull!")
        return pd.DataFrame()
        
    # ── Step 4: Compute Spatial and Pressure Metric Components ──
    print("Step 4: Computing raw Voronoi, density, nearest defender distance, and zone value ...")
    
    metrics = receipts_filtered.apply(
        lambda row: pd.Series({
            "voronoi_area": _get_clipped_voronoi_area(row["all_players_xy"], (row["rec_x"], row["rec_y"]), row["defenders_xy"]),
            "defensive_density": _get_defensive_density((row["rec_x"], row["rec_y"]), row["defenders_xy"], DENSITY_RADIUS),
            "nearest_defender_distance": _get_nearest_defender_distance((row["rec_x"], row["rec_y"]), row["defenders_xy"]),
            "zone_value": _zone_value(row["rec_x"])
        }),
        axis=1
    )
    receipts_filtered = pd.concat([receipts_filtered, metrics], axis=1)
    
    # ── Step 5: Normalize Components and Score ──
    print("Step 5: Normalizing components and calculating final RAI scores ...")
    
    # Voronoi Area: higher area is more space/advantageous -> higher value
    min_vor = receipts_filtered["voronoi_area"].min()
    max_vor = receipts_filtered["voronoi_area"].max()
    receipts_filtered["voronoi_area_norm"] = (receipts_filtered["voronoi_area"] - min_vor) / max((max_vor - min_vor), 1.0)
    
    # Defensive Density: higher density is harder -> higher difficulty
    min_den = receipts_filtered["defensive_density"].min()
    max_den = receipts_filtered["defensive_density"].max()
    receipts_filtered["density_norm"] = (receipts_filtered["defensive_density"] - min_den) / max((max_den - min_den), 1.0)
    
    # Nearest Defender Distance: smaller distance is harder -> higher difficulty
    min_near = receipts_filtered["nearest_defender_distance"].min()
    max_near = receipts_filtered["nearest_defender_distance"].max()
    receipts_filtered["nearest_defender_norm"] = (max_near - receipts_filtered["nearest_defender_distance"]) / max((max_near - min_near), 1.0)
    
    # Hull Area: smaller hull area is more compact block/harder -> higher difficulty
    min_hull = receipts_filtered["hull_area"].min()
    max_hull = receipts_filtered["hull_area"].max()
    receipts_filtered["hull_area_norm"] = (max_hull - receipts_filtered["hull_area"]) / max((max_hull - min_hull), 1.0)
    
    # Zone value is already normalized to [0,1] inside the helper, so no further normalization needed
    
    # Compute composite Difficulty Context Index
    receipts_filtered["difficulty_context"] = (
        WEIGHTS_DIFFICULTY["density"] * receipts_filtered["density_norm"] +
        WEIGHTS_DIFFICULTY["nearest"] * receipts_filtered["nearest_defender_norm"] +
        WEIGHTS_DIFFICULTY["hull"] * receipts_filtered["hull_area_norm"] +
        WEIGHTS_DIFFICULTY["zone"] * receipts_filtered["zone_value"]
    )
    
    # Compute final Reception Ability Index (RAI) score
    receipts_filtered["score"] = (
        WEIGHTS_RAI["voronoi"] * receipts_filtered["voronoi_area_norm"] +
        WEIGHTS_RAI["difficulty"] * receipts_filtered["difficulty_context"]
    )
    
    # Select columns to preserve
    output_cols = [
        "id", "match_id", "player", "player_id", "team", "team_id",
        "period", "minute", "second", "timestamp", "possession",
        "rec_x", "rec_y", "hull_area", "voronoi_area", "defensive_density",
        "nearest_defender_distance", "zone_value",
        "voronoi_area_norm", "density_norm", "nearest_defender_norm", "hull_area_norm",
        "difficulty_context", "score"
    ]
    output_cols = [c for c in output_cols if c in receipts_filtered.columns]
    result = receipts_filtered[output_cols].copy()
    
    return result

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[2]
    EVENTS_PATH = BASE / "data" / "processed" / "events.parquet"
    OUTPUT_PATH = BASE / "data" / "processed" / "reception_ability_index.parquet"
    
    print("=" * 60)
    print("RECEPTION ABILITY INDEX (RAI) METRIC PIPELINE")
    print("=" * 60)
    print(f"Loading events from {EVENTS_PATH} ...")
    events = pd.read_parquet(EVENTS_PATH)
    print(f"  Loaded {len(events):,} rows, {events['id'].nunique():,} unique events")
    print()
    
    result = compute_reception_ability_index(events)
    
    if len(result) > 0:
        # ── Sanity checks ──
        print("\nRunning sanity checks ...")
        assert result["score"].between(0.0, 1.0).all(), "RAI score out of [0, 1] range!"
        assert result["score"].notna().all(), "NaN scores found!"
        assert result["difficulty_context"].between(0.0, 1.0).all(), "difficulty_context out of [0, 1] range!"
        assert result["voronoi_area_norm"].between(0.0, 1.0).all(), "voronoi_area_norm out of [0, 1] range!"
        print("  ✓ All sanity checks passed")
        
        # ── Save ──
        result.to_parquet(OUTPUT_PATH, index=False)
        print(f"\nSaved {len(result):,} RAI-evaluated receipts to: {OUTPUT_PATH}")
        
        # ── Quick summary ──
        print("\nRAI Score Distribution:")
        print(result["score"].describe().round(3))
        
        print("\nTop 10 Receptions by RAI Score:")
        top10_cols = ["player", "team", "minute", "rec_x", "rec_y", "defensive_density",
                      "nearest_defender_distance", "voronoi_area", "zone_value", "score"]
        top10 = result.nlargest(10, "score")[top10_cols].copy()
        top10["player"] = top10["player"].apply(_get_short_name)
        print(top10.to_string(index=False))
    
    print("\nDone ✓")
