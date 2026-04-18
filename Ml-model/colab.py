# ============================================================
# 🌾  AI CROP RECOMMENDATION + GENETIC BREEDING SIMULATOR
# Unified Production System — Fully Debugged & Corrected
# ============================================================
#
# FIXES APPLIED (all 8 issues resolved):
#
#  1. Wrong result variable in print_genetic_report() calls
#     → each test now passes its own result variable
#  2. Invalid scenario key "heatwaves" → "heatwave"
#     → added validate_scenario() with hard error
#  3. Invalid soil type "dry" → valid types only
#     → added validate_soil_type() with hard error
#  4. Copy-paste result variable confusion
#     → each test block is self-contained and clearly labelled
#  5. Label/parameter mismatch
#     → every test title reflects the exact parameters used
#  6. No multi-scenario testing
#     → 3 states × 3 scenarios = 9 automated runs in a loop
#  7. Silent fallbacks for invalid state/season/scenario/soil
#     → all now raise ValueError with a clear diagnostic message
#  8. Genetic layer applied before report printing
#     → extend_rankings() always called inside predict(),
#        result dict always contains GeneticCropResult objects
# ============================================================

# ── stdlib ───────────────────────────────────────────────────
import csv
import math
import os
import logging
import time
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional
import pickle

# ── third-party ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not found — falling back to RandomForest.")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ============================================================
# ⚙️  CONFIGURATION
# ============================================================
@dataclass
class Config:
    # ── Dataset paths ─────────────────────────────────────────
    data_path:       str = "crop_production.csv"
    trait_csv_path:  str = "crop_traits.csv"

    # ── Data cleaning ─────────────────────────────────────────
    yield_upper_quantile: float = 0.99
    yield_lower_quantile: float = 0.01

    # ── Model ─────────────────────────────────────────────────
    test_size:    float = 0.2
    random_state: int   = 42

    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators":    500,
        "max_depth":       7,
        "learning_rate":   0.05,
        "subsample":       0.8,
        "colsample_bytree":0.8,
        "min_child_weight":3,
        "reg_alpha":       0.1,
        "reg_lambda":      1.0,
        "n_jobs":         -1,
        "random_state":    42,
    })

    # ── NASA POWER ────────────────────────────────────────────
    nasa_start_date: str = "20190101"
    nasa_end_date:   str = "20211231"
    api_timeout:     int = 15
    api_retries:     int = 3

    # ── Valid enumerations (used by validators) ───────────────
    valid_scenarios: tuple = ("drought", "heatwave", "flood", "normal")
    valid_soil_types: tuple = (
        "clay", "sandy", "loamy", "silty", "peaty", "chalky", "unknown"
    )

    # ── Scenario climate modifiers ────────────────────────────
    scenarios: dict = field(default_factory=lambda: {
        "drought":  {"temp_delta": +2.0, "rain_factor": 0.55, "humidity_factor": 0.65},
        "heatwave": {"temp_delta": +5.0, "rain_factor": 0.75, "humidity_factor": 0.80},
        "flood":    {"temp_delta": +1.0, "rain_factor": 1.70, "humidity_factor": 1.25},
        "normal":   {"temp_delta":  0.0, "rain_factor": 1.00, "humidity_factor": 1.00},
    })

    # ── Soil profiles ─────────────────────────────────────────
    soil_profiles: dict = field(default_factory=lambda: {
        "clay":   {"water_retention": 0.90, "drainage": 0.25, "fertility": 0.80},
        "sandy":  {"water_retention": 0.15, "drainage": 0.95, "fertility": 0.35},
        "loamy":  {"water_retention": 0.72, "drainage": 0.70, "fertility": 0.92},
        "silty":  {"water_retention": 0.80, "drainage": 0.50, "fertility": 0.72},
        "peaty":  {"water_retention": 0.95, "drainage": 0.35, "fertility": 0.60},
        "chalky": {"water_retention": 0.28, "drainage": 0.82, "fertility": 0.48},
        "unknown":{"water_retention": 0.60, "drainage": 0.60, "fertility": 0.65},
    })

    # ── Hybrid score weights (must sum to 1.0) ────────────────
    score_weights: dict = field(default_factory=lambda: {
        "ml_rank": 0.30,
        "climate": 0.25,
        "soil":    0.20,
        "trait":   0.15,
        "domain":  0.10,
    })


cfg = Config()


# ============================================================
# 🛡️  INPUT VALIDATORS
# ============================================================

def validate_scenario(scenario: str) -> str:
    """
    Normalise and validate the scenario string.
    Raises ValueError for any value not in cfg.valid_scenarios.
    """
    s = scenario.strip().lower()
    if s not in cfg.valid_scenarios:
        raise ValueError(
            f"[Validation] Invalid scenario '{scenario}'. "
            f"Valid options: {cfg.valid_scenarios}"
        )
    return s


def validate_soil_type(soil_type: str) -> str:
    """
    Normalise and validate the soil type string.
    Raises ValueError for any value not in cfg.valid_soil_types.
    """
    s = soil_type.strip().lower()
    if s not in cfg.valid_soil_types:
        raise ValueError(
            f"[Validation] Invalid soil type '{soil_type}'. "
            f"Valid options: {cfg.valid_soil_types}"
        )
    return s


def validate_state(state: str, known_states: list[str]) -> str:
    """
    Validate that state exists in the training data.
    Raises ValueError instead of silently falling back.
    """
    if state not in known_states:
        raise ValueError(
            f"[Validation] State '{state}' is not in training data. "
            f"Available states (sample): {known_states[:10]}"
        )
    return state


def validate_season(season: str, known_seasons: list[str]) -> str:
    season_clean = season.strip()

    # normalize known seasons
    normalized = [s.strip() for s in known_seasons]

    if season_clean not in normalized:
        raise ValueError(
            f"[Validation] Season '{season}' is not in training data. "
            f"Available seasons: {normalized}"
        )

    return season_clean


# ============================================================
# 🌿  DOMAIN KNOWLEDGE TABLES
# ============================================================

@dataclass(frozen=True)
class CropClimateSpec:
    """Agronomic climate tolerance envelope for one crop."""
    temp_min:          float
    temp_max:          float
    temp_opt_lo:       float
    temp_opt_hi:       float
    rain_min:          float   # mm/day
    rain_max:          float
    rain_opt_lo:       float
    rain_opt_hi:       float
    water_need:        float   # 0 (very low) → 1 (very high)
    drought_tolerance: float
    flood_tolerance:   float
    heat_tolerance:    float
    soil_affinity:     dict = field(default_factory=dict)


CROP_SPECS: dict[str, CropClimateSpec] = {
    "Sugarcane": CropClimateSpec(
        20, 40, 24, 32, 3.5, 12.0, 4.0, 8.0,
        water_need=0.95, drought_tolerance=0.05, flood_tolerance=0.40,
        heat_tolerance=0.65,
        soil_affinity={"loamy":1.0,"clay":0.8,"silty":0.75,"sandy":0.2,"peaty":0.5,"chalky":0.3},
    ),
    "Rice": CropClimateSpec(
        20, 38, 22, 32, 3.0, 12.0, 4.0, 9.0,
        water_need=0.90, drought_tolerance=0.10, flood_tolerance=0.80,
        heat_tolerance=0.55,
        soil_affinity={"clay":1.0,"silty":0.85,"loamy":0.75,"sandy":0.15,"peaty":0.60,"chalky":0.25},
    ),
    "Jute": CropClimateSpec(
        22, 38, 24, 34, 4.0, 12.0, 5.0, 9.0,
        water_need=0.85, drought_tolerance=0.10, flood_tolerance=0.70,
        heat_tolerance=0.60,
        soil_affinity={"loamy":1.0,"silty":0.85,"clay":0.70,"sandy":0.20,"peaty":0.55,"chalky":0.20},
    ),
    "Banana": CropClimateSpec(
        20, 38, 24, 34, 3.0, 10.0, 4.0, 8.0,
        water_need=0.85, drought_tolerance=0.15, flood_tolerance=0.45,
        heat_tolerance=0.65,
        soil_affinity={"loamy":1.0,"silty":0.80,"clay":0.65,"sandy":0.20,"peaty":0.55,"chalky":0.25},
    ),
    "Maize": CropClimateSpec(
        15, 35, 18, 30, 1.5, 7.0, 2.0, 5.0,
        water_need=0.55, drought_tolerance=0.45, flood_tolerance=0.30,
        heat_tolerance=0.55,
        soil_affinity={"loamy":1.0,"silty":0.85,"sandy":0.60,"clay":0.65,"peaty":0.50,"chalky":0.45},
    ),
    "Soyabean": CropClimateSpec(
        15, 32, 18, 28, 2.0, 6.0, 2.5, 4.5,
        water_need=0.55, drought_tolerance=0.40, flood_tolerance=0.25,
        heat_tolerance=0.40,
        soil_affinity={"loamy":1.0,"silty":0.85,"clay":0.70,"sandy":0.50,"peaty":0.55,"chalky":0.40},
    ),
    "Cotton": CropClimateSpec(
        20, 38, 22, 32, 1.0, 5.5, 1.5, 3.5,
        water_need=0.50, drought_tolerance=0.55, flood_tolerance=0.15,
        heat_tolerance=0.70,
        soil_affinity={"loamy":1.0,"clay":0.80,"sandy":0.50,"silty":0.70,"peaty":0.40,"chalky":0.35},
    ),
    "Potato": CropClimateSpec(
        10, 25, 12, 22, 1.5, 5.0, 2.0, 4.0,
        water_need=0.60, drought_tolerance=0.30, flood_tolerance=0.20,
        heat_tolerance=0.15,
        soil_affinity={"loamy":1.0,"silty":0.85,"sandy":0.75,"clay":0.50,"peaty":0.70,"chalky":0.35},
    ),
    "Onion": CropClimateSpec(
        13, 32, 15, 28, 1.0, 4.0, 1.5, 3.0,
        water_need=0.50, drought_tolerance=0.40, flood_tolerance=0.25,
        heat_tolerance=0.50,
        soil_affinity={"loamy":1.0,"sandy":0.80,"silty":0.75,"clay":0.55,"peaty":0.50,"chalky":0.40},
    ),
    "Tomato": CropClimateSpec(
        18, 32, 20, 28, 1.5, 5.0, 2.0, 4.0,
        water_need=0.60, drought_tolerance=0.30, flood_tolerance=0.20,
        heat_tolerance=0.40,
        soil_affinity={"loamy":1.0,"sandy":0.75,"silty":0.80,"clay":0.55,"peaty":0.55,"chalky":0.40},
    ),
    "Wheat": CropClimateSpec(
        10, 25, 12, 22, 1.0, 5.0, 1.5, 3.5,
        water_need=0.45, drought_tolerance=0.50, flood_tolerance=0.20,
        heat_tolerance=0.25,
        soil_affinity={"loamy":1.0,"clay":0.80,"silty":0.85,"sandy":0.50,"peaty":0.50,"chalky":0.60},
    ),
    "Groundnut": CropClimateSpec(
        20, 35, 22, 30, 1.5, 5.0, 2.0, 4.0,
        water_need=0.45, drought_tolerance=0.60, flood_tolerance=0.20,
        heat_tolerance=0.65,
        soil_affinity={"sandy":1.0,"loamy":0.90,"silty":0.70,"clay":0.45,"peaty":0.45,"chalky":0.40},
    ),
    "Bajra": CropClimateSpec(
        20, 42, 25, 36, 0.4, 3.5, 0.6, 2.0,
        water_need=0.20, drought_tolerance=0.92, flood_tolerance=0.15,
        heat_tolerance=0.90,
        soil_affinity={"sandy":1.0,"loamy":0.85,"chalky":0.75,"silty":0.65,"clay":0.50,"peaty":0.35},
    ),
    "Jowar": CropClimateSpec(
        18, 40, 22, 35, 0.5, 4.0, 0.8, 2.5,
        water_need=0.25, drought_tolerance=0.88, flood_tolerance=0.20,
        heat_tolerance=0.85,
        soil_affinity={"sandy":0.95,"loamy":0.90,"chalky":0.80,"clay":0.60,"silty":0.70,"peaty":0.40},
    ),
    "Ragi": CropClimateSpec(
        18, 38, 22, 32, 0.6, 4.0, 0.8, 2.5,
        water_need=0.25, drought_tolerance=0.80, flood_tolerance=0.25,
        heat_tolerance=0.75,
        soil_affinity={"loamy":1.0,"sandy":0.85,"clay":0.65,"silty":0.70,"peaty":0.50,"chalky":0.60},
    ),
    "Moong": CropClimateSpec(
        18, 38, 22, 32, 0.6, 3.5, 0.8, 2.2,
        water_need=0.25, drought_tolerance=0.75, flood_tolerance=0.20,
        heat_tolerance=0.70,
        soil_affinity={"loamy":1.0,"sandy":0.85,"silty":0.75,"clay":0.60,"peaty":0.45,"chalky":0.50},
    ),
    "Urad": CropClimateSpec(
        18, 38, 22, 32, 0.6, 3.5, 0.8, 2.2,
        water_need=0.30, drought_tolerance=0.70, flood_tolerance=0.20,
        heat_tolerance=0.65,
        soil_affinity={"loamy":1.0,"sandy":0.80,"silty":0.75,"clay":0.60,"peaty":0.45,"chalky":0.50},
    ),
    "Arhar/Tur": CropClimateSpec(
        18, 38, 22, 32, 0.8, 4.0, 1.0, 2.5,
        water_need=0.30, drought_tolerance=0.72, flood_tolerance=0.20,
        heat_tolerance=0.65,
        soil_affinity={"loamy":1.0,"sandy":0.75,"clay":0.65,"silty":0.70,"peaty":0.45,"chalky":0.50},
    ),
    "Sunflower": CropClimateSpec(
        15, 35, 18, 28, 0.8, 4.0, 1.2, 3.0,
        water_need=0.40, drought_tolerance=0.60, flood_tolerance=0.25,
        heat_tolerance=0.60,
        soil_affinity={"loamy":1.0,"sandy":0.80,"silty":0.75,"clay":0.60,"peaty":0.50,"chalky":0.50},
    ),
}

# ── Scenario hard rules ───────────────────────────────────────
@dataclass
class ScenarioRule:
    boost:      float = 0.0
    penalty:    float = 0.0
    hard_block: bool  = False
    reason:     str   = ""


SCENARIO_RULES: dict[str, dict[str, ScenarioRule]] = {
    "drought": {
        "Sugarcane": ScenarioRule(hard_block=True,  reason="Extremely water-intensive"),
        "Rice":      ScenarioRule(penalty=0.50,      reason="High water requirement"),
        "Jute":      ScenarioRule(penalty=0.45,      reason="Needs high moisture"),
        "Banana":    ScenarioRule(penalty=0.40,      reason="High water need"),
        "Bajra":     ScenarioRule(boost=0.50,        reason="Excellent drought tolerance"),
        "Jowar":     ScenarioRule(boost=0.45,        reason="Very drought-tolerant"),
        "Ragi":      ScenarioRule(boost=0.35,        reason="Drought-tolerant millet"),
        "Groundnut": ScenarioRule(boost=0.30,        reason="Good drought tolerance"),
        "Moong":     ScenarioRule(boost=0.25,        reason="Short-duration, water-efficient"),
        "Cotton":    ScenarioRule(boost=0.20,        reason="Moderate drought tolerance"),
        "Sunflower": ScenarioRule(boost=0.20,        reason="Deep roots, drought-tolerant"),
    },
    "heatwave": {
        "Wheat":     ScenarioRule(hard_block=True,   reason="Heat-sensitive — terminal heat damage"),
        "Potato":    ScenarioRule(hard_block=True,   reason="Tuber formation fails above 28°C"),
        "Sugarcane": ScenarioRule(penalty=0.35,      reason="High water need in heat"),
        "Rice":      ScenarioRule(penalty=0.30,      reason="Spikelet sterility above 35°C"),
        "Soyabean":  ScenarioRule(penalty=0.30,      reason="Pod-fill stress at high temp"),
        "Bajra":     ScenarioRule(boost=0.50,        reason="Thrives in extreme heat"),
        "Jowar":     ScenarioRule(boost=0.45,        reason="C4 crop, heat-adapted"),
        "Cotton":    ScenarioRule(boost=0.30,        reason="High heat tolerance"),
        "Groundnut": ScenarioRule(boost=0.25,        reason="Good heat tolerance"),
        "Ragi":      ScenarioRule(boost=0.20,        reason="Heat-tolerant millet"),
    },
    "flood": {
        "Wheat":     ScenarioRule(hard_block=True,   reason="Waterlogging lethal"),
        "Potato":    ScenarioRule(hard_block=True,   reason="Tuber rot in waterlogged soil"),
        "Groundnut": ScenarioRule(hard_block=True,   reason="Pod rot in waterlogged soil"),
        "Bajra":     ScenarioRule(penalty=0.40,      reason="Sandy-soil crop, poor flood tolerance"),
        "Cotton":    ScenarioRule(penalty=0.40,      reason="Very poor waterlogging tolerance"),
        "Sunflower": ScenarioRule(penalty=0.35,      reason="Root rot risk"),
        "Rice":      ScenarioRule(boost=0.50,        reason="Paddy is flood-tolerant"),
        "Jute":      ScenarioRule(boost=0.40,        reason="Thrives in high moisture"),
        "Arhar/Tur": ScenarioRule(boost=0.15,        reason="Moderate flood tolerance"),
    },
    "normal": {},
}


# ============================================================
# 🌦️  CLIMATE DATA
# ============================================================
@dataclass
class ClimateData:
    temperature: float
    rainfall:    float
    humidity:    float
    source:      str = "NASA POWER"

    def apply_scenario(self, scenario: str) -> "ClimateData":
        sc = cfg.scenarios[scenario]          # KeyError impossible — already validated
        return ClimateData(
            temperature=self.temperature + sc["temp_delta"],
            rainfall   =self.rainfall    * sc["rain_factor"],
            humidity   =self.humidity    * sc["humidity_factor"],
            source     =f"{self.source} [{scenario}]",
        )

    def __str__(self):
        return (f"🌡  {self.temperature:.1f}°C | "
                f"🌧  {self.rainfall:.2f} mm/day | "
                f"💧 {self.humidity:.1f}% RH")


def fetch_climate(lat: float, lon: float) -> ClimateData:
    """NASA POWER API with exponential backoff and fallback."""
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,PRECTOTCORR,RH2M&community=AG"
        f"&longitude={lon}&latitude={lat}&format=JSON"
        f"&start={cfg.nasa_start_date}&end={cfg.nasa_end_date}"
    )
    for attempt in range(1, cfg.api_retries + 1):
        try:
            log.info(f"  Fetching NASA climate (attempt {attempt}/{cfg.api_retries})...")
            r = requests.get(url, timeout=cfg.api_timeout)
            r.raise_for_status()
            params = r.json()["properties"]["parameter"]

            def _avg(d: dict) -> float:
                vals = [v for v in d.values() if v is not None and v > -900]
                return float(np.mean(vals)) if vals else 0.0

            return ClimateData(
                temperature=_avg(params["T2M"]),
                rainfall   =_avg(params["PRECTOTCORR"]),
                humidity   =_avg(params["RH2M"]),
            )
        except Exception as e:
            log.warning(f"  NASA API attempt {attempt} failed: {e}")
            if attempt < cfg.api_retries:
                time.sleep(2 ** attempt)

    log.warning("  All NASA API attempts failed — using regional fallback values.")
    return ClimateData(temperature=25.0, rainfall=2.5, humidity=65.0,
                       source="Fallback defaults")


# ============================================================
# 🪨  SOIL PROFILE
# ============================================================
@dataclass
class SoilProfile:
    soil_type:       str
    water_retention: float
    drainage:        float
    fertility:       float

    @classmethod
    def from_type(cls, soil_type: str) -> "SoilProfile":
        """soil_type must already be validated before calling this."""
        return cls(soil_type=soil_type, **cfg.soil_profiles[soil_type])

    def score_for_crop(self, crop: str, climate: ClimateData) -> float:
        """
        Soil suitability score in [0, 1] that accounts for:
        1. Crop-specific soil affinity
        2. Waterlogging risk
        3. Drought risk
        4. Fertility contribution
        """
        spec = CROP_SPECS.get(crop)

        affinity = (spec.soil_affinity.get(self.soil_type, 0.55)
                    if spec and spec.soil_affinity else 0.65)

        if spec:
            wl_exposure = max(0, climate.rainfall - 3.0) * (1 - self.drainage)
            wl_penalty  = float(np.clip(wl_exposure * (1 - spec.flood_tolerance) * 0.06, 0, 0.35))

            dr_exposure = max(0, 3.0 - climate.rainfall) * (1 - self.water_retention)
            dr_penalty  = float(np.clip(dr_exposure * spec.water_need * 0.12, 0, 0.35))
        else:
            wl_penalty = dr_penalty = 0.0

        fertility_bonus = (self.fertility - 0.5) * 0.15
        return float(np.clip(affinity - wl_penalty - dr_penalty + fertility_bonus, 0.05, 1.0))


# ============================================================
# 🌡️  DOMAIN SCORING ENGINE
# ============================================================

def climate_suitability_score(crop: str, climate: ClimateData) -> float:
    """Smooth 0–1 score based on proximity to the crop's optimal climate."""
    spec = CROP_SPECS.get(crop)
    if spec is None:
        return 0.60

    def _range_score(val, lo_hard, lo_opt, hi_opt, hi_hard):
        if lo_opt <= val <= hi_opt:
            return 1.0
        if val < lo_opt:
            return float(np.interp(val, [lo_hard, lo_opt], [0.0, 1.0]))
        return float(np.interp(val, [hi_opt, hi_hard], [1.0, 0.0]))

    t_score = _range_score(climate.temperature,
                           spec.temp_min, spec.temp_opt_lo,
                           spec.temp_opt_hi, spec.temp_max)
    r_score = _range_score(climate.rainfall,
                           spec.rain_min, spec.rain_opt_lo,
                           spec.rain_opt_hi, spec.rain_max)
    return float(np.sqrt(np.clip(t_score, 0, 1) * np.clip(r_score, 0, 1)))


def domain_adj_score(crop: str, scenario: str) -> tuple[float, str]:
    """
    Scenario-based domain adjustment.
    Returns (score in 0–1, reason string).
    Neutral base = 0.5. Boosts raise it; penalties lower it.
    """
    rule = SCENARIO_RULES.get(scenario, {}).get(crop, ScenarioRule())
    raw  = 0.5 + rule.boost - rule.penalty
    return float(np.clip(raw, 0.0, 1.0)), rule.reason


def is_hard_blocked(crop: str, scenario: str) -> tuple[bool, str]:
    rule = SCENARIO_RULES.get(scenario, {}).get(crop, ScenarioRule())
    return rule.hard_block, rule.reason


# ============================================================
# 📦  DATA LOADING & FEATURE ENGINEERING
# ============================================================

def load_and_engineer(path: str) -> pd.DataFrame:
    """Load crop production CSV and build all engineered features."""
    log.info(f"Loading production data: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = {"State_Name", "Crop_Year", "Season", "Crop", "Area", "Production"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df[(df["Area"] > 0) & (df["Production"] > 0)].copy()
    df.dropna(subset=["Crop", "Season", "State_Name"], inplace=True)

    df["Yield"]    = df["Production"] / df["Area"]
    lo = df["Yield"].quantile(cfg.yield_lower_quantile)
    hi = df["Yield"].quantile(cfg.yield_upper_quantile)
    df = df[(df["Yield"] >= lo) & (df["Yield"] <= hi)]
    df["LogYield"] = np.log1p(df["Yield"])

    df["Crop_Year"] = pd.to_numeric(df["Crop_Year"], errors="coerce")
    yr_min, yr_max = df["Crop_Year"].min(), df["Crop_Year"].max()
    df["Year_Norm"] = (df["Crop_Year"] - yr_min) / (yr_max - yr_min + 1e-6)
    df["Log_Area"]  = np.log1p(df["Area"])

    crop_stats = df.groupby("Crop")["Yield"].agg(
        Crop_Mean_Yield="mean", Crop_Std_Yield="std", Crop_Median_Yield="median"
    ).reset_index()
    df = df.merge(crop_stats, on="Crop", how="left")

    state_stats = df.groupby("State_Name")["Yield"].agg(
        State_Mean_Yield="mean"
    ).reset_index()
    df = df.merge(state_stats, on="State_Name", how="left")

    season_rank = (df.groupby("Season")["Yield"].mean()
                     .rank(ascending=False).to_dict())
    df["Season_Rank"] = df["Season"].map(season_rank).fillna(3.0)

    log.info(f"  Engineered dataset: {len(df):,} rows | "
             f"{df['Crop'].nunique()} crops | "
             f"{df['State_Name'].nunique()} states")
    return df


# ============================================================
# 🤖  CROP MODEL
# ============================================================

class CropModel:
    """Encapsulates encoding, training, and inference."""

    def __init__(self):
        self.model          = None
        self.feature_cols:  list[str]         = []
        self.label_encoders:dict[str, LabelEncoder] = {}
        self.crop_list:     list[str]         = []
        self.season_list:   list[str]         = []
        self.state_list:    list[str]         = []
        self.metrics:       dict              = {}
        self._df_ref:       Optional[pd.DataFrame] = None

    # ── Encoding ──────────────────────────────────────────────
    def _fit_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["Crop", "Season", "State_Name"]:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        return df

    def _encode_single(self, crop: str, season: str, state: str) -> dict:
        out = {}
        for col, val in [("Crop", crop), ("Season", season), ("State_Name", state)]:
            le = self.label_encoders.get(col)
            out[col + "_enc"] = (int(le.transform([val])[0])
                                 if le and val in le.classes_ else -1)
        return out

    # ── Training ──────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> "CropModel":
        df = df.copy()
        df = self._fit_encode(df)
        self._df_ref     = df
        self.crop_list   = sorted(df["Crop"].unique().tolist())
        self.season_list = sorted(df["Season"].unique().tolist())
        self.state_list  = sorted(df["State_Name"].unique().tolist())

        feature_cols = [
            "Crop_enc", "Season_enc", "State_Name_enc",
            "Log_Area", "Year_Norm",
            "Crop_Mean_Yield", "Crop_Std_Yield", "Crop_Median_Yield",
            "State_Mean_Yield", "Season_Rank",
        ]
        self.feature_cols = feature_cols

        X = df[feature_cols].fillna(0)
        y = df["LogYield"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state
        )

        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(**cfg.xgb_params, verbosity=0)
            log.info("  Training XGBoost...")
        else:
            self.model = RandomForestRegressor(
                n_estimators=300, max_depth=15, n_jobs=-1,
                random_state=cfg.random_state
            )
            log.info("  Training RandomForest...")

        self.model.fit(X_tr, y_tr)

        y_pred = np.expm1(self.model.predict(X_te))
        y_true = np.expm1(y_te)
        self.metrics = {
            "MAE": round(mean_absolute_error(y_true, y_pred), 2),
            "R2":  round(r2_score(y_true, y_pred), 3),
        }
        log.info(f"  ✅ Model trained | MAE={self.metrics['MAE']} | R²={self.metrics['R2']}")
        return self

    # ── Single-crop inference ─────────────────────────────────
    def predict_yield(
        self,
        crop:   str,
        season: str,
        state:  str,
        area:   float = 1.0,
        year:   int   = 2023,
    ) -> float:
        df    = self._df_ref
        enc   = self._encode_single(crop, season, state)
        yr_min, yr_max = df["Crop_Year"].min(), df["Crop_Year"].max()
        year_norm = float(np.clip((year - yr_min) / (yr_max - yr_min + 1e-6), 0, 1.5))

        cr = df[df["Crop"]       == crop]["Yield"]
        sr = df[df["State_Name"] == state]["Yield"]
        srm = df.groupby("Season")["Yield"].mean().rank(ascending=False).to_dict()

        row = {
            "Crop_enc":          enc["Crop_enc"],
            "Season_enc":        enc["Season_enc"],
            "State_Name_enc":    enc["State_Name_enc"],
            "Log_Area":          float(np.log1p(area)),
            "Year_Norm":         year_norm,
            "Crop_Mean_Yield":   float(cr.mean()   if len(cr) else 0),
            "Crop_Std_Yield":    float(cr.std()    if len(cr) > 1 else 0),
            "Crop_Median_Yield": float(cr.median() if len(cr) else 0),
            "State_Mean_Yield":  float(sr.mean()   if len(sr) else 0),
            "Season_Rank":       float(srm.get(season, 3.0)),
        }
        return float(np.expm1(self.model.predict(pd.DataFrame([row]))[0]))

    def feature_importance(self) -> pd.Series:
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(
                self.model.feature_importances_, index=self.feature_cols
            ).sort_values(ascending=False)
        return pd.Series(dtype=float)


# ============================================================
# 🧬  GENETIC TRAIT LAYER
# ============================================================

# Module-level trait dictionary — populated by load_trait_dict()
TRAIT_DICT: dict[str, dict] = {}

_MAX_TEMP_DIFF = 20.0   # °C
_MAX_RAIN_DIFF =  8.0   # mm/day

SOIL_MULTIPLIER_MAP: dict[str, float] = {
    "sandy": 0.55, "loamy": 0.90, "clay": 0.85,
    "silty": 0.80, "peaty": 0.75, "chalky": 0.60, "unknown": 0.70,
}


def load_trait_dict(csv_path: str = "crop_traits.csv") -> dict[str, dict]:
    """Load crop_traits.csv into TRAIT_DICT format."""
    log.info(f"Loading trait dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required = {
        "Crop", "drought_resistance", "heat_tolerance", "flood_tolerance",
        "water_requirement", "nutrient_efficiency", "growth_duration_days",
        "yield_potential", "pest_resistance", "soil_adaptability",
        "temperature_optimum", "rainfall_optimum",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[TraitLoader] Missing columns: {missing}")

    df["Crop"] = df["Crop"].str.strip()
    trait_cols = [c for c in df.columns if c != "Crop"]
    for col in trait_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        result[row["Crop"]] = {col: float(row[col]) for col in trait_cols}

    log.info(f"  ✅ Traits loaded: {len(result)} crops")
    return result


def compute_trait_score(
    crop:     str,
    climate:  ClimateData,
    soil:     SoilProfile,
    scenario: str,          # already validated
) -> float:
    """
    Genetic trait suitability score in [0, 1].

    Weights: scenario=0.40 | climate=0.30 | soil=0.20 | nutrient=0.10
    Penalty: drought → subtract (water_requirement × 0.3) before clip
    """
    if crop not in TRAIT_DICT:
        return 0.5

    t = TRAIT_DICT[crop]

    # 1. Scenario alignment
    if scenario == "drought":
        scenario_trait = t["drought_resistance"]
    elif scenario == "heatwave":
        scenario_trait = t["heat_tolerance"]
    elif scenario == "flood":
        scenario_trait = t["flood_tolerance"]
    else:
        scenario_trait = (
            t["drought_resistance"] + t["heat_tolerance"] + t["flood_tolerance"]
        ) / 3.0

    # 2. Climate compatibility — exp-decay geometric mean
    temp_score    = math.exp(-abs(climate.temperature - t["temperature_optimum"]) / _MAX_TEMP_DIFF)
    rain_score    = math.exp(-abs(climate.rainfall    - t["rainfall_optimum"])    / _MAX_RAIN_DIFF)
    climate_comp  = math.sqrt(temp_score * rain_score)

    # 3. Soil compatibility
    soil_mult  = SOIL_MULTIPLIER_MAP.get(soil.soil_type, 0.70)
    soil_comp  = float(np.clip(t["soil_adaptability"] * soil_mult, 0.0, 1.0))

    # 4. Nutrient efficiency
    nutrient   = float(np.clip(t["nutrient_efficiency"], 0.0, 1.0))

    # Water penalty (drought only)
    penalty    = t["water_requirement"] * 0.3 if scenario == "drought" else 0.0

    raw = (0.40 * scenario_trait
         + 0.30 * climate_comp
         + 0.20 * soil_comp
         + 0.10 * nutrient
         - penalty)
    return float(np.clip(raw, 0.0, 1.0))


def compute_genetic_fitness(trait_score: float, ml_rank_score: float) -> float:
    """fitness = 0.6 × trait_score + 0.4 × ml_rank_score  (both in [0,1])"""
    return float(np.clip(0.6 * trait_score + 0.4 * ml_rank_score, 0.0, 1.0))


# ============================================================
# 📦  CROP RESULT — EXTENDED WITH GENETIC FIELDS
# ============================================================

@dataclass
class CropResult:
    """
    Unified result object.  Carries both the original 4-component
    scores AND the genetic layer additions.  final_score always
    uses the 5-component formula.
    """
    crop:             str
    raw_yield:        float

    # Original scoring components
    ml_rank_score:    float
    climate_score:    float
    soil_score:       float
    domain_adj_score: float

    # Genetic additions (populated by extend_rankings)
    trait_score:      float = 0.0
    genetic_fitness:  float = 0.0

    # Composite
    final_score:      float = 0.0

    # Metadata
    rank:             int   = 0
    blocked:          bool  = False
    block_reason:     str   = ""
    domain_reason:    str   = ""
    confidence:       str   = ""

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def _confidence_label(score: float) -> str:
    if score >= 0.75: return "⭐ Excellent"
    if score >= 0.60: return "✅ Good"
    if score >= 0.45: return "⚠️  Fair"
    if score >= 0.30: return "🔶 Marginal"
    return "❌ Not Recommended"


# ============================================================
# 🏆  HYBRID SCORER
# ============================================================

def hybrid_score(
    crop_model: CropModel,
    climate:    ClimateData,
    soil:       SoilProfile,
    scenario:   str,
    state:      str,
    season:     str,
    year:       int   = 2023,
    area:       float = 1.0,
) -> list[CropResult]:
    """
    Compute the 5-component hybrid score for every crop in
    crop_model.crop_list and return a sorted list of CropResult.

    Pipeline
    ────────
    1. ML yield prediction → percentile rank (0–1)
    2. climate_suitability_score   (domain rules)
    3. SoilProfile.score_for_crop  (soil × climate)
    4. domain_adj_score            (scenario rules)
    5. compute_trait_score         (genetic layer)
    6. compute_genetic_fitness     (trait + ML rank)
    7. Weighted composite → final_score
    8. Hard-block disqualification
    9. Sort + assign ranks
    """
    w = cfg.score_weights

    # ── Step 1: raw yields + percentile rank ─────────────────
    raw_yields = {
        crop: crop_model.predict_yield(crop, season, state, area, year)
        for crop in crop_model.crop_list
    }
    yields_arr = np.array(list(raw_yields.values()))
    rank_arr   = yields_arr.argsort().argsort() / max(len(yields_arr) - 1, 1)
    ml_rank_map = {crop: float(rank_arr[i])
                   for i, crop in enumerate(raw_yields)}

    # ── Steps 2–7: per-crop ───────────────────────────────────
    results: list[CropResult] = []
    for crop in crop_model.crop_list:
        blocked, block_reason = is_hard_blocked(crop, scenario)

        clim_s = climate_suitability_score(crop, climate)
        soil_s = soil.score_for_crop(crop, climate)
        dom_s, dom_reason = domain_adj_score(crop, scenario)
        ml_s   = ml_rank_map[crop]

        # Genetic layer
        trait_s   = compute_trait_score(crop, climate, soil, scenario)
        gen_fit   = compute_genetic_fitness(trait_s, ml_s)

        # 5-component final score
        final = float(np.clip(
            w["ml_rank"] * ml_s
          + w["climate"] * clim_s
          + w["soil"]    * soil_s
          + w["trait"]   * trait_s
          + w["domain"]  * dom_s,
            0.0, 1.0
        ))

        results.append(CropResult(
            crop             = crop,
            raw_yield        = raw_yields[crop],
            ml_rank_score    = ml_s,
            climate_score    = clim_s,
            soil_score       = soil_s,
            domain_adj_score = dom_s,
            trait_score      = trait_s,
            genetic_fitness  = gen_fit,
            final_score      = final,
            blocked          = blocked,
            block_reason     = block_reason,
            domain_reason    = dom_reason,
        ))

    # ── Sort: active by final_score desc; blocked at end ─────
    active  = sorted([r for r in results if not r.blocked],
                     key=lambda r: r.final_score, reverse=True)
    blocked_list = [r for r in results if r.blocked]

    for i, r in enumerate(active, 1):
        r.rank       = i
        r.confidence = _confidence_label(r.final_score)
    for i, r in enumerate(blocked_list, len(active) + 1):
        r.rank       = i
        r.confidence = "🚫 Blocked"

    return active + blocked_list


# ============================================================
# 🔮  MAIN PREDICT FUNCTION
# ============================================================

def predict(
    crop_model:    CropModel,
    lat:           float,
    lon:           float,
    scenario:      str,
    soil_type:     str,
    state:         str,
    season:        str,
    user_crop:     Optional[str] = None,
    year:          int           = 2023,
    area:          float         = 1.0,
    fetch_weather: bool          = True,
) -> dict:
    """
    Full end-to-end prediction.

    All inputs are validated before use — no silent fallbacks.
    Returns a dict ready for print_report() and print_genetic_report().
    """
    # ── Input validation ──────────────────────────────────────
    scenario  = validate_scenario(scenario)
    soil_type = validate_soil_type(soil_type)
    state     = validate_state(state, crop_model.state_list)
    season    = validate_season(season, crop_model.season_list)

    # ── Climate ───────────────────────────────────────────────
    raw_climate = (fetch_climate(lat, lon) if fetch_weather
                   else ClimateData(25.0, 2.5, 65.0, "Offline mode"))
    climate = raw_climate.apply_scenario(scenario)

    # ── Soil ──────────────────────────────────────────────────
    soil = SoilProfile.from_type(soil_type)

    # ── Hybrid scoring (genetic layer included) ───────────────
    rankings = hybrid_score(
        crop_model, climate, soil, scenario, state, season, year, area
    )

    active = [r for r in rankings if not r.blocked]
    best   = active[0] if active else rankings[0]

    # ── User-crop lookup ──────────────────────────────────────
    user_result = None
    if user_crop:
        matches = [r for r in rankings if r.crop.lower() == user_crop.lower()]
        user_result = matches[0] if matches else None
        if not user_result:
            log.warning(f"  User crop '{user_crop}' not found in crop list.")

    return {
        "climate":          climate,
        "soil":             soil,
        "scenario":         scenario,
        "state":            state,
        "season":           season,
        "rankings":         rankings,
        "best_crop":        best,
        "user_crop_result": user_result,
        "score_weights":    cfg.score_weights,
    }


# ============================================================
# 📋  STANDARD REPORT
# ============================================================

def print_report(result: dict, top_n: int = 10, show_blocked: bool = True):
    """Print the standard hybrid scoring report."""
    climate  = result["climate"]
    soil     = result["soil"]
    scenario = result["scenario"]
    rankings = result["rankings"]
    best     = result["best_crop"]
    user_res = result["user_crop_result"]
    weights  = result["score_weights"]

    W = 68
    print(f"\n{'═'*W}")
    print(f"  🌾  CROP RECOMMENDATION  |  Scenario: {scenario.upper()}")
    print(f"  State: {result['state']}  |  Season: {result['season']}")
    print(f"{'═'*W}")
    print(f"\n📡 Climate  : {climate}")
    print(f"   Source   : {climate.source}")
    print(f"\n🪨  Soil     : {soil.soil_type.title()}")
    print(f"   Fertility={soil.fertility:.0%}  "
          f"Drainage={soil.drainage:.0%}  "
          f"Water-Retention={soil.water_retention:.0%}")

    print(f"\n⚖️  Score Weights:")
    for k, v in weights.items():
        bar = "█" * int(v * 20)
        print(f"   {k:<12} {bar:<20} {v:.0%}")

    print(f"\n{'─'*W}")
    print(f"  🏆  BEST CROP: {best.crop}")
    print(f"     Final Score    : {best.final_score:.3f}  {best.confidence}")
    print(f"     ML Rank        : {best.ml_rank_score:.3f} | "
          f"Climate: {best.climate_score:.3f} | "
          f"Soil: {best.soil_score:.3f} | "
          f"Trait: {best.trait_score:.3f} | "
          f"Domain: {best.domain_adj_score:.3f}")
    print(f"{'─'*W}")

    active = [r for r in rankings if not r.blocked]
    print(f"\n📊 TOP {min(top_n, len(active))} CROPS:")
    hdr = (f"  {'#':<4} {'Crop':<22} {'Final':>6} "
           f"{'ML%':>5} {'Clim':>5} {'Soil':>5} {'Trait':>6} {'Dom':>5}  Note")
    print(hdr)
    print(f"  {'─'*3} {'─'*21} {'─'*6} "
          f"{'─'*5} {'─'*5} {'─'*5} {'─'*6} {'─'*5}  {'─'*16}")
    for r in active[:top_n]:
        note = r.domain_reason[:18] if r.domain_reason else ""
        print(f"  {r.rank:<4} {r.crop:<22} {r.final_score:>6.3f} "
              f"{r.ml_rank_score:>5.2f} {r.climate_score:>5.2f} "
              f"{r.soil_score:>5.2f} {r.trait_score:>6.3f} "
              f"{r.domain_adj_score:>5.2f}  {note}")

    if show_blocked:
        blocked = [r for r in rankings if r.blocked]
        if blocked:
            print(f"\n🚫 HARD-BLOCKED CROPS:")
            for r in blocked:
                print(f"   ✗ {r.crop:<22}  Reason: {r.block_reason}")

    if user_res:
        print(f"\n{'─'*W}")
        print(f"  🎯  YOUR CROP: {user_res.crop}")
        if user_res.blocked:
            print(f"     STATUS     : 🚫 BLOCKED — {user_res.block_reason}")
            print(f"     ⚠️  NOT recommended under {scenario} conditions.")
        else:
            print(f"     Rank       : {user_res.rank} / {len(active)}")
            print(f"     Final Score: {user_res.final_score:.3f}  {user_res.confidence}")
            print(f"     ML:{user_res.ml_rank_score:.3f}  "
                  f"Clim:{user_res.climate_score:.3f}  "
                  f"Soil:{user_res.soil_score:.3f}  "
                  f"Trait:{user_res.trait_score:.3f}  "
                  f"Domain:{user_res.domain_adj_score:.3f}")
        print(f"{'─'*W}")
    print()


# ============================================================
# 🧬  GENETIC REPORT
# ============================================================

# ── Trait labels for natural-language descriptions ────────────
_TRAIT_LABELS = {
    "drought_resistance": ("drought-resistant",    "drought resistance"),
    "heat_tolerance":     ("heat-tolerant",         "heat tolerance"),
    "flood_tolerance":    ("flood-tolerant",         "flood tolerance"),
    "yield_potential":    ("high-yield",             "yield potential"),
    "pest_resistance":    ("pest-resistant",         "pest resistance"),
    "nutrient_efficiency":("nutrient-efficient",     "nutrient efficiency"),
    "water_requirement":  ("low-water-requirement",  "water efficiency"),
    "soil_adaptability":  ("soil-adaptable",         "soil adaptability"),
}


def _dominant_trait(crop: str, exclude: Optional[set] = None) -> tuple[str, str]:
    traits = {k: v for k, v in TRAIT_DICT.get(crop, {}).items()
              if k in _TRAIT_LABELS}
    if exclude:
        traits = {k: v for k, v in traits.items() if k not in exclude}
    if not traits:
        return "yield_potential", "high-yield"
    best_key = max(traits, key=traits.get)
    return best_key, _TRAIT_LABELS[best_key][0]


def select_parents(
    rankings:         list[CropResult],
    top_n_candidates: int = 10,
    top_pairs:        int = 3,
) -> list[dict]:
    """
    Select the best parent pairs for genetic crossing.

    pair_score = 0.5 × diversity_norm + 0.5 × avg_genetic_fitness
    diversity  = |drought_diff| + |heat_diff| + |pest_diff|
    """
    candidates = [r for r in rankings if not r.blocked][:top_n_candidates]
    if len(candidates) < 2:
        log.warning("  [ParentSelection] Need ≥ 2 active crops.")
        return []

    pairs = []
    for r_a, r_b in combinations(candidates, 2):
        t_a = TRAIT_DICT.get(r_a.crop, {})
        t_b = TRAIT_DICT.get(r_b.crop, {})

        diversity = (
            abs(t_a.get("drought_resistance", 0.5) - t_b.get("drought_resistance", 0.5))
          + abs(t_a.get("heat_tolerance",     0.5) - t_b.get("heat_tolerance",     0.5))
          + abs(t_a.get("pest_resistance",    0.5) - t_b.get("pest_resistance",    0.5))
        )
        div_norm    = float(np.clip(diversity / 3.0, 0.0, 1.0))
        avg_fitness = (r_a.genetic_fitness + r_b.genetic_fitness) / 2.0
        pair_score  = 0.5 * div_norm + 0.5 * avg_fitness

        pairs.append({
            "crop_a":       r_a.crop,
            "crop_b":       r_b.crop,
            "diversity":    round(div_norm,    4),
            "avg_fitness":  round(avg_fitness, 4),
            "pair_score":   round(pair_score,  4),
            "fitness_a":    round(r_a.genetic_fitness, 4),
            "fitness_b":    round(r_b.genetic_fitness, 4),
            "trait_score_a":round(r_a.trait_score, 4),
            "trait_score_b":round(r_b.trait_score, 4),
            "traits_a":     t_a,
            "traits_b":     t_b,
        })

    pairs.sort(key=lambda p: p["pair_score"], reverse=True)
    return pairs[:top_pairs]


def generate_breeding_strategy(pair: dict) -> str:
    """Generate a human-readable crossing recommendation for one pair."""
    crop_a, crop_b = pair["crop_a"], pair["crop_b"]
    t_a, t_b       = pair["traits_a"], pair["traits_b"]

    key_a, adj_a = _dominant_trait(crop_a)
    key_b, adj_b = _dominant_trait(crop_b, exclude={key_a})

    val_a = t_a.get(key_a, 0.5)
    val_b = t_b.get(key_b, 0.5)

    lab_a = _TRAIT_LABELS[key_a][1]
    lab_b = _TRAIT_LABELS[key_b][1]

    hy_yield = min(1.0,
        (t_a.get("yield_potential", 0.5) + t_b.get("yield_potential", 0.5)) / 2.0
        + pair["diversity"] * 0.10           # heterosis bonus
    )
    exp_fitness = min(1.0, pair["avg_fitness"] + pair["diversity"] * 0.05)

    return (
        f"Cross {crop_a} (high {lab_a}, score {val_a:.2f}) "
        f"with {crop_b} (high {lab_b}, score {val_b:.2f}) "
        f"to produce a {adj_a} × {adj_b} high-yield hybrid.\n"
        f"   Predicted hybrid yield potential : {hy_yield:.2f}\n"
        f"   Expected hybrid fitness          : {exp_fitness:.2f}\n"
        f"   Genetic diversity index          : {pair['diversity']:.2f}"
    )


def print_genetic_report(
    result:  dict,
    top_n:   int  = 10,
    n_pairs: int  = 3,
    show_blocked: bool = True,
):
    """
    Print the genetic-enhanced addendum.

    IMPORTANT: result["rankings"] must already contain CropResult
    objects with trait_score and genetic_fitness populated
    (i.e. produced by predict() which calls hybrid_score()
    which includes the genetic layer inline).
    """
    rankings = result["rankings"]
    best     = result["best_crop"]
    user_res = result["user_crop_result"]
    scenario = result["scenario"]
    climate  = result.get("climate")
    soil     = result.get("soil")

    W = 70
    print(f"\n{'═'*W}")
    print("  🧬  GENETIC TRAIT & BREEDING INTELLIGENCE REPORT")
    print(f"  State: {result['state']}  |  Scenario: {scenario.upper()}")
    print(f"{'═'*W}")

    if climate:
        print(f"\n📡 Climate : {climate}")
    if soil:
        print(f"🪨  Soil    : {soil.soil_type.title()}")

    print(f"\n⚖️  Five-Component Score Formula:")
    print(f"   0.30×ML_Rank + 0.25×Climate + 0.20×Soil + 0.15×Trait + 0.10×Domain")

    # ── Best crop ─────────────────────────────────────────────
    if best:
        print(f"\n{'─'*W}")
        print(f"  🏆  BEST CROP: {best.crop}")
        print(f"     Final Score     : {best.final_score:.3f}  {best.confidence}")
        print(f"     Trait Score     : {best.trait_score:.3f}")
        print(f"     Genetic Fitness : {best.genetic_fitness:.3f}")
        print(f"     ML:{best.ml_rank_score:.3f}  Clim:{best.climate_score:.3f}  "
              f"Soil:{best.soil_score:.3f}  Domain:{best.domain_adj_score:.3f}")
        print(f"{'─'*W}")

    # ── Rankings table ────────────────────────────────────────
    active  = [r for r in rankings if not r.blocked]
    blocked = [r for r in rankings if r.blocked]

    print(f"\n📊 TOP {min(top_n, len(active))} CROPS — Genetic-Enhanced Ranking:")
    hdr = (f"  {'#':<4} {'Crop':<22} {'Final':>6} "
           f"{'Trait':>6} {'GFit':>6} {'ML%':>5} {'Clim':>5} {'Soil':>5}  Confidence")
    print(hdr)
    print(f"  {'─'*3} {'─'*21} {'─'*6} "
          f"{'─'*6} {'─'*6} {'─'*5} {'─'*5} {'─'*5}  {'─'*14}")
    for r in active[:top_n]:
        print(
            f"  {r.rank:<4} {r.crop:<22} {r.final_score:>6.3f} "
            f"{r.trait_score:>6.3f} {r.genetic_fitness:>6.3f} "
            f"{r.ml_rank_score:>5.2f} {r.climate_score:>5.2f} "
            f"{r.soil_score:>5.2f}  {r.confidence}"
        )

    if show_blocked and blocked:
        print(f"\n🚫 HARD-BLOCKED CROPS:")
        for r in blocked:
            print(f"   ✗ {r.crop:<22}  "
                  f"Trait={r.trait_score:.3f}  Reason: {r.block_reason}")

    # ── User crop ─────────────────────────────────────────────
    if user_res:
        print(f"\n{'─'*W}")
        print(f"  🎯  YOUR CROP: {user_res.crop}")
        if user_res.blocked:
            print(f"     STATUS          : 🚫 BLOCKED — {user_res.block_reason}")
            print(f"     Trait Score     : {user_res.trait_score:.3f}")
            print(f"     Genetic Fitness : {user_res.genetic_fitness:.3f}")
            print(f"     ⚠️  Genetically capable but agronomically blocked "
                  f"under {scenario}.")
        else:
            print(f"     Rank            : {user_res.rank} / {len(active)}")
            print(f"     Final Score     : {user_res.final_score:.3f}  {user_res.confidence}")
            print(f"     Trait Score     : {user_res.trait_score:.3f}")
            print(f"     Genetic Fitness : {user_res.genetic_fitness:.3f}")
        print(f"{'─'*W}")

    # ── Parent selection ──────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  🧬  PARENT SELECTION FOR BREEDING")
    print(f"{'═'*W}")

    pairs = select_parents(rankings, top_n_candidates=10, top_pairs=n_pairs)

    if not pairs:
        print("  ⚠️  Not enough active crops for parent selection.")
    else:
        for idx, pair in enumerate(pairs, 1):
            print(f"\n  Pair #{idx}  ──  {pair['crop_a']}  ×  {pair['crop_b']}")
            print(f"  {'─'*40}")
            print(f"   Pair Score        : {pair['pair_score']:.4f}")
            print(f"   Genetic Diversity : {pair['diversity']:.4f}")
            print(f"   Avg Fitness       : {pair['avg_fitness']:.4f}")
            print(f"   Fitness ({pair['crop_a'][:12]:<12}): "
                  f"{pair['fitness_a']:.4f}  Trait={pair['trait_score_a']:.4f}")
            print(f"   Fitness ({pair['crop_b'][:12]:<12}): "
                  f"{pair['fitness_b']:.4f}  Trait={pair['trait_score_b']:.4f}")

    # ── Breeding strategies ───────────────────────────────────
    print(f"\n{'═'*W}")
    print("  🌱  BREEDING RECOMMENDATIONS")
    print(f"{'═'*W}")

    if not pairs:
        print("  ⚠️  No pairs available.")
    else:
        for idx, pair in enumerate(pairs, 1):
            strategy = generate_breeding_strategy(pair)
            print(f"\n  Strategy #{idx}:")
            for line in strategy.split("\n"):
                print(f"   {line}")

    print(f"\n{'═'*W}\n")


# ============================================================
# 💾  RESULTS STORAGE & CSV EXPORT
# ============================================================

def store_result_summary(result: dict) -> dict:
    """
    Extract a flat summary row from one predict() result.
    Suitable for building a results table and exporting to CSV.
    """
    best = result["best_crop"]
    top5 = [r for r in result["rankings"] if not r.blocked][:5]

    return {
        "state":                result["state"],
        "season":               result["season"],
        "scenario":             result["scenario"],
        "soil_type":            result["soil"].soil_type,
        "temperature_c":        round(result["climate"].temperature, 2),
        "rainfall_mm_day":      round(result["climate"].rainfall,    2),
        "humidity_pct":         round(result["climate"].humidity,    2),
        "best_crop":            best.crop,
        "best_final_score":     round(best.final_score,      4),
        "best_ml_rank":         round(best.ml_rank_score,    4),
        "best_climate_score":   round(best.climate_score,    4),
        "best_soil_score":      round(best.soil_score,       4),
        "best_trait_score":     round(best.trait_score,      4),
        "best_domain_score":    round(best.domain_adj_score, 4),
        "best_genetic_fitness": round(best.genetic_fitness,  4),
        "best_confidence":      best.confidence,
        "top5_crops":           " | ".join(r.crop for r in top5),
    }


def export_results_csv(
    summaries: list[dict],
    output_path: str = "scenario_results.csv",
):
    """Write list of summary dicts to a CSV file."""
    if not summaries:
        log.warning("No summaries to export.")
        return

    fieldnames = list(summaries[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    log.info(f"  ✅ Results exported → {output_path}  ({len(summaries)} rows)")


# ============================================================
# 🧪  MAIN — COMPLETE TEST SUITE
# ============================================================

if __name__ == "__main__":

    # ── 0. Load trait dictionary ─────────────────────────────
    TRAIT_DICT.update(load_trait_dict(cfg.trait_csv_path))

    # ── 1. Load production data & train model ────────────────
    df    = load_and_engineer(cfg.data_path)
    model = CropModel().train(df)
    with open("crop_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model saved as crop_model.pkl")

    log.info(f"\n📈 Model performance: {model.metrics}")

    # ── 2. Define the 3 fixed test cases (corrected) ─────────
    #
    #  BUG FIX #1 / #4 / #5:  Each test uses its OWN result
    #  variable.  Labels match the exact parameters used.
    #  BUG FIX #2:  "heatwaves" → "heatwave"
    #  BUG FIX #3:  "dry" → valid soil type
    #  BUG FIX #7:  validate_* functions raise ValueError on
    #               bad input instead of silently falling back.
    #  BUG FIX #8:  hybrid_score() integrates genetic layer
    #               inline, so result dicts are always complete
    #               before any report function is called.

    # ── Test 1: Rajasthan | Heatwave | Sandy ──────────────────
    print("\n" + "=" * 68)
    print("TEST 1 — Rajasthan | Heatwave | Sandy soil | User: Sugarcane")
    print("=" * 68)
    result1 = predict(
        crop_model    = model,
        lat           = 27.0238,    # Jaipur
        lon           = 74.2179,
        scenario      = "heatwave", # ← was "heatwaves" (BUG #2)
        soil_type     = "sandy",    # ← was "dry" (BUG #3)
        state         = "Rajasthan",
        season        = "Kharif",
        user_crop     = "Sugarcane",
        year          = 2024,
        fetch_weather = True,
    )
    print_report(result1, top_n=10)          # ← uses result1 (BUG #1/#4)
    print_genetic_report(result1, top_n=10)  # ← uses result1 (BUG #1/#4)

    # ── Test 2: Maharashtra | Drought | Loamy ─────────────────
    print("=" * 68)
    print("TEST 2 — Maharashtra | Drought | Loamy soil | User: Rice")
    print("=" * 68)
    result2 = predict(
        crop_model    = model,
        lat           = 21.1458,    # Nagpur
        lon           = 79.0882,
        scenario      = "drought",
        soil_type     = "loamy",
        state         = "Maharashtra",
        season        = "Kharif",
        user_crop     = "Rice",
        year          = 2024,
        fetch_weather = True,
    )
    print_report(result2, top_n=10)          # ← uses result2 (BUG #1/#4)
    print_genetic_report(result2, top_n=10)  # ← uses result2 (BUG #1/#4)

    # ── Test 3: West Bengal | Flood | Clay ────────────────────
    print("=" * 68)
    print("TEST 3 — West Bengal | Flood | Clay soil | User: Wheat")
    print("=" * 68)
    result3 = predict(
        crop_model    = model,
        lat           = 22.9868,    # Kolkata
        lon           = 87.8550,
        scenario      = "flood",
        soil_type     = "clay",
        state         = "West Bengal",
        season        = "Kharif",
        user_crop     = "Wheat",
        year          = 2024,
        fetch_weather = True,
    )
    print_report(result3, top_n=10)          # ← uses result3 (BUG #1/#4)
    print_genetic_report(result3, top_n=10)  # ← uses result3 (BUG #1/#4)

    # ── 3. Multi-scenario loop: 3 states × 3 scenarios = 9 ───
    #  BUG FIX #6: Implements the required loop-based testing.
    print("\n" + "=" * 68)
    print("  🔁  MULTI-SCENARIO MATRIX  (3 states × 3 scenarios)")
    print("=" * 68)

    TEST_MATRIX = [
        # (label,           lat,      lon,      state,         soil,    season)
        ("Rajasthan",     27.0238, 74.2179, "Rajasthan",    "sandy",  "Kharif"),
        ("Maharashtra",   21.1458, 79.0882, "Maharashtra",  "loamy",  "Kharif"),
        ("West Bengal",   22.9868, 87.8550, "West Bengal",  "clay",   "Kharif"),
    ]

    MULTI_SCENARIOS = ["drought", "heatwave", "flood"]  # all valid — BUG #2 prevented

    all_summaries: list[dict] = []

    for label, lat, lon, state, soil_type, season in TEST_MATRIX:
        for scenario in MULTI_SCENARIOS:
            print(f"\n{'─'*68}")
            print(f"  ▶  {label.upper()} | {scenario.upper()} | {soil_type.upper()} soil")
            print(f"{'─'*68}")

            # validate_scenario / validate_soil_type already called inside predict()
            result = predict(
                crop_model    = model,
                lat           = lat,
                lon           = lon,
                scenario      = scenario,   # always valid — from MULTI_SCENARIOS
                soil_type     = soil_type,  # always valid — from TEST_MATRIX
                state         = state,
                season        = season,
                year          = 2024,
                fetch_weather = True,
            )

            # Compact summary to console
            best = result["best_crop"]
            top3 = [r for r in result["rankings"] if not r.blocked][:3]
            top3_str = " > ".join(
                f"{r.crop}({r.final_score:.3f})" for r in top3
            )
            print(f"  ✅ Best   : {best.crop}  [{best.confidence}]  "
                  f"Score={best.final_score:.3f}")
            print(f"  📊 Top 3  : {top3_str}")
            print(f"  🌡  Climate: {result['climate']}")

            # Full genetic report for this scenario
            print_genetic_report(result, top_n=5, n_pairs=2)

            # Collect for CSV export
            all_summaries.append(store_result_summary(result))

    # ── 4. Export results to CSV ──────────────────────────────
    export_results_csv(all_summaries, "scenario_results.csv")

    # ── 5. Print summary table ────────────────────────────────
    if all_summaries:
        df_summary = pd.DataFrame(all_summaries)
        print(f"\n{'='*68}")
        print("  📋  SCENARIO RESULTS SUMMARY")
        print(f"{'='*68}")
        cols = ["state", "scenario", "soil_type", "best_crop",
                "best_final_score", "best_trait_score", "best_genetic_fitness",
                "best_confidence"]
        pd.set_option("display.width", 140)
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.max_rows", 20)
        print(df_summary[cols].to_string(index=False))
        print(f"\n  → Full data saved to: scenario_results.csv")

    print(f"\n{'='*68}")
    print("  ✅  ALL TESTS COMPLETE")
    print(f"{'='*68}\n")