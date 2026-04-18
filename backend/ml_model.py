import math
import os
import pickle

import numpy as np
import pandas as pd


MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "crop_model.pkl")


class CropModel:
    """Compatibility class for pickle files saved from __main__.CropModel."""

    pass


class CropModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "CropModel":
            return CropModel
        return super().find_class(module, name)


CROP_ALIASES = {
    "bajra": "Bajra",
    "jowar": "Jowar",
    "corn": "Maize",
    "maize": "Maize",
    "onion": "Onion",
    "soybean": "Soyabean",
    "soyabean": "Soyabean",
    "cotton": "Cotton(lint)",
    "rice": "Rice",
    "sugarcane": "Sugarcane",
    "wheat": "Wheat",
}


SOIL_PROFILES = {
    "sandy": {"fertility": 0.35, "drainage": 0.95, "water_retention": 0.15},
    "loamy": {"fertility": 0.92, "drainage": 0.70, "water_retention": 0.72},
    "clay": {"fertility": 0.80, "drainage": 0.25, "water_retention": 0.90},
    "black": {"fertility": 0.86, "drainage": 0.55, "water_retention": 0.82},
    "red": {"fertility": 0.55, "drainage": 0.78, "water_retention": 0.42},
}

SCENARIO_CLIMATE = {
    "normal": {"temperature": 27.0, "rainfall_mm_day": 3.5, "humidity": 60.0},
    "drought": {"temperature": 28.2, "rainfall_mm_day": 1.5, "humidity": 38.0},
    "heatwave": {"temperature": 31.0, "rainfall_mm_day": 2.0, "humidity": 45.0},
    "flood": {"temperature": 27.0, "rainfall_mm_day": 7.0, "humidity": 84.0},
}

TRAIT_BASE = {
    "Bajra": {"drought": 0.95, "heat": 0.92, "flood": 0.25, "water": 0.20, "soil": 0.80, "yield": 0.50},
    "Jowar": {"drought": 0.90, "heat": 0.88, "flood": 0.30, "water": 0.25, "soil": 0.72, "yield": 0.55},
    "Rice": {"drought": 0.15, "heat": 0.45, "flood": 0.95, "water": 0.85, "soil": 0.80, "yield": 0.70},
    "Wheat": {"drought": 0.35, "heat": 0.25, "flood": 0.10, "water": 0.35, "soil": 0.65, "yield": 0.65},
    "Sugarcane": {"drought": 0.05, "heat": 0.55, "flood": 0.80, "water": 0.95, "soil": 0.70, "yield": 0.90},
    "Maize": {"drought": 0.65, "heat": 0.68, "flood": 0.45, "water": 0.45, "soil": 0.75, "yield": 0.75},
    "Onion": {"drought": 0.70, "heat": 0.50, "flood": 0.35, "water": 0.35, "soil": 0.80, "yield": 0.70},
    "Jute": {"drought": 0.20, "heat": 0.45, "flood": 0.92, "water": 0.82, "soil": 0.68, "yield": 0.60},
    "Banana": {"drought": 0.35, "heat": 0.65, "flood": 0.75, "water": 0.90, "soil": 0.72, "yield": 0.80},
    "Potato": {"drought": 0.35, "heat": 0.15, "flood": 0.10, "water": 0.55, "soil": 0.70, "yield": 0.75},
    "Groundnut": {"drought": 0.70, "heat": 0.55, "flood": 0.10, "water": 0.25, "soil": 0.65, "yield": 0.65},
    "Cabbage": {"drought": 0.45, "heat": 0.45, "flood": 0.55, "water": 0.55, "soil": 0.80, "yield": 0.62},
    "Cauliflower": {"drought": 0.42, "heat": 0.45, "flood": 0.55, "water": 0.55, "soil": 0.80, "yield": 0.60},
    "Papaya": {"drought": 0.50, "heat": 0.62, "flood": 0.45, "water": 0.55, "soil": 0.65, "yield": 0.85},
    "Grapes": {"drought": 0.50, "heat": 0.60, "flood": 0.35, "water": 0.45, "soil": 0.62, "yield": 0.80},
    "Pineapple": {"drought": 0.45, "heat": 0.58, "flood": 0.55, "water": 0.65, "soil": 0.65, "yield": 0.80},
    "Tapioca": {"drought": 0.55, "heat": 0.55, "flood": 0.45, "water": 0.45, "soil": 0.60, "yield": 0.65},
}

SCENARIO_NOTES = {
    ("drought", "Bajra"): "Excellent drought tolerance",
    ("drought", "Jowar"): "Very drought-tolerant",
    ("heatwave", "Bajra"): "Thrives in extreme heat",
    ("heatwave", "Jowar"): "C4 crop, heat-adapted",
    ("flood", "Rice"): "Paddy is flood-tolerant",
    ("flood", "Jute"): "Thrives in high moisture",
}

HARD_BLOCKS = {
    "drought": {"Sugarcane": "Extremely water-intensive"},
    "heatwave": {
        "Potato": "Tuber formation fails above 28°C",
        "Wheat": "Heat-sensitive - terminal heat damage",
    },
    "flood": {
        "Groundnut": "Pod rot in waterlogged soil",
        "Potato": "Tuber rot in waterlogged soil",
        "Wheat": "Waterlogging lethal",
    },
}


def load_crop_model():
    with open(MODEL_PATH, "rb") as file:
        artifact = CropModelUnpickler(file).load()
    return _prepare_crop_model(artifact)


def _clean(value):
    return str(value or "").strip()


def _normalize(value):
    return _clean(value).lower()


def _prepare_crop_model(artifact):
    if getattr(artifact, "_runtime_ready", False):
        return artifact

    df_ref = artifact._df_ref
    encoders = artifact.label_encoders

    artifact._feature_cols = list(artifact.feature_cols)
    artifact._crop_classes = list(encoders["Crop"].classes_)
    artifact._label_lookup = {
        name: {_normalize(label): label for label in encoder.classes_}
        for name, encoder in encoders.items()
    }
    artifact._encoded_lookup = {}
    for name, encoder in encoders.items():
        classes = list(encoder.classes_)
        encoded = encoder.transform(classes)
        artifact._encoded_lookup[name] = {
            label: int(code) for label, code in zip(classes, encoded)
        }

    artifact._year_min = int(df_ref["Crop_Year"].min())
    artifact._year_max = int(df_ref["Crop_Year"].max())
    artifact._ref_lookup = {}
    for column in ("Crop", "State_Name", "Season"):
        normalized = df_ref[column].astype(str).str.strip().str.lower()
        first_rows = normalized.drop_duplicates(keep="first")
        artifact._ref_lookup[column] = {
            label: df_ref.loc[index]
            for index, label in first_rows.items()
            if label
        }

    artifact._runtime_ready = True
    return artifact


def _positive_float(value, field_name):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc

    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than 0.")
    return parsed


def _resolve_label(value, classes, field_name, aliases=None, lookup=None):
    raw = _clean(value)
    if not raw:
        raise ValueError(f"{field_name} is required for ML prediction.")

    normalized = _normalize(raw)
    if aliases and normalized in aliases:
        raw = aliases[normalized]
        normalized = _normalize(raw)

    if lookup:
        match = lookup.get(normalized)
        if match is not None:
            return match

    for item in classes:
        if _normalize(item) == normalized:
            return item

    allowed = ", ".join(_clean(item) for item in list(classes)[:8])
    raise ValueError(f"Unsupported {field_name}: {raw}. Try one of: {allowed}.")


def _first_matching_row(df, column, label):
    matches = df[df[column].astype(str).str.strip().str.lower() == _normalize(label)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _get_ref_row(artifact, column, label):
    lookup = getattr(artifact, "_ref_lookup", {}).get(column)
    if lookup:
        row = lookup.get(_normalize(label))
        if row is not None:
            return row
    return _first_matching_row(artifact._df_ref, column, label)


def _year_norm_for_artifact(artifact, crop_year):
    if hasattr(artifact, "_year_min") and hasattr(artifact, "_year_max"):
        min_year = artifact._year_min
        max_year = artifact._year_max
    else:
        min_year = int(artifact._df_ref["Crop_Year"].min())
        max_year = int(artifact._df_ref["Crop_Year"].max())
    year = min(max(int(crop_year), min_year), max_year)
    if max_year == min_year:
        return 0.0
    return (year - min_year) / (max_year - min_year)


def _encoded_value(artifact, encoder_name, value):
    lookup = getattr(artifact, "_encoded_lookup", {}).get(encoder_name)
    if lookup and value in lookup:
        return lookup[value]
    return artifact.label_encoders[encoder_name].transform([value])[0]


def _predict_log_yields(artifact, feature_frame):
    ordered = feature_frame[artifact._feature_cols]
    get_booster = getattr(artifact.model, "get_booster", None)
    if callable(get_booster):
        try:
            values = ordered.to_numpy(dtype=np.float32, copy=False)
            return get_booster().inplace_predict(values)
        except Exception:
            pass
    return artifact.model.predict(ordered)


def predict_crop_yield(artifact, data):
    artifact = _prepare_crop_model(artifact)
    df_ref = artifact._df_ref
    encoders = artifact.label_encoders
    label_lookup = artifact._label_lookup

    crop = _resolve_label(
        data.get("crop_type"),
        encoders["Crop"].classes_,
        "crop",
        CROP_ALIASES,
        label_lookup.get("Crop"),
    )
    season = _resolve_label(
        data.get("season", "Kharif"),
        encoders["Season"].classes_,
        "season",
        lookup=label_lookup.get("Season"),
    )
    state = _resolve_label(
        data.get("state") or data.get("location"),
        encoders["State_Name"].classes_,
        "state",
        lookup=label_lookup.get("State_Name"),
    )

    area = _positive_float(data.get("area"), "Area")
    crop_year = int(float(data.get("crop_year", df_ref["Crop_Year"].max())))

    crop_row = _get_ref_row(artifact, "Crop", crop)
    state_row = _get_ref_row(artifact, "State_Name", state)
    season_row = _get_ref_row(artifact, "Season", season)

    if crop_row is None or state_row is None or season_row is None:
        raise ValueError("The selected crop, state, or season is not available in the model data.")

    feature_values = {
        "Crop_enc": _encoded_value(artifact, "Crop", crop),
        "Season_enc": _encoded_value(artifact, "Season", season),
        "State_Name_enc": _encoded_value(artifact, "State_Name", state),
        "Log_Area": math.log1p(area),
        "Year_Norm": _year_norm_for_artifact(artifact, crop_year),
        "Crop_Mean_Yield": crop_row["Crop_Mean_Yield"],
        "Crop_Std_Yield": crop_row["Crop_Std_Yield"],
        "Crop_Median_Yield": crop_row["Crop_Median_Yield"],
        "State_Mean_Yield": state_row["State_Mean_Yield"],
        "Season_Rank": season_row["Season_Rank"],
    }

    model_input = pd.DataFrame([{col: feature_values[col] for col in artifact._feature_cols}])
    predicted_log_yield = float(_predict_log_yields(artifact, model_input)[0])
    predicted_yield = max(0.0, float(np.expm1(predicted_log_yield)))

    return {
        "predicted_yield": round(predicted_yield, 2),
        "unit": "yield per hectare",
        "crop": _clean(crop),
        "state": _clean(state),
        "season": _clean(season),
        "crop_year": crop_year,
        "metrics": getattr(artifact, "metrics", {}),
    }


def prediction_from_report(artifact, crop_report):
    user_crop = (crop_report or {}).get("user_crop")
    if not user_crop:
        return {"error": "The selected crop is not available in the ranked crop report."}

    return {
        "predicted_yield": user_crop.get("predicted_yield"),
        "unit": "yield per hectare",
        "crop": user_crop.get("crop"),
        "state": crop_report.get("state"),
        "season": crop_report.get("season"),
        "crop_year": crop_report.get("crop_year"),
        "metrics": getattr(artifact, "metrics", {}),
    }


def _scenario_from_data(data):
    return _clean(data.get("scenario") or "normal").lower()


def _soil_from_data(data):
    return _clean(data.get("soil_type") or "loamy").lower()


def _climate_for_report(weather, scenario):
    base = SCENARIO_CLIMATE.get(scenario, SCENARIO_CLIMATE["normal"]).copy()
    if weather:
        if weather.get("temperature") is not None:
            base["temperature"] = float(weather["temperature"])
        if weather.get("humidity") is not None:
            base["humidity"] = float(weather["humidity"])
    return base


def _trait_profile(crop):
    return TRAIT_BASE.get(
        _clean(crop),
        {"drought": 0.50, "heat": 0.50, "flood": 0.50, "water": 0.50, "soil": 0.50, "yield": 0.50},
    )


def _trait_score(crop, scenario):
    trait = _trait_profile(crop)
    if scenario == "drought":
        return 0.65 * trait["drought"] + 0.20 * (1 - trait["water"]) + 0.15 * trait["soil"]
    if scenario == "heatwave":
        return 0.65 * trait["heat"] + 0.20 * trait["drought"] + 0.15 * trait["soil"]
    if scenario == "flood":
        return 0.60 * trait["flood"] + 0.25 * trait["water"] + 0.15 * trait["soil"]
    return 0.35 * trait["yield"] + 0.25 * trait["soil"] + 0.20 * trait["drought"] + 0.20 * trait["heat"]


def _climate_score(crop, scenario, climate):
    trait = _trait_profile(crop)
    if scenario == "drought":
        return min(1.0, 0.65 * trait["drought"] + 0.25 * (1 - trait["water"]) + 0.10)
    if scenario == "heatwave":
        heat_penalty = 0.15 if climate["temperature"] > 32 else 0.0
        return max(0.0, min(1.0, 0.70 * trait["heat"] + 0.20 * trait["drought"] + 0.10 - heat_penalty))
    if scenario == "flood":
        return min(1.0, 0.70 * trait["flood"] + 0.20 * trait["water"] + 0.10)
    return min(1.0, 0.45 * trait["heat"] + 0.35 * trait["drought"] + 0.20 * trait["flood"])


def _soil_score(crop, soil_type):
    profile = SOIL_PROFILES.get(soil_type, SOIL_PROFILES["loamy"])
    trait = _trait_profile(crop)
    drainage_need = 1 - trait["water"]
    retention_need = trait["water"]
    return max(
        0.0,
        min(
            1.0,
            0.40 * (1 - abs(profile["drainage"] - drainage_need))
            + 0.35 * (1 - abs(profile["water_retention"] - retention_need))
            + 0.25 * profile["fertility"],
        ),
    )


def _domain_score(crop, scenario):
    crop_name = _clean(crop)
    strong = {
        "drought": {"Bajra": 1.0, "Jowar": 0.95, "Urad": 0.75, "Maize": 0.70},
        "heatwave": {"Bajra": 1.0, "Jowar": 0.95, "Maize": 0.75},
        "flood": {"Rice": 1.0, "Jute": 0.90, "Sugarcane": 0.70, "Banana": 0.65},
        "normal": {},
    }
    weak = {
        "drought": {"Rice": 0.0, "Sugarcane": 0.0},
        "heatwave": {"Wheat": 0.0, "Potato": 0.0},
        "flood": {"Wheat": 0.0, "Potato": 0.0, "Groundnut": 0.0},
        "normal": {},
    }
    if crop_name in strong.get(scenario, {}):
        return strong[scenario][crop_name]
    if crop_name in weak.get(scenario, {}):
        return weak[scenario][crop_name]
    return 0.50


def _confidence(score):
    if score >= 0.75:
        return "Excellent"
    if score >= 0.60:
        return "Good"
    if score >= 0.40:
        return "Marginal"
    return "Poor"


def _encode_value(encoder, value):
    return encoder.transform([value])[0]


def _build_feature_frame_for_crops(artifact, crops, state, season, area, crop_year):
    artifact = _prepare_crop_model(artifact)
    state_row = _get_ref_row(artifact, "State_Name", state)
    season_row = _get_ref_row(artifact, "Season", season)
    if state_row is None or season_row is None:
        raise ValueError("The selected state or season is not available in the model data.")

    state_enc = _encoded_value(artifact, "State_Name", state)
    season_enc = _encoded_value(artifact, "Season", season)
    log_area = math.log1p(area)
    year_norm = _year_norm_for_artifact(artifact, crop_year)
    rows = []

    for crop in crops:
        crop_row = _get_ref_row(artifact, "Crop", crop)
        if crop_row is None:
            continue
        values = {
            "Crop_enc": _encoded_value(artifact, "Crop", crop),
            "Season_enc": season_enc,
            "State_Name_enc": state_enc,
            "Log_Area": log_area,
            "Year_Norm": year_norm,
            "Crop_Mean_Yield": crop_row["Crop_Mean_Yield"],
            "Crop_Std_Yield": crop_row["Crop_Std_Yield"],
            "Crop_Median_Yield": crop_row["Crop_Median_Yield"],
            "State_Mean_Yield": state_row["State_Mean_Yield"],
            "Season_Rank": season_row["Season_Rank"],
            "_crop": _clean(crop),
        }
        rows.append(values)

    return pd.DataFrame(rows)


def generate_crop_report(artifact, data, weather=None):
    artifact = _prepare_crop_model(artifact)
    df_ref = artifact._df_ref
    encoders = artifact.label_encoders
    label_lookup = artifact._label_lookup
    scenario = _scenario_from_data(data)
    soil_type = _soil_from_data(data)
    climate = _climate_for_report(weather, scenario)
    soil_profile = SOIL_PROFILES.get(soil_type, SOIL_PROFILES["loamy"])

    user_crop = _resolve_label(
        data.get("crop_type"),
        encoders["Crop"].classes_,
        "crop",
        CROP_ALIASES,
        label_lookup.get("Crop"),
    )
    state = _resolve_label(
        data.get("state") or data.get("location"),
        encoders["State_Name"].classes_,
        "state",
        lookup=label_lookup.get("State_Name"),
    )
    season = _resolve_label(
        data.get("season", "Kharif"),
        encoders["Season"].classes_,
        "season",
        lookup=label_lookup.get("Season"),
    )
    area = _positive_float(data.get("area"), "Area")
    crop_year = int(float(data.get("crop_year", df_ref["Crop_Year"].max())))

    crops = artifact._crop_classes
    feature_df = _build_feature_frame_for_crops(artifact, crops, state, season, area, crop_year)
    if feature_df.empty:
        raise ValueError("No crop rows are available for the selected model data.")

    predicted_yields = np.expm1(_predict_log_yields(artifact, feature_df))
    min_yield = float(np.min(predicted_yields))
    max_yield = float(np.max(predicted_yields))
    span = max(max_yield - min_yield, 1e-9)

    scored = []
    blocked_reasons = HARD_BLOCKS.get(scenario, {})
    for crop, predicted_yield in zip(feature_df["_crop"], predicted_yields):
        ml_rank = (float(predicted_yield) - min_yield) / span
        climate_score = _climate_score(crop, scenario, climate)
        soil_score = _soil_score(crop, soil_type)
        trait_score = _trait_score(crop, scenario)
        domain_score = _domain_score(crop, scenario)
        final_score = (
            0.30 * ml_rank
            + 0.25 * climate_score
            + 0.20 * soil_score
            + 0.15 * trait_score
            + 0.10 * domain_score
        )
        hard_block_reason = blocked_reasons.get(crop)
        if hard_block_reason:
            final_score = min(final_score, 0.20)

        genetic_fitness = 0.55 * trait_score + 0.25 * ml_rank + 0.20 * soil_score
        scored.append({
            "crop": crop,
            "predicted_yield": round(float(predicted_yield), 2),
            "final_score": round(float(final_score), 3),
            "ml_rank": round(float(ml_rank), 3),
            "climate": round(float(climate_score), 3),
            "soil": round(float(soil_score), 3),
            "trait": round(float(trait_score), 3),
            "domain": round(float(domain_score), 3),
            "genetic_fitness": round(float(genetic_fitness), 3),
            "confidence": _confidence(final_score),
            "note": SCENARIO_NOTES.get((scenario, crop), ""),
            "blocked_reason": hard_block_reason,
        })

    ranked = sorted(scored, key=lambda row: row["final_score"], reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index

    top_crops = ranked[:10]
    best_crop = top_crops[0] if top_crops else None
    blocked = [row for row in ranked if row["blocked_reason"]]
    user_entry = next((row for row in ranked if row["crop"].lower() == _clean(user_crop).lower()), None)

    parents_pool = [row for row in ranked[:12] if not row["blocked_reason"]]
    parent_pairs = []
    for i, first in enumerate(parents_pool):
        for second in parents_pool[i + 1:]:
            diversity = min(1.0, abs(first["trait"] - second["trait"]) + 0.20)
            avg_fitness = (first["genetic_fitness"] + second["genetic_fitness"]) / 2
            pair_score = 0.55 * avg_fitness + 0.45 * diversity
            parent_pairs.append({
                "parents": [first["crop"], second["crop"]],
                "pair_score": round(pair_score, 3),
                "genetic_diversity": round(diversity, 3),
                "avg_fitness": round(avg_fitness, 3),
                "expected_hybrid_fitness": round(min(1.0, avg_fitness + 0.04), 3),
                "predicted_hybrid_yield": round((first["predicted_yield"] + second["predicted_yield"]) / 2, 2),
            })

    parent_pairs = sorted(parent_pairs, key=lambda row: row["pair_score"], reverse=True)[:3]
    breeding_recommendations = [
        {
            "strategy": index,
            "text": (
                f"Cross {pair['parents'][0]} with {pair['parents'][1]} to combine "
                f"scenario fitness with complementary trait diversity."
            ),
            "predicted_hybrid_yield": pair["predicted_hybrid_yield"],
            "expected_hybrid_fitness": pair["expected_hybrid_fitness"],
            "genetic_diversity": pair["genetic_diversity"],
        }
        for index, pair in enumerate(parent_pairs, start=1)
    ]

    return {
        "scenario": scenario,
        "state": _clean(state),
        "season": _clean(season),
        "crop_year": crop_year,
        "soil_type": soil_type,
        "climate": {
            "temperature": round(climate["temperature"], 1),
            "rainfall_mm_day": round(climate["rainfall_mm_day"], 2),
            "humidity": round(climate["humidity"], 1),
            "source": f"OpenWeather + scenario profile [{scenario}]",
        },
        "soil_profile": {
            "fertility": round(soil_profile["fertility"], 2),
            "drainage": round(soil_profile["drainage"], 2),
            "water_retention": round(soil_profile["water_retention"], 2),
        },
        "weights": {"ml_rank": 0.30, "climate": 0.25, "soil": 0.20, "trait": 0.15, "domain": 0.10},
        "best_crop": best_crop,
        "top_crops": top_crops,
        "blocked_crops": blocked[:8],
        "user_crop": user_entry,
        "parent_pairs": parent_pairs,
        "breeding_recommendations": breeding_recommendations,
    }
