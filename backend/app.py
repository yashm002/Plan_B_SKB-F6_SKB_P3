from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
import os
import requests
import sys

from location_data import CITY_COORDS_BY_STATE
from ml_model import generate_crop_report, load_crop_model, prediction_from_report

# Load env
load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ENABLE_AI_RECOMMENDATION = os.getenv("ENABLE_AI_RECOMMENDATION", "true").lower() == "true"
AI_RECOMMENDATION_TIMEOUT = float(os.getenv("AI_RECOMMENDATION_TIMEOUT", "3"))
WEATHER_REQUEST_TIMEOUT = float(os.getenv("WEATHER_REQUEST_TIMEOUT", "4"))
WEATHER_CACHE_TTL_SECONDS = max(60, int(os.getenv("WEATHER_CACHE_TTL_SECONDS", "900")))
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://localhost:3001",
    ).split(",")
    if origin.strip()
]


# Gemini config
gemini_model = None
if ENABLE_AI_RECOMMENDATION:
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

try:
    crop_model_artifact = load_crop_model()
    crop_model_error = None
except Exception as e:
    crop_model_artifact = None
    crop_model_error = (
        f"{str(e)}. Backend Python: {sys.executable}. "
        "Start Flask with backend\\venv\\Scripts\\python.exe app.py "
        "or install backend requirements in the Python environment you are using."
    )
    print("Crop model load error:", e)

app = Flask(__name__)
CORS(app, origins=FRONTEND_ORIGINS)




# 🌦 Weather API
def get_weather_and_coords(location):
    normalized_location = str(location or "").strip()
    cache_bucket = int(datetime.now(timezone.utc).timestamp() // WEATHER_CACHE_TTL_SECONDS)
    coords, weather_summary = _get_weather_and_coords_cached(normalized_location, cache_bucket)
    return (
        coords.copy() if coords else None,
        weather_summary.copy() if weather_summary else None,
    )


@lru_cache(maxsize=256)
def _get_weather_and_coords_cached(location, _cache_bucket):
    try:
        url = "http://api.openweathermap.org/data/2.5/weather"
        response = requests.get(
            url,
            params={"q": location, "appid": WEATHER_API_KEY, "units": "metric"},
            timeout=WEATHER_REQUEST_TIMEOUT,
        )
        data = response.json()

        if str(data.get("cod")) != "200":
            return None, {
                "temperature": None,
                "humidity": None,
                "weather": data.get("message")
            }

        coords = {
            "lat": data.get("coord", {}).get("lat"),
            "lon": data.get("coord", {}).get("lon")
        }

        weather_summary = {
            "temperature": data.get("main", {}).get("temp"),
            "humidity": data.get("main", {}).get("humidity"),
            "weather": data.get("weather", [{}])[0].get("description")
        }

        return coords, weather_summary

    except Exception as e:
        print("Weather Error:", e)
        return None, {
            "temperature": None,
            "humidity": None,
            "weather": "Error"
        }


# 🌍 Language mapping
def get_language_instruction(lang):
    if lang == "hi":
        return "Respond in Hindi."
    elif lang == "mr":
        return "Respond in Marathi."
    return "Respond in English."


# 🤖 Gemini AI
def generate_recommendation(data, weather, language, model_prediction=None, crop_report=None):
    if not ENABLE_AI_RECOMMENDATION or gemini_model is None:
        return (
            "AI summary is disabled. Review the crop ranking, blocked crops, and breeding suggestions below."
        )

    try:
        lang_instruction = get_language_instruction(language)
        if not crop_report:
            return "Model report is not available yet. Weather data and ML yield prediction are shown below."

        compact_report = {
            "scenario": crop_report.get("scenario"),
            "state": crop_report.get("state"),
            "season": crop_report.get("season"),
            "soil_type": crop_report.get("soil_type"),
            "climate": crop_report.get("climate"),
            "best_crop": crop_report.get("best_crop"),
            "user_crop": crop_report.get("user_crop"),
            "top_crops": crop_report.get("top_crops", [])[:5],
            "blocked_crops": crop_report.get("blocked_crops", [])[:4],
            "breeding_recommendations": crop_report.get("breeding_recommendations", [])[:3],
            "single_crop_prediction": model_prediction,
        }

        prompt = f"""
You are an expert agricultural advisor. Summarize this crop model output for a farmer.

{lang_instruction}

Rules:
- Keep it under 120 words.
- Mention best crop, user's crop status, key risk, and one breeding suggestion.
- Do not invent data not present in JSON.
- Use simple practical language.

Model JSON:
{json.dumps(compact_report, ensure_ascii=False)}
"""

        response = gemini_model.generate_content(
            prompt,
            request_options={"timeout": AI_RECOMMENDATION_TIMEOUT},
        )
        return response.text

    except Exception as e:
        error_message = str(e)
        if "429" in error_message or "quota" in error_message.lower():
            return (
                "AI summary quota is temporarily exhausted. Review the model ranking and breeding report below."
            )
        return "AI summary is unavailable right now. Review the model ranking and breeding report below."







# 🔥 MAIN API
@app.route('/api/crop-data', methods=['POST'])

def submit_crop_data():
   
    data = request.json

    location = data.get("location")
    language = data.get("language", "en")

    if not location:
        return jsonify({"error": "Location required"}), 400

    coords, weather_summary = get_weather_and_coords(location)

    if crop_model_artifact is None:
        model_prediction = {"error": f"Crop model not loaded: {crop_model_error}"}
        crop_report = None
    else:
        try:
            crop_report = generate_crop_report(crop_model_artifact, data, weather_summary)
            model_prediction = prediction_from_report(crop_model_artifact, crop_report)
        except Exception as e:
            model_prediction = {"error": str(e)}
            crop_report = None

    result = {
        "status": "success",
        "message": "Processed successfully",
        "location": location,
        "coordinates": coords,
        "weather_summary": weather_summary,
        "submitted_data": data,
        "model_prediction": model_prediction,
        "crop_report": crop_report,
        "recommendation": None,
       
    }

    

    return jsonify(result)


@app.route('/api/ai-summary', methods=['POST'])
def ai_summary():
    payload = request.json or {}
    crop_report = payload.get("crop_report")

    if not crop_report:
        return jsonify({"error": "Crop report required for AI summary."}), 400

    submitted_data = payload.get("submitted_data") or {}
    language = payload.get("language") or submitted_data.get("language", "en")
    weather_summary = payload.get("weather_summary")
    model_prediction = payload.get("model_prediction")

    recommendation = generate_recommendation(
        submitted_data,
        weather_summary,
        language,
        model_prediction,
        crop_report,
    )

    return jsonify({
        "status": "success",
        "recommendation": recommendation,
    })


@app.route('/api/model-health', methods=['GET'])
def model_health():
    return jsonify({
        "model_loaded": crop_model_artifact is not None,
        "model_error": crop_model_error,
        "python": sys.executable,
    })



if __name__ == "__main__":
    app.run(debug=True, port=5000)
