from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests
import google.generativeai as genai

# Load env
load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini config
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:3001"])

# 🌦 Weather API
def get_weather_and_coords(location):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
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
def generate_recommendation(data, weather, language):
    try:
        lang_instruction = get_language_instruction(language)

        prompt = f"""
You are an expert agricultural advisor.

{lang_instruction}

Inputs:
- Crop: {data.get('crop_type')}
- Soil: {data.get('soil_type')}
- Irrigation: {data.get('irrigation')}
- User Temp: {data.get('temperature')}
- Rainfall: {data.get('rainfall')}
- Current Temp: {weather.get('temperature')}
- Humidity: {weather.get('humidity')}
- Weather: {weather.get('weather')}

Give:
1. Crop suitability
2. Better crop suggestion
3. Irrigation advice
4. Soil tips

Keep it short and practical.
"""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        import traceback
        print("Gemini Error Details:")
        traceback.print_exc()
        return f"AI recommendation not available. (Error: {str(e)})"


# 🔥 MAIN API
@app.route('/api/crop-data', methods=['POST'])
def submit_crop_data():
    data = request.json

    location = data.get("location")
    language = data.get("language", "en")

    if not location:
        return jsonify({"error": "Location required"}), 400

    coords, weather_summary = get_weather_and_coords(location)

    recommendation = generate_recommendation(data, weather_summary, language)

    return jsonify({
        "status": "success",
        "message": "Processed successfully",
        "location": location,
        "coordinates": coords,
        "weather_summary": weather_summary,
        "submitted_data": data,
        "recommendation": recommendation
    })


if __name__ == "__main__":
    app.run(debug=True,port=5000)