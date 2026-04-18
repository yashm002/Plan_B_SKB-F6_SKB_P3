export default function ResultCard({ response }) {
  if (!response) return null;

  if (response.error) {
    return (
      <div
        className="glass result animate-fade-in"
        style={{ borderColor: "#ef5350", background: "#ffebee" }}
      >
        ❌ {response.error}
      </div>
    );
  }

  const { weather_summary, recommendation, location } = response;

  // ✅ Convert **bold** → <strong>
  const formattedRecommendation = recommendation?.replace(
    /\*\*(.*?)\*\*/g,
    "<strong>$1</strong>"
  );

  return (
    <div className="glass result animate-fade-in">
      <h3>📍 {location}</h3>

      {/* 🌦️ Weather Info */}
      <div className="weather-info">
        <div
          className={`weather-card ${
            weather_summary?.temperature > 30
              ? "hot"
              : weather_summary?.temperature < 20
              ? "cool"
              : ""
          }`}
        >
          <span>Temperature</span>
          <strong>{weather_summary?.temperature}°C</strong>
        </div>

        <div
          className={`weather-card ${
            weather_summary?.humidity > 70 ? "humid" : ""
          }`}
        >
          <span>Humidity</span>
          <strong>{weather_summary?.humidity}%</strong>
        </div>

        <div className="weather-card">
          <span>Condition</span>
          <strong>{weather_summary?.weather}</strong>
        </div>
      </div>

      {/* 🤖 Recommendation */}
      <div className="recommendation-section">
        <h4>🤖 Smart Recommendation</h4>
        <div className="recommendation-content">
          <p
            style={{ whiteSpace: "pre-wrap" }}
            dangerouslySetInnerHTML={{ __html: formattedRecommendation }}
          />
        </div>
      </div>
    </div>
  );
}