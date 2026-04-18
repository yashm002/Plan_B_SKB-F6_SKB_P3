export default function ResultCard({ response, onAiSummarize, summaryLoading }) {
  if (!response) return null;

  if (response.error) {
    return (
      <div
        className="glass result animate-fade-in"
        style={{ borderColor: "#ef5350", background: "#ffebee" }}
      >
        Error: {response.error}
      </div>
    );
  }

  const { weather_summary, recommendation, location, model_prediction, crop_report } = response;

  return (
    <div className="glass result animate-fade-in">
      <h3>{location}</h3>

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
          <strong>{weather_summary?.temperature}Â°C</strong>
        </div>

        <div className={`weather-card ${weather_summary?.humidity > 70 ? "humid" : ""}`}>
          <span>Humidity</span>
          <strong>{weather_summary?.humidity}%</strong>
        </div>

        <div className="weather-card">
          <span>Condition</span>
          <strong>{weather_summary?.weather}</strong>
        </div>

        {model_prediction && !model_prediction.error && (
          <div className="weather-card">
            <span>Predicted Yield</span>
            <strong>
              {model_prediction.predicted_yield} {model_prediction.unit}
            </strong>
          </div>
        )}
      </div>

      {model_prediction?.error && (
        <div
          className="recommendation-content"
          style={{ border: "1px solid #ef5350", marginBottom: "20px" }}
        >
          ML prediction unavailable: {model_prediction.error}
        </div>
      )}

      {crop_report && (
        <div className="recommendation-section">
          <h4>Model Crop Ranking</h4>

          <div className="report-grid">
            {crop_report.best_crop && (
              <div className="report-highlight">
                <span>Best crop</span>
                <strong>{crop_report.best_crop.crop}</strong>
                <small>
                  Score {crop_report.best_crop.final_score} | {crop_report.best_crop.confidence}
                </small>
              </div>
            )}

            {crop_report.user_crop && (
              <div className="report-highlight">
                <span>Your crop</span>
                <strong>{crop_report.user_crop.crop}</strong>
                <small>
                  Rank {crop_report.user_crop.rank} | Score {crop_report.user_crop.final_score}
                  {crop_report.user_crop.blocked_reason
                    ? ` | Blocked: ${crop_report.user_crop.blocked_reason}`
                    : ` | ${crop_report.user_crop.confidence}`}
                </small>
              </div>
            )}
          </div>

          <div className="report-meta">
            <span>Scenario: {crop_report.scenario}</span>
            <span>Soil: {crop_report.soil_type}</span>
            <span>
              Climate: {crop_report.climate?.temperature}Â°C, {crop_report.climate?.humidity}% RH
            </span>
          </div>

          <div className="report-table-wrap">
            <table className="report-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Crop</th>
                  <th>Final</th>
                  <th>ML</th>
                  <th>Climate</th>
                  <th>Soil</th>
                  <th>Trait</th>
                </tr>
              </thead>
              <tbody>
                {crop_report.top_crops?.slice(0, 10).map((crop) => (
                  <tr key={crop.crop}>
                    <td>{crop.rank}</td>
                    <td>
                      <strong>{crop.crop}</strong>
                      {crop.note && <small>{crop.note}</small>}
                    </td>
                    <td>{crop.final_score}</td>
                    <td>{crop.ml_rank}</td>
                    <td>{crop.climate}</td>
                    <td>{crop.soil}</td>
                    <td>{crop.trait}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {!!crop_report.blocked_crops?.length && (
            <div className="blocked-list">
              <h5>Hard-blocked crops</h5>
              {crop_report.blocked_crops.map((crop) => (
                <p key={crop.crop}>
                  <strong>{crop.crop}</strong>: {crop.blocked_reason}
                </p>
              ))}
            </div>
          )}

          {!!crop_report.breeding_recommendations?.length && (
            <div className="blocked-list">
              <h5>Breeding suggestions</h5>
              {crop_report.breeding_recommendations.map((item) => (
                <p key={item.strategy}>
                  <strong>Strategy {item.strategy}:</strong> {item.text}
                </p>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="recommendation-section">
        <div className="summary-header">
          <h4>AI Summary</h4>
          <button
            type="button"
            className="ai-summary-btn"
            onClick={onAiSummarize}
            disabled={summaryLoading || !crop_report}
          >
            {summaryLoading ? "Summarizing..." : recommendation ? "Refresh AI summary" : "AI summarize"}
          </button>
        </div>

        <div className={`recommendation-content ${!recommendation ? "is-empty" : ""}`}>
          {summaryLoading ? (
            <p className="summary-placeholder">Preparing the AI summary...</p>
          ) : recommendation ? (
            <p className="summary-text">{recommendation}</p>
          ) : (
            <p className="summary-placeholder">
              Click AI summarize when you want a Gemini explanation of the model ranking.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
