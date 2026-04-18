import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "motion/react";
import { Joyride, STATUS } from "react-joyride";
import api from "./api";
import "./App.css";

import Header from "./components/Header";
import LanguageToggle from "./components/LanguageToggle";
import CropForm from "./components/CropForm";
import ResultCard from "./components/ResultCard";

const TOUR_STEPS = [
  {
    target: '[data-tour="language"]',
    content: "Choose the language for the advice.",
  },
  {
    target: '[data-tour="crop"]',
    content: "Select the crop you want to evaluate.",
  },
  {
    target: '[data-tour="state"]',
    content: "Pick the state first. The city list updates from this choice.",
  },
  {
    target: '[data-tour="location"]',
    content: "Choose the city for weather and location-aware results.",
  },
  {
    target: '[data-tour="area"]',
    content: "Enter the cultivated area used by the ML yield model.",
  },
  {
    target: '[data-tour="submit"]',
    content: "Run the weather and ML checks first.",
  },
  {
    target: "body",
    placement: "center",
    content: "After submitting, the weather and model results appear below. Use AI summarize only when you want a Gemini summary.",
  },
];

function App() {
  const { i18n } = useTranslation();
  const [language, setLanguage] = useState(i18n.language);
  const [runTour, setRunTour] = useState(false);

  const [formData, setFormData] = useState({
    crop_type: "",
    state: "",
    season: "",
    scenario: "normal",
    area: "",
    crop_year: "2015",
    rainfall: "",
    soil_type: "",
    irrigation: "",
    location: "",
  });

  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const joyrideStyles = useMemo(
    () => ({
      options: {
        primaryColor: "#2e7d32",
        textColor: "#1f2933",
        zIndex: 10000,
      },
      buttonNext: {
        borderRadius: 8,
      },
      buttonBack: {
        color: "#2e7d32",
      },
    }),
    []
  );

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((current) => ({
      ...current,
      [name]: value,
      ...(name === "state" ? { location: "" } : {}),
    }));
  };

  const handleLanguageChange = (langCode) => {
    setLanguage(langCode);
    i18n.changeLanguage(langCode);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSummaryLoading(false);
    setResponse(null);

    try {
      const res = await api.post("/crop-data", {
        ...formData,
        language,
      });

      setResponse(res.data);
    } catch (err) {
      setResponse({ error: "Unable to reach the server. Please check if the backend is running." });
    }

    setLoading(false);
  };

  const handleAiSummarize = async () => {
    if (!response?.crop_report) return;

    const summaryPayload = {
      submitted_data: response.submitted_data,
      weather_summary: response.weather_summary,
      model_prediction: response.model_prediction,
      crop_report: response.crop_report,
      language,
    };

    setSummaryLoading(true);

    try {
      const res = await api.post("/ai-summary", summaryPayload);
      setResponse((current) => ({
        ...current,
        recommendation: res.data.recommendation,
      }));
    } catch (err) {
      setResponse((current) => ({
        ...current,
        recommendation: "AI summary is unavailable right now. Please try again.",
      }));
    }

    setSummaryLoading(false);
  };

  const handleJoyrideCallback = ({ status }) => {
    if ([STATUS.FINISHED, STATUS.SKIPPED].includes(status)) {
      setRunTour(false);
    }
  };

  return (
    <div className="container">
      <Joyride
        callback={handleJoyrideCallback}
        continuous
        run={runTour}
        scrollToFirstStep
        showProgress
        showSkipButton
        steps={TOUR_STEPS}
        styles={joyrideStyles}
      />

      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Header />
      </motion.div>

      <motion.div 
        className="glass form-container"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <div className="form-actions">
          <div data-tour="language">
            <LanguageToggle language={language} setLanguage={handleLanguageChange} />
          </div>
          <button type="button" className="tour-btn" onClick={() => setRunTour(true)}>
            Start Tour
          </button>
        </div>

        <CropForm
          formData={formData}
          handleChange={handleChange}
          handleSubmit={handleSubmit}
          loading={loading}
        />
      </motion.div>

      <AnimatePresence>
        {response && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5 }}
          >
            <div data-tour="result">
              <ResultCard
                response={response}
                onAiSummarize={handleAiSummarize}
                summaryLoading={summaryLoading}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
