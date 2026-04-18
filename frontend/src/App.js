import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import api from "./api";
import "./App.css";
import { motion, AnimatePresence } from "motion/react";

import Header from "./components/Header";
import LanguageToggle from "./components/LanguageToggle";
import CropForm from "./components/CropForm";
import ResultCard from "./components/ResultCard";

function App() {
  const { i18n } = useTranslation();
  const [language, setLanguage] = useState(i18n.language);

  const [formData, setFormData] = useState({
    crop_type: "",
    rainfall: "",
    soil_type: "",
    irrigation: "",
    location: "",
  });

  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleLanguageChange = (langCode) => {
    setLanguage(langCode);
    i18n.changeLanguage(langCode);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
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

  return (
    <div className="container">
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
        <LanguageToggle language={language} setLanguage={handleLanguageChange} />

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
            <ResultCard response={response} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;