import React, { useState } from 'react';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    crop_type: '',
    temperature: '',
    rainfall: '',
    soil_type: '',
    irrigation: '',
    location: ''
  });
  
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResponse(null);

    try {
      const response = await fetch('http://localhost:5000/api/submit-crop-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });

      const data = await response.json();
      
      if (response.ok) {
        setResponse(data);
      } else {
        setResponse({ error: data.error || 'Something went wrong' });
      }
    } catch (error) {
      setResponse({ error: 'Failed to connect to backend' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Autonomous Multi Omics Fusion for Climate Resilient</h1>
        <h2>Genomic Selection and Crop Breeding Optimization</h2>
      </header>
      
      <main className="container">
        <div className="form-container">
          <h3>Crop Parameters Input</h3>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="crop_type">Crop Type:</label>
              <select
                id="crop_type"
                name="crop_type"
                value={formData.crop_type}
                onChange={handleChange}
                required
              >
                <option value="">Select Crop Type</option>
                <option value="wheat">Wheat</option>
                <option value="rice">Rice</option>
                <option value="corn">Corn</option>
                <option value="soybean">Soybean</option>
                <option value="barley">Barley</option>
                <option value="sorghum">Sorghum</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="temperature">Temperature (°C):</label>
              <input
                type="number"
                id="temperature"
                name="temperature"
                value={formData.temperature}
                onChange={handleChange}
                step="0.1"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="rainfall">Rainfall (mm/year):</label>
              <input
                type="number"
                id="rainfall"
                name="rainfall"
                value={formData.rainfall}
                onChange={handleChange}
                step="0.1"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="soil_type">Soil Type:</label>
              <select
                id="soil_type"
                name="soil_type"
                value={formData.soil_type}
                onChange={handleChange}
                required
              >
                <option value="">Select Soil Type</option>
                <option value="clay">Clay</option>
                <option value="sandy">Sandy</option>
                <option value="loamy">Loamy</option>
                <option value="silty">Silty</option>
                <option value="peaty">Peaty</option>
                <option value="chalky">Chalky</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="irrigation">Irrigation Method:</label>
              <select
                id="irrigation"
                name="irrigation"
                value={formData.irrigation}
                onChange={handleChange}
                required
              >
                <option value="">Select Irrigation Method</option>
                <option value="drip">Drip Irrigation</option>
                <option value="sprinkler">Sprinkler</option>
                <option value="flood">Flood Irrigation</option>
                <option value="center_pivot">Center Pivot</option>
                <option value="furrow">Furrow</option>
                <option value="none">No Irrigation</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="location">Location:</label>
              <input
                type="text"
                id="location"
                name="location"
                value={formData.location}
                onChange={handleChange}
                placeholder="Enter location (e.g., California, USA)"
                required
              />
            </div>

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? 'Submitting...' : 'Submit Crop Data'}
            </button>
          </form>

          {response && (
            <div className={`response ${response.error ? 'error' : 'success'}`}>
              {response.error ? (
                <p>Error: {response.error}</p>
              ) : (
                <div>
                  <h4>Submission Successful!</h4>
                  <p>{response.message}</p>
                  <pre>{JSON.stringify(response.submitted_data, null, 2)}</pre>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
