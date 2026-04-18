import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { CITY_OPTIONS_BY_STATE, STATE_OPTIONS } from '../data/locationOptions';

export default function CropForm({ formData, handleChange, handleSubmit, loading }) {
  const { t } = useTranslation();
  const seasons = ['Kharif', 'Rabi', 'Summer', 'Winter', 'Autumn', 'Whole Year'];
  const scenarios = ['normal', 'drought', 'heatwave', 'flood'];
  const cityOptions = useMemo(
    () => CITY_OPTIONS_BY_STATE[formData.state] || [],
    [formData.state]
  );

  return (
    <form onSubmit={handleSubmit} className="crop-form">
      <div className="form-group" data-tour="crop">
        <label>{t('cropForm.cropType')}</label>
        <select name="crop_type" value={formData.crop_type} onChange={handleChange} required>
          <option value="">{t('cropForm.selectCrop')}</option>
          <option value="rice">{t('crops.rice')}</option>
          <option value="wheat">{t('crops.wheat')}</option>
          <option value="corn">{t('crops.corn')}</option>
          <option value="soybean">{t('crops.soybean')}</option>
          <option value="cotton">{t('crops.cotton')}</option>
          <option value="sugarcane">Sugarcane</option>
          <option value="bajra">Bajra</option>
          <option value="jowar">Jowar</option>
          <option value="onion">Onion</option>
        </select>
      </div>

      <div className="form-group" data-tour="state">
        <label>{t('cropForm.state', { defaultValue: 'State' })}</label>
        <select name="state" value={formData.state} onChange={handleChange} required>
          <option value="">{t('cropForm.selectState', { defaultValue: 'Select State' })}</option>
          {STATE_OPTIONS.map((state) => (
            <option key={state} value={state}>{state}</option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>{t('cropForm.season', { defaultValue: 'Season' })}</label>
        <select name="season" value={formData.season} onChange={handleChange} required>
          <option value="">{t('cropForm.selectSeason', { defaultValue: 'Select Season' })}</option>
          {seasons.map((season) => (
            <option key={season} value={season}>{season}</option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>{t('cropForm.scenario', { defaultValue: 'Climate Scenario' })}</label>
        <select name="scenario" value={formData.scenario} onChange={handleChange} required>
          {scenarios.map((scenario) => (
            <option key={scenario} value={scenario}>
              {scenario.charAt(0).toUpperCase() + scenario.slice(1)}
            </option>
          ))}
        </select>
      </div>

      <div className="form-group" data-tour="area">
        <label>{t('cropForm.area', { defaultValue: 'Area' })}</label>
        <input
          type="number"
          name="area"
          min="0.01"
          step="0.01"
          placeholder={t('cropForm.areaPlaceholder', { defaultValue: 'e.g. 100' })}
          value={formData.area}
          onChange={handleChange}
          required
        />
      </div>

      <div className="form-group">
        <label>{t('cropForm.cropYear', { defaultValue: 'Crop Year' })}</label>
        <input
          type="number"
          name="crop_year"
          min="1997"
       
          value={formData.crop_year}
          onChange={handleChange}
          required
        />
      </div>

      <div className="form-group">
        <label>{t('cropForm.soilType')}</label>
        <select name="soil_type" value={formData.soil_type} onChange={handleChange} required>
          <option value="">{t('cropForm.selectSoil')}</option>
          <option value="clay">{t('soilTypes.clay')}</option>
          <option value="sandy">{t('soilTypes.sandy')}</option>
          <option value="loamy">{t('soilTypes.loamy')}</option>
          <option value="black">{t('soilTypes.black')}</option>
          <option value="red">{t('soilTypes.red')}</option>
        </select>
      </div>

      <div className="form-group">
        <label>{t('cropForm.irrigationMethod')}</label>
        <select name="irrigation" value={formData.irrigation} onChange={handleChange} required>
          <option value="">{t('cropForm.selectMethod')}</option>
          <option value="drip">{t('irrigationMethods.drip')}</option>
          <option value="sprinkler">{t('irrigationMethods.sprinkler')}</option>
          <option value="surface">{t('irrigationMethods.surface')}</option>
          <option value="manual">{t('irrigationMethods.manual')}</option>
        </select>
      </div>

      <div className="form-group">
        <label>{t('cropForm.annualRainfall')}</label>
        <input 
          type="number" 
          name="rainfall" 
          placeholder={t('cropForm.rainfallPlaceholder')} 
          value={formData.rainfall} 
          onChange={handleChange} 
          required
        />
      </div>

      <div className="form-group" data-tour="location">
        <label>{t('cropForm.farmLocation')}</label>
        <select
          name="location"
          value={formData.location}
          onChange={handleChange}
          disabled={!formData.state}
          required
        >
          <option value="">
            {formData.state
              ? t('cropForm.selectCity', { defaultValue: 'Select City' })
              : t('cropForm.selectStateFirst', { defaultValue: 'Select state first' })}
          </option>
          {cityOptions.map((city) => (
            <option key={city} value={city}>{city}</option>
          ))}
        </select>
      </div>

      <button className="btn" disabled={loading} data-tour="submit">
        {loading ? (
          <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "10px" }}>
             {t('cropForm.analyzingData')}
          </span>
        ) : (
          t('cropForm.getRecommendation')
        )}
      </button>
    </form>
  );
}
