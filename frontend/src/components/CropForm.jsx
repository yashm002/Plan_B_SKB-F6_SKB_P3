import { useTranslation } from 'react-i18next';

export default function CropForm({ formData, handleChange, handleSubmit, loading }) {
  const { t } = useTranslation();
  return (
    <form onSubmit={handleSubmit} className="crop-form">
      <div className="form-group">
        <label>{t('cropForm.cropType')}</label>
        <select name="crop_type" value={formData.crop_type} onChange={handleChange} required>
          <option value="">{t('cropForm.selectCrop')}</option>
          <option value="rice">{t('crops.rice')}</option>
          <option value="wheat">{t('crops.wheat')}</option>
          <option value="corn">{t('crops.corn')}</option>
          <option value="soybean">{t('crops.soybean')}</option>
          <option value="cotton">{t('crops.cotton')}</option>
        </select>
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

      <div className="form-group">
        <label>{t('cropForm.farmLocation')}</label>
        <input 
          type="text" 
          name="location" 
          placeholder={t('cropForm.locationPlaceholder')} 
          value={formData.location} 
          onChange={handleChange} 
          required 
        />
      </div>

      <button className="btn" disabled={loading}>
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