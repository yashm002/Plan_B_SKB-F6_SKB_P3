import { useTranslation } from 'react-i18next';

export default function Header() {
  const { t } = useTranslation();
  
  return (
    <header className="header" style={{ textAlign: "center", marginBottom: "40px" }}>
      <h1 style={{ fontSize: "2.5rem", fontWeight: "700", color: "#ffffff", marginBottom: "10px" }}>
        🌱 {t('header.title')}
      </h1>
      <p style={{ fontSize: "1.1rem", color: "#ffffff", fontWeight: "400" }}>
        {t('header.subtitle')}
      </p>
    </header>
  );
}