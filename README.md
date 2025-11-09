# 🔬 GARCH Monte Carlo Equity Explorer

Ein professionelles Streamlit-Tool für Monte-Carlo-Simulationen von Aktienkursen mit GARCH(1,1) Volatilitätsmodellierung.

## 📋 Übersicht

Dieses Tool kombiniert **GARCH(1,1)** (Generalized Autoregressive Conditional Heteroskedasticity) mit **Geometric Brownian Motion (GBM)**, um realistische Aktienkurssimulationen mit zeitvariabler Volatilität zu erstellen.

### Hauptfunktionen

- ✅ **GARCH(1,1) Volatilitätsmodellierung**: Erfasst Volatility Clustering und zeitvariante Marktbedingungen
- ✅ **Monte Carlo Simulationen**: Tausende von möglichen Preispfaden in die Zukunft
- ✅ **Risikoanalyse**: Value-at-Risk (VaR) und Expected Shortfall (ES/CVaR)
- ✅ **Rolling Statistics**: 30- und 60-Tage rollende Drift und Volatilität
- ✅ **P/E Ratio Adjustment**: Optional fundamentale Bewertung in Drift integrieren
- ✅ **Interactive Visualisierungen**: Plotly Fan Charts, Histogramme, und mehr
- ✅ **CSV Export**: Exportiere alle simulierten Pfade für weitere Analysen

## 🚀 Installation

### Voraussetzungen

- Python 3.8+
- pip

### Abhängigkeiten installieren

```bash
pip install streamlit yfinance pandas numpy plotly arch scipy
```

Oder mit einer `requirements.txt`:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit>=1.28.0
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
arch>=6.2.0
scipy>=1.11.0
```

## 💻 Verwendung

### App starten

```bash
streamlit run streanlit.py
```

Die App öffnet sich automatisch in Ihrem Browser unter `http://localhost:8501`

### Workflow

1. **Ticker auswählen**: Geben Sie ein Yahoo Finance Ticker-Symbol ein (z.B. AAPL, MSFT, TSLA)
2. **Zeitraum definieren**: Wählen Sie Start- und Enddatum für historische Daten
3. **Simulation konfigurieren**:
   - **Horizon (Tage)**: Wie weit in die Zukunft simulieren (z.B. 252 Tage = 1 Jahr)
   - **Schritte pro Pfad**: Zeitauflösung der Simulation
   - **Anzahl Pfade**: Mehr Pfade = genauere Statistiken, aber längere Rechenzeit
   - **Random Seed**: Für reproduzierbare Ergebnisse
4. **Optional**: P/E Ratio Adjustment aktivieren
5. **Klick auf "🚀 Daten laden & simulieren"**

## 📊 Tabs und Visualisierungen

### Tab 1: Fan Chart & Pfade
- Historische Kursentwicklung (schwarz)
- 50 zufällig ausgewählte Simulationspfade (transparent)
- Quantile Ribbons (5%-95%, 10%-90%, 25%-75%)
- Median-Pfad (blau gestrichelt)
- Rote vertikale Linie markiert das Horizon-Datum

### Tab 2: Risikoanalyse (VaR/ES)
- **Value-at-Risk (VaR)**: Maximaler erwarteter Verlust bei gegebenem Konfidenzniveau
- **Expected Shortfall (ES)**: Durchschnittlicher Verlust im Tail-Risk-Szenario
- Rendite-Verteilungs-Histogramm mit VaR/ES Markierungen

### Tab 3: Verteilung Endpreise
- Histogram der simulierten Endpreise nach dem Horizon
- Aktueller Preis als Referenzlinie
- Quantile-Tabelle (1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%)

### Tab 4: Rolling Statistiken
- 30-Tage und 60-Tage rollende Drift (μ)
- 30-Tage und 60-Tage rollende Volatilität (σ)
- Historische Durchschnitte als Referenzlinien
- Aktuelle Werte als Metriken

### Tab 5: Daten Export
- CSV-Download aller simulierten Pfade
- Statistik-Zusammenfassung
- Datenformat: Datum × Pfade

## 🔧 Technische Details

### GARCH(1,1) Modell

Das GARCH(1,1) Modell prognostiziert die bedingte Varianz:

```
h_t+1 = ω + α·ε²_t + β·h_t
```

Wo:
- **ω** (omega): Konstante
- **α** (alpha): Gewicht vergangener Schocks
- **β** (beta): Gewicht vergangener Volatilität
- **h_t**: Bedingte Varianz zum Zeitpunkt t

**Vorteile:**
- Erfasst Volatility Clustering
- Zeitvariante Volatilitätsschätzung
- Realistischere Prognosen als konstante Volatilität

### GBM mit GARCH-Volatilität

```python
S_t+1 = S_t × exp((μ - 0.5σ²_t)·dt + σ_t·√dt·Z_t)
```

Wo:
- **S_t**: Preis zum Zeitpunkt t
- **μ**: Drift (annualisierte erwartete Rendite)
- **σ_t**: GARCH-prognostizierte Volatilität (zeitvariant!)
- **dt**: Zeitschrittgröße
- **Z_t**: Standard-Normalverteilte Zufallsvariable

### P/E Ratio Adjustment

Wenn aktiviert:
```
μ_adjusted = μ_historical × adjustment_factor
```

Wo:
```
adjustment_factor = clip(sector_PE / stock_PE, 0.5, 1.5)
```

- Niedriger P/E → Höhere erwartete Rendite (unterbewertet)
- Hoher P/E → Niedrigere erwartete Rendite (überbewertet)

## 📈 Beispiel-Anwendungsfälle

### 1. Risikomanagement
- Berechne VaR für Portfolio-Positionen
- Stress-Testing mit verschiedenen Horizont-Zeiträumen
- Expected Shortfall für Tail-Risk-Management

### 2. Optionsbewertung (indicativ)
- Verständnis möglicher Preisbewegungen
- Implizite Volatilitätsschätzung
- Szenario-Planung für Optionsstrategien

### 3. Investment Planning
- Langfristige Kursziele visualisieren
- Wahrscheinlichkeit verschiedener Outcomes
- Risiko-Rendite-Profile verstehen

### 4. Research & Education
- GARCH-Modellierung lernen
- Monte Carlo Simulationen verstehen
- Zeitvariante Volatilität explorieren

## 🎯 Parameter-Empfehlungen

| Verwendungszweck | Horizon | Schritte | Pfade | Seed |
|------------------|---------|----------|-------|------|
| Quick Check | 30-60 | 30-60 | 500 | 42 |
| Tägliche Analyse | 252 | 252 | 1000-2000 | 42 |
| Detaillierte Studie | 252-504 | 252-504 | 5000+ | 42 |
| Forschung | 504+ | 504+ | 10000+ | variabel |

**Rechenzeit** (ungefähr):
- 1000 Pfade × 252 Schritte: ~2-5 Sekunden
- 5000 Pfade × 252 Schritte: ~10-20 Sekunden
- 10000 Pfade × 504 Schritte: ~1-2 Minuten

## ⚠️ Wichtige Hinweise

### Limitationen

- **Keine Dividenden**: Modell ignoriert Dividendenzahlungen
- **Keine Splits**: Stock Splits werden nicht berücksichtigt
- **Normalverteilung**: GBM nimmt log-normale Renditen an (keine Fat Tails perfekt modelliert)
- **Konstante Drift**: μ bleibt konstant (keine Regime-Switches)
- **Historische Basis**: GARCH basiert auf Vergangenheitsdaten

### Risiko-Disclaimer

⚠️ **WICHTIG**: Dieses Tool ist ausschließlich für **Bildungs- und Forschungszwecke** gedacht.

- ❌ **KEINE ANLAGEBERATUNG**
- ❌ **KEINE GARANTIE** für zukünftige Ergebnisse
- ❌ **KEINE EMPFEHLUNG** zum Kauf/Verkauf von Wertpapieren

Vergangenheitsperformance ist kein Indikator für zukünftige Ergebnisse. Alle Simulationen sind rein illustrativ.

## 🐛 Bekannte Issues

### Pandas Timestamp Errors
**Problem**: Bei älteren Pandas-Versionen können Timestamp-Arithmetik-Fehler auftreten.

**Lösung**: Aktualisieren Sie auf Pandas ≥2.0
```bash
pip install --upgrade pandas
```

### GARCH Fitting Failures
**Problem**: GARCH kann bei zu kurzen Zeitreihen oder extrem volatilen Daten fehlschlagen.

**Lösung**:
- Verwenden Sie längere historische Zeiträume (min. 1 Jahr)
- Prüfen Sie, ob Daten lückenlos sind

### Memory Issues bei vielen Pfaden
**Problem**: >10000 Pfade können RAM-Probleme verursachen.

**Lösung**:
- Reduzieren Sie Anzahl der Pfade
- Erhöhen Sie verfügbaren RAM
- Führen Sie Simulationen in Batches durch

## 🔄 Updates & Versionen

### Version 1.0 (Aktuell)
- ✅ GARCH(1,1) Volatilitätsmodellierung
- ✅ Monte Carlo GBM Simulationen
- ✅ VaR & Expected Shortfall
- ✅ Rolling Statistics
- ✅ P/E Adjustment
- ✅ Interactive Plotly Charts
- ✅ CSV Export

### Geplante Features
- [ ] Regime-Switching Modelle
- [ ] Jump-Diffusion für Extremereignisse
- [ ] Portfolio-Simulationen (Multivariate)
- [ ] GARCH-Varianten (EGARCH, GJR-GARCH)
- [ ] Makro-Indikator Integration

## 📚 Literatur & Referenzen

### GARCH Models
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"
- Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity"

### Monte Carlo Simulation
- Glasserman, P. (2004). "Monte Carlo Methods in Financial Engineering"
- Boyle, P. P. (1977). "Options: A Monte Carlo Approach"

### Risk Management
- Jorion, P. (2006). "Value at Risk: The New Benchmark for Managing Financial Risk"
- McNeil, A. J., et al. (2005). "Quantitative Risk Management"

## 🤝 Contribution

Dieses Tool ist ein Bildungsprojekt. Verbesserungsvorschläge und Feedback sind willkommen!

## 📄 Lizenz

Dieses Projekt ist für Bildungs- und Forschungszwecke frei verfügbar.


---

**Built with**: Streamlit, yfinance, arch, plotly, pandas, numpy
**Data Source**: Yahoo Finance
**⚠️ No Investment Advice**