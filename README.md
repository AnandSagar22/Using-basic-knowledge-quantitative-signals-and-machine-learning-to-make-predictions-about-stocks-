# Predicting Next-Quarter Stock Performance using Fundamentals, Quantitative Signals and Machine Learning

This project combines **fundamental analysis**, **market sentiment**, and **machine learning** to study and predict next-quarter stock performance.

The workflow follows a data pipeline that ingests raw inputs, engineers quarterly features, integrates sentiment indicators, trains models, and evaluates performance through backtesting.

---

## Project Overview

The goal of this project is to build a structured stock analysis pipeline that:

* collects and prepares company and market data,
* generates financial and sentiment-based features,
* merges all inputs into model-ready datasets,
* trains predictive models,
* and evaluates results using backtesting.

The project is organized into modular Python scripts so each stage can be run independently.

---

## Repository Structure

```text
├── data_ingestion.py
├── feature_engineering.py
├── fundamental_analysis.py
├── sentiment_analysis.py
├── model_and_backtest.py
├── features_quarterly.csv
├── ff_factors_monthly.csv
├── market_sentiment_range.csv
├── market_sentiment_results.csv
├── ticker_cik_map.csv
├── tickers.csv
├── sentiment_bar_chart.png
├── sentiment_vix_trend.png
```

---

## File Descriptions

### Python Scripts

**`data_ingestion.py`**
Handles raw data collection and preparation. This script is typically used to fetch or load the required stock, financial, and related datasets.

**`fundamental_analysis.py`**
Computes fundamental metrics from company financial data and prepares the basic financial analysis inputs.

**`sentiment_analysis.py`**
Processes market/news sentiment data and generates sentiment-related outputs used in the model.

**`feature_engineering.py`**
Creates model-ready features by combining financial data, technical signals, sentiment variables, and other derived metrics.

**`model_and_backtest.py`**
Trains machine learning models and evaluates them using backtesting to measure predictive performance.

---

### Data Files

**`features_quarterly.csv`**
Final quarterly feature dataset used for modeling.

**`ff_factors_monthly.csv`**
Monthly Fama-French factor data used for factor-based analysis or benchmarking.

**`market_sentiment_range.csv`**
Contains sentiment values across a defined market date range.

**`market_sentiment_results.csv`**
Stores processed sentiment analysis results.

**`ticker_cik_map.csv`**
Mapping file connecting stock tickers with CIK identifiers.

**`tickers.csv`**
List of stock tickers used in the analysis.

---

### Visualization Outputs

**`sentiment_bar_chart.png`**
Bar chart visualizing sentiment distribution or summary.

**`sentiment_vix_trend.png`**
Trend visualization comparing sentiment behavior with VIX or volatility-related movement.

---

## Workflow

1. **Data ingestion**
   Load or collect the required datasets.

2. **Fundamental analysis**
   Compute company-level financial metrics.

3. **Sentiment analysis**
   Extract and process sentiment indicators from market or news data.

4. **Feature engineering**
   Combine all inputs and create quarterly model features.

5. **Model training and backtesting**
   Train predictive models and evaluate future stock performance.

---

## How to Run

Run the scripts in the following order:

```bash
python data_ingestion.py
python fundamental_analysis.py
python sentiment_analysis.py
python feature_engineering.py
python model_and_backtest.py
```

Depending on your local setup, some scripts may depend on the output of previous ones.

---

## Requirements

Typical Python libraries used in this project may include:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* yfinance
* requests

Install them with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn yfinance requests
```

---

## Notes

* Ensure all CSV files are placed in the correct working directory before running the scripts.
* Some scripts may require internet access for data download.
* The model results depend on the quality of the input data, feature selection, and the chosen backtesting setup.

---

## Research Context

This project is aligned with research on stock return prediction using:

* fundamental variables,
* factor-based finance signals,
* sentiment indicators,
* and machine learning methods.

It is intended for academic research and experimentation.

---

## Author

Prepared for research and academic project work.

---

## License

Add your preferred license here if you plan to publish or share the repository publicly.
