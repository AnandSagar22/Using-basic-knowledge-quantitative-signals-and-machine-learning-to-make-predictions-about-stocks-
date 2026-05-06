# senti.py

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Optional, Sequence, Literal, Dict, List, Any

import feedparser
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types


START_DATE = "2025-12-01"
END_DATE = "2026-01-31"

OUTPUT_CSV = "market_sentiment_range.csv"
PLOT_LINE_FILE = "sentiment_vix_trend.png"
PLOT_BAR_FILE = "sentiment_bar_chart.png"

NEWS_QUERY = (
    "market OR markets OR economy OR inflation OR rates OR investors OR recession "
    'OR "central bank" OR equities OR risk OR rally OR selloff'
)


class MarketSentimentResult(BaseModel):
    sentiment_label: Literal["Fearful", "Greedy", "Neutral"]
    score: float = Field(..., ge=-1.0, le=1.0)
    crowd_behavior: str
    market_vibe: str


class MarketSentimentAnalyzer:
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
    ) -> None:
        key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY or pass gemini_api_key=..."
            )
        self.client = genai.Client(api_key=key)
        self.model = model

    @staticmethod
    def _normalize_vix(vix: float) -> float:
        vix = float(vix)
        raw = (20.0 - vix) / 15.0
        return max(-1.0, min(1.0, raw))

    @staticmethod
    def _final_label(score: float) -> str:
        if score > 0.15:
            return "Greedy"
        if score < -0.15:
            return "Fearful"
        return "Neutral"

    @staticmethod
    def _clean_text(text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _force_five_words(text: str) -> str:
        words = re.findall(r"[A-Za-z0-9']+", text or "")
        if not words:
            return "Investors staying cautious and defensive"
        return " ".join(words[:5])

    @staticmethod
    def _force_one_word(text: str) -> str:
        words = re.findall(r"[A-Za-z0-9']+", text or "")
        if not words:
            return "Neutral"
        return words[0]

    def analyze(self, vix: float, headlines: Sequence[str]) -> dict:
        if not headlines:
            raise ValueError("headlines cannot be empty")

        cleaned_headlines = [h.strip() for h in headlines if h and h.strip()]
        if not cleaned_headlines:
            raise ValueError("headlines contain no usable text")

        headlines_text = "\n".join(f"- {h}" for h in cleaned_headlines)

        prompt = f"""
You are a behavioral finance expert.

Analyze the overall psychological state of the market using VIX and headlines.
Ignore individual stock performance.
Focus only on global fear, greed, uncertainty, and crowd behavior.

VIX: {vix}

Market Headlines:
{headlines_text}

Return ONLY valid JSON with this exact schema:
{{
  "sentiment_label": "Fearful | Greedy | Neutral",
  "score": -1.0 to 1.0,
  "crowd_behavior": "Exactly 5 words, no punctuation",
  "market_vibe": "One word"
}}

Rules:
- Use VIX as a fear gauge.
- Mixed headlines + middling VIX => neutral.
- Risk-off headlines + high VIX => fearful.
- Risk-on headlines + low VIX => greedy.
- No markdown, no explanation, no extra keys.
""".strip()

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=MarketSentimentResult.model_json_schema(),
            ),
        )

        raw_text = self._clean_text(response.text or "")
        if not raw_text:
            raise RuntimeError("Gemini returned an empty response")

        llm_result = MarketSentimentResult.model_validate_json(raw_text)

        vix_component = self._normalize_vix(vix)
        combined_score = round((0.70 * llm_result.score) + (0.30 * vix_component), 3)
        combined_score = max(-1.0, min(1.0, combined_score))

        final_result = MarketSentimentResult(
            sentiment_label=self._final_label(combined_score),
            score=combined_score,
            crowd_behavior=self._force_five_words(llm_result.crowd_behavior),
            market_vibe=self._force_one_word(llm_result.market_vibe),
        )

        return final_result.model_dump()


def generate_date_range(start_date: str, end_date: str) -> List[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def fetch_google_news_headlines_for_date(date_str: str, max_items: int = 20) -> List[str]:
    query = NEWS_QUERY.replace(" ", "+")
    rss_url = (
        f"https://news.google.com/rss/search?q={query}"
        f"+after:{date_str}+before:{date_str}&hl=en-IN&gl=IN&ceid=IN:en"
    )

    feed = feedparser.parse(rss_url)

    headlines: List[str] = []
    seen = set()

    for entry in feed.entries[:max_items]:
        title = (entry.get("title") or "").strip()
        if title and title not in seen:
            seen.add(title)
            headlines.append(title)

    return headlines


def fetch_vix_series(start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    raw = yf.download(
        "^VIX",
        start=start_date,
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        interval="1d",
    )

    if raw.empty:
        raise RuntimeError("Could not download VIX data from Yahoo Finance.")

    # Flatten multi-index columns if yfinance returns them
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [
            "_".join([str(x) for x in col if x is not None and str(x) != ""]).strip("_")
            for col in raw.columns.to_flat_index()
        ]

    raw = raw.reset_index()

    # Handle either normal or flattened column names
    if "Date" not in raw.columns:
        raise RuntimeError(f"Unexpected VIX format. Columns found: {list(raw.columns)}")

    close_col = None
    for candidate in ["Close", "Close_^VIX", "^VIX_Close"]:
        if candidate in raw.columns:
            close_col = candidate
            break

    if close_col is None:
        close_col = next((c for c in raw.columns if "Close" in c), None)

    if close_col is None:
        raise RuntimeError(f"Could not find VIX close column. Columns: {list(raw.columns)}")

    vix_df = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"]).dt.strftime("%Y-%m-%d"),
        "vix": pd.to_numeric(raw[close_col], errors="coerce"),
    }).dropna()

    all_dates = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    all_dates["date"] = all_dates["date"].dt.strftime("%Y-%m-%d")

    merged = all_dates.merge(vix_df, on="date", how="left")
    merged["vix"] = merged["vix"].ffill().bfill()

    return merged


def build_market_sentiment_dataset(
    analyzer: MarketSentimentAnalyzer,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    date_list = generate_date_range(start_date, end_date)
    vix_df = fetch_vix_series(start_date, end_date)

    rows: List[Dict[str, Any]] = []

    for date_str in date_list:
        vix_row = vix_df[vix_df["date"] == date_str]
        vix_value = float(vix_row.iloc[0]["vix"]) if not vix_row.empty else None

        headlines = fetch_google_news_headlines_for_date(date_str)

        if not headlines:
            rows.append(
                {
                    "date": date_str,
                    "vix": vix_value,
                    "sentiment_label": "Neutral",
                    "score": 0.0,
                    "crowd_behavior": "No headlines available today",
                    "market_vibe": "Neutral",
                    "headline_count": 0,
                    "status": "no_headlines",
                }
            )
            continue

        try:
            result = analyzer.analyze(vix=vix_value, headlines=headlines)
            rows.append(
                {
                    "date": date_str,
                    "vix": vix_value,
                    "sentiment_label": result["sentiment_label"],
                    "score": float(result["score"]),
                    "crowd_behavior": result["crowd_behavior"],
                    "market_vibe": result["market_vibe"],
                    "headline_count": len(headlines),
                    "status": "ok",
                }
            )
        except Exception as e:
            rows.append(
                {
                    "date": date_str,
                    "vix": vix_value,
                    "sentiment_label": "Neutral",
                    "score": 0.0,
                    "crowd_behavior": "Analysis failed due error",
                    "market_vibe": "Neutral",
                    "headline_count": len(headlines),
                    "status": f"analysis_error: {str(e)}",
                }
            )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def save_results(df: pd.DataFrame, csv_path: str = OUTPUT_CSV) -> None:
    df.to_csv(csv_path, index=False)


def plot_sentiment_and_vix(df: pd.DataFrame, output_file: str = PLOT_LINE_FILE) -> None:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df["date"], df["score"], marker="o", label="Sentiment Score")
    ax1.axhline(0, linestyle="--")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sentiment Score (-1 to 1)")
    ax1.set_title("Market Sentiment vs VIX")

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["vix"], marker="s", linestyle="--", label="VIX")
    ax2.set_ylabel("VIX")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()


def plot_sentiment_bars(df: pd.DataFrame, output_file: str = PLOT_BAR_FILE) -> None:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(df["date"].dt.strftime("%Y-%m-%d"), df["score"])
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Score (-1 to 1)")
    ax.set_title("Daily Market Sentiment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()


def main() -> None:
    analyzer = MarketSentimentAnalyzer(
        gemini_api_key=None,  # or paste your Gemini key here
        model="gemini-2.5-flash",
    )

    try:
        df = build_market_sentiment_dataset(
            analyzer=analyzer,
            start_date=START_DATE,
            end_date=END_DATE,
        )

        print("\nRESULTS\n")
        print(df.to_string(index=False))

        save_results(df, OUTPUT_CSV)
        print(f"\nSaved dataset to: {OUTPUT_CSV}")

        plot_sentiment_and_vix(df, PLOT_LINE_FILE)
        print(f"Saved line plot to: {PLOT_LINE_FILE}")

        plot_sentiment_bars(df, PLOT_BAR_FILE)
        print(f"Saved bar plot to: {PLOT_BAR_FILE}")

    except (ValidationError, ValueError, RuntimeError, FileNotFoundError, Exception) as e:
        print(json.dumps({"error": str(e)}, indent=2))


if __name__ == "__main__":
    main()