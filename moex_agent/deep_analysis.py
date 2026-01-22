"""
MOEX Agent - Глубокий анализ исторических данных

Анализ 4 лет данных для выявления паттернов и оптимизации стратегии.

Usage:
    python -m moex_agent.deep_analysis
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import load_config
from .features import build_feature_frame
from .storage import connect

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("moex_agent.deep_analysis")


def load_all_candles(conn) -> pd.DataFrame:
    """Загрузка всех свечей."""
    logger.info("Загрузка свечей...")
    q = """
    SELECT secid, ts, open, high, low, close, value, volume
    FROM candles
    WHERE interval = 1
    ORDER BY ts, secid
    """
    df = pd.read_sql_query(q, conn)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    logger.info(f"Загружено {len(df):,} свечей")
    return df


def analyze_market_regimes(df: pd.DataFrame) -> Dict:
    """Анализ рыночных режимов (бычий/медвежий/боковой)."""
    logger.info("Анализ рыночных режимов...")

    # Агрегируем по дням
    daily = df.groupby([df["ts"].dt.date, "secid"]).agg({
        "close": "last",
        "value": "sum",
        "volume": "sum"
    }).reset_index()

    # Считаем общий индекс (среднее по всем бумагам)
    market_daily = daily.groupby("ts").agg({
        "close": "mean",
        "value": "sum"
    }).reset_index()
    market_daily.columns = ["date", "avg_close", "total_value"]

    # Расчёт returns
    market_daily["return"] = market_daily["avg_close"].pct_change()

    # Определяем режимы (20-дневное окно)
    market_daily["sma20"] = market_daily["avg_close"].rolling(20).mean()
    market_daily["sma50"] = market_daily["avg_close"].rolling(50).mean()
    market_daily["volatility"] = market_daily["return"].rolling(20).std()

    # Режим: bull (sma20 > sma50), bear (sma20 < sma50), sideways (близко)
    def classify_regime(row):
        if pd.isna(row["sma20"]) or pd.isna(row["sma50"]):
            return "unknown"
        diff_pct = (row["sma20"] - row["sma50"]) / row["sma50"] * 100
        if diff_pct > 2:
            return "bull"
        elif diff_pct < -2:
            return "bear"
        else:
            return "sideways"

    market_daily["regime"] = market_daily.apply(classify_regime, axis=1)

    # Статистика по режимам
    regime_stats = market_daily.groupby("regime").agg({
        "return": ["mean", "std", "count"],
        "volatility": "mean"
    })

    regimes = {
        "bull_days": int((market_daily["regime"] == "bull").sum()),
        "bear_days": int((market_daily["regime"] == "bear").sum()),
        "sideways_days": int((market_daily["regime"] == "sideways").sum()),
        "total_days": len(market_daily),
        "avg_daily_return": float(market_daily["return"].mean() * 100),
        "avg_volatility": float(market_daily["volatility"].mean() * 100),
    }

    logger.info(f"Режимы: bull={regimes['bull_days']}, bear={regimes['bear_days']}, sideways={regimes['sideways_days']}")

    return regimes


def analyze_seasonality(df: pd.DataFrame) -> Dict:
    """Анализ сезонности (дни недели, месяцы, часы)."""
    logger.info("Анализ сезонности...")

    df = df.copy()
    df["hour"] = df["ts"].dt.hour
    df["weekday"] = df["ts"].dt.weekday
    df["month"] = df["ts"].dt.month
    df["return"] = df.groupby("secid")["close"].pct_change()

    # По часам
    hourly = df.groupby("hour")["return"].agg(["mean", "std", "count"])
    best_hour = hourly["mean"].idxmax()
    worst_hour = hourly["mean"].idxmin()

    # По дням недели
    weekday_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    daily = df.groupby("weekday")["return"].agg(["mean", "std", "count"])
    best_weekday = daily["mean"].idxmax()
    worst_weekday = daily["mean"].idxmin()

    # По месяцам
    monthly = df.groupby("month")["return"].agg(["mean", "std", "count"])
    best_month = monthly["mean"].idxmax()
    worst_month = monthly["mean"].idxmin()

    seasonality = {
        "best_hour": int(best_hour),
        "worst_hour": int(worst_hour),
        "best_weekday": weekday_names[best_weekday],
        "worst_weekday": weekday_names[worst_weekday],
        "best_month": int(best_month),
        "worst_month": int(worst_month),
        "hourly_returns": {int(h): float(r * 100) for h, r in hourly["mean"].items()},
        "weekday_returns": {weekday_names[w]: float(r * 100) for w, r in daily["mean"].items()},
        "monthly_returns": {int(m): float(r * 100) for m, r in monthly["mean"].items()},
    }

    logger.info(f"Лучший час: {best_hour}:00, Лучший день: {weekday_names[best_weekday]}, Лучший месяц: {best_month}")

    return seasonality


def analyze_tickers(df: pd.DataFrame) -> Dict:
    """Анализ отдельных тикеров."""
    logger.info("Анализ тикеров...")

    ticker_stats = {}

    for secid in df["secid"].unique():
        ticker_df = df[df["secid"] == secid].copy()
        ticker_df["return"] = ticker_df["close"].pct_change()

        # Базовая статистика
        returns = ticker_df["return"].dropna()
        if len(returns) < 100:
            continue

        # Sharpe ratio (annualized)
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 60 * 9)  # 9 часов торгов

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252 * 60 * 9)

        # Средний оборот
        avg_turnover = ticker_df["value"].mean()

        ticker_stats[secid] = {
            "total_candles": len(ticker_df),
            "avg_return": float(returns.mean() * 100),
            "volatility": float(volatility * 100),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown * 100),
            "avg_turnover": float(avg_turnover),
            "skewness": float(stats.skew(returns)),
            "kurtosis": float(stats.kurtosis(returns)),
        }

    # Сортируем по Sharpe
    sorted_tickers = sorted(ticker_stats.items(), key=lambda x: x[1]["sharpe"], reverse=True)

    logger.info(f"Топ-5 по Sharpe: {[t[0] for t in sorted_tickers[:5]]}")

    return {
        "tickers": ticker_stats,
        "top_sharpe": [t[0] for t in sorted_tickers[:10]],
        "worst_sharpe": [t[0] for t in sorted_tickers[-10:]],
    }


def analyze_correlations(df: pd.DataFrame) -> Dict:
    """Анализ корреляций между тикерами."""
    logger.info("Анализ корреляций...")

    # Pivot таблица с returns
    pivot = df.pivot_table(index="ts", columns="secid", values="close")
    returns = pivot.pct_change().dropna()

    # Матрица корреляций
    corr_matrix = returns.corr()

    # Средняя корреляция
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()

    # Наименее коррелированные пары (для диверсификации)
    corr_pairs = []
    for i, t1 in enumerate(corr_matrix.columns):
        for j, t2 in enumerate(corr_matrix.columns):
            if i < j:
                corr_pairs.append((t1, t2, corr_matrix.loc[t1, t2]))

    corr_pairs.sort(key=lambda x: x[2])

    logger.info(f"Средняя корреляция: {avg_corr:.3f}")

    return {
        "avg_correlation": float(avg_corr),
        "least_correlated": [(p[0], p[1], float(p[2])) for p in corr_pairs[:10]],
        "most_correlated": [(p[0], p[1], float(p[2])) for p in corr_pairs[-10:]],
    }


def analyze_patterns(df: pd.DataFrame) -> Dict:
    """Анализ паттернов (momentum, mean reversion)."""
    logger.info("Анализ паттернов...")

    results = {
        "momentum": {},
        "mean_reversion": {},
    }

    for secid in df["secid"].unique()[:10]:  # Топ 10 для скорости
        ticker_df = df[df["secid"] == secid].copy()
        ticker_df["return"] = ticker_df["close"].pct_change()
        ticker_df["prev_return_5"] = ticker_df["return"].rolling(5).sum().shift(1)
        ticker_df["next_return_5"] = ticker_df["return"].rolling(5).sum().shift(-5)

        valid = ticker_df.dropna(subset=["prev_return_5", "next_return_5"])

        if len(valid) < 100:
            continue

        # Корреляция между прошлыми и будущими returns
        corr = valid["prev_return_5"].corr(valid["next_return_5"])

        # Momentum: после роста - рост?
        up_prev = valid[valid["prev_return_5"] > 0]
        if len(up_prev) > 0:
            momentum_signal = (up_prev["next_return_5"] > 0).mean()
        else:
            momentum_signal = 0.5

        # Mean reversion: после падения - рост?
        down_prev = valid[valid["prev_return_5"] < -0.01]
        if len(down_prev) > 0:
            reversion_signal = (down_prev["next_return_5"] > 0).mean()
        else:
            reversion_signal = 0.5

        results["momentum"][secid] = float(momentum_signal)
        results["mean_reversion"][secid] = float(reversion_signal)

    # Средние значения
    avg_momentum = np.mean(list(results["momentum"].values()))
    avg_reversion = np.mean(list(results["mean_reversion"].values()))

    results["avg_momentum_signal"] = float(avg_momentum)
    results["avg_reversion_signal"] = float(avg_reversion)

    logger.info(f"Momentum signal: {avg_momentum:.2%}, Mean reversion: {avg_reversion:.2%}")

    return results


def analyze_volume_patterns(df: pd.DataFrame) -> Dict:
    """Анализ паттернов объёма."""
    logger.info("Анализ объёмных паттернов...")

    df = df.copy()
    df["return"] = df.groupby("secid")["close"].pct_change()
    df["volume_sma"] = df.groupby("secid")["volume"].transform(lambda x: x.rolling(20).mean())
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # Высокий объём (> 2x среднего)
    high_volume = df[df["volume_ratio"] > 2]
    high_vol_return = high_volume["return"].mean()

    # Низкий объём (< 0.5x среднего)
    low_volume = df[df["volume_ratio"] < 0.5]
    low_vol_return = low_volume["return"].mean()

    # Volume spike -> будущий return
    df["next_return"] = df.groupby("secid")["return"].shift(-5)
    volume_spikes = df[df["volume_ratio"] > 3]
    spike_next_return = volume_spikes["next_return"].mean()

    results = {
        "high_volume_avg_return": float(high_vol_return * 100) if not np.isnan(high_vol_return) else 0,
        "low_volume_avg_return": float(low_vol_return * 100) if not np.isnan(low_vol_return) else 0,
        "volume_spike_next_return": float(spike_next_return * 100) if not np.isnan(spike_next_return) else 0,
        "high_volume_candles": len(high_volume),
        "volume_spike_candles": len(volume_spikes),
    }

    logger.info(f"High volume return: {results['high_volume_avg_return']:.4f}%, Spike next return: {results['volume_spike_next_return']:.4f}%")

    return results


def generate_report(analysis: Dict) -> str:
    """Генерация текстового отчёта."""
    report = []
    report.append("=" * 60)
    report.append("ГЛУБОКИЙ АНАЛИЗ MOEX (4 ГОДА)")
    report.append("=" * 60)

    # Общая статистика
    report.append(f"\n📊 ОБЩАЯ СТАТИСТИКА")
    report.append(f"   Всего свечей: {analysis['total_candles']:,}")
    report.append(f"   Период: {analysis['period']}")
    report.append(f"   Тикеров: {analysis['total_tickers']}")

    # Рыночные режимы
    regimes = analysis["regimes"]
    report.append(f"\n📈 РЫНОЧНЫЕ РЕЖИМЫ")
    report.append(f"   Бычий рынок: {regimes['bull_days']} дней ({regimes['bull_days']/regimes['total_days']*100:.1f}%)")
    report.append(f"   Медвежий рынок: {regimes['bear_days']} дней ({regimes['bear_days']/regimes['total_days']*100:.1f}%)")
    report.append(f"   Боковик: {regimes['sideways_days']} дней ({regimes['sideways_days']/regimes['total_days']*100:.1f}%)")
    report.append(f"   Средний дневной return: {regimes['avg_daily_return']:.4f}%")
    report.append(f"   Средняя волатильность: {regimes['avg_volatility']:.2f}%")

    # Сезонность
    season = analysis["seasonality"]
    report.append(f"\n🗓 СЕЗОННОСТЬ")
    report.append(f"   Лучший час: {season['best_hour']}:00")
    report.append(f"   Худший час: {season['worst_hour']}:00")
    report.append(f"   Лучший день: {season['best_weekday']}")
    report.append(f"   Худший день: {season['worst_weekday']}")
    report.append(f"   Лучший месяц: {season['best_month']}")
    report.append(f"   Худший месяц: {season['worst_month']}")

    # Тикеры
    tickers = analysis["tickers"]
    report.append(f"\n🏆 ЛУЧШИЕ ТИКЕРЫ (по Sharpe)")
    for t in tickers["top_sharpe"][:5]:
        stats = tickers["tickers"][t]
        report.append(f"   {t}: Sharpe={stats['sharpe']:.2f}, Vol={stats['volatility']:.1f}%, DD={stats['max_drawdown']:.1f}%")

    report.append(f"\n⚠️ ХУДШИЕ ТИКЕРЫ (по Sharpe)")
    for t in tickers["worst_sharpe"][:5]:
        stats = tickers["tickers"][t]
        report.append(f"   {t}: Sharpe={stats['sharpe']:.2f}, Vol={stats['volatility']:.1f}%, DD={stats['max_drawdown']:.1f}%")

    # Корреляции
    corr = analysis["correlations"]
    report.append(f"\n🔗 КОРРЕЛЯЦИИ")
    report.append(f"   Средняя корреляция: {corr['avg_correlation']:.3f}")
    report.append(f"   Наименее коррелированные:")
    for t1, t2, c in corr["least_correlated"][:3]:
        report.append(f"      {t1} - {t2}: {c:.3f}")

    # Паттерны
    patterns = analysis["patterns"]
    report.append(f"\n🔄 ПАТТЕРНЫ")
    report.append(f"   Momentum signal: {patterns['avg_momentum_signal']:.1%}")
    report.append(f"   Mean reversion signal: {patterns['avg_reversion_signal']:.1%}")

    # Объём
    volume = analysis["volume"]
    report.append(f"\n📊 ОБЪЁМНЫЕ ПАТТЕРНЫ")
    report.append(f"   High volume return: {volume['high_volume_avg_return']:.4f}%")
    report.append(f"   Volume spike → next return: {volume['volume_spike_next_return']:.4f}%")

    report.append("\n" + "=" * 60)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Глубокий анализ MOEX")
    parser.add_argument("--output", default="data/deep_analysis.json", help="Output file")
    args = parser.parse_args()

    cfg = load_config()
    conn = connect(cfg.sqlite_path)

    # Загрузка данных
    df = load_all_candles(conn)

    if len(df) == 0:
        logger.error("Нет данных для анализа")
        return

    # Анализ
    analysis = {
        "total_candles": len(df),
        "total_tickers": df["secid"].nunique(),
        "period": f"{df['ts'].min().date()} - {df['ts'].max().date()}",
        "generated_at": datetime.now().isoformat(),
    }

    analysis["regimes"] = analyze_market_regimes(df)
    analysis["seasonality"] = analyze_seasonality(df)
    analysis["tickers"] = analyze_tickers(df)
    analysis["correlations"] = analyze_correlations(df)
    analysis["patterns"] = analyze_patterns(df)
    analysis["volume"] = analyze_volume_patterns(df)

    # Сохранение
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Анализ сохранён: {output_path}")

    # Генерация отчёта
    report = generate_report(analysis)
    print(report)

    # Сохранение текстового отчёта
    report_path = output_path.with_suffix(".txt")
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Отчёт сохранён: {report_path}")

    conn.close()


if __name__ == "__main__":
    main()
