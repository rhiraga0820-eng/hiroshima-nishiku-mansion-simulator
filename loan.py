from __future__ import annotations

import pandas as pd


def normalize_scenario_table(
    table: pd.DataFrame,
    start_col: str,
    end_col: str,
    value_col: str,
    forecast_years: int,
) -> list[dict]:
    if table is None or table.empty:
        return []

    scenarios = []

    clean = table.dropna(subset=[start_col, end_col, value_col])

    for _, row in clean.iterrows():
        start_year = int(row[start_col])
        end_year = int(row[end_col])
        value = float(row[value_col])

        if start_year < 1:
            raise ValueError("開始年は1以上にしてください。")
        if end_year < start_year:
            raise ValueError("終了年は開始年以上にしてください。")
        if start_year > forecast_years:
            continue

        scenarios.append(
            {
                "start_year": start_year,
                "end_year": min(end_year, forecast_years),
                "value": value,
            }
        )

    scenarios.sort(key=lambda x: (x["start_year"], x["end_year"]))

    for previous, current in zip(scenarios, scenarios[1:]):
        if current["start_year"] <= previous["end_year"]:
            raise ValueError("シナリオ期間が重複しています。")

    return scenarios


def _get_value(year: int, scenarios: list[dict], default: float = 0.0) -> float:
    for scenario in scenarios:
        if scenario["start_year"] <= year <= scenario["end_year"]:
            return float(scenario["value"])
    return default


def build_policy_rate_series(
    current_value: float,
    forecast_years: int,
    scenarios: list[dict],
    floor: float = -1.0,
    ceiling: float = 10.0,
) -> list[float]:
    values = [float(current_value)]

    for year in range(1, forecast_years + 1):
        annual_change = _get_value(year, scenarios, 0.0)
        next_value = values[-1] + annual_change
        next_value = min(max(next_value, floor), ceiling)
        values.append(float(next_value))

    return values


def build_cpi_series(
    current_value: float,
    forecast_years: int,
    scenarios: list[dict],
) -> list[float]:
    values = [float(current_value)]

    for year in range(1, forecast_years + 1):
        annual_growth = _get_value(year, scenarios, 0.0)
        next_value = values[-1] * (1 + annual_growth / 100)

        if next_value <= 0:
            raise ValueError("CPIが0以下になるシナリオは設定できません。")

        values.append(float(next_value))

    return values
