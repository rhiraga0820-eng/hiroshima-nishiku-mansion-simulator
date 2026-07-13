from __future__ import annotations

import pandas as pd


def build_model_input(
    bundle: dict,
    station: str,
    distance: float,
    layout: str,
    area: float,
    building_age: float,
    renovation: int,
    policy_rate: float,
    cpi: float,
    structure: str,
    city_plan: str,
) -> pd.DataFrame:
    station_rank = float(bundle["station_rank_map"][station])
    rooms = float(bundle["layout_room_map"][layout])

    interaction = bundle.get("interaction", {})
    rank_mean = float(interaction["rank_mean"])
    distance_mean = float(interaction["distance_mean"])

    rank_centered = station_rank - rank_mean
    distance_centered = float(distance) - distance_mean
    interaction_value = rank_centered * distance_centered

    input_df = pd.DataFrame(
        {
            "駅ランク_c": [rank_centered],
            "徒歩分数_c": [distance_centered],
            "駅ランク×徒歩": [interaction_value],
            "部屋数": [rooms],
            "面積（㎡）": [float(area)],
            "築年数": [float(building_age)],
            "改装": [int(renovation)],
            "政策金利": [float(policy_rate)],
            "広島市CPI": [float(cpi)],
            "建物の構造": [str(structure)],
            "都市計画": [str(city_plan)],
        }
    )

    return input_df[bundle["model_features"]]


def predict_price(bundle: dict, model_input: pd.DataFrame) -> float:
    value = float(bundle["model"].predict(model_input)[0])
    return max(value, 0.0)


def check_training_range(bundle: dict, raw_values: dict) -> list[str]:
    warnings = []
    training_range = bundle.get("training_range", {})

    for column, value in raw_values.items():
        limits = training_range.get(column)
        if not limits:
            continue

        minimum = float(limits["min"])
        maximum = float(limits["max"])

        if value < minimum or value > maximum:
            warnings.append(
                f"{column}={value:.2f}（学習範囲 {minimum:.2f}～{maximum:.2f}）"
            )

    return warnings
