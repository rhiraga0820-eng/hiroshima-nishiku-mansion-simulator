from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.loan import build_annual_loan_schedule
from src.prediction import (
    build_model_input,
    check_training_range,
    predict_price,
)
from src.scenarios import (
    build_cpi_series,
    build_policy_rate_series,
    normalize_scenario_table,
)


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "mansion_xgb_interaction_bundle_v5.joblib"


@st.cache_resource
def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "モデルファイルが見つかりません。\n\n"
            f"配置先: {MODEL_PATH}\n\n"
            "Colabで作成した mansion_xgb_interaction_bundle_v5.joblib を "
            "model フォルダへ入れてください。"
        )
    return joblib.load(MODEL_PATH)


def format_man_yen(value_yen: float) -> str:
    return f"{value_yen / 10_000:,.0f}万円"


def preferred_index(options: list, preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


def scenario_default(bundle: dict, key: str, fallback):
    return bundle.get("scenario_defaults", {}).get(key, fallback)


def loan_default(bundle: dict, key: str, fallback):
    return bundle.get("loan_defaults", {}).get(key, fallback)


st.set_page_config(
    page_title="広島マンション価格・売却シミュレーター",
    page_icon="🏢",
    layout="wide",
)

st.title("広島マンション価格・売却シミュレーター")
st.caption(
    "XGBoostによる現在価格・将来価格予測、政策金利・CPIシナリオ、"
    "住宅ローン残債、売却費用控除後の手残りを最大50年先まで試算します。"
)

try:
    bundle = load_bundle()
except Exception as exc:
    st.error("モデルを読み込めませんでした。")
    st.exception(exc)
    st.stop()

metrics = bundle.get("metrics", {})
test_r2 = metrics.get("test_r2", float("nan"))
test_mae = metrics.get("test_mae_man_yen")
if test_mae is None:
    test_mae = metrics.get("test_mae", 0) / 10_000

st.caption(
    f"モデル: {bundle.get('model_version', '不明')} ／ "
    f"Test R²: {test_r2:.3f} ／ "
    f"Test MAE: {test_mae:.1f}万円"
)

layout_options = list(bundle["layout_options"])
station_options = list(bundle["station_options"])
structure_options = list(bundle["structure_options"])
city_plan_options = list(bundle["city_plan_options"])

tab_property, tab_scenario, tab_loan, tab_result, tab_model = st.tabs(
    ["物件条件", "経済シナリオ", "ローン条件", "予測結果", "モデル情報"]
)

with tab_property:
    st.subheader("物件条件")

    c1, c2, c3 = st.columns(3)

    with c1:
        station = st.selectbox(
            "最寄駅",
            station_options,
            index=preferred_index(station_options, "横川"),
        )
        distance = st.number_input(
            "駅徒歩分数",
            min_value=0.0,
            max_value=120.0,
            value=5.0,
            step=1.0,
        )
        layout = st.selectbox(
            "間取り",
            layout_options,
            index=preferred_index(layout_options, "3LDK"),
        )

    with c2:
        area = st.number_input(
            "専有面積（㎡）",
            min_value=10.0,
            max_value=300.0,
            value=70.0,
            step=1.0,
        )
        building_age = st.number_input(
            "現在の築年数",
            min_value=0,
            max_value=120,
            value=10,
            step=1,
        )
        renovation_label = st.radio(
            "改装",
            ["なし", "あり"],
            horizontal=True,
        )
        renovation = 1 if renovation_label == "あり" else 0

    with c3:
        structure = st.selectbox(
            "建物の構造",
            structure_options,
            index=preferred_index(structure_options, "ＲＣ"),
        )
        city_plan = st.selectbox(
            "都市計画",
            city_plan_options,
            index=preferred_index(city_plan_options, "商業"),
        )
        st.info(
            f"駅ランク: {float(bundle['station_rank_map'][station]):.0f}"
        )
        st.caption(
            f"{layout} はモデル内部で部屋数 "
            f"{float(bundle['layout_room_map'][layout]):g} として処理します。"
        )

with tab_scenario:
    st.subheader("政策金利・CPIシナリオ")

    forecast_years = st.slider(
        "予測期間",
        min_value=1,
        max_value=int(scenario_default(bundle, "forecast_years_max", 50)),
        value=25,
        step=1,
    )

    c1, c2 = st.columns(2)
    with c1:
        current_policy_rate = st.number_input(
            "現在の政策金利（%）",
            value=float(scenario_default(bundle, "policy_rate_current", 0.5)),
            step=0.05,
            format="%.2f",
        )
    with c2:
        current_cpi = st.number_input(
            "現在の広島市CPI",
            value=float(scenario_default(bundle, "cpi_current", 105.0)),
            step=0.1,
            format="%.1f",
        )

    st.markdown("#### 政策金利")
    st.caption(
        "年間変化はパーセントポイントです。"
        "例：0.10なら毎年0.10ポイント上昇、-0.05なら毎年0.05ポイント低下。"
    )

    default_policy = pd.DataFrame(
        [
            {"開始年": 1, "終了年": 5, "年間変化（pt）": 0.10},
            {"開始年": 6, "終了年": 15, "年間変化（pt）": 0.00},
            {"開始年": 16, "終了年": 50, "年間変化（pt）": 0.00},
        ]
    )

    policy_scenarios = st.data_editor(
        default_policy,
        num_rows="dynamic",
        use_container_width=True,
        key="policy_scenarios",
        column_config={
            "開始年": st.column_config.NumberColumn(min_value=1, step=1),
            "終了年": st.column_config.NumberColumn(min_value=1, step=1),
            "年間変化（pt）": st.column_config.NumberColumn(step=0.05, format="%.2f"),
        },
    )

    st.markdown("#### 広島市CPI")
    st.caption(
        "年間上昇率は複利で適用します。"
        "例：1.0なら毎年1.0%上昇、-0.5なら毎年0.5%低下。"
    )

    default_cpi = pd.DataFrame(
        [
            {"開始年": 1, "終了年": 10, "年間上昇率（%）": 1.0},
            {"開始年": 11, "終了年": 30, "年間上昇率（%）": 0.5},
            {"開始年": 31, "終了年": 50, "年間上昇率（%）": 0.0},
        ]
    )

    cpi_scenarios = st.data_editor(
        default_cpi,
        num_rows="dynamic",
        use_container_width=True,
        key="cpi_scenarios",
        column_config={
            "開始年": st.column_config.NumberColumn(min_value=1, step=1),
            "終了年": st.column_config.NumberColumn(min_value=1, step=1),
            "年間上昇率（%）": st.column_config.NumberColumn(step=0.1, format="%.2f"),
        },
    )

with tab_loan:
    st.subheader("住宅ローン条件")

    c1, c2, c3 = st.columns(3)

    with c1:
        purchase_price_man = st.number_input(
            "購入価格（万円）",
            min_value=0.0,
            value=float(loan_default(bundle, "purchase_price_yen", 50_000_000)) / 10_000,
            step=100.0,
        )
        down_payment_man = st.number_input(
            "頭金（万円）",
            min_value=0.0,
            value=float(loan_default(bundle, "down_payment_yen", 0)) / 10_000,
            step=100.0,
        )

    with c2:
        loan_term_years = st.number_input(
            "借入期間（年）",
            min_value=1,
            max_value=50,
            value=int(loan_default(bundle, "loan_term_years", 50)),
            step=1,
        )
        loan_rate = st.number_input(
            "住宅ローン金利（年率%・固定）",
            min_value=0.0,
            max_value=20.0,
            value=float(loan_default(bundle, "annual_interest_rate_percent", 2.0)),
            step=0.05,
            format="%.2f",
        )

    with c3:
        bonus_payment_man = st.number_input(
            "ボーナス時の追加返済額（万円／回）",
            min_value=0.0,
            value=float(loan_default(bundle, "bonus_payment_per_time_yen", 0)) / 10_000,
            step=10.0,
        )
        bonus_times = st.selectbox(
            "ボーナス返済回数／年",
            [0, 1, 2],
            index=[0, 1, 2].index(
                int(loan_default(bundle, "bonus_payments_per_year", 2))
            ),
        )
        selling_cost_rate = st.number_input(
            "売却費用率（%）",
            min_value=0.0,
            max_value=20.0,
            value=float(scenario_default(bundle, "selling_cost_rate_percent", 3.5)),
            step=0.1,
            format="%.1f",
        )

    loan_amount = max((purchase_price_man - down_payment_man) * 10_000, 0.0)
    st.metric("借入額", format_man_yen(loan_amount))
    st.caption(
        "ボーナス返済は、毎月返済に加えて元本を追加返済する簡易方式です。"
    )

with tab_result:
    st.subheader("価格・残債・売却後手残り")

    run = st.button(
        "シミュレーションを実行",
        type="primary",
        use_container_width=True,
    )

    if run:
        try:
            policy_table = normalize_scenario_table(
                policy_scenarios,
                start_col="開始年",
                end_col="終了年",
                value_col="年間変化（pt）",
                forecast_years=forecast_years,
            )
            cpi_table = normalize_scenario_table(
                cpi_scenarios,
                start_col="開始年",
                end_col="終了年",
                value_col="年間上昇率（%）",
                forecast_years=forecast_years,
            )

            policy_series = build_policy_rate_series(
                current_value=current_policy_rate,
                forecast_years=forecast_years,
                scenarios=policy_table,
                floor=-1.0,
                ceiling=10.0,
            )
            cpi_series = build_cpi_series(
                current_value=current_cpi,
                forecast_years=forecast_years,
                scenarios=cpi_table,
            )

            prediction_rows = []

            for year in range(forecast_years + 1):
                future_age = building_age + year

                model_input = build_model_input(
                    bundle=bundle,
                    station=station,
                    distance=distance,
                    layout=layout,
                    area=area,
                    building_age=future_age,
                    renovation=renovation,
                    policy_rate=policy_series[year],
                    cpi=cpi_series[year],
                    structure=structure,
                    city_plan=city_plan,
                )

                price = predict_price(bundle, model_input)

                warnings = check_training_range(
                    bundle=bundle,
                    raw_values={
                        "最寄駅ランク（0～10）": float(bundle["station_rank_map"][station]),
                        "最寄駅：距離（分）": float(distance),
                        "部屋数": float(bundle["layout_room_map"][layout]),
                        "面積（㎡）": float(area),
                        "築年数": float(future_age),
                        "改装": float(renovation),
                        "政策金利": float(policy_series[year]),
                        "広島市CPI": float(cpi_series[year]),
                    },
                )

                prediction_rows.append(
                    {
                        "経過年": year,
                        "築年数": future_age,
                        "政策金利（%）": policy_series[year],
                        "広島市CPI": cpi_series[year],
                        "予測売却価格（円）": price,
                        "範囲外警告": " / ".join(warnings),
                    }
                )

            result_df = pd.DataFrame(prediction_rows)

            loan_df = build_annual_loan_schedule(
                principal_yen=loan_amount,
                annual_rate_percent=loan_rate,
                term_years=int(loan_term_years),
                forecast_years=int(forecast_years),
                bonus_payment_per_time_yen=bonus_payment_man * 10_000,
                bonus_payments_per_year=int(bonus_times),
            )

            result_df = result_df.merge(loan_df, on="経過年", how="left")
            result_df["ローン残債（円）"] = result_df["ローン残債（円）"].fillna(0.0)
            result_df["売却費用（円）"] = (
                result_df["予測売却価格（円）"] * selling_cost_rate / 100
            )
            result_df["売却後手残り（円）"] = (
                result_df["予測売却価格（円）"]
                - result_df["売却費用（円）"]
                - result_df["ローン残債（円）"]
            )

            row0 = result_df.iloc[0]
            rowf = result_df.iloc[-1]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("現在の予測価格", format_man_yen(row0["予測売却価格（円）"]))
            m2.metric(
                f"{forecast_years}年後の予測価格",
                format_man_yen(rowf["予測売却価格（円）"]),
            )
            m3.metric(
                f"{forecast_years}年後のローン残債",
                format_man_yen(rowf["ローン残債（円）"]),
            )
            m4.metric(
                f"{forecast_years}年後の売却後手残り",
                format_man_yen(rowf["売却後手残り（円）"]),
            )

            chart_df = result_df.set_index("経過年")[
                ["予測売却価格（円）", "ローン残債（円）", "売却後手残り（円）"]
            ] / 10_000
            chart_df.columns = [
                "予測売却価格（万円）",
                "ローン残債（万円）",
                "売却後手残り（万円）",
            ]
            st.line_chart(chart_df, use_container_width=True)

            display_years = sorted(
                set([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, forecast_years])
            )
            display_years = [year for year in display_years if year <= forecast_years]

            summary = result_df[result_df["経過年"].isin(display_years)].copy()
            summary["予測売却価格（万円）"] = (
                summary["予測売却価格（円）"] / 10_000
            ).round(1)
            summary["ローン残債（万円）"] = (
                summary["ローン残債（円）"] / 10_000
            ).round(1)
            summary["売却費用（万円）"] = (
                summary["売却費用（円）"] / 10_000
            ).round(1)
            summary["売却後手残り（万円）"] = (
                summary["売却後手残り（円）"] / 10_000
            ).round(1)

            st.dataframe(
                summary[
                    [
                        "経過年",
                        "築年数",
                        "政策金利（%）",
                        "広島市CPI",
                        "予測売却価格（万円）",
                        "ローン残債（万円）",
                        "売却費用（万円）",
                        "売却後手残り（万円）",
                        "範囲外警告",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            csv = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "年次結果をCSVでダウンロード",
                data=csv,
                file_name="mansion_forecast_result.csv",
                mime="text/csv",
            )

            warnings_df = result_df[result_df["範囲外警告"] != ""]
            if not warnings_df.empty:
                first_year = int(warnings_df.iloc[0]["経過年"])
                st.warning(
                    f"{first_year}年後以降の一部条件が学習範囲外です。"
                    "長期予測は参考シナリオとして扱ってください。"
                )

        except Exception as exc:
            st.error("シミュレーション中にエラーが発生しました。")
            st.exception(exc)

    else:
        st.info(
            "物件条件、経済シナリオ、ローン条件を設定してから"
            "「シミュレーションを実行」を押してください。"
        )

with tab_model:
    st.subheader("モデル情報")

    st.json(
        {
            "model_version": bundle.get("model_version"),
            "metrics": bundle.get("metrics"),
            "environment": bundle.get("environment"),
        }
    )

    st.markdown("#### 特徴量重要度")

    importance = bundle.get("feature_importance", {})
    if importance:
        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=["特徴量", "重要度"],
        ).sort_values("重要度", ascending=False)

        st.bar_chart(
            importance_df.set_index("特徴量")["重要度"],
            use_container_width=True,
        )
        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("特徴量重要度はモデルファイルに保存されていません。")

    st.markdown("#### 注意事項")
    limitations = bundle.get(
        "limitations",
        [
            "長期予測は学習データ範囲外を含む参考値です。",
            "XGBoostは学習範囲外を線形外挿しません。",
            "税金や修繕費等は完全には反映していません。",
        ],
    )
    for item in limitations:
        st.write(f"- {item}")
