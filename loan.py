from __future__ import annotations

import pandas as pd


def calculate_monthly_payment(
    principal_yen: float,
    annual_rate_percent: float,
    term_months: int,
) -> float:
    if principal_yen <= 0 or term_months <= 0:
        return 0.0

    monthly_rate = annual_rate_percent / 100 / 12

    if monthly_rate == 0:
        return principal_yen / term_months

    factor = (1 + monthly_rate) ** term_months

    return (
        principal_yen
        * monthly_rate
        * factor
        / (factor - 1)
    )


def build_annual_loan_schedule(
    principal_yen: float,
    annual_rate_percent: float,
    term_years: int,
    forecast_years: int,
    bonus_payment_per_time_yen: float = 0.0,
    bonus_payments_per_year: int = 0,
) -> pd.DataFrame:
    principal_yen = max(float(principal_yen), 0.0)
    term_months = int(term_years * 12)
    monthly_rate = annual_rate_percent / 100 / 12

    payment = calculate_monthly_payment(
        principal_yen=principal_yen,
        annual_rate_percent=annual_rate_percent,
        term_months=term_months,
    )

    if bonus_payments_per_year == 1:
        bonus_months = {12}
    elif bonus_payments_per_year == 2:
        bonus_months = {6, 12}
    else:
        bonus_months = set()

    balance = principal_yen

    rows = [
        {
            "経過年": 0,
            "ローン残債（円）": balance,
        }
    ]

    total_simulation_months = int(forecast_years * 12)

    for month in range(1, total_simulation_months + 1):
        if month > term_months or balance <= 0:
            balance = 0.0

        else:
            interest = balance * monthly_rate
            principal_payment = max(payment - interest, 0.0)
            balance = max(balance - principal_payment, 0.0)

            month_in_year = ((month - 1) % 12) + 1

            if (
                month_in_year in bonus_months
                and bonus_payment_per_time_yen > 0
                and balance > 0
            ):
                balance = max(
                    balance - float(bonus_payment_per_time_yen),
                    0.0,
                )

        if month % 12 == 0:
            rows.append(
                {
                    "経過年": month // 12,
                    "ローン残債（円）": balance,
                }
            )

    return pd.DataFrame(rows)
