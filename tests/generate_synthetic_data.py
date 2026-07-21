#!/usr/bin/env python3
"""Generates small, schema-faithful synthetic CSVs for the 8 Home Credit
tables, with real referential integrity (shared SK_ID_CURR, unique
SK_ID_BUREAU/SK_ID_PREV, history tables referencing real previous-application
IDs). Used by CI and local development to exercise the dbt pipeline and the
modeling pipeline without the real (non-redistributable) Kaggle dataset.

Only the columns actually consumed by warehouse/models/staging/*.sql are
generated - the real CSVs have ~122 columns on application_train/test alone;
this fixture is deliberately narrower since it exists to prove the pipeline
logic is correct, not to mimic every column.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT = Path(__file__).resolve().parent / "fixtures" / "raw"


def generate(out_dir: Path, n_applicants: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    sk_id_curr = np.arange(100001, 100001 + n_applicants)
    n_train = int(n_applicants * 0.8)
    is_train = np.zeros(n_applicants, dtype=bool)
    is_train[:n_train] = True
    rng.shuffle(is_train)

    doc_cols = {f"FLAG_DOCUMENT_{i}": rng.binomial(1, 0.02, n_applicants) for i in [2, *range(3, 22)]}
    building_cols = {
        col: np.where(rng.random(n_applicants) < 0.4, np.nan, rng.random(n_applicants))
        for col in [
            "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG",
            "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG",
            "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG",
            "NONLIVINGAREA_AVG",
        ]
    }
    bureau_req_cols = {
        col: rng.poisson(0.3, n_applicants)
        for col in [
            "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
            "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
        ]
    }

    ext_source = lambda: np.where(rng.random(n_applicants) < 0.15, np.nan, rng.random(n_applicants))
    days_employed = rng.integers(-15000, 0, n_applicants).astype(float)
    days_employed[rng.random(n_applicants) < 0.1] = 365243  # known "pensioner" sentinel

    amt_income = rng.lognormal(11.8, 0.5, n_applicants)
    amt_credit = rng.lognormal(12.8, 0.6, n_applicants)

    application = pd.DataFrame({
        "SK_ID_CURR": sk_id_curr,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n_applicants, p=[0.9, 0.1]),
        "CODE_GENDER": rng.choice(["M", "F"], n_applicants),
        "FLAG_OWN_CAR": rng.choice(["Y", "N"], n_applicants),
        "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n_applicants),
        "CNT_CHILDREN": rng.poisson(0.5, n_applicants),
        "AMT_INCOME_TOTAL": amt_income,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_credit / rng.uniform(8, 30, n_applicants),
        "AMT_GOODS_PRICE": amt_credit * rng.uniform(0.85, 1.0, n_applicants),
        "NAME_TYPE_SUITE": rng.choice(["Unaccompanied", "Family", "Spouse, partner"], n_applicants),
        "NAME_INCOME_TYPE": rng.choice(["Working", "Commercial associate", "Pensioner", "State servant"], n_applicants),
        "NAME_EDUCATION_TYPE": rng.choice(
            ["Secondary / secondary special", "Higher education", "Incomplete higher"], n_applicants
        ),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married", "Civil marriage", "Widow"], n_applicants),
        "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents", "Rented apartment"], n_applicants),
        "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.07, n_applicants),
        "DAYS_BIRTH": -rng.integers(7000, 25000, n_applicants),
        "DAYS_EMPLOYED": days_employed,
        "DAYS_REGISTRATION": -rng.integers(0, 15000, n_applicants).astype(float),
        "DAYS_ID_PUBLISH": -rng.integers(0, 6000, n_applicants),
        "OWN_CAR_AGE": np.where(rng.random(n_applicants) < 0.6, np.nan, rng.integers(0, 30, n_applicants)),
        "FLAG_MOBIL": 1,
        "FLAG_EMP_PHONE": rng.binomial(1, 0.8, n_applicants),
        "FLAG_WORK_PHONE": rng.binomial(1, 0.2, n_applicants),
        "FLAG_CONT_MOBILE": rng.binomial(1, 0.95, n_applicants),
        "FLAG_PHONE": rng.binomial(1, 0.3, n_applicants),
        "FLAG_EMAIL": rng.binomial(1, 0.1, n_applicants),
        "OCCUPATION_TYPE": rng.choice(["Laborers", "Sales staff", "Core staff", "Managers", None], n_applicants),
        "CNT_FAM_MEMBERS": rng.integers(1, 5, n_applicants).astype(float),
        "REGION_RATING_CLIENT": rng.integers(1, 4, n_applicants),
        "REGION_RATING_CLIENT_W_CITY": rng.integers(1, 4, n_applicants),
        "WEEKDAY_APPR_PROCESS_START": rng.choice(["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY"], n_applicants),
        "HOUR_APPR_PROCESS_START": rng.integers(6, 21, n_applicants),
        "REG_REGION_NOT_LIVE_REGION": rng.binomial(1, 0.05, n_applicants),
        "REG_REGION_NOT_WORK_REGION": rng.binomial(1, 0.05, n_applicants),
        "LIVE_REGION_NOT_WORK_REGION": rng.binomial(1, 0.05, n_applicants),
        "REG_CITY_NOT_LIVE_CITY": rng.binomial(1, 0.08, n_applicants),
        "REG_CITY_NOT_WORK_CITY": rng.binomial(1, 0.08, n_applicants),
        "LIVE_CITY_NOT_WORK_CITY": rng.binomial(1, 0.08, n_applicants),
        "ORGANIZATION_TYPE": rng.choice(["Business Entity Type 3", "Self-employed", "Government", "XNA"], n_applicants),
        "EXT_SOURCE_1": ext_source(),
        "EXT_SOURCE_2": ext_source(),
        "EXT_SOURCE_3": ext_source(),
        "OBS_30_CNT_SOCIAL_CIRCLE": rng.poisson(1.5, n_applicants),
        "DEF_30_CNT_SOCIAL_CIRCLE": rng.poisson(0.1, n_applicants),
        "OBS_60_CNT_SOCIAL_CIRCLE": rng.poisson(1.5, n_applicants),
        "DEF_60_CNT_SOCIAL_CIRCLE": rng.poisson(0.1, n_applicants),
        "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 4000, n_applicants),
        "TOTALAREA_MODE": rng.uniform(0, 1, n_applicants),
        "WALLSMATERIAL_MODE": rng.choice(["Panel", "Block", "Stone, brick", None], n_applicants),
        "HOUSETYPE_MODE": rng.choice(["block of flats", None], n_applicants),
        "FONDKAPREMONT_MODE": rng.choice(["reg oper account", None], n_applicants),
        "EMERGENCYSTATE_MODE": rng.choice(["No", "Yes", None], n_applicants),
        **doc_cols,
        **building_cols,
        **bureau_req_cols,
    })
    application["TARGET"] = np.where(is_train, rng.binomial(1, 0.08, n_applicants), np.nan)

    app_train = application[is_train].drop(columns=[]).copy()
    app_test = application[~is_train].drop(columns=["TARGET"]).copy()

    app_train.to_csv(out_dir / "application_train.csv", index=False)
    app_test.to_csv(out_dir / "application_test.csv", index=False)

    # --- bureau + bureau_balance ---
    bureau_rows = []
    bureau_balance_rows = []
    sk_id_bureau_counter = 5000001
    for curr_id in sk_id_curr:
        n_bureau = rng.poisson(2)
        for _ in range(n_bureau):
            sk_id_bureau = sk_id_bureau_counter
            sk_id_bureau_counter += 1
            days_credit = -int(rng.integers(0, 3000))
            amt_credit_sum = float(rng.lognormal(11, 1))
            bureau_rows.append({
                "SK_ID_BUREAU": sk_id_bureau,
                "SK_ID_CURR": curr_id,
                "CREDIT_ACTIVE": rng.choice(["Active", "Closed", "Sold"], p=[0.4, 0.55, 0.05]),
                "CREDIT_CURRENCY": "currency 1",
                "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Car loan", "Mortgage"]),
                "DAYS_CREDIT": days_credit,
                "CREDIT_DAY_OVERDUE": int(rng.poisson(0.2)),
                "DAYS_CREDIT_ENDDATE": days_credit + int(rng.integers(30, 2000)),
                "DAYS_ENDDATE_FACT": days_credit + int(rng.integers(30, 2000)) if rng.random() < 0.5 else np.nan,
                "DAYS_CREDIT_UPDATE": -int(rng.integers(0, 365)),
                "CNT_CREDIT_PROLONG": int(rng.poisson(0.05)),
                "AMT_CREDIT_MAX_OVERDUE": float(rng.exponential(500)) if rng.random() < 0.3 else np.nan,
                "AMT_CREDIT_SUM": amt_credit_sum,
                "AMT_CREDIT_SUM_DEBT": amt_credit_sum * rng.uniform(0, 0.9),
                "AMT_CREDIT_SUM_LIMIT": float(rng.exponential(1000)) if rng.random() < 0.3 else np.nan,
                "AMT_CREDIT_SUM_OVERDUE": float(rng.exponential(50)) if rng.random() < 0.05 else 0.0,
                "AMT_ANNUITY": float(rng.exponential(5000)) if rng.random() < 0.5 else np.nan,
            })
            n_months = rng.integers(0, 24)
            for m in range(n_months):
                status = rng.choice(["0", "1", "2", "C", "X"], p=[0.7, 0.1, 0.03, 0.12, 0.05])
                bureau_balance_rows.append({
                    "SK_ID_BUREAU": sk_id_bureau,
                    "MONTHS_BALANCE": -m,
                    "STATUS": status,
                })
    pd.DataFrame(bureau_rows).to_csv(out_dir / "bureau.csv", index=False)
    pd.DataFrame(bureau_balance_rows).to_csv(out_dir / "bureau_balance.csv", index=False)

    # --- previous_application + pos_cash/credit_card/installments ---
    prev_rows, pos_rows, cc_rows, inst_rows = [], [], [], []
    sk_id_prev_counter = 2000001
    for curr_id in sk_id_curr:
        n_prev = rng.poisson(1.5)
        for _ in range(n_prev):
            sk_id_prev = sk_id_prev_counter
            sk_id_prev_counter += 1
            amt_application = float(rng.lognormal(11.5, 0.6))
            amt_credit = amt_application * rng.uniform(0.9, 1.1)
            contract_status = rng.choice(["Approved", "Refused", "Canceled"], p=[0.6, 0.25, 0.15])
            days_decision = -int(rng.integers(0, 2500))
            is_cash = rng.random() < 0.7
            prev_rows.append({
                "SK_ID_PREV": sk_id_prev,
                "SK_ID_CURR": curr_id,
                "NAME_CONTRACT_TYPE": "Cash loans" if is_cash else "Consumer loans",
                "NAME_CONTRACT_STATUS": contract_status,
                "NAME_CASH_LOAN_PURPOSE": "XAP",
                "NAME_PAYMENT_TYPE": "Cash through the bank",
                "CODE_REJECT_REASON": "XAP" if contract_status == "Approved" else rng.choice(["HC", "LIMIT", "SCO"]),
                "NAME_CLIENT_TYPE": rng.choice(["New", "Repeater", "Refreshed"], p=[0.2, 0.7, 0.1]),
                "NAME_GOODS_CATEGORY": "XNA",
                "NAME_PORTFOLIO": "Cash" if is_cash else "POS",
                "NAME_PRODUCT_TYPE": "x-sell",
                "CHANNEL_TYPE": "Credit and cash offices",
                "NAME_SELLER_INDUSTRY": "XNA",
                "NAME_YIELD_GROUP": "middle",
                "PRODUCT_COMBINATION": "Cash X-Sell: middle",
                "AMT_ANNUITY": amt_credit / rng.uniform(6, 24),
                "AMT_APPLICATION": amt_application,
                "AMT_CREDIT": amt_credit,
                "AMT_DOWN_PAYMENT": amt_application * rng.uniform(0, 0.2) if not is_cash else np.nan,
                "AMT_GOODS_PRICE": amt_application,
                "RATE_DOWN_PAYMENT": rng.uniform(0, 0.2) if not is_cash else np.nan,
                "RATE_INTEREST_PRIMARY": np.nan,
                "RATE_INTEREST_PRIVILEGED": np.nan,
                "CNT_PAYMENT": float(rng.integers(6, 36)) if contract_status == "Approved" else np.nan,
                "DAYS_DECISION": days_decision,
                "DAYS_FIRST_DRAWING": 365243,
                "DAYS_FIRST_DUE": days_decision + 10 if contract_status == "Approved" else 365243,
                "DAYS_LAST_DUE_1ST_VERSION": days_decision + 400 if contract_status == "Approved" else 365243,
                "DAYS_LAST_DUE": days_decision + 380 if contract_status == "Approved" else 365243,
                "DAYS_TERMINATION": days_decision + 390 if contract_status == "Approved" else 365243,
                "FLAG_LAST_APPL_PER_CONTRACT": "Y",
                "NFLAG_LAST_APPL_IN_DAY": 1,
                "NFLAG_INSURED_ON_APPROVAL": rng.binomial(1, 0.3) if contract_status == "Approved" else np.nan,
                "SELLERPLACE_AREA": int(rng.integers(-1, 500)),
            })

            if contract_status != "Approved":
                continue

            if is_cash:
                n_months = int(rng.integers(1, 20))
                cnt_instalment = float(rng.integers(6, 36))
                for m in range(n_months):
                    pos_rows.append({
                        "SK_ID_PREV": sk_id_prev,
                        "SK_ID_CURR": curr_id,
                        "MONTHS_BALANCE": -m,
                        "NAME_CONTRACT_STATUS": "Active" if m > 0 else rng.choice(["Active", "Completed"], p=[0.8, 0.2]),
                        "CNT_INSTALMENT": cnt_instalment,
                        "CNT_INSTALMENT_FUTURE": max(cnt_instalment - m, 0),
                        "SK_DPD": int(rng.poisson(0.5)) if rng.random() < 0.1 else 0,
                        "SK_DPD_DEF": 0,
                    })
            else:
                n_months = int(rng.integers(1, 15))
                credit_limit = float(rng.choice([45000, 90000, 135000, 180000]))
                for m in range(n_months):
                    balance = credit_limit * rng.uniform(0, 0.9)
                    cc_rows.append({
                        "SK_ID_PREV": sk_id_prev,
                        "SK_ID_CURR": curr_id,
                        "MONTHS_BALANCE": -m,
                        "NAME_CONTRACT_STATUS": "Active",
                        "AMT_BALANCE": balance,
                        "AMT_CREDIT_LIMIT_ACTUAL": credit_limit,
                        "AMT_DRAWINGS_ATM_CURRENT": float(rng.exponential(2000)),
                        "AMT_DRAWINGS_CURRENT": float(rng.exponential(3000)),
                        "AMT_DRAWINGS_OTHER_CURRENT": 0.0,
                        "AMT_DRAWINGS_POS_CURRENT": float(rng.exponential(1000)),
                        "AMT_INST_MIN_REGULARITY": balance * 0.05,
                        "AMT_PAYMENT_CURRENT": balance * rng.uniform(0.05, 1.1),
                        "AMT_PAYMENT_TOTAL_CURRENT": balance * rng.uniform(0.05, 1.1),
                        "AMT_RECEIVABLE_PRINCIPAL": balance * 0.95,
                        "AMT_RECIVABLE": balance,
                        "AMT_TOTAL_RECEIVABLE": balance,
                        "CNT_DRAWINGS_ATM_CURRENT": int(rng.poisson(0.5)),
                        "CNT_DRAWINGS_CURRENT": int(rng.poisson(1)),
                        "CNT_DRAWINGS_OTHER_CURRENT": 0,
                        "CNT_DRAWINGS_POS_CURRENT": int(rng.poisson(0.5)),
                        "CNT_INSTALMENT_MATURE_CUM": m,
                        "SK_DPD": int(rng.poisson(0.5)) if rng.random() < 0.1 else 0,
                        "SK_DPD_DEF": 0,
                    })

            n_instalments = int(rng.integers(1, 24))
            for i in range(1, n_instalments + 1):
                days_instalment = days_decision + 30 * i
                days_late = int(rng.normal(-2, 5))
                amt_instalment = float(rng.lognormal(8, 0.5))
                inst_rows.append({
                    "SK_ID_PREV": sk_id_prev,
                    "SK_ID_CURR": curr_id,
                    "NUM_INSTALMENT_VERSION": 1,
                    "NUM_INSTALMENT_NUMBER": i,
                    "DAYS_INSTALMENT": days_instalment,
                    "DAYS_ENTRY_PAYMENT": days_instalment + days_late,
                    "AMT_INSTALMENT": amt_instalment,
                    "AMT_PAYMENT": amt_instalment * rng.uniform(0.95, 1.0),
                })

    pd.DataFrame(prev_rows).to_csv(out_dir / "previous_application.csv", index=False)
    pd.DataFrame(pos_rows).to_csv(out_dir / "POS_CASH_balance.csv", index=False)
    pd.DataFrame(cc_rows).to_csv(out_dir / "credit_card_balance.csv", index=False)
    pd.DataFrame(inst_rows).to_csv(out_dir / "installments_payments.csv", index=False)

    print(f"Synthetic fixtures written to {out_dir}")
    print(f"  application_train: {len(app_train)} rows, application_test: {len(app_test)} rows")
    print(f"  bureau: {len(bureau_rows)} rows, bureau_balance: {len(bureau_balance_rows)} rows")
    print(f"  previous_application: {len(prev_rows)} rows")
    print(f"  POS_CASH_balance: {len(pos_rows)} rows, credit_card_balance: {len(cc_rows)} rows")
    print(f"  installments_payments: {len(inst_rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-applicants", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.out_dir, n_applicants=args.n_applicants, seed=args.seed)
