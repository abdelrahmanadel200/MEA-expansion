"""
AMECATH MEA — SAP S/4HANA Intelligent Enterprise Layer Simulator
=================================================================
Simulates three production-grade enterprise intelligence layers:

  Layer 1 — Real-Time Order Prioritization Engine
    Scores and ranks incoming sales orders against live margin,
    backlog urgency, and ATP inventory availability, mirroring the
    logic a VBAK/VBAP + MARD ATP-check would execute in S/4HANA.

  Layer 2 — Market-Linked Profitability Roadmap
    Computes net margin per order/market, dynamically adjusting for
    real-time freight mode costs and spot FX rates, reading from a
    Universal Journal (ACDOCA-style) cost ledger.

  Layer 3 — Predictive Lead Time AI Model
    Trains a scikit-learn GradientBoostingRegressor on historical
    routing, customs delay, and port-bottleneck features to predict
    delivery lead times and ETA dates for incoming orders.

Schemas modelled:
  VBAK / VBAP  — Sales order header / item
  LIKP / LIPS  — Delivery header / item
  MARC / MARD  — Plant material / storage-location stock (ATP)
  ACDOCA       — Universal Journal line items (CO-PA profitability)

Usage:
  python sap_intelligent_layer.py

No external visualization libraries required — runs on pandas,
numpy, and scikit-learn only, with a formatted terminal dashboard.
"""

from __future__ import annotations

import textwrap
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PALETTE / TERMINAL HELPERS  (pure ASCII — no 3rd-party colour libs)
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
DIM  = "\033[2m"
CYAN = "\033[96m"
GREEN= "\033[92m"
GOLD = "\033[93m"
RED  = "\033[91m"
RESET= "\033[0m"


def _hr(char: str = "─", width: int = 80) -> str:
    return char * width


def _banner(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{_hr()}")
    print(f"  {title}")
    print(f"{_hr()}{RESET}")


def _section(title: str) -> None:
    print(f"\n{BOLD}{GOLD}▸ {title}{RESET}")


def _ok(msg: str) -> None:
    print(f"  {GREEN}✔{RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {GOLD}⚠{RESET}  {msg}")


def _table(df: pd.DataFrame, max_rows: int = 20) -> None:
    """Render a DataFrame as a clean fixed-width terminal table."""
    subset = df.head(max_rows).copy()
    col_widths = {c: max(len(str(c)), subset[c].astype(str).map(len).max()) for c in subset.columns}
    header = "  " + "  ".join(str(c).ljust(col_widths[c]) for c in subset.columns)
    print(f"{DIM}{header}{RESET}")
    print(f"  {_hr('-', len(header) - 2)}")
    for _, row in subset.iterrows():
        print("  " + "  ".join(str(row[c]).ljust(col_widths[c]) for c in subset.columns))
    if len(df) > max_rows:
        print(f"  {DIM}… {len(df) - max_rows} additional rows omitted{RESET}")


# ===========================================================================
# SECTION 1 — REFERENCE / MASTER DATA
# ===========================================================================

# Region & country master (mirrors T005 / customer master geography)
COUNTRY_REGION: dict[str, str] = {
    "Saudi Arabia": "GCC",   "UAE": "GCC",   "Qatar": "GCC",
    "Kuwait":       "GCC",   "Bahrain": "GCC","Oman": "GCC",
    "Kenya":        "East Africa",   "Tanzania": "East Africa",
    "Nigeria":      "West Africa",   "Ghana": "West Africa",
    "South Africa": "Southern Africa",
}

# Exchange rates vs USD (live spot simulation — replace with FX-service call)
FX_RATES: dict[str, float] = {
    "USD": 1.00, "SAR": 3.75, "AED": 3.67, "QAR": 3.64,
    "KWD": 0.31, "OMR": 0.38, "BHD": 0.38, "KES": 128.5,
    "NGN": 1550.0, "GHS": 15.8, "ZAR": 18.6, "TZS": 2530.0,
}

# Per-unit air vs sea freight cost lookup by destination country (USD)
FREIGHT_COST: dict[str, dict[str, float]] = {
    "Saudi Arabia": {"Air": 3.80, "Sea": 1.10},
    "UAE":          {"Air": 3.20, "Sea": 0.95},
    "Qatar":        {"Air": 3.50, "Sea": 1.00},
    "Kuwait":       {"Air": 4.00, "Sea": 1.20},
    "Bahrain":      {"Air": 3.60, "Sea": 1.05},
    "Oman":         {"Air": 4.20, "Sea": 1.25},
    "Kenya":        {"Air": 5.10, "Sea": 1.80},
    "Tanzania":     {"Air": 5.40, "Sea": 2.00},
    "Nigeria":      {"Air": 6.20, "Sea": 2.30},
    "Ghana":        {"Air": 6.00, "Sea": 2.20},
    "South Africa": {"Air": 5.80, "Sea": 1.60},
}

# Default freight day assumptions
TRANSIT_DAYS: dict[str, dict[str, int]] = {
    "Air": {"GCC": 3, "East Africa": 4, "West Africa": 5, "Southern Africa": 4},
    "Sea": {"GCC": 22, "East Africa": 30, "West Africa": 35, "Southern Africa": 28},
}

CUSTOMS_DAYS: dict[str, int] = {
    "Saudi Arabia": 4,  "UAE": 2,   "Qatar": 3,   "Kuwait": 4,
    "Bahrain": 3,       "Oman": 5,  "Kenya": 7,   "Tanzania": 9,
    "Nigeria": 14,      "Ghana": 10,"South Africa": 6, "Tanzania": 9,
}

# ---------------------------------------------------------------------------
# Product cost master (mirrors MBEW — material valuation)
# ---------------------------------------------------------------------------
@dataclass
class ProductMaster:
    material_id: str
    description: str
    standard_cost_usd: float   # manufacturing COGS per unit
    list_price_usd: float      # standard sales price
    lead_time_days: int        # production lead time (SAP MARC field)
    shelf_life_days: int       # sterilisation shelf life
    family: str                # "HD_Acute" | "HD_Chronic" | "PD"

PRODUCT_MASTER: dict[str, ProductMaster] = {
    "DLC-1430-A":  ProductMaster("DLC-1430-A",  "Acute HD Catheter 13F/30cm",          22.0,  80.0, 7, 730, "HD_Acute"),
    "DLC-1445-A":  ProductMaster("DLC-1445-A",  "Acute HD Catheter 14F/45cm",          24.5,  87.0, 7, 730, "HD_Acute"),
    "SMART-DLC-A": ProductMaster("SMART-DLC-A", "SMART Grooves HD Catheter Acute",     28.0,  98.0, 8, 730, "HD_Acute"),
    "PERM-P2TC":   ProductMaster("PERM-P2TC",   "Permthane Chronic Twin-Lumen",        55.0, 185.0,10, 730, "HD_Chronic"),
    "PERM-PXDLC":  ProductMaster("PERM-PXDLC",  "Permthane X-Split Tip Chronic",      60.0, 198.0,10, 730, "HD_Chronic"),
    "PD-SWAN-A":   ProductMaster("PD-SWAN-A",   "PD Swan-Neck Catheter + Cuff",       38.0, 135.0, 8, 730, "PD"),
    "PD-COIL-A":   ProductMaster("PD-COIL-A",   "PD Coiled-Tip Catheter + Extension", 40.0, 142.0, 8, 730, "PD"),
}


# ===========================================================================
# SECTION 2 — SAP TABLE SIMULATORS
# ===========================================================================

def build_vbak_vbap(n_orders: int = 40, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate SAP VBAK (sales order header) and VBAP (sales order item).
    Returns (vbak_df, vbap_df).
    """
    rng = np.random.default_rng(seed)
    products   = list(PRODUCT_MASTER.keys())
    countries  = list(FREIGHT_COST.keys())
    modes      = ["Air", "Sea"]
    customers  = [
        "King Faisal Specialist Hospital",  "Rashid Hospital Dubai",
        "Hamad General Hospital",           "Ibn Sina Hospital Kuwait",
        "Muhimbili National Hospital",      "Lagos University Teaching Hospital",
        "Groote Schuur Hospital",           "Aga Khan Nairobi",
        "Al-Ahli Hospital Doha",            "Mediclinic City Dubai",
    ]

    today = date.today()
    order_dates = [today - timedelta(days=int(d)) for d in rng.integers(0, 45, n_orders)]

    vbak_rows = []
    vbap_rows = []

    for i in range(n_orders):
        vbeln = f"ORD-{10000 + i}"
        country  = countries[i % len(countries)]
        region   = COUNTRY_REGION.get(country, "GCC")
        customer = customers[i % len(customers)]
        mode     = modes[rng.integers(0, 2)]
        ord_date = order_dates[i]
        req_date = ord_date + timedelta(days=int(rng.integers(14, 45)))

        vbak_rows.append({
            "VBELN":      vbeln,
            "ERDAT":      ord_date,               # order creation date
            "VDATU":      req_date,               # requested delivery date
            "KUNNR":      customer,               # sold-to party
            "COUNTRY":    country,
            "REGION":     region,
            "SHIP_MODE":  mode,
            "DOC_STATUS": rng.choice(["Open", "Open", "Open", "Partially Delivered"]),
        })

        # 1-2 line items per order
        n_items = rng.integers(1, 3)
        for pos in range(n_items):
            mat = products[rng.integers(0, len(products))]
            pm  = PRODUCT_MASTER[mat]
            qty = int(rng.integers(100, 5001))
            vbap_rows.append({
                "VBELN":       vbeln,
                "POSNR":       (pos + 1) * 10,       # item number (SAP convention)
                "MATNR":       mat,
                "ARKTX":       pm.description,
                "KWMENG":      qty,                   # order qty
                "NETPR":       pm.list_price_usd,     # net price / unit
                "WERKS":       "AMEC_EG",             # plant = AMECATH Egypt
                "COUNTRY":     country,
                "REGION":      region,
                "SHIP_MODE":   mode,
                "PROD_FAMILY": pm.family,
            })

    vbak = pd.DataFrame(vbak_rows)
    vbap = pd.DataFrame(vbap_rows)
    return vbak, vbap


def build_mard_marc(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate MARD (warehouse stock) and MARC (plant-material data).
    Returns (mard_df, marc_df).
    ATP (Available-to-Promise) = unrestricted + in-transit - open demand.
    """
    rng = np.random.default_rng(seed)
    rows_mard, rows_marc = [], []
    plant = "AMEC_EG"
    sloc  = "FG01"          # finished-goods storage location

    for mat_id, pm in PRODUCT_MASTER.items():
        unrestricted  = int(rng.integers(500, 8000))
        in_transit    = int(rng.integers(0, 2000))
        open_demand   = int(rng.integers(200, 3000))
        atp           = max(unrestricted + in_transit - open_demand, 0)

        rows_mard.append({
            "MATNR":        mat_id,
            "WERKS":        plant,
            "LGORT":        sloc,
            "LABST":        unrestricted,      # unrestricted stock
            "TRANSIT":      in_transit,
            "OPEN_DEMAND":  open_demand,
            "ATP":          atp,
        })
        rows_marc.append({
            "MATNR":    mat_id,
            "WERKS":    plant,
            "DZEIT":    pm.lead_time_days,     # in-house production time
            "MTVFP":    "02",                  # checking rule: ATP+planning
            "BESKZ":    "E",                   # procurement type: in-house
        })

    return pd.DataFrame(rows_mard), pd.DataFrame(rows_marc)


def build_acdoca(vbap: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Simulate ACDOCA (Universal Journal) — one CO-PA profitability line per
    sales order item, recording revenue, COGS, freight, and contribution.
    This mirrors S/4HANA's single-source-of-truth ledger concept.
    """
    rng = np.random.default_rng(seed)
    rows = []
    today = date.today()

    for _, item in vbap.iterrows():
        pm      = PRODUCT_MASTER[item["MATNR"]]
        country = item["COUNTRY"]
        mode    = item["SHIP_MODE"]
        qty     = item["KWMENG"]

        fc_map    = FREIGHT_COST.get(country, {"Air": 5.0, "Sea": 2.0})
        fx        = FX_RATES.get("USD", 1.0)   # all in USD for simplicity

        revenue   = qty * pm.list_price_usd
        cogs      = qty * pm.standard_cost_usd
        freight   = qty * fc_map.get(mode, 3.0)
        tax_rate  = rng.uniform(0.05, 0.15)
        tax       = revenue * tax_rate
        contrib   = revenue - cogs - freight
        net_margin_usd = contrib - tax
        net_margin_pct = round(net_margin_usd / revenue * 100, 2) if revenue else 0

        # small Monte-Carlo noise for realism (+/- 3 pp)
        noise = rng.uniform(-0.03, 0.03) * revenue
        net_margin_usd += noise
        net_margin_pct  = round(net_margin_usd / revenue * 100, 2)

        rows.append({
            "RBUKRS":           "AMEC",              # company code
            "GJAHR":            today.year,           # fiscal year
            "DOCNR":            item["VBELN"],        # document number
            "POSNR":            item["POSNR"],
            "MATNR":            item["MATNR"],
            "PROD_FAMILY":      pm.family,
            "COUNTRY":          country,
            "REGION":           item["REGION"],
            "SHIP_MODE":        mode,
            "QTY":              qty,
            "REVENUE_USD":      round(revenue, 2),
            "COGS_USD":         round(cogs, 2),
            "FREIGHT_USD":      round(freight, 2),
            "TAX_USD":          round(tax, 2),
            "NET_MARGIN_USD":   round(net_margin_usd, 2),
            "NET_MARGIN_PCT":   net_margin_pct,
            "FX_RATE":          fx,
            "POST_DATE":        today,
        })

    return pd.DataFrame(rows)


def build_likp_lips(vbak: pd.DataFrame, vbap: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate LIKP (delivery header) and LIPS (delivery item).
    Covers completed and in-transit deliveries for the historical
    training set used by the lead-time ML model.
    """
    rng = np.random.default_rng(seed)
    rows_likp, rows_lips = [], []
    today = date.today()

    for _, order in vbak.iterrows():
        vbeln    = order["VBELN"]
        country  = order["COUNTRY"]
        region   = order["REGION"]
        mode     = order["SHIP_MODE"]
        ord_date = order["ERDAT"]

        transit_base = TRANSIT_DAYS[mode].get(region, 5)
        customs_base = CUSTOMS_DAYS.get(country, 5)
        noise_t  = int(rng.integers(-2, 5))
        noise_c  = int(rng.integers(-1, 7))

        actual_transit  = max(transit_base + noise_t, 1)
        actual_customs  = max(customs_base + noise_c, 1)
        prod_days       = int(rng.integers(5, 14))
        total_lead_time = prod_days + actual_transit + actual_customs

        ship_date    = ord_date + timedelta(days=prod_days)
        delivery_date= ship_date + timedelta(days=actual_transit + actual_customs)

        port_bottleneck = 1 if (country in ["Nigeria", "Ghana"] and mode == "Sea") else 0
        hormuz_risk     = 1 if (region == "GCC" and mode == "Sea") else 0
        customs_delay   = actual_customs

        rows_likp.append({
            "VBELN_DEL":       f"DEL-{vbeln[4:]}",
            "VBELN_ORDER":     vbeln,
            "COUNTRY":         country,
            "REGION":          region,
            "SHIP_MODE":       mode,
            "SHIP_DATE":       ship_date,
            "DELIVERY_DATE":   delivery_date,
            "PROD_DAYS":       prod_days,
            "TRANSIT_DAYS_ACT":actual_transit,
            "CUSTOMS_DAYS_ACT":actual_customs,
            "TOTAL_LEAD_DAYS": total_lead_time,
            "PORT_BOTTLENECK": port_bottleneck,
            "HORMUZ_RISK":     hormuz_risk,
        })

        items = vbap[vbap["VBELN"] == vbeln]
        for _, item in items.iterrows():
            rows_lips.append({
                "VBELN_DEL": f"DEL-{vbeln[4:]}",
                "POSNR":     item["POSNR"],
                "MATNR":     item["MATNR"],
                "LFIMG":     item["KWMENG"],   # delivery qty
            })

    return pd.DataFrame(rows_likp), pd.DataFrame(rows_lips)


# ===========================================================================
# LAYER 1 — REAL-TIME ORDER PRIORITIZATION ENGINE
# ===========================================================================

def compute_priority_score(
    order_row: pd.Series,
    acdoca: pd.DataFrame,
    mard: pd.DataFrame,
) -> dict:
    """
    Score a single incoming order on three sub-dimensions:

      Margin Score     (0-40 pts): contribution margin as % of revenue
      Urgency Score    (0-35 pts): days to requested delivery date
      ATP Score        (0-25 pts): stock availability vs order quantity

    Returns a dict of sub-scores and a composite priority score (0-100).
    """
    vbeln   = order_row["VBELN"]
    req_date= order_row["VDATU"]
    country = order_row["COUNTRY"]

    # --- Margin sub-score ---
    acdo_items = acdoca[acdoca["DOCNR"] == vbeln]
    if acdo_items.empty:
        margin_pct = 0.0
    else:
        tot_rev  = acdo_items["REVENUE_USD"].sum()
        tot_nm   = acdo_items["NET_MARGIN_USD"].sum()
        margin_pct = (tot_nm / tot_rev * 100) if tot_rev else 0.0

    # Sigmoid-style scoring: >35% → full 40 pts, <10% → near 0
    margin_score = min(40, max(0, (margin_pct / 35) * 40))

    # --- Urgency sub-score ---
    days_to_req = (req_date - date.today()).days
    if days_to_req <= 7:
        urgency_score = 35
    elif days_to_req <= 14:
        urgency_score = 28
    elif days_to_req <= 30:
        urgency_score = 18
    elif days_to_req <= 60:
        urgency_score = 8
    else:
        urgency_score = 2

    # --- ATP (Available-to-Promise) sub-score ---
    # Sum required qty across all line items; check against available stock
    items_needed = acdoca[acdoca["DOCNR"] == vbeln][["MATNR", "QTY"]]
    atp_ratios = []
    for _, need in items_needed.iterrows():
        stock = mard.loc[mard["MATNR"] == need["MATNR"], "ATP"]
        atp_val = int(stock.values[0]) if not stock.empty else 0
        ratio = min(atp_val / need["QTY"], 1.0) if need["QTY"] > 0 else 1.0
        atp_ratios.append(ratio)
    avg_atp_ratio = np.mean(atp_ratios) if atp_ratios else 0.0
    atp_score = avg_atp_ratio * 25

    composite = round(margin_score + urgency_score + atp_score, 2)

    return {
        "VBELN":           vbeln,
        "COUNTRY":         country,
        "REQ_DATE":        req_date,
        "DAYS_TO_REQ":     days_to_req,
        "NET_MARGIN_PCT":  round(margin_pct, 2),
        "MARGIN_SCORE":    round(margin_score, 2),
        "URGENCY_SCORE":   round(urgency_score, 2),
        "ATP_RATIO":       round(avg_atp_ratio, 3),
        "ATP_SCORE":       round(atp_score, 2),
        "PRIORITY_SCORE":  composite,
        "PRIORITY_BAND":   (
            "🔴 CRITICAL" if composite >= 75 else
            "🟡 HIGH"     if composite >= 50 else
            "🟢 STANDARD" if composite >= 25 else
            "⬜ DEFER"
        ),
    }


def run_prioritization_engine(
    vbak: pd.DataFrame,
    acdoca: pd.DataFrame,
    mard: pd.DataFrame,
) -> pd.DataFrame:
    """Layer 1 entry point. Scores all open orders and returns a sorted queue."""
    open_orders = vbak[vbak["DOC_STATUS"].str.startswith("Open")]
    scores = [compute_priority_score(row, acdoca, mard) for _, row in open_orders.iterrows()]
    queue  = pd.DataFrame(scores).sort_values("PRIORITY_SCORE", ascending=False).reset_index(drop=True)
    queue.index += 1   # rank from 1
    queue.index.name = "RANK"
    return queue


# ===========================================================================
# LAYER 2 — MARKET-LINKED PROFITABILITY ROADMAP
# ===========================================================================

def compute_profitability_roadmap(
    acdoca: pd.DataFrame,
    freight_shock_pct: float = 0.0,
    fx_override: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Aggregate ACDOCA into a market-level P&L, applying:
      - Optional freight cost shock (e.g., +30% for Hormuz disruption)
      - Optional spot FX override for currency sensitivity analysis

    Parameters
    ----------
    acdoca            : Universal Journal dataframe from build_acdoca()
    freight_shock_pct : percentage increase to apply to FREIGHT_USD (e.g. 0.30)
    fx_override       : dict of {country: multiplier} for FX adjustment

    Returns
    -------
    DataFrame with one row per COUNTRY×PROD_FAMILY, including
    contribution margin, adjusted net margin, and margin band.
    """
    df = acdoca.copy()

    # Apply freight shock
    if freight_shock_pct != 0.0:
        df["FREIGHT_USD"] = df["FREIGHT_USD"] * (1 + freight_shock_pct)
        df["NET_MARGIN_USD"] = df["REVENUE_USD"] - df["COGS_USD"] - df["FREIGHT_USD"] - df["TAX_USD"]

    # Apply FX override (adjusts revenue USD)
    if fx_override:
        for country, multiplier in fx_override.items():
            mask = df["COUNTRY"] == country
            df.loc[mask, "REVENUE_USD"] *= multiplier
            df.loc[mask, "NET_MARGIN_USD"] = (
                df.loc[mask, "REVENUE_USD"]
                - df.loc[mask, "COGS_USD"]
                - df.loc[mask, "FREIGHT_USD"]
                - df.loc[mask, "TAX_USD"]
            )

    # Aggregate by Country × Product Family
    grp = df.groupby(["COUNTRY", "REGION", "PROD_FAMILY", "SHIP_MODE"]).agg(
        TOTAL_REVENUE_USD  = ("REVENUE_USD",    "sum"),
        TOTAL_COGS_USD     = ("COGS_USD",       "sum"),
        TOTAL_FREIGHT_USD  = ("FREIGHT_USD",    "sum"),
        TOTAL_TAX_USD      = ("TAX_USD",        "sum"),
        NET_MARGIN_USD     = ("NET_MARGIN_USD", "sum"),
        ORDER_COUNT        = ("DOCNR",          "nunique"),
    ).reset_index()

    grp["NET_MARGIN_PCT"] = (grp["NET_MARGIN_USD"] / grp["TOTAL_REVENUE_USD"] * 100).round(2)
    grp["FREIGHT_AS_PCT_REV"] = (grp["TOTAL_FREIGHT_USD"] / grp["TOTAL_REVENUE_USD"] * 100).round(2)

    grp["MARGIN_BAND"] = grp["NET_MARGIN_PCT"].apply(
        lambda m: "🟢 HEALTHY (>30%)"    if m > 30 else
                  "🟡 SQUEEZED (20-30%)" if m >= 20 else
                  "🔴 AT RISK (<20%)"
    )

    return grp.sort_values("NET_MARGIN_USD", ascending=False).reset_index(drop=True)


# ===========================================================================
# LAYER 3 — PREDICTIVE LEAD TIME AI MODEL
# ===========================================================================

REGION_ENCODER = LabelEncoder()
MODE_ENCODER   = LabelEncoder()

FEATURE_COLS = [
    "SHIP_MODE_ENC", "REGION_ENC", "PROD_DAYS",
    "PORT_BOTTLENECK", "HORMUZ_RISK", "CUSTOMS_DAYS_ACT",
]


def train_lead_time_model(
    likp: pd.DataFrame,
) -> tuple[GradientBoostingRegressor, float]:
    """
    Train a GradientBoostingRegressor on historical delivery data (LIKP).
    Features: shipping mode, region, production days, port/Hormuz risk,
              customs delay.
    Target:   TOTAL_LEAD_DAYS (actual end-to-end lead time).

    Returns (fitted_model, MAE_days).
    """
    df = likp.copy()
    df["SHIP_MODE_ENC"] = MODE_ENCODER.fit_transform(df["SHIP_MODE"])
    df["REGION_ENC"]    = REGION_ENCODER.fit_transform(df["REGION"])

    X = df[FEATURE_COLS].values
    y = df["TOTAL_LEAD_DAYS"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)

    return model, round(mae, 2)


def predict_lead_time(
    model: GradientBoostingRegressor,
    country: str,
    region: str,
    ship_mode: str,
    prod_days: int,
    order_date: date = date.today(),
) -> dict:
    """
    Predict delivery lead time and ETA for a single incoming order.

    Parameters
    ----------
    model       : fitted GradientBoostingRegressor from train_lead_time_model()
    country     : destination country (used for customs profile)
    region      : GCC / East Africa / West Africa / Southern Africa
    ship_mode   : "Air" or "Sea"
    prod_days   : estimated production days from factory calendar
    order_date  : order creation date (defaults to today)

    Returns
    -------
    dict with predicted lead time, component breakdown, and ETA date.
    """
    port_bottleneck = 1 if (country in ["Nigeria", "Ghana"] and ship_mode == "Sea") else 0
    hormuz_risk     = 1 if (region == "GCC" and ship_mode == "Sea") else 0
    customs_est     = CUSTOMS_DAYS.get(country, 5)

    try:
        mode_enc   = int(MODE_ENCODER.transform([ship_mode])[0])
        region_enc = int(REGION_ENCODER.transform([region])[0])
    except ValueError:
        # Unseen label — use nearest class
        mode_enc   = 0
        region_enc = 0

    features = np.array([[mode_enc, region_enc, prod_days,
                           port_bottleneck, hormuz_risk, customs_est]])
    predicted_days = max(int(round(float(model.predict(features)[0]))), 1)
    eta_date       = order_date + timedelta(days=predicted_days)

    return {
        "COUNTRY":            country,
        "REGION":             region,
        "SHIP_MODE":          ship_mode,
        "PROD_DAYS":          prod_days,
        "PORT_BOTTLENECK":    port_bottleneck,
        "HORMUZ_RISK":        hormuz_risk,
        "CUSTOMS_EST_DAYS":   customs_est,
        "PREDICTED_LEAD_DAYS":predicted_days,
        "ETA_DATE":           eta_date.strftime("%Y-%m-%d"),
        "RISK_FLAG": (
            "⛔ HIGH RISK — consider Air Freight" if (hormuz_risk and ship_mode == "Sea") else
            "⚠ ELEVATED — port delays possible"   if port_bottleneck else
            "✅ NORMAL"
        ),
    }


# ===========================================================================
# TERMINAL DASHBOARD
# ===========================================================================

def print_layer_1(queue: pd.DataFrame) -> None:
    _banner("LAYER 1 — REAL-TIME ORDER PRIORITIZATION ENGINE")

    _section("Priority Queue (Top 15 Open Orders)")
    display_cols = [
        "VBELN", "COUNTRY", "DAYS_TO_REQ", "NET_MARGIN_PCT",
        "ATP_RATIO", "PRIORITY_SCORE", "PRIORITY_BAND",
    ]
    _table(queue[display_cols].head(15))

    _section("Band Summary")
    for band, grp in queue.groupby("PRIORITY_BAND"):
        print(f"  {band:<25}  {len(grp):>3} orders")

    _section("Highest-Priority Order — Detail")
    top = queue.iloc[0]
    for k, v in top.items():
        print(f"  {DIM}{k:<25}{RESET}  {v}")


def print_layer_2(roadmap: pd.DataFrame, roadmap_shocked: pd.DataFrame) -> None:
    _banner("LAYER 2 — MARKET-LINKED PROFITABILITY ROADMAP (ACDOCA Universal Journal)")

    _section("Baseline Profitability by Country × Product Family")
    cols = ["COUNTRY", "PROD_FAMILY", "SHIP_MODE",
            "TOTAL_REVENUE_USD", "NET_MARGIN_USD", "NET_MARGIN_PCT",
            "FREIGHT_AS_PCT_REV", "MARGIN_BAND"]
    _table(roadmap[cols].head(15))

    _section("Hormuz Disruption Scenario (+30% Sea Freight Cost Shock)")
    merged = roadmap[["COUNTRY", "PROD_FAMILY", "SHIP_MODE", "NET_MARGIN_PCT"]].copy()
    merged = merged.rename(columns={"NET_MARGIN_PCT": "BASELINE_MARGIN_%"})
    shocked_sub = roadmap_shocked[["COUNTRY", "PROD_FAMILY", "SHIP_MODE", "NET_MARGIN_PCT", "MARGIN_BAND"]].copy()
    shocked_sub = shocked_sub.rename(columns={"NET_MARGIN_PCT": "SHOCKED_MARGIN_%"})
    delta = merged.merge(shocked_sub, on=["COUNTRY", "PROD_FAMILY", "SHIP_MODE"], how="inner")
    delta["MARGIN_DELTA_%"] = (delta["SHOCKED_MARGIN_%"] - delta["BASELINE_MARGIN_%"]).round(2)
    sea_delta = delta[delta["SHIP_MODE"] == "Sea"].sort_values("MARGIN_DELTA_%")
    _table(sea_delta[["COUNTRY","PROD_FAMILY","BASELINE_MARGIN_%","SHOCKED_MARGIN_%","MARGIN_DELTA_%","MARGIN_BAND"]].head(12))

    at_risk = roadmap_shocked[roadmap_shocked["MARGIN_BAND"].str.startswith("🔴")]
    if not at_risk.empty:
        _warn(f"{len(at_risk)} country×product combinations fall below 20% net margin under shock scenario.")


def print_layer_3(
    mae: float,
    model: GradientBoostingRegressor,
    predictions: list[dict],
) -> None:
    _banner("LAYER 3 — PREDICTIVE LEAD TIME & ETA AI MODEL (GradientBoostingRegressor)")

    _section("Model Performance")
    _ok(f"Training complete — Mean Absolute Error: {mae} days")

    _section("Feature Importance (TOTAL_LEAD_DAYS drivers)")
    importances = model.feature_importances_
    for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {feat:<22}  {bar:<26}  {imp:.3f}")

    _section("Incoming Order ETA Predictions")
    pred_df = pd.DataFrame(predictions)
    _table(pred_df[["COUNTRY","SHIP_MODE","PROD_DAYS","PREDICTED_LEAD_DAYS","ETA_DATE","RISK_FLAG"]])


def print_summary(queue: pd.DataFrame, roadmap: pd.DataFrame) -> None:
    _banner("EXECUTIVE SUMMARY DASHBOARD")

    total_rev = roadmap["TOTAL_REVENUE_USD"].sum()
    avg_margin = roadmap["NET_MARGIN_PCT"].mean()
    critical_count = (queue["PRIORITY_BAND"] == "🔴 CRITICAL").sum()
    at_risk_markets = roadmap[roadmap["MARGIN_BAND"].str.startswith("🔴")]["COUNTRY"].nunique()

    rows = [
        ("Total Pipeline Revenue (USD)",     f"${total_rev:>14,.0f}"),
        ("Average Net Margin (%)",            f"{avg_margin:>14.1f}%"),
        ("CRITICAL Priority Orders",          f"{critical_count:>14}"),
        ("Markets at Margin Risk (<20%)",     f"{at_risk_markets:>14}"),
    ]
    print(f"\n  {'Metric':<40} {'Value':>15}")
    print(f"  {_hr('-', 57)}")
    for label, val in rows:
        print(f"  {label:<40} {val:>15}")

    print(f"\n  {BOLD}Recommended Actions:{RESET}")
    if critical_count > 0:
        print(f"  {RED}▸ Release CRITICAL orders to logistics immediately — {critical_count} orders pending{RESET}")
    if at_risk_markets > 0:
        print(f"  {GOLD}▸ Review freight mode for {at_risk_markets} at-risk market(s) — switch Sea→Air to recover margin{RESET}")
    print(f"  {GREEN}▸ Replenish ATP stock for top 5 priority orders before next MRP run{RESET}")
    print()


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def main() -> None:
    _banner("AMECATH MEA — SAP S/4HANA INTELLIGENT ENTERPRISE LAYER SIMULATOR")
    print(f"  {DIM}Initialising data layer …{RESET}")

    # ---- Build SAP table snapshots ----
    vbak, vbap = build_vbak_vbap(n_orders=60)
    mard, marc = build_mard_marc()
    acdoca     = build_acdoca(vbap)
    likp, lips = build_likp_lips(vbak, vbap)

    _ok(f"VBAK/VBAP  : {len(vbak)} orders / {len(vbap)} items")
    _ok(f"MARD/MARC  : {len(mard)} material-plant records")
    _ok(f"ACDOCA     : {len(acdoca)} Universal Journal lines")
    _ok(f"LIKP/LIPS  : {len(likp)} deliveries / {len(lips)} items")

    # ---- Layer 1 ----
    queue = run_prioritization_engine(vbak, acdoca, mard)
    print_layer_1(queue)

    # ---- Layer 2 ----
    roadmap         = compute_profitability_roadmap(acdoca)
    roadmap_shocked = compute_profitability_roadmap(acdoca, freight_shock_pct=0.30)
    print_layer_2(roadmap, roadmap_shocked)

    # ---- Layer 3 ----
    model, mae = train_lead_time_model(likp)

    incoming_orders = [
        {"country": "Saudi Arabia", "region": "GCC",            "mode": "Air", "prod_days": 8},
        {"country": "Saudi Arabia", "region": "GCC",            "mode": "Sea", "prod_days": 8},
        {"country": "Nigeria",      "region": "West Africa",    "mode": "Air", "prod_days": 10},
        {"country": "Nigeria",      "region": "West Africa",    "mode": "Sea", "prod_days": 10},
        {"country": "Kenya",        "region": "East Africa",    "mode": "Air", "prod_days": 7},
        {"country": "South Africa", "region": "Southern Africa","mode": "Air", "prod_days": 9},
    ]
    predictions = [
        predict_lead_time(model, o["country"], o["region"], o["mode"], o["prod_days"])
        for o in incoming_orders
    ]
    print_layer_3(mae, model, predictions)

    # ---- Executive summary ----
    print_summary(queue, roadmap)


if __name__ == "__main__":
    main()
