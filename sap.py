from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AMECATH MEA — SAP S/4HANA Intelligent Simulator",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# REFERENCE / MASTER DATA
# ---------------------------------------------------------------------------
COUNTRY_REGION: dict[str, str] = {
    "Saudi Arabia": "GCC",
    "UAE": "GCC",
    "Qatar": "GCC",
    "Kuwait": "GCC",
    "Bahrain": "GCC",
    "Oman": "GCC",
    "Kenya": "East Africa",
    "Tanzania": "East Africa",
    "Nigeria": "West Africa",
    "Ghana": "West Africa",
    "South Africa": "Southern Africa",
}

FX_RATES: dict[str, float] = {
    "USD": 1.00,
    "SAR": 3.75,
    "AED": 3.67,
    "QAR": 3.64,
    "KWD": 0.31,
    "OMR": 0.38,
    "BHD": 0.38,
    "KES": 128.5,
    "NGN": 1550.0,
    "GHS": 15.8,
    "ZAR": 18.6,
    "TZS": 2530.0,
}

FREIGHT_COST: dict[str, dict[str, float]] = {
    "Saudi Arabia": {"Air": 3.80, "Sea": 1.10},
    "UAE": {"Air": 3.20, "Sea": 0.95},
    "Qatar": {"Air": 3.50, "Sea": 1.00},
    "Kuwait": {"Air": 4.00, "Sea": 1.20},
    "Bahrain": {"Air": 3.60, "Sea": 1.05},
    "Oman": {"Air": 4.20, "Sea": 1.25},
    "Kenya": {"Air": 5.10, "Sea": 1.80},
    "Tanzania": {"Air": 5.40, "Sea": 2.00},
    "Nigeria": {"Air": 6.20, "Sea": 2.30},
    "Ghana": {"Air": 6.00, "Sea": 2.20},
    "South Africa": {"Air": 5.80, "Sea": 1.60},
}

TRANSIT_DAYS: dict[str, dict[str, int]] = {
    "Air": {"GCC": 3, "East Africa": 4, "West Africa": 5, "Southern Africa": 4},
    "Sea": {
        "GCC": 22,
        "East Africa": 30,
        "West Africa": 35,
        "Southern Africa": 28,
    },
}

CUSTOMS_DAYS: dict[str, int] = {
    "Saudi Arabia": 4,
    "UAE": 2,
    "Qatar": 3,
    "Kuwait": 4,
    "Bahrain": 3,
    "Oman": 5,
    "Kenya": 7,
    "Tanzania": 9,
    "Nigeria": 14,
    "Ghana": 10,
    "South Africa": 6,
}


@dataclass
class ProductMaster:
  material_id: str
  description: str
  standard_cost_usd: float
  list_price_usd: float
  lead_time_days: int
  shelf_life_days: int
  family: str


PRODUCT_MASTER: dict[str, ProductMaster] = {
  "DLC-1430-A": ProductMaster(
      "DLC-1430-A", "Acute HD Catheter 13F/30cm", 22.0, 80.0, 7, 730, "HD_Acute"
  ),
  "DLC-1445-A": ProductMaster(
      "DLC-1445-A", "Acute HD Catheter 14F/45cm", 24.5, 87.0, 7, 730, "HD_Acute"
  ),
  "SMART-DLC-A": ProductMaster(
      "SMART-DLC-A",
      "SMART Grooves HD Catheter Acute",
      28.0,
      98.0,
      8,
      730,
      "HD_Acute",
  ),
  "PERM-P2TC": ProductMaster(
      "PERM-P2TC", "Permthane Chronic Twin-Lumen", 55.0, 185.0, 10, 730, "HD_Chronic"
  ),
  "PERM-PXDLC": ProductMaster(
      "PERM-PXDLC",
      "Permthane X-Split Tip Chronic",
      60.0,
      198.0,
      10,
      730,
      "HD_Chronic",
  ),
  "PD-SWAN-A": ProductMaster(
      "PD-SWAN-A", "PD Swan-Neck Catheter + Cuff", 38.0, 135.0, 8, 730, "PD"
  ),
  "PD-COIL-A": ProductMaster(
      "PD-COIL-A",
      "PD Coiled-Tip Catheter + Extension",
      40.0,
      142.0,
      8,
      730,
      "PD",
  ),
}

# ---------------------------------------------------------------------------
# SAP TABLE SIMULATORS (Cached for Performance)
# ---------------------------------------------------------------------------


@st.cache_data
def load_sap_data():
  rng = np.random.default_rng(42)
  products = list(PRODUCT_MASTER.keys())
  countries = list(FREIGHT_COST.keys())
  modes = ["Air", "Sea"]
  customers = [
      "King Faisal Specialist Hospital",
      "Rashid Hospital Dubai",
      "Hamad General Hospital",
      "Ibn Sina Hospital Kuwait",
      "Muhimbili National Hospital",
      "Lagos University Teaching Hospital",
      "Groote Schuur Hospital",
      "Aga Khan Nairobi",
      "Al-Ahli Hospital Doha",
      "Mediclinic City Dubai",
  ]

  today = date.today()
  n_orders = 60
  order_dates = [
      today - timedelta(days=int(d)) for d in rng.integers(0, 45, n_orders)
  ]

  vbak_rows, vbap_rows = [], []
  for i in range(n_orders):
    vbeln = f"ORD-{10000 + i}"
    country = countries[i % len(countries)]
    region = COUNTRY_REGION.get(country, "GCC")
    customer = customers[i % len(customers)]
    mode = modes[rng.integers(0, 2)]
    ord_date = order_dates[i]
    req_date = ord_date + timedelta(days=int(rng.integers(14, 45)))

    vbak_rows.append({
        "VBELN": vbeln,
        "ERDAT": ord_date,
        "VDATU": req_date,
        "KUNNR": customer,
        "COUNTRY": country,
        "REGION": region,
        "SHIP_MODE": mode,
        "DOC_STATUS": rng.choice(
            ["Open", "Open", "Open", "Partially Delivered"]
        ),
    })

    for pos in range(rng.integers(1, 3)):
      mat = products[rng.integers(0, len(products))]
      pm = PRODUCT_MASTER[mat]
      qty = int(rng.integers(100, 5001))
      vbap_rows.append({
          "VBELN": vbeln,
          "POSNR": (pos + 1) * 10,
          "MATNR": mat,
          "ARKTX": pm.description,
          "KWMENG": qty,
          "NETPR": pm.list_price_usd,
          "WERKS": "AMEC_EG",
          "COUNTRY": country,
          "REGION": region,
          "SHIP_MODE": mode,
          "PROD_FAMILY": pm.family,
      })

  vbak = pd.DataFrame(vbak_rows)
  vbap = pd.DataFrame(vbap_rows)

  # MARD / MARC
  rows_mard, rows_marc = [], []
  for mat_id, pm in PRODUCT_MASTER.items():
    unrestricted = int(rng.integers(500, 8000))
    in_transit = int(rng.integers(0, 2000))
    open_demand = int(rng.integers(200, 3000))
    atp = max(unrestricted + in_transit - open_demand, 0)
    rows_mard.append({
        "MATNR": mat_id,
        "WERKS": "AMEC_EG",
        "LGORT": "FG01",
        "LABST": unrestricted,
        "TRANSIT": in_transit,
        "OPEN_DEMAND": open_demand,
        "ATP": atp,
    })
    rows_marc.append({
        "MATNR": mat_id,
        "WERKS": "AMEC_EG",
        "DZEIT": pm.lead_time_days,
        "MTVFP": "02",
        "BESKZ": "E",
    })

  mard = pd.DataFrame(rows_mard)
  marc = pd.DataFrame(rows_marc)

  # ACDOCA
  acdo_rows = []
  for _, item in vbap.iterrows():
    pm = PRODUCT_MASTER[item["MATNR"]]
    country = item["COUNTRY"]
    mode = item["SHIP_MODE"]
    qty = item["KWMENG"]
    fc_map = FREIGHT_COST.get(country, {"Air": 5.0, "Sea": 2.0})

    revenue = qty * pm.list_price_usd
    cogs = qty * pm.standard_cost_usd
    freight = qty * fc_map.get(mode, 3.0)
    tax = revenue * rng.uniform(0.05, 0.15)
    contrib = revenue - cogs - freight
    net_margin_usd = contrib - tax + rng.uniform(-0.03, 0.03) * revenue
    net_margin_pct = (
        round(net_margin_usd / revenue * 100, 2) if revenue else 0
    )

    acdo_rows.append({
        "RBUKRS": "AMEC",
        "GJAHR": today.year,
        "DOCNR": item["VBELN"],
        "POSNR": item["POSNR"],
        "MATNR": item["MATNR"],
        "PROD_FAMILY": pm.family,
        "COUNTRY": country,
        "REGION": item["REGION"],
        "SHIP_MODE": mode,
        "QTY": qty,
        "REVENUE_USD": round(revenue, 2),
        "COGS_USD": round(cogs, 2),
        "FREIGHT_USD": round(freight, 2),
        "TAX_USD": round(tax, 2),
        "NET_MARGIN_USD": round(net_margin_usd, 2),
        "NET_MARGIN_PCT": net_margin_pct,
    })
  acdoca = pd.DataFrame(acdo_rows)

  # LIKP / LIPS
  likp_rows, lips_rows = [], []
  for _, order in vbak.iterrows():
    vbeln = order["VBELN"]
    country = order["COUNTRY"]
    region = order["REGION"]
    mode = order["SHIP_MODE"]
    ord_date = order["ERDAT"]

    transit_base = TRANSIT_DAYS[mode].get(region, 5)
    customs_base = CUSTOMS_DAYS.get(country, 5)
    actual_transit = max(transit_base + int(rng.integers(-2, 5)), 1)
    actual_customs = max(customs_base + int(rng.integers(-1, 7)), 1)
    prod_days = int(rng.integers(5, 14))
    total_lead_time = prod_days + actual_transit + actual_customs

    ship_date = ord_date + timedelta(days=prod_days)
    delivery_date = ship_date + timedelta(
        days=actual_transit + actual_customs
    )

    port_bottleneck = (
        1 if (country in ["Nigeria", "Ghana"] and mode == "Sea") else 0
    )
    hormuz_risk = 1 if (region == "GCC" and mode == "Sea") else 0

    likp_rows.append({
        "VBELN_DEL": f"DEL-{vbeln[4:]}",
        "VBELN_ORDER": vbeln,
        "COUNTRY": country,
        "REGION": region,
        "SHIP_MODE": mode,
        "SHIP_DATE": ship_date,
        "DELIVERY_DATE": delivery_date,
        "PROD_DAYS": prod_days,
        "TRANSIT_DAYS_ACT": actual_transit,
        "CUSTOMS_DAYS_ACT": actual_customs,
        "TOTAL_LEAD_DAYS": total_lead_time,
        "PORT_BOTTLENECK": port_bottleneck,
        "HORMUZ_RISK": hormuz_risk,
    })
    for _, item in vbap[vbap["VBELN"] == vbeln].iterrows():
      lips_rows.append({
          "VBELN_DEL": f"DEL-{vbeln[4:]}",
          "POSNR": item["POSNR"],
          "MATNR": item["MATNR"],
          "LFIMG": item["KWMENG"],
      })

  return vbak, vbap, mard, marc, acdoca, pd.DataFrame(likp_rows), pd.DataFrame(lips_rows)


vbak, vbap, mard, marc, acdoca, likp, lips = load_sap_data()

# ---------------------------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------------------------
st.title("⚡ AMECATH MEA — SAP S/4HANA Intelligent Enterprise Layer")
st.markdown(
    "*Production-grade enterprise intelligence layers simulator running live"
    " in-memory.*"
)

tab1, tab2, tab3 = st.tabs([
    "Layer 1: Real-Time Order Prioritization",
    "Layer 2: Profitability Roadmap (ACDOCA)",
    "Layer 3: Predictive Lead Time AI",
])

# --- TAB 1: LAYER 1 ---
with tab1:
  st.subheader("Layer 1 — Real-Time Order Prioritization Engine (VBAK/VBAP/MARD)")


  def compute_priority_score(order_row, acdoca_df, mard_df):
    vbeln = order_row["VBELN"]
    req_date = order_row["VDATU"]
    country = order_row["COUNTRY"]

    acdo_items = acdoca_df[acdoca_df["DOCNR"] == vbeln]
    margin_pct = (
        (
            acdo_items["NET_MARGIN_USD"].sum()
            / acdo_items["REVENUE_USD"].sum()
            * 100
        )
        if not acdo_items.empty and acdo_items["REVENUE_USD"].sum() > 0
        else 0.0
    )
    margin_score = min(40, max(0, (margin_pct / 35) * 40))

    days_to_req = (req_date - date.today()).days
    urgency_score = (
        35
        if days_to_req <= 7
        else (
            28
            if days_to_req <= 14
            else 18 if days_to_req <= 30 else 8 if days_to_req <= 60 else 2
        )
    )

    items_needed = acdoca_df[acdoca_df["DOCNR"] == vbeln][["MATNR", "QTY"]]
    atp_ratios = []
    for _, need in items_needed.iterrows():
      stock = mard_df.loc[mard_df["MATNR"] == need["MATNR"], "ATP"]
      atp_val = int(stock.values[0]) if not stock.empty else 0
      atp_ratios.append(
          min(atp_val / need["QTY"], 1.0) if need["QTY"] > 0 else 1.0
      )
    avg_atp = np.mean(atp_ratios) if atp_ratios else 0.0
    atp_score = avg_atp * 25
    composite = round(margin_score + urgency_score + atp_score, 2)

    return {
        "VBELN": vbeln,
        "COUNTRY": country,
        "DAYS_TO_REQ": days_to_req,
        "NET_MARGIN_PCT": round(margin_pct, 2),
        "PRIORITY_SCORE": composite,
        "PRIORITY_BAND": (
            "🔴 CRITICAL"
            if composite >= 75
            else (
                "🟡 HIGH"
                if composite >= 50
                else "🟢 STANDARD" if composite >= 25 else "⬜ DEFER"
            )
        ),
    }


  open_orders = vbak[vbak["DOC_STATUS"].str.startswith("Open")]
  queue = pd.DataFrame(
      [compute_priority_score(row, acdoca, mard) for _, row in open_orders.iterrows()]
  ).sort_values("PRIORITY_SCORE", ascending=False)
  queue.index = range(1, len(queue) + 1)

  st.dataframe(queue, use_container_width=True)

# --- TAB 2: LAYER 2 ---
with tab2:
  st.subheader("Layer 2 — Market-Linked Profitability Roadmap (ACDOCA)")

  shock_pct = st.slider("Simulate Sea Freight Cost Shock (%)", 0, 50, 30)

  df_acdo = acdoca.copy()
  if shock_pct > 0:
    df_acdo.loc[df_acdo["SHIP_MODE"] == "Sea", "FREIGHT_USD"] *= 1 + (
        shock_pct / 100.0
    )
    df_acdo["NET_MARGIN_USD"] = (
        df_acdo["REVENUE_USD"]
        - df_acdo["COGS_USD"]
        - df_acdo["FREIGHT_USD"]
        - df_acdo["TAX_USD"]
    )

  roadmap = (
      df_acdo.groupby(["COUNTRY", "REGION", "PROD_FAMILY", "SHIP_MODE"])
      .agg(
          TOTAL_REVENUE_USD=("REVENUE_USD", "sum"),
          NET_MARGIN_USD=("NET_MARGIN_USD", "sum"),
          ORDER_COUNT=("DOCNR", "nunique"),
      )
      .reset_index()
  )
  roadmap["NET_MARGIN_PCT"] = (
      roadmap["NET_MARGIN_USD"] / roadmap["TOTAL_REVENUE_USD"] * 100
  ).round(2)
  roadmap["MARGIN_BAND"] = roadmap["NET_MARGIN_PCT"].apply(
      lambda m: (
          "🟢 HEALTHY (>30%)"
          if m > 30
          else "🟡 SQUEEZED (20-30%)" if m >= 20 else "🔴 AT RISK (<20%)"
      )
  )

  st.dataframe(
      roadmap.sort_values("NET_MARGIN_USD", ascending=False),
      use_container_width=True,
  )

# --- TAB 3: LAYER 3 ---
with tab3:
  st.subheader("Layer 3 — Predictive Lead Time AI Model (Gradient Boosting)")

  # Train Model
  df_ml = likp.copy()
  m_enc, r_enc = LabelEncoder(), LabelEncoder()
  df_ml["SHIP_MODE_ENC"] = m_enc.fit_transform(df_ml["SHIP_MODE"])
  df_ml["REGION_ENC"] = r_enc.fit_transform(df_ml["REGION"])

  features = [
      "SHIP_MODE_ENC",
      "REGION_ENC",
      "PROD_DAYS",
      "PORT_BOTTLENECK",
      "HORMUZ_RISK",
      "CUSTOMS_DAYS_ACT",
  ]
  X, y = df_ml[features].values, df_ml["TOTAL_LEAD_DAYS"].values
  X_tr, X_te, y_tr, y_te = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  model = GradientBoostingRegressor(
      n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42
  )
  model.fit(X_tr, y_tr)
  mae = mean_absolute_error(y_te, model.predict(X_te))

  st.success(
      f"Model Trained Successfully! Mean Absolute Error (MAE):"
      f" {mae:.2f} Days"
  )

  col1, col2, col3 = st.columns(3)
  with col1:
    sel_country = st.selectbox(
        "Destination Country", list(FREIGHT_COST.keys())
    )
  with col2:
    sel_mode = st.selectbox("Shipping Mode", ["Air", "Sea"])
  with col3:
    sel_prod_days = st.slider("Production Days", 5, 20, 8)

  reg = COUNTRY_REGION.get(sel_country, "GCC")
  port_b = 1 if (sel_country in ["Nigeria", "Ghana"] and sel_mode == "Sea") else 0
  hormuz = 1 if (reg == "GCC" and sel_mode == "Sea") else 1 if reg == "GCC" else 0
  customs = CUSTOMS_DAYS.get(sel_country, 5)

  try:
    m_val = int(m_enc.transform([sel_mode])[0])
    r_val = int(r_enc.transform([reg])[0])
  except:
    m_val, r_val = 0, 0

  pred_days = max(
      int(
          round(
              float(
                  model.predict(
                      np.array([[m_val, r_val, sel_prod_days, port_b, hormuz, customs]])
                  )[0]
              )
          )
      ),
      1,
  )
  eta = date.today() + timedelta(days=pred_days)

  st.metric(label="Predicted Total Lead Time", value=f"{pred_days} Days")
  st.info(f"Estimated Delivery Date (ETA): **{eta.strftime('%Y-%m-%d')}**")
