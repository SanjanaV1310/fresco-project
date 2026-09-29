"""
Synthetic restaurant demand/inventory/supplier dataset generator.

Backbone approach: one demand table (modeled on Favorita's real schema:
date, store, item, family/category, perishable flag, onpromotion flag,
unit_sales) drives everything else. Inventory and Supplier fields are
DERIVED from this same table, keyed on the same item_id, so every agent
shares one consistent identifier space -- no cross-dataset mapping needed.

This is a documented synthetic stand-in (network access to Kaggle is
blocked in this environment) built to match Favorita's real structure and
realistic restaurant-industry parameter ranges (USDA LAFA-informed shelf
life, industry-typical lead times/MOQs). Swap in the real Favorita
train.csv later using the same column names and this pipeline keeps working.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. ITEM MASTER — one simulated restaurant, ~54 SKUs across 9 families
#    relevant to food service. Perishable flag + shelf life follow
#    USDA LAFA-informed spoilage patterns per category.
# ---------------------------------------------------------------------------
families = {
    "PRODUCE":      dict(n=8,  perishable=1, shelf_life=(3, 6),   base_demand=(15, 45), price=(0.8, 3.5)),
    "MEATS":        dict(n=6,  perishable=1, shelf_life=(3, 5),   base_demand=(8, 25),  price=(4.0, 12.0)),
    "POULTRY":      dict(n=4,  perishable=1, shelf_life=(2, 4),   base_demand=(10, 30), price=(2.5, 6.0)),
    "SEAFOOD":      dict(n=4,  perishable=1, shelf_life=(1, 3),   base_demand=(5, 15),  price=(6.0, 18.0)),
    "DAIRY":        dict(n=6,  perishable=1, shelf_life=(7, 14),  base_demand=(10, 30), price=(2.0, 6.0)),
    "BAKERY":       dict(n=5,  perishable=1, shelf_life=(2, 4),   base_demand=(12, 35), price=(1.5, 5.0)),
    "EGGS":         dict(n=2,  perishable=1, shelf_life=(21, 35), base_demand=(8, 20),  price=(2.5, 4.5)),
    "GROCERY_DRY":  dict(n=12, perishable=0, shelf_life=(90, 270),base_demand=(5, 20),  price=(1.0, 8.0)),
    "BEVERAGES":    dict(n=7,  perishable=0, shelf_life=(120, 365),base_demand=(10, 40),price=(1.0, 10.0)),
}

item_rows = []
item_id = 1000
for fam, cfg in families.items():
    for i in range(cfg["n"]):
        item_rows.append(dict(
            item_id=item_id,
            item_name=f"{fam.title().replace('_',' ')} Item {i+1}",
            family=fam,
            perishable=cfg["perishable"],
            shelf_life_days=int(rng.integers(cfg["shelf_life"][0], cfg["shelf_life"][1] + 1)),
            base_demand=rng.uniform(*cfg["base_demand"]),
            unit_price=round(rng.uniform(*cfg["price"]), 2),
        ))
        item_id += 1

items_df = pd.DataFrame(item_rows)

# ---------------------------------------------------------------------------
# 2. DEMAND TABLE — daily unit sales per item, single store ("STORE_1"),
#    1 full year, with weekly seasonality (restaurant-relevant: weekend
#    spikes), a winter-holiday bump, random promotions, and noise.
#    Modeled on Favorita's real train.csv schema:
#    date, store_nbr, item_nbr, unit_sales, onpromotion
# ---------------------------------------------------------------------------
start = date(2025, 10, 1)
n_days = 365
dates = [start + timedelta(days=i) for i in range(n_days)]

demand_rows = []
for _, item in items_df.iterrows():
    # weekly pattern: Fri/Sat/Sun higher (restaurant traffic)
    weekday_mult = {0: 0.9, 1: 0.85, 2: 0.9, 3: 1.0, 4: 1.25, 5: 1.4, 6: 1.15}
    # promotion runs ~8% of days, boosts demand 20-50%
    on_promo_days = set(rng.choice(n_days, size=int(n_days * 0.08), replace=False))

    for d_idx, d in enumerate(dates):
        wd = d.weekday()
        seasonal = weekday_mult[wd]
        # winter holiday bump late Nov - early Jan
        if (d.month == 11 and d.day >= 20) or (d.month == 12) or (d.month == 1 and d.day <= 2):
            seasonal *= 1.3
        promo = d_idx in on_promo_days
        promo_mult = rng.uniform(1.2, 1.5) if promo else 1.0

        mean_demand = item["base_demand"] * seasonal * promo_mult
        # intermittent demand for slower-moving/perishable-limited items
        if rng.random() < 0.03:
            units = 0
        else:
            units = max(0, int(rng.normal(mean_demand, mean_demand * 0.25)))

        demand_rows.append((d.isoformat(), "STORE_1", item["item_id"], units, promo))

demand_df = pd.DataFrame(demand_rows, columns=["date", "store_nbr", "item_id", "unit_sales", "onpromotion"])

# ---------------------------------------------------------------------------
# 3. INVENTORY PARAMETERS — derived FROM the demand table itself, per item:
#    safety stock = z * rolling std dev of that item's own daily demand
#    (z=1.65 -> ~95% service level), reorder point = safety stock + mean
#    demand over the supplier's own lead time (joined below).
# ---------------------------------------------------------------------------
demand_stats = (
    demand_df.groupby("item_id")["unit_sales"]
    .agg(mean_daily_demand="mean", std_daily_demand="std")
    .reset_index()
)

# ---------------------------------------------------------------------------
# 4. SUPPLIER PARAMETERS — synthetic, but keyed to the SAME item_id and
#    parameterized by category-realistic ranges (perishable items get
#    shorter lead times & smaller MOQs; dry/beverage items get longer
#    lead times & larger MOQs, consistent with real supplier behavior).
# ---------------------------------------------------------------------------
lead_time_ranges = {
    "PRODUCE": (1, 2), "MEATS": (1, 3), "POULTRY": (1, 2), "SEAFOOD": (1, 2),
    "DAIRY": (1, 3), "BAKERY": (1, 2), "EGGS": (2, 4),
    "GROCERY_DRY": (3, 7), "BEVERAGES": (3, 7),
}
moq_ranges = {
    "PRODUCE": (10, 30), "MEATS": (5, 20), "POULTRY": (5, 20), "SEAFOOD": (5, 15),
    "DAIRY": (10, 25), "BAKERY": (10, 25), "EGGS": (5, 15),
    "GROCERY_DRY": (24, 72), "BEVERAGES": (24, 96),
}
supplier_names = ["Gordon Fresh Co.", "Sysco Regional", "US Foods Direct", "Local Farm Collective", "Reinhart FoodService"]

supplier_rows = []
for _, item in items_df.iterrows():
    fam = item["family"]
    lt_lo, lt_hi = lead_time_ranges[fam]
    moq_lo, moq_hi = moq_ranges[fam]
    supplier_rows.append(dict(
        item_id=item["item_id"],
        supplier_name=rng.choice(supplier_names),
        unit_cost=round(item["unit_price"] * rng.uniform(0.55, 0.75), 2),  # cost < menu/retail price
        lead_time_days=int(rng.integers(lt_lo, lt_hi + 1)),
        moq_units=int(rng.integers(moq_lo, moq_hi + 1)),
        availability_pct=round(rng.uniform(0.92, 1.0), 3),
    ))
suppliers_df = pd.DataFrame(supplier_rows)

# Inventory params joined with supplier lead time for reorder point
inv_df = items_df[["item_id", "family", "perishable", "shelf_life_days"]].merge(demand_stats, on="item_id")
inv_df = inv_df.merge(suppliers_df[["item_id", "lead_time_days"]], on="item_id")
inv_df["std_daily_demand"] = inv_df["std_daily_demand"].fillna(0)
Z = 1.65
inv_df["safety_stock_units"] = np.ceil(Z * inv_df["std_daily_demand"] * np.sqrt(inv_df["lead_time_days"])).astype(int)
inv_df["reorder_point_units"] = np.ceil(
    inv_df["mean_daily_demand"] * inv_df["lead_time_days"] + inv_df["safety_stock_units"]
).astype(int)
inv_df = inv_df.drop(columns=["lead_time_days"])  # lead time lives in suppliers table; avoid duplicate column drift

# ---------------------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------------------
items_df.to_csv("items.csv", index=False)
demand_df.to_csv("demand.csv", index=False)
inv_df.to_csv("inventory_params.csv", index=False)
suppliers_df.to_csv("suppliers.csv", index=False)

print("items:", items_df.shape)
print("demand:", demand_df.shape)
print("inventory_params:", inv_df.shape)
print("suppliers:", suppliers_df.shape)
print(items_df.head())
print(demand_df.head())
print(inv_df.head())
print(suppliers_df.head())
