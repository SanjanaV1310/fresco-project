# Dataset for Multi-Agent Inventory & Waste Reduction System

## What this is

A single-store, one-year synthetic restaurant dataset built to match the real structure of Kaggle's Corporación Favorita Grocery Sales Forecasting dataset (date, store, item, family/category, unit sales, promotion flag), with Inventory and Supplier fields **derived directly from the same demand table** rather than pulled from a second unrelated dataset.

**Why synthetic rather than the real Favorita CSV:** Kaggle is not reachable from this environment's network, and no real dataset publicly combines demand + inventory + supplier + perishability at SKU level for restaurants (see Requirements Statement, Section 2.3). This generator produces a fully consistent stand-in you can start building against today. The column names and structure match Favorita's real schema exactly, so if you later download the real dataset from Kaggle yourself, you can substitute it in with no pipeline changes.

## Files

### `items.csv` (54 rows) — item master
| Column | Description |
|---|---|
| `item_id` | Unique SKU identifier (1000–1053) |
| `item_name` | Human-readable name |
| `family` | Category: PRODUCE, MEATS, POULTRY, SEAFOOD, DAIRY, BAKERY, EGGS, GROCERY_DRY, BEVERAGES |
| `perishable` | 1 if perishable, 0 if shelf-stable |
| `shelf_life_days` | Days until spoilage/expiration |
| `base_demand` | Underlying average daily demand used to generate `demand.csv` |
| `unit_price` | Menu/retail unit price |

### `demand.csv` (19,710 rows = 54 items × 365 days) — Demand Forecasting Agent's input
| Column | Description |
|---|---|
| `date` | Calendar date (2025-10-01 through 2026-09-30) |
| `store_nbr` | Single simulated store, `STORE_1` |
| `item_id` | Joins to `items.csv` |
| `unit_sales` | Units sold that day |
| `onpromotion` | Whether the item was on promotion that day (~8% of days) |

Built-in realistic patterns: weekend demand spike (Fri/Sat/Sun highest, consistent with restaurant traffic), a Nov 20–Jan 2 holiday demand bump, promotion-driven spikes (+20–50%), and ~3% zero-sale days for intermittent-demand realism.

### `inventory_params.csv` (54 rows) — Inventory Agent's input
| Column | Description |
|---|---|
| `item_id`, `family`, `perishable`, `shelf_life_days` | From `items.csv` |
| `mean_daily_demand`, `std_daily_demand` | Computed directly from that item's own row in `demand.csv` |
| `safety_stock_units` | `1.65 × std_daily_demand × sqrt(lead_time_days)` — standard safety-stock formula targeting ~95% service level, using each item's own demand variability |
| `reorder_point_units` | `mean_daily_demand × lead_time_days + safety_stock_units` |

Shelf life ranges follow USDA Loss-Adjusted Food Availability (LAFA) category patterns (e.g., produce/meat/poultry/seafood: 1–6 days; dairy: 7–14 days; dry goods/beverages: 90–365 days) — these are realistic category-level assumptions, not pulled from LAFA's raw rows.

### `suppliers.csv` (54 rows) — Supplier Agent's input
| Column | Description |
|---|---|
| `item_id` | Joins to `items.csv` / `inventory_params.csv` |
| `supplier_name` | One of 5 simulated suppliers |
| `unit_cost` | Wholesale cost (55–75% of retail `unit_price`) |
| `lead_time_days` | Category-realistic: 1–2 days for produce/seafood, up to 3–7 days for dry goods/beverages |
| `moq_units` | Minimum order quantity, category-realistic (smaller for perishables, larger for dry/beverage) |
| `availability_pct` | Simulated fill rate (92–100%) |

## Why every agent shares one identifier space

All four files key on the same `item_id`. There is no cross-dataset mapping step, no category-name reconciliation, and no risk of the Optimization/Critic agents receiving inconsistent identifiers from different sources — this directly avoids the dataset-consistency downsides discussed earlier (mismatched taxonomies, no real correlation between demand/spoilage/supply, scale mismatches).

## Known limitations (state these plainly in your report)

- Demand, promotion, and seasonality patterns are simulated, not drawn from real restaurant point-of-sale data.
- Supplier lead times/costs/MOQs are synthetic, parameterized by category-realistic ranges rather than real vendor contracts.
- Correlation between demand shocks and supply disruptions (e.g., a lead-time spike during a demand spike) is not modeled in this version — if you want to test the Critic Agent under stress, consider manually injecting a few such events.
- If you later swap in the real Favorita `train.csv`, keep the same column names (`date`, `store_nbr`, `item_id`, `unit_sales`, `onpromotion`) and the Inventory/Supplier derivation logic (`generate.py`) will still work unchanged — just skip the demand-generation step and start from `inventory_params.csv`'s derivation logic directly on the real data.

## Regenerating or modifying

The full generation script (`generate.py`) is included. Adjust `families` (item counts, shelf life ranges, price ranges), `n_days`, or the seasonality/promotion logic to change dataset size or realism, then rerun.
