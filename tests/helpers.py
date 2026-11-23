"""Small hand-built inputs so Optimization/Critic tests don't depend on the real data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

START = "2026-10-01"


def make_forecast(units=30.0, L=2, cycle=7, rel=0.15, item_id=1, start=START):
    n = L + cycle
    days = [{"date": f"d{i}", "units": float(units), "lower": None, "upper": None, "promo": False}
            for i in range(n)]
    return {"item_id": item_id, "forecast_start": start, "horizon_days": n,
            "daily_forecast": days, "method": "weekday_level",
            "error_scale": {"daily_relative": 0.25, "total_relative": rel,
                            "window_relative": rel},
            "cover_window": {"start": L, "end": L + cycle, "units": units * cycle},
            "rationale": "test forecast"}


def make_inventory(stock=0, shelf=30, price=4.0, ss=0, last_offset=None, item_id=1,
                   in_transit=None, start=START):
    batches = []
    if stock:
        lo = shelf - 1 if last_offset is None else last_offset
        batches = [{"units": stock, "received": start, "last_sellable": start,
                    "last_offset": lo, "days_left": lo + 1}]
    return {"item_id": item_id, "as_of": start, "current_stock": stock, "batches": batches,
            "in_transit": in_transit or [], "in_transit_units": sum(o["units"] for o in (in_transit or [])),
            "expiring_soon": [], "shelf_life_days": shelf, "unit_price": price,
            "safety_stock_units": ss, "reorder_point_units": 0, "flags": [], "rationale": "test"}


def make_supplier(L=2, moq=10, cost=2.5, avail=1.0, item_id=1):
    return {"item_id": item_id, "supplier_name": "Test Co", "unit_cost": cost,
            "lead_time_days": L, "moq_units": moq, "availability_pct": avail,
            "expected_fill": avail, "risk_flags": [], "terms_changed": False,
            "rationale": "test"}
