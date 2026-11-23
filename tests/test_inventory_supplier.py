"""Inventory Agent (FR2) and Supplier Agent (FR3)."""
import pytest

from tests.helpers import *  # noqa: F401,F403  (sets sys.path)
from agents.inventory_agent import InventoryAgent
from agents.supplier_agent import SupplierAgent


# ------------------------------------------------------------- inventory
def test_shelf_life_boundaries():
    """Received on day D with shelf life 3: sellable D, D+1, D+2; binned at the end of D+2."""
    inv = InventoryAgent()
    item = 1000                                   # shelf life 3
    inv.receive(item, 10, "2026-10-01")
    assert inv.sell(item, 1, "2026-10-03") == 1   # last sellable day: still sellable
    assert inv.expire("2026-10-02") == {}         # day before: nothing thrown away
    assert inv.expire("2026-10-03") == {item: 9}  # end of last day: the rest is waste
    assert inv.run(item, as_of="2026-10-04")["current_stock"] == 0


def test_sells_oldest_first():
    inv = InventoryAgent()
    inv.receive(1000, 5, "2026-10-01")
    inv.receive(1000, 5, "2026-10-02")
    assert inv.sell(1000, 6, "2026-10-02") == 6
    left = inv.batches[1000]
    assert len(left) == 1 and left[0]["received"].isoformat() == "2026-10-02" and left[0]["units"] == 4


def test_sell_limited_to_stock():
    inv = InventoryAgent()
    inv.receive(1000, 5, "2026-10-01")
    assert inv.sell(1000, 50, "2026-10-01") == 5


def test_expired_stock_cannot_be_sold():
    inv = InventoryAgent()
    inv.receive(1000, 5, "2026-10-01")
    assert inv.sell(1000, 5, "2026-10-04") == 0


def test_status_flags_and_expiry_warning():
    inv = InventoryAgent()
    inv.receive(1000, 10, "2026-10-01")           # shelf 3 -> last sellable 10-03
    s = inv.run(1000, as_of="2026-10-02")
    assert s["current_stock"] == 10
    assert s["expiring_soon"] == [{"units": 10, "days_left": 2}]
    assert "units nearing expiration" in s["flags"]
    assert "below safety stock" in s["flags"]     # 10 < 28
    assert s["batches"][0]["last_offset"] == 1


def test_safety_stock_is_configurable():
    inv = InventoryAgent()
    inv.receive(1000, 10, "2026-10-01")
    inv.set_safety_stock(1000, 5)
    assert "below safety stock" not in inv.run(1000, as_of="2026-10-02")["flags"]
    with pytest.raises(ValueError):
        inv.set_safety_stock(1000, -1)


def test_in_transit_tracking():
    inv = InventoryAgent()
    inv.add_in_transit(1000, 50, "2026-10-05")
    s = inv.run(1000, as_of="2026-10-01")
    assert s["in_transit"] == [{"units": 50, "arrives_on": "2026-10-05", "offset": 4}]
    assert inv.pop_due(1000, "2026-10-04") == []
    assert len(inv.pop_due(1000, "2026-10-05")) == 1
    assert inv.run(1000, as_of="2026-10-05")["in_transit"] == []


def test_stock_correction_takes_shortfall_from_oldest():
    inv = InventoryAgent()
    inv.receive(1001, 10, "2026-10-01")           # shelf 6
    inv.receive(1001, 10, "2026-10-03")
    adj = inv.set_count(1001, 14, "2026-10-04")
    assert adj["adjustment"] == -6
    assert sorted(b["units"] for b in inv.batches[1001]) == [4, 10]
    assert inv.set_count(1001, 20, "2026-10-04")["adjustment"] == 6


def test_manual_count_does_not_touch_ledger():
    inv = InventoryAgent()
    s = inv.run(1000, as_of="2026-10-01", current_stock=12)
    assert s["current_stock"] == 12 and inv.batches[1000] == []
    assert s["batches"][0]["days_left"] == 2      # unknown age assumed half-fresh (shelf 3)


def test_unknown_item_rejected():
    with pytest.raises(ValueError):
        InventoryAgent().run(99999, as_of="2026-10-01")


# -------------------------------------------------------------- supplier
def test_supplier_contract():
    r = SupplierAgent().run(1000)
    for k in ("unit_cost", "lead_time_days", "moq_units", "availability_pct", "expected_fill",
              "supplier_name", "rationale"):
        assert k in r
    assert r["moq_units"] == 21 and r["lead_time_days"] == 2


def test_supplier_update_logged_and_reset():
    s = SupplierAgent()
    rec = s.apply_update(1000, reason="price rise", unit_cost=2.5)
    assert rec["changes"]["unit_cost"] == {"old": 1.83, "new": 2.5}
    assert s.run(1000)["unit_cost"] == 2.5 and s.run(1000)["terms_changed"]
    assert "price rise" in s.run(1000)["rationale"]
    s.reset()
    assert s.run(1000)["unit_cost"] == 1.83 and s.change_log == []


@pytest.mark.parametrize("bad", [{"availability_pct": 1.5}, {"unit_cost": -1},
                                 {"lead_time_days": 0}, {"moq_units": 2.5}, {"colour": "red"}])
def test_supplier_rejects_bad_updates(bad):
    with pytest.raises(ValueError):
        SupplierAgent().apply_update(1000, **bad)


def test_supplier_quote_respects_moq():
    q = SupplierAgent().quote(1000, 10)
    assert not q["meets_moq"]
    assert SupplierAgent().quote(1000, 21)["meets_moq"]
    assert SupplierAgent().quote(1000, 0)["meets_moq"]
