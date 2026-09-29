"""
Loads and validates the four core data files, confirming they all
share a consistent item_id key -- this is the "environment sanity
check" that proves the dev environment + data are both working
end to end before any agent logic is built.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_items() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "items.csv")


def load_demand() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "demand.csv", parse_dates=["date"])


def load_inventory_params() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "inventory_params.csv")


def load_suppliers() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "suppliers.csv")


def load_all() -> dict:
    return {
        "items": load_items(),
        "demand": load_demand(),
        "inventory_params": load_inventory_params(),
        "suppliers": load_suppliers(),
    }


def validate_consistency(data: dict) -> None:
    """Confirms every table shares the same item_id keys -- the whole
    point of the single-backbone dataset design."""
    item_ids = set(data["items"]["item_id"])
    demand_ids = set(data["demand"]["item_id"])
    inv_ids = set(data["inventory_params"]["item_id"])
    sup_ids = set(data["suppliers"]["item_id"])

    assert item_ids == demand_ids, "items/demand item_id mismatch"
    assert item_ids == inv_ids, "items/inventory_params item_id mismatch"
    assert item_ids == sup_ids, "items/suppliers item_id mismatch"
    print(f"Consistency check passed: {len(item_ids)} items, all keys aligned.")


if __name__ == "__main__":
    data = load_all()
    for name, df in data.items():
        print(f"{name}: {df.shape[0]} rows, {df.shape[1]} cols")
    validate_consistency(data)
