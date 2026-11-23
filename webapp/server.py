"""Local web app for the class demo.   python -m webapp.server   ->  http://localhost:8000

Standard library only (no Flask, no internet needed). One in-memory session: the page plans a
week, lets you change inputs (supplier terms, stock count, budget) and shows the replan.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from agents.data_loader import load_items
from agents.demand_agent import DemandForecastingAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.supplier_agent import SupplierAgent
from demo import AS_OF, stocked_ledger

STATIC = Path(__file__).parent / "static"
ITEMS = load_items().set_index("item_id")
LOCK = threading.Lock()
DEMAND = DemandForecastingAgent()          # shared: keeps the backtest cache warm
S: dict = {}


def reset():
    S["orch"] = OrchestratorAgent(demand_agent=DEMAND, inventory_agent=stocked_ledger(DEMAND),
                                  supplier_agent=SupplierAgent())
    S["plan"], S["cycle"], S["budget"] = None, 7, None


def view(plan: dict) -> dict:
    rows = []
    for i, it in plan["items"].items():
        o, inv, sup = it["optimization_result"], it["inventory_status"], it["supplier_info"]
        rows.append({
            "item_id": i, "name": ITEMS.loc[i, "item_name"], "family": ITEMS.loc[i, "family"],
            "order": it["final_order_units"], "cost": o["order_cost"], "status": it["status"],
            "lead_time": sup["lead_time_days"], "moq": sup["moq_units"], "unit_cost": sup["unit_cost"],
            "shelf": inv["shelf_life_days"], "stock": inv["current_stock"],
            "forecast": o["window_demand"]["forecast"], "error": o["window_demand"]["relative_error"],
            "service": o["expected"]["service_level"], "waste": o["expected"]["waste_units"],
            "binding": o["binding_constraints"],
            "warnings": [w["code"] for w in it["critic_result"]["warnings"]],
            "explanation": it["explanation"], "rounds": it["rounds"],
            "trace": [{"agent": t["agent"], "round": t.get("round"), "summary": t["summary"]} for t in it["trace"]],
            "daily": [{"date": d["date"], "units": d["units"], "lower": d["lower"], "upper": d["upper"]}
                      for d in it["demand_forecast"]["daily_forecast"]],
        })
    return {"decision_date": plan["decision_date"], "as_of": AS_OF, "cycle": S["cycle"],
            "budget": S["budget"], "summary": plan["summary"], "basket": plan["basket"],
            "replan_log": plan["replan_log"], "items": rows}


def do_plan(cycle, budget):
    S["cycle"], S["budget"] = cycle, budget
    S["plan"] = S["orch"].plan_week(as_of=AS_OF, cycle_days=cycle, budget=budget)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = urlparse(self.path).path
        with LOCK:
            if path == "/api/plan":
                if S["plan"] is None:
                    do_plan(7, None)
                return self.send(200, view(S["plan"]))
            if path == "/api/evaluation":
                return self.send(200, json.loads((ROOT / "reports/system_evaluation_data.json").read_text()))
        files = {"/": "index.html", "/index.html": "index.html",
                 "/system_evaluation.png": "../../reports/system_evaluation.png"}
        if path in files:
            f = (STATIC / files[path]).resolve()
            ctype = "image/png" if f.suffix == ".png" else "text/html; charset=utf-8"
            return self.send(200, f.read_bytes(), ctype)
        self.send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        path = urlparse(self.path).path
        try:
            with LOCK:
                if path == "/api/reset":
                    reset()
                    do_plan(7, None)
                elif path == "/api/plan":
                    do_plan(int(data.get("cycle", 7)), data.get("budget") or None)
                elif path == "/api/replan":
                    if S["plan"] is None:
                        do_plan(S["cycle"], S["budget"])
                    S["plan"] = S["orch"].replan(S["plan"], data)
                    if data["type"] == "budget_change":
                        S["budget"] = data["budget"]
                else:
                    return self.send(404, {"error": "not found"})
                return self.send(200, view(S["plan"]))
        except Exception as e:                       # show the message on the page, don't crash
            return self.send(400, {"error": f"{type(e).__name__}: {e}"})


def main(port=8000):
    reset()
    do_plan(7, None)
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"FRESCO demo running at http://localhost:{port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
