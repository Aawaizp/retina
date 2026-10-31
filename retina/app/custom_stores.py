"""Create local, store-specific profiles from details and a checked product CSV.

The profile records which facts the user supplied. Everything copied from a generic
store profile remains an estimate; a custom store is never presented as observed sales.
"""

import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
BUILTIN = ROOT / "stores"
BASE = {
    "supermarket": "generic_supermarket",
    "supercenter": "retina_supercenter",
    "neighborhood": "neighborhood_market",
    "convenience": "convenience",
}
CUSTOM_ID = re.compile(r"custom-[a-z0-9-]{1,45}-[0-9a-f]{10}\Z")


def custom_root():
    return Path(os.environ.get("RETINA_CUSTOM_STORES", ROOT / "data" / "custom_stores"))


def profile_path(store_id):
    if not CUSTOM_ID.fullmatch(store_id):
        return None
    return custom_root() / store_id / "profile.json"


def custom_ids():
    root = custom_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and profile_path(p.name) and
                  (p / "profile.json").is_file())


class StoreSetup(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    city: str = Field(default="", max_length=80)
    state: str = Field(default="", max_length=40)
    store_type: Literal["supermarket", "supercenter", "neighborhood", "convenience"] = "supermarket"
    open_hour: int = Field(ge=0, le=23)
    close_hour: int = Field(ge=1, le=24)
    shoppers_per_week: int = Field(ge=7, le=200_000)
    checkout_lanes: int = Field(ge=1, le=100)
    products_csv: str = ""
    hourly_counts: dict[str, int] | None = None
    start_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


def create_store(setup: StoreSetup):
    """Validate the upload, then save an immutable profile and its original product file."""
    from retina.products.check import check_file, report_text
    from retina.products.convert import save

    if setup.close_hour <= setup.open_hour:
        raise ValueError("Closing hour must be later than opening hour (use 24 for midnight).")
    if setup.start_date:
        try:
            date.fromisoformat(setup.start_date)
        except ValueError as exc:
            raise ValueError("First simulated date must be a real date (YYYY-MM-DD).") from exc
    raw = setup.products_csv.encode("utf-8")
    if len(raw) > 20_000_000:
        raise ValueError("Product CSV is over 20 MB; upload at most 50,000 products.")
    counts = setup.hourly_counts
    if counts:
        if any(not h.isdigit() or not setup.open_hour <= int(h) < setup.close_hour or v < 0
               for h, v in counts.items()) or not any(counts.values()):
            raise ValueError("Hourly counts must be nonnegative and within the store's open hours.")
    details = setup.model_dump(exclude={"products_csv"})
    digest = hashlib.sha256(json.dumps(details, sort_keys=True).encode("utf-8") + raw).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9]+", "-", setup.name.lower()).strip("-")[:45].strip("-") or "store"
    store_id = f"custom-{slug}-{digest}"
    folder = custom_root() / store_id
    existing = profile_path(store_id)
    if existing.is_file():
        profile = json.loads(existing.read_text(encoding="utf-8"))
        summary = profile.get("custom_store", {}).get("product_summary") or {}
        return {"id": store_id, "name": profile["name"], "warnings": [], "existing": True,
                "products": summary.get("uploaded_products"), "aisles": summary.get("aisles"),
                "demand_covered": summary.get("demand_covered")}

    # Check before creating a permanent store. The reader accepts ordinary CSV encodings.
    root = custom_root()
    root.mkdir(parents=True, exist_ok=True)
    checked, summary, catalog_file = None, None, None
    if raw:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="check_", dir=root) as tmp:
            candidate = Path(tmp) / "products.csv"
            candidate.write_bytes(raw)
            checked = check_file(candidate)
        if not checked.ok:
            raise ValueError(report_text(checked, max_lines=15))

        folder.mkdir(parents=True, exist_ok=True)
        original = folder / "products.csv"
        original.write_bytes(raw)
        checked.file = str(original)
        catalog, _, summary = save(checked, original, "only")
        try:
            catalog_file = catalog.relative_to(ROOT)
        except ValueError:  # a test or a caller using a separate local product directory
            catalog_file = catalog
    profile = json.loads((BUILTIN / f"{BASE[setup.store_type]}.json").read_text(encoding="utf-8"))
    place = ", ".join(x for x in (setup.city.strip(), setup.state.strip()) if x) or "the US"
    profile.update(id=store_id, name=setup.name.strip(),
                   description=f"User-configured {setup.store_type} in {place}. "
                               "All shopper activity and sales are simulated.",
                   engine="realtime")
    if catalog_file is not None:
        profile.update(catalog_file=str(catalog_file).replace("\\", "/"), catalog_kind="only")
    profile["simulation"]["n_shoppers"] = "auto"
    profile["simulation"]["days"] = 365
    if setup.start_date:
        profile["simulation"]["start_date"] = setup.start_date
    if catalog_file is not None:
        profile.setdefault("format", {})["assortment_share"] = 1.0  # upload = exact stocked range
    profile["traffic"].update(enabled=True, open_hours=[setup.open_hour, setup.close_hour],
                              real_shoppers_per_week=setup.shoppers_per_week, npc_scale=1.0,
                              hourly_counts=counts or None)
    if counts:
        # Hourly counts are a typical weekday. The traffic engine uses their sum for volume.
        profile["traffic"]["real_shoppers_per_week"] = round(sum(counts.values()) *
                                                                  (5 + 2 * profile["traffic"]["weekend_vs_weekday"]))
    profile.setdefault("floor", {})["checkout_lanes"] = setup.checkout_lanes
    profile["labels"] = {key: "ESTIMATED - inherited from the generic store-type profile; edit when measured "
                               "store facts are available" for key in profile.get("labels", {})}
    profile["labels"].update({
        "catalog_file": ("USER_PROVIDED - products, categories and prices in products.csv; not independently verified"
                         if catalog_file is not None else
                         "ESTIMATED store assortment - a generic US catalog, not this retailer's actual products"),
        "traffic.open_hours": "USER_PROVIDED - opening and closing hours entered in store setup",
        "traffic.real_shoppers_per_week": "USER_PROVIDED - weekly visitors entered in store setup" if not counts else
                                          "USER_PROVIDED (derived) - typical weekday hourly counts",
        "traffic.hourly_curve": "USER_PROVIDED - typical weekday hourly counts" if counts else
                                "ESTIMATED - generic US shopping-hour pattern scaled to user-provided hours",
        "floor": "USER_PROVIDED checkout lanes; other layout and walking settings ESTIMATED from store type",
        "engine": "SIMULATED - shared real-time store clock, student model decisions at shelves",
        "simulation.start_date": ("USER_PROVIDED - first simulated date entered in store setup" if setup.start_date else
                                  "ESTIMATED - inherited from the generic store-type profile"),
    })
    profile["custom_store"] = {"city": setup.city.strip(), "state": setup.state.strip(),
                               "store_type": setup.store_type, "product_summary": summary}
    folder.mkdir(parents=True, exist_ok=True)
    tmp_profile = folder / "profile.tmp"
    tmp_profile.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    tmp_profile.replace(folder / "profile.json")
    return {"id": store_id, "name": profile["name"],
            "warnings": [m for _, m in checked.warnings] if checked else [],
            "existing": False, "products": summary["uploaded_products"] if summary else None,
            "aisles": summary["aisles"] if summary else None,
            "demand_covered": summary["demand_covered"] if summary else None}
