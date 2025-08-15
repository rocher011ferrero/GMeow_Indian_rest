#!/usr/bin/env python3
"""
multi_city_rois_full.py

Complete multi-city ROI analysis pipeline (London & Bath example), fully hardened:
- Grid generation
- FSA FHRS establishments fetch with caching
- Optional polite menu scraping
- Monte Carlo simulations
- Chart + PDF report generation
- Heatmap generation per point
- Parallel processing support
"""
from __future__ import annotations
import os
import sys
import math
import time
import argparse
import logging
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import urllib.robotparser as robotparser
import multiprocessing as mp

import requests
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from shapely.geometry import Point
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# -------------------
# Logging
# -------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("multi_city_rois")

# -------------------
# Constants & polite defaults
# -------------------
USER_AGENT = "roi-research-bot/0.1 (+mailto:you@yourdomain.example)"
NOMINATIM_SLEEP = 1.0   # seconds between geocode requests
SCRAPE_SLEEP = 1.5      # seconds between scrapes
FSA_HEADERS = {"x-api-version": "2", "User-Agent": USER_AGENT}
CACHE_DIRNAME = ".roi_cache"

CITY_BBOXES = {
    "london": [51.286760, 51.691874, -0.510375, 0.334015],
    "bath": [51.3500, 51.4250, -2.5350, -2.3500],
}

# -------------------
# Utilities
# -------------------
def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def cache_get_path(key: str) -> str:
    ensure_dir(CACHE_DIRNAME)
    safe = key.replace("/", "_").replace(":", "_").replace(" ", "_")
    return os.path.join(CACHE_DIRNAME, f"{safe}.json")

def cache_load(key: str):
    p = cache_get_path(key)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def cache_save(key: str, obj):
    p = cache_get_path(key)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f)
    except Exception:
        pass

# -------------------
# Robust numeric conversion helper
# -------------------
def safe_to_numeric(series: pd.Series, default: Optional[float] = None) -> pd.Series:
    if series is None:
        s = pd.Series(dtype="float64")
        return s if default is None else s.fillna(default)
    def _truncate(x):
        try:
            if isinstance(x, str) and len(x) > 200:
                return x[:200]
            return x
        except Exception:
            return x
    cleaned = series.map(_truncate)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if default is not None:
        numeric = numeric.fillna(default)
    return numeric

# -------------------
# Geocoding via Nominatim (cached)
# -------------------
def geocode_nominatim(query: str, sleep_sec: float = NOMINATIM_SLEEP):
    key = f"geocode::{query}"
    cached = cache_load(key)
    if cached:
        return cached["lat"], cached["lng"], cached.get("address")
    geolocator = Nominatim(user_agent=USER_AGENT, timeout=10)
    rate = RateLimiter(geolocator.geocode, min_delay_seconds=sleep_sec)
    loc = rate(query + ", UK")
    if loc is None:
        raise RuntimeError(f"Nominatim: could not geocode '{query}'")
    out = {"lat": float(loc.latitude), "lng": float(loc.longitude), "address": loc.address}
    cache_save(key, out)
    return out["lat"], out["lng"], out["address"]

# -------------------
# NEW: Reverse geocoding (cached) + safe label helper
# -------------------
def reverse_geocode_name(lat: float, lng: float, sleep_sec: float = NOMINATIM_SLEEP) -> str:
    """
    Return a human-readable location name for (lat, lng).
    Uses Nominatim reverse geocoding with on-disk caching.
    """
    key = f"revgeo::{lat:.5f},{lng:.5f}"
    cached = cache_load(key)
    if cached and "name" in cached:
        return cached["name"]
    geolocator = Nominatim(user_agent=USER_AGENT, timeout=10)
    rate = RateLimiter(geolocator.reverse, min_delay_seconds=sleep_sec)
    loc = rate((lat, lng))
    if loc is None or not getattr(loc, "address", None):
        name = f"{lat:.5f},{lng:.5f}"
    else:
        name = loc.address
    cache_save(key, {"name": name})
    return name

def safe_label_from_name(name: str) -> str:
    """
    Make a filesystem-safe label from a location name.
    """
    safe = name
    for ch in r'\/:*?"<>|':
        safe = safe.replace(ch, "_")
    safe = "_".join(safe.split())  # collapse whitespace to underscores
    return safe[:120]  # keep it short-ish

# -------------------
# Generate grid
# -------------------
def generate_grid_for_bbox(min_lat, max_lat, min_lon, max_lon, spacing_m=1000) -> List[Tuple[float,float]]:
    lat_step_deg = spacing_m / 111000.0
    lats = []
    lat = min_lat
    while lat <= max_lat:
        lats.append(lat)
        lat += lat_step_deg
    pts = []
    for lat in lats:
        lon_step_deg = spacing_m / (111000.0 * math.cos(math.radians(lat)) + 1e-9)
        lon = min_lon
        while lon <= max_lon:
            pts.append((round(lat,6), round(lon,6)))
            lon += lon_step_deg
    return pts

# -------------------
# FSA FHRS establishments near point (public API)
# -------------------
def fsa_establishments(lat: float, lng: float, max_km: int = 2) -> pd.DataFrame:
    cache_key = f"fsa::{lat:.5f}_{lng:.5f}_{max_km}"
    cached = cache_load(cache_key)
    if cached:
        try:
            return pd.DataFrame(cached)
        except Exception:
            pass
    url = "https://api.ratings.food.gov.uk/establishments"
    params = {"latitude": lat, "longitude": lng, "maxDistanceLimit": max_km}
    try:
        r = requests.get(url, params=params, headers=FSA_HEADERS, timeout=20)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        logger.warning("FSA API failed for %.5f,%.5f: %s", lat, lng, e)
        return pd.DataFrame()
    rows = []
    for e in j.get("establishments", []):
        address_parts = [e.get(k) for k in ("AddressLine1","AddressLine2","AddressLine3","PostCode") if e.get(k)]
        address = ", ".join(address_parts)
        rows.append({
            "name": e.get("BusinessName") or "",
            "fhrs_id": e.get("FHRSID"),
            "lat": e.get("geocode", {}).get("latitude"),
            "lng": e.get("geocode", {}).get("longitude"),
            "rating_raw": e.get("RatingValue"),
            "local_authority": e.get("LocalAuthorityName") or "",
            "address": address,
            "business_type": e.get("BusinessType") or ""
        })
    df = pd.DataFrame(rows)
    df["rating"] = safe_to_numeric(df.get("rating_raw"), default=np.nan)
    df["lat"] = safe_to_numeric(df.get("lat"), default=np.nan)
    df["lng"] = safe_to_numeric(df.get("lng"), default=np.nan)
    try:
        cache_save(cache_key, df.to_dict(orient="records"))
    except Exception:
        pass
    return df

# -------------------
# robots.txt check
# -------------------
def can_fetch_url(url: str, ua: str = USER_AGENT) -> bool:
    try:
        parsed = robotparser.RobotFileParser()
        host_root = "/".join(url.split("/", 3)[:3])
        robots_url = host_root + "/robots.txt"
        parsed.set_url(robots_url)
        parsed.read()
        return parsed.can_fetch(ua, url)
    except Exception:
        return False

# -------------------
# Polite scraping example (menu prices)
# -------------------
def scrape_menu_prices(url: str) -> Dict:
    if not can_fetch_url(url):
        logger.info("robots.txt disallows scraping %s - skipping", url)
        return {}
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception as e:
        logger.warning("Scrape failed for %s: %s", url, e)
        return {}
    soup = BeautifulSoup(r.text, "lxml")
    text = soup.get_text(" ", strip=True)
    import re
    prices = re.findall(r"\£\s*\d+(?:\.\d{1,2})?", text)
    try:
        prices = [float(p.replace("£", "").strip()) for p in prices]
    except Exception:
        prices = []
    time.sleep(SCRAPE_SLEEP)
    if not prices:
        return {}
    return {"url": url, "n_prices": len(prices), "mean_price": float(np.mean(prices)), "median_price": float(np.median(prices))}

# -------------------
# Infer priors from competitor signals and optional menu scrapes
# -------------------
def infer_priors_from_data(df_places: pd.DataFrame, menu_scrapes: List[Dict]) -> Dict:
    """
    Robustly infer priors. Cleans ratings to protect against non-numeric FHRS values
    like 'AwaitingInspection', 'Exempt', or concatenated junk strings.
    """
    priors: Dict[str, float] = {}

    # 1) Try to use menu-derived spend if available
    medians = [s["median_price"] for s in menu_scrapes if s and "median_price" in s]
    if medians:
        priors["avg_spend_mean"] = float(np.mean(medians)) * 1.6
        priors["avg_spend_sd"] = max(1.5, 0.2 * priors["avg_spend_mean"])
    else:
        # 2) Use ratings as a proxy for spend level (after strict cleaning)
        ratings_src = None
        if "rating" in df_places.columns:
            ratings_src = df_places["rating"]
        elif "rating_raw" in df_places.columns:
            ratings_src = df_places["rating_raw"]

        avg_rating = np.nan
        if ratings_src is not None:
            ratings_clean = pd.to_numeric(ratings_src, errors="coerce")
            # Remove inf/-inf and keep within FHRS-like [0, 5]
            ratings_clean = ratings_clean.replace([np.inf, -np.inf], np.nan)
            ratings_clean = ratings_clean[(ratings_clean >= 0) & (ratings_clean <= 5)]
            if ratings_clean.notna().any():
                avg_rating = float(ratings_clean.mean())

        if not np.isnan(avg_rating):
            base = 14.0 + (avg_rating - 3.5) * 4.0
            priors["avg_spend_mean"] = base
            priors["avg_spend_sd"] = max(2.5, 0.25 * base)
        else:
            priors["avg_spend_mean"] = 18.0
            priors["avg_spend_sd"] = 4.0

    # 3) Customers/day heuristic
    if "user_ratings_total" in df_places.columns and df_places["user_ratings_total"].dropna().size > 0:
        try:
            urt = pd.to_numeric(df_places["user_ratings_total"], errors="coerce").replace({0: np.nan})
            if urt.dropna().size > 0:
                avg_reviews = float(urt.dropna().mean())
            else:
                avg_reviews = 100.0
        except Exception:
            avg_reviews = 100.0
        customers_mean = max(20.0, (avg_reviews * 100.0) / 365.0)
        priors["customers_per_day_mean"] = customers_mean
        priors["customers_per_day_sd"] = max(10.0, 0.25 * customers_mean)
    else:
        competitor_count = 0 if df_places.empty else len(df_places)
        if competitor_count < 5:
            priors["customers_per_day_mean"] = 160.0
            priors["customers_per_day_sd"] = 40.0
        elif competitor_count < 15:
            priors["customers_per_day_mean"] = 120.0
            priors["customers_per_day_sd"] = 35.0
        else:
            priors["customers_per_day_mean"] = 90.0
            priors["customers_per_day_sd"] = 30.0

    # 4) Cost priors
    priors["food_cost_pct_mean"] = 0.30
    priors["food_cost_pct_sd"] = 0.05
    priors["labor_cost_pct_mean"] = 0.28
    priors["labor_cost_pct_sd"] = 0.05

    return priors

# -------------------
# Monte Carlo simulation
# -------------------
def run_monte_carlo(n_sims: int,
                    days_open_per_year: int,
                    avg_spend_mean: float, avg_spend_sd: float,
                    customers_mean: float, customers_sd: float,
                    food_cost_pct_mean: float, food_cost_pct_sd: float,
                    labor_cost_pct_mean: float, labor_cost_pct_sd: float,
                    rent_per_year: float, other_fixed_per_year: float,
                    capex: float, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    avg_spend = rng.normal(avg_spend_mean, avg_spend_sd, size=n_sims)
    avg_spend = np.clip(avg_spend, 5.0, None)
    customers = rng.normal(customers_mean, customers_sd, size=n_sims)
    customers = np.clip(customers, 1.0, None)

    revenue = avg_spend * customers * days_open_per_year
    food_pct = np.clip(rng.normal(food_cost_pct_mean, food_cost_pct_sd, size=n_sims), 0.05, 0.6)
    labor_pct = np.clip(rng.normal(labor_cost_pct_mean, labor_cost_pct_sd, size=n_sims), 0.05, 0.6)
    food_cost = revenue * food_pct
    labor_cost = revenue * labor_pct
    ebitda = revenue - (food_cost + labor_cost + rent_per_year + other_fixed_per_year)
    depreciation = capex / 10.0
    taxable = ebitda - depreciation
    tax = np.where(taxable > 0, taxable * 0.19, 0.0)
    net_profit = taxable - tax
    roi = np.where(capex > 0, net_profit / capex, np.nan)
    payback = np.where(ebitda > 0, capex / ebitda, np.nan)

    df = pd.DataFrame({
        "revenue": revenue,
        "ebitda": ebitda,
        "net_profit": net_profit,
        "roi": roi,
        "payback_years": payback,
        "avg_spend": avg_spend,
        "customers_per_day": customers
    })
    return df

# -------------------
# Plotting helpers
# -------------------
def plot_simulation_charts(sim_df: pd.DataFrame, out_prefix: str) -> List[str]:
    charts = []
    plt.figure(figsize=(8,4.5))
    plt.hist(sim_df["revenue"].clip(lower=-1e9, upper=1e9), bins=80)
    plt.title("Simulated Annual Revenue distribution")
    plt.xlabel("Annual revenue (£)")
    rev_png = f"{out_prefix}_revenue_hist.png"
    plt.tight_layout()
    plt.savefig(rev_png); plt.close(); charts.append(rev_png)

    plt.figure(figsize=(8,4.5))
    plt.hist(sim_df["ebitda"].clip(lower=-1e9, upper=1e9), bins=80)
    plt.title("Simulated EBITDA distribution")
    plt.xlabel("EBITDA (£)")
    ebit_png = f"{out_prefix}_ebitda_hist.png"
    plt.tight_layout()
    plt.savefig(ebit_png); plt.close(); charts.append(ebit_png)

    plt.figure(figsize=(5,4))
    roi_vals = sim_df["roi"].dropna()
    if roi_vals.size == 0:
        roi_vals = pd.Series([0.0])
    plt.boxplot(roi_vals, vert=True)
    plt.title("ROI distribution (net_profit / capex)")
    roi_png = f"{out_prefix}_roi_box.png"
    plt.tight_layout()
    plt.savefig(roi_png); plt.close(); charts.append(roi_png)

    return charts

# -------------------
# PDF report builder
# -------------------
def build_pdf_report(title: str, location_label: str, charts: List[str], heatmap_html: Optional[str], summary_csv: str, out_pdf: str):
    c = canvas.Canvas(out_pdf, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h-60, title)
    c.setFont("Helvetica", 9)
    c.drawString(40, h-80, f"Location: {location_label}    Generated: {datetime.utcnow().isoformat()} UTC")
    y = h-120
    for img in charts:
        try:
            c.drawImage(img, 40, y-220, width=520, height=200)
            y -= 220 + 8
            if y < 140:
                c.showPage(); y = h-60
        except Exception as e:
            logger.warning("Could not draw image %s: %s", img, e)
    c.showPage()
    c.setFont("Helvetica", 9)
    c.drawString(40, h-60, "Simulation summary (first 20 lines):")
    try:
        lines = open(summary_csv).read().splitlines()[:20]
        y = h-80
        for ln in lines:
            c.drawString(40, y, ln[:110]); y -= 10
            if y < 40:
                c.showPage(); y = h-60
    except Exception as e:
        c.drawString(40, h-80, f"Could not read {summary_csv}: {e}")
    c.save()

# -------------------
# Process a single grid point
# -------------------
def process_point(args_tuple):
    (lat, lon, city, idx, params) = args_tuple
    output_dir = params["output_dir"]
    radius_m = params["radius_m"]
    n_sims = params["n_sims"]
    days_open = params["days_open"]
    rent_per_year = params["rent_per_year"]
    other_fixed_per_year = params["other_fixed_per_year"]
    capex = params["capex"]
    menu_sample_urls = params.get("menu_sample_urls", [])

    # --- derive human-readable name from coordinates ---
    try:
        place_name = reverse_geocode_name(lat, lon)
    except Exception as _e:
        place_name = f"{lat:.5f},{lon:.5f}"
    safe_place = safe_label_from_name(place_name)

    # Use the place name (not london_75/city_idx) in outputs
    label = f"{safe_place}_{lat:.5f}_{lon:.5f}"
    prefix = os.path.join(output_dir, label)
    ensure_dir(output_dir)

    try:
        fsa_df = fsa_establishments(lat, lon, max_km=math.ceil(radius_m/1000.0))
        if fsa_df is None:
            fsa_df = pd.DataFrame()

        menu_scrapes = []
        for url in menu_sample_urls:
            try:
                res = scrape_menu_prices(url)
                if res:
                    menu_scrapes.append(res)
            except Exception as e:
                logger.debug("Menu scrape failure for %s: %s", url, e)

        priors = infer_priors_from_data(fsa_df, menu_scrapes)
        sim_df = run_monte_carlo(n_sims, days_open,
                                 priors["avg_spend_mean"], priors["avg_spend_sd"],
                                 priors["customers_per_day_mean"], priors["customers_per_day_sd"],
                                 priors["food_cost_pct_mean"], priors["food_cost_pct_sd"],
                                 priors["labor_cost_pct_mean"], priors["labor_cost_pct_sd"],
                                 rent_per_year, other_fixed_per_year, capex)
        summary_csv = f"{prefix}_summary.csv"
        sim_df.to_csv(summary_csv, index=False)
        charts = plot_simulation_charts(sim_df, prefix)

        # Heatmap
        m = folium.Map(location=[lat, lon], zoom_start=15)
        if not fsa_df.empty:
            heat_data = [[row["lat"], row["lng"]] for idx,row in fsa_df.iterrows() if not pd.isna(row["lat"]) and not pd.isna(row["lng"])]
            if heat_data:
                HeatMap(heat_data).add_to(m)
        heatmap_html = f"{prefix}_heatmap.html"
        m.save(heatmap_html)

        pdf_file = f"{prefix}_report.pdf"
        # show the human-readable place in the PDF header
        build_pdf_report("ROI Simulation Report", f"{place_name} ({lat:.5f},{lon:.5f})", charts, heatmap_html, summary_csv, pdf_file)
        logger.info("Completed point %s: PDF %s", label, pdf_file)
        return pdf_file
    except Exception as e:
        logger.exception("Failed processing point %s: %s", label, e)
        return None

# -------------------
# Main CLI entry
# -------------------
def main():
    p = argparse.ArgumentParser(description="Grid-level multi-location ROI pipeline (London & Bath example)")
    p.add_argument("--cities", default="london,bath", help="Comma-separated city keys: london,bath")
    p.add_argument("--grid_m", type=int, default=1000, help="Grid spacing in meters")
    p.add_argument("--radius_m", type=int, default=1200, help="Catchment radius to query FSA (meters)")
    p.add_argument("--n_sims", type=int, default=3000, help="Monte Carlo draws per location (reduce for speed)")
    p.add_argument("--days_open", type=int, default=320)
    p.add_argument("--rent_per_year", type=float, default=60000.0)
    p.add_argument("--other_fixed_per_year", type=float, default=20000.0)
    p.add_argument("--capex", type=float, default=140000.0)
    p.add_argument("--menu_sample_urls", default="", help="Optional pipe-separated sample URLs to scrape for menu prices")
    p.add_argument("--output_dir", default="./multi_reports")
    p.add_argument("--parallel", type=int, default=2, help="Number of parallel workers (careful with rate limits)")
    args = p.parse_args()

    cities = [c.strip().lower() for c in args.cities.split(",") if c.strip()]
    invalid = [c for c in cities if c not in CITY_BBOXES]
    if invalid:
        logger.error("Unknown cities: %s. Supported: %s", invalid, list(CITY_BBOXES.keys()))
        sys.exit(1)

    menu_sample_urls = [u.strip() for u in args.menu_sample_urls.split("|") if u.strip()]
    ensure_dir(args.output_dir)

    all_points = []
    for city in cities:
        min_lat, max_lat, min_lon, max_lon = CITY_BBOXES[city]
        pts = generate_grid_for_bbox(min_lat, max_lat, min_lon, max_lon, spacing_m=args.grid_m)
        logger.info("City %s -> %d grid points (spacing %dm)", city, len(pts), args.grid_m)
        for idx, (lat, lon) in enumerate(pts):
            params_dict = {
                "output_dir": args.output_dir,
                "radius_m": args.radius_m,
                "n_sims": args.n_sims,
                "days_open": args.days_open,
                "rent_per_year": args.rent_per_year,
                "other_fixed_per_year": args.other_fixed_per_year,
                "capex": args.capex,
                "menu_sample_urls": menu_sample_urls
            }
            all_points.append((lat, lon, city, idx, params_dict))

    logger.info("Starting processing of %d points with %d workers", len(all_points), args.parallel)
    results = []

    # Use imap_unordered to safely handle argument tuples in multiprocessing
    if args.parallel <= 1:
        for a in all_points:
            results.append(process_point(a))
    else:
        with mp.Pool(processes=args.parallel) as pool:
            for r in pool.imap_unordered(process_point, all_points):
                results.append(r)

    master = pd.DataFrame(results)
    master_csv = os.path.join(args.output_dir, "master_summary.csv")
    master.to_csv(master_csv, index=False)
    logger.info("Master summary saved to %s", master_csv)
    print("Done. Individual reports in", args.output_dir)
    print("Master summary:", master_csv)

if __name__ == "__main__":
    main()
