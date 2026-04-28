#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import shutil
import warnings
from pathlib import Path
from datetime import datetime
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, box
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from PIL import Image
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# CONFIG
# -----------------------------
CENTRAL_EUROPE_BBOX_4326 = {
    "lon_min": 2.0,
    "lon_max": 28.5,
    "lat_min": 42.0,
    "lat_max": 56.5,
}

EUROPE_BBOX_4326 = {
    "lon_min": -25.0,
    "lon_max": 45.0,
    "lat_min": 34.0,
    "lat_max": 72.0,
}

INTENSITY_COLUMN = "col5"

FIG_BG = "black"
AX_BG = "black"
LAND_COLOR = "black"
BORDER_COLOR = "white"
TEXT_COLOR = "white"
GRID_COLOR = "#666666"

NATURAL_EARTH_COUNTRIES_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
NATURAL_EARTH_PROVINCES_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"

LOCAL_TIMEZONE = "Europe/Prague"

GRID_CELL_KM = 100
GRID_FORECAST_MIN_FRACTION = 0.50

FORECAST_STYLE = {
    "tstorm":   {"label": "T-storm Risk (<5%)", "color": "#90EE90", "rank": 1},
    "general":  {"label": "General Risk (5%)",  "color": "#2E8B57", "rank": 2},
    "slight":   {"label": "Slight Risk (15%)",  "color": "#FFD700", "rank": 3},
    "enhanced": {"label": "Enhanced Risk (30%)","color": "#FF8C00", "rank": 4},
    "moderate": {"label": "Moderate Risk (45%)","color": "#8B0000", "rank": 5},
    "severe":   {"label": "Severe Risk (>50%)", "color": "#FF00FF", "rank": 6},
    "nonrisk":  {"label": "Non-risk",           "color": "#555555", "rank": 0},
}

RISK_ORDER = ["tstorm", "general", "slight", "enhanced", "moderate", "severe"]

COUNTRY_ALIASES = {
    "czechia": ["czechia", "czech republic", "cesko", "česko", "cz"],
    "slovakia": ["slovakia", "slovak republic", "slovensko", "sk"],
    "austria": ["austria", "osterreich", "österreich", "at"],
    "germany": ["germany", "deutschland", "de"],
    "poland": ["poland", "polska", "pl"],
    "hungary": ["hungary", "magyarorszag", "magyarország", "hu"],
    "romania": ["romania", "ro"],
    "ukraine": ["ukraine", "ua"],
    "france": ["france", "fr"],
    "italy": ["italy", "italia", "it"],
    "switzerland": ["switzerland", "schweiz", "suisse", "ch"],
    "belgium": ["belgium", "belgie", "belgique", "be"],
    "netherlands": ["netherlands", "holland", "nl"],
    "luxembourg": ["luxembourg", "lu"],
    "slovenia": ["slovenia", "slovenija", "si"],
    "croatia": ["croatia", "hrvatska", "hr"],
    "serbia": ["serbia", "srbija", "rs"],
    "bosnia and herzegovina": ["bosnia and herzegovina", "bosnia", "bih", "ba"],
}

# -----------------------------
# ARGUMENT PARSING
# -----------------------------
def looks_like_geojson(s: str) -> bool:
    s = str(s).lower()
    return s.endswith(".geojson") or s.endswith(".json")

def looks_like_time_filter(s: str) -> bool:
    s = str(s).strip().lower()
    if s in {"risk", "gif", "gif1h", "gif30"}:
        return False
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        pass
    if "-" in s:
        a, b = s.split("-", 1)
        try:
            float(a)
            float(b)
            return True
        except ValueError:
            return False
    return False

def parse_args():
    if len(sys.argv) < 2:
        print("Usage examples:")
        print("  python anal.py strikes.csv")
        print("  python anal.py strikes.csv forecast.geojson")
        print("  python anal.py strikes.csv 18")
        print("  python anal.py strikes.csv 18-20")
        print("  python anal.py strikes.csv czechia")
        print("  python anal.py strikes.csv forecast.geojson 8 czechia")
        print("  python anal.py strikes.csv forecast.geojson risk")
        print("  python anal.py strikes.csv forecast.geojson gif")
        print("  python anal.py strikes.csv forecast.geojson gif1h")
        print("  python anal.py strikes.csv forecast.geojson 18 risk gif")
        raise SystemExit(1)

    csv_path = Path(sys.argv[1])
    forecast_path = None
    time_filter = None
    country_filter = None
    zoom_mode = None
    gif_mode = None
    region_mode = None

    extras = sys.argv[2:]

    for arg in extras:
        arg_clean = str(arg).strip().lower()

        if looks_like_geojson(arg) and forecast_path is None:
            forecast_path = Path(arg)
        elif arg_clean == "risk" and zoom_mode is None:
            zoom_mode = "risk"
        elif arg_clean in {"gif", "gif30"} and gif_mode is None:
            gif_mode = "30min"
        elif arg_clean == "gif1h" and gif_mode is None:
            gif_mode = "1h"
        elif arg_clean == "orp" and region_mode is None:
            region_mode = "orp"
        elif looks_like_time_filter(arg) and time_filter is None:
            time_filter = arg
        elif country_filter is None:
            country_filter = arg
        else:
            if country_filter is None:
                country_filter = arg
            else:
                country_filter += f" {arg}"

    return csv_path, forecast_path, time_filter, country_filter, zoom_mode, gif_mode, region_mode

# -----------------------------
# LOAD CSV
# -----------------------------
def load_csv(csv_path):
    return pd.read_csv(
        csv_path,
        header=None,
        names=[
            "latitude",
            "longitude",
            "strike_time",
            "col4",
            "col5",
            "col6",
            "col7",
        ],
    )

# -----------------------------
# CLEAN DATA
# -----------------------------
def normalize_dataframe(df):
    df = df.copy()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["strike_time"] = pd.to_datetime(df["strike_time"], errors="coerce", utc=True)

    for c in ["col4", "col5", "col6", "col7"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude", "strike_time"]).copy()
    df = df.reset_index(drop=True)
    df["strike_id"] = np.arange(len(df))
    df["local_time"] = to_local_time(df["strike_time"])
    df["local_date"] = df["local_time"].dt.date
    return df

# -----------------------------
# FILTER CENTRAL EUROPE
# -----------------------------
def filter_ce(df):
    b = CENTRAL_EUROPE_BBOX_4326
    out = df[
        (df["longitude"] >= b["lon_min"]) &
        (df["longitude"] <= b["lon_max"]) &
        (df["latitude"] >= b["lat_min"]) &
        (df["latitude"] <= b["lat_max"])
    ].copy()
    out = out.reset_index(drop=True)
    out["strike_id"] = np.arange(len(out))
    return out

def filter_europe(df):
    b = EUROPE_BBOX_4326
    out = df[
        (df["longitude"] >= b["lon_min"]) &
        (df["longitude"] <= b["lon_max"]) &
        (df["latitude"] >= b["lat_min"]) &
        (df["latitude"] <= b["lat_max"])
    ].copy()
    out = out.reset_index(drop=True)
    out["strike_id"] = np.arange(len(out))
    return out

# -----------------------------
# TIME
# -----------------------------
def to_local_time(series):
    return series.dt.tz_convert(LOCAL_TIMEZONE)

def extract_date_from_filename(path_like):
    name = Path(path_like).stem
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if not m:
        return None
    try:
        return pd.to_datetime(m.group(1)).date()
    except Exception:
        return None

def filter_to_local_day_from_filename(df, csv_path):
    target_date = extract_date_from_filename(csv_path)
    if target_date is None or df.empty:
        return df

    out = df.copy()
    if "local_time" not in out.columns:
        out["local_time"] = to_local_time(out["strike_time"])
    out["local_date"] = out["local_time"].dt.date
    out = out[out["local_date"] == target_date].copy()
    out = out.reset_index(drop=True)
    out["strike_id"] = np.arange(len(out))
    return out

def parse_time_filter_spec(spec):
    if spec is None:
        return None

    s = str(spec).strip()
    if not s:
        return None

    if "-" in s:
        parts = s.split("-", 1)
        try:
            start_h = float(parts[0])
            end_h = float(parts[1])
        except ValueError:
            raise ValueError(f"Invalid time filter: {spec}")
        return ("range", start_h, end_h)

    try:
        start_h = float(s)
    except ValueError:
        raise ValueError(f"Invalid time filter: {spec}")

    return ("from", start_h)

def apply_time_filter(df, spec):
    parsed = parse_time_filter_spec(spec)
    if parsed is None:
        return df.copy()

    out = df.copy()
    out["local_time"] = to_local_time(out["strike_time"])
    out["hour_float"] = (
        out["local_time"].dt.hour +
        out["local_time"].dt.minute / 60.0 +
        out["local_time"].dt.second / 3600.0 +
        out["local_time"].dt.microsecond / 3_600_000_000.0
    )

    if parsed[0] == "from":
        start_h = parsed[1]
        out = out[out["hour_float"] >= start_h].copy()
    else:
        start_h, end_h = parsed[1], parsed[2]
        out = out[(out["hour_float"] >= start_h) & (out["hour_float"] < end_h)].copy()

    out = out.drop(columns=["local_time", "hour_float"], errors="ignore")
    out = out.reset_index(drop=True)
    out["strike_id"] = np.arange(len(out))
    return out

# -----------------------------
# MAP LOADING / PROJECTION
# -----------------------------
def load_map(bbox_geom=None):
    world = gpd.read_file(NATURAL_EARTH_COUNTRIES_URL)

    if "CONTINENT" in world.columns:
        europe = world[world["CONTINENT"] == "Europe"].copy()
    elif "continent" in world.columns:
        europe = world[world["continent"] == "Europe"].copy()
    else:
        europe = world.copy()

    if europe.crs is None:
        europe = europe.set_crs(epsg=4326)
    else:
        europe = europe.to_crs(epsg=4326)

    if bbox_geom is not None:
        europe = europe[europe.geometry.intersects(bbox_geom)].copy()
        if not europe.empty:
            europe["geometry"] = europe.geometry.intersection(bbox_geom)
            europe = europe[europe.geometry.notna()].copy()
            europe = europe[~europe.geometry.is_empty].copy()

    return europe.to_crs(epsg=3857)

def load_provinces(bbox_geom=None):
    provinces = gpd.read_file(NATURAL_EARTH_PROVINCES_URL).copy()

    if provinces.crs is None:
        provinces = provinces.set_crs(epsg=4326)

    provinces = provinces.to_crs(epsg=4326)

    if bbox_geom is None:
        b = CENTRAL_EUROPE_BBOX_4326
        bbox_geom = box(b["lon_min"], b["lat_min"], b["lon_max"], b["lat_max"])

    provinces = provinces[provinces.geometry.intersects(bbox_geom)].copy()

    provinces = provinces.to_crs(epsg=3857)
    return provinces

def load_orp(orp_path="orp.geojson", bbox_geom=None):
    gdf = gpd.read_file(orp_path).copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    gdf = gdf.to_crs(epsg=4326)

    if bbox_geom is None:
        b = CENTRAL_EUROPE_BBOX_4326
        bbox_geom = box(b["lon_min"], b["lat_min"], b["lon_max"], b["lat_max"])

    gdf = gdf[gdf.geometry.intersects(bbox_geom)].copy()

    return gdf.to_crs(epsg=3857)

def make_gdf(df):
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )
    return gdf.to_crs(epsg=3857)

def get_bounds_3857(bbox_4326=None):
    if bbox_4326 is None:
        bbox_4326 = CENTRAL_EUROPE_BBOX_4326

    pts = gpd.GeoSeries(
        [
            Point(bbox_4326["lon_min"], bbox_4326["lat_min"]),
            Point(bbox_4326["lon_max"], bbox_4326["lat_max"]),
        ],
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    return pts.iloc[0].x, pts.iloc[0].y, pts.iloc[1].x, pts.iloc[1].y

# -----------------------------
# COUNTRY FILTER / ZOOM
# -----------------------------
def normalize_country_name(name: str) -> str:
    s = str(name).strip().lower()
    for canonical, aliases in COUNTRY_ALIASES.items():
        if s == canonical or s in aliases:
            return canonical
    return s

def find_country_geometry(countries_gdf, requested_name):
    if requested_name is None:
        return None, None

    requested = normalize_country_name(requested_name)

    name_cols = [c for c in ["NAME", "ADMIN", "NAME_EN", "SOVEREIGNT", "name", "admin"] if c in countries_gdf.columns]
    if not name_cols:
        return None, None

    for _, row in countries_gdf.iterrows():
        values = []
        for col in name_cols:
            v = row.get(col)
            if pd.notna(v):
                values.append(str(v))

        for v in values:
            nv = normalize_country_name(v)
            if requested == nv:
                display_name = values[0]
                return row.geometry, display_name

    return None, None

def filter_to_country(lightning_gdf, country_geom):
    if lightning_gdf.empty or country_geom is None:
        return lightning_gdf.copy()

    mask = lightning_gdf.geometry.intersects(country_geom)
    out = lightning_gdf[mask].copy().reset_index(drop=True)
    out["strike_id"] = np.arange(len(out))
    return out

def clip_regions_to_country(regions_gdf, country_geom, use_representative_point=False, crop=False):
    if regions_gdf is None or regions_gdf.empty or country_geom is None:
        return regions_gdf

    out = regions_gdf.copy()

    if use_representative_point:
        reps = out.geometry.representative_point()
        mask = reps.within(country_geom) | reps.intersects(country_geom)
        out = out[mask].copy()
    else:
        out = out[out.geometry.intersects(country_geom)].copy()

    if crop and not out.empty:
        try:
            out["geometry"] = out.geometry.intersection(country_geom)
            out = out[out.geometry.notna()].copy()
            out = out[~out.geometry.is_empty].copy()
        except Exception:
            pass

    return out

def get_zoom_bounds(zoom_geom=None):
    if zoom_geom is None:
        return get_bounds_3857()

    minx, miny, maxx, maxy = zoom_geom.bounds
    dx = maxx - minx
    dy = maxy - miny

    dx = max(dx, 50000)
    dy = max(dy, 50000)

    padx = max(dx * 0.08, 15000)
    pady = max(dy * 0.08, 15000)
    return minx - padx, miny - pady, maxx + padx, maxy + pady

def get_risk_zoom_geometry(forecast_gdf):
    if forecast_gdf is None or forecast_gdf.empty:
        return None

    try:
        geom = forecast_gdf.geometry.union_all()
    except Exception:
        geom = forecast_gdf.unary_union

    if geom is None or geom.is_empty:
        return None

    return geom

def get_bbox_zoom_geometry(bbox_4326):
    geom = gpd.GeoSeries([
        box(
            bbox_4326["lon_min"],
            bbox_4326["lat_min"],
            bbox_4326["lon_max"],
            bbox_4326["lat_max"],
        )
    ], crs="EPSG:4326").to_crs(epsg=3857)
    return geom.iloc[0]

# -----------------------------
# STYLE
# -----------------------------
def style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.25, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

def draw_base(ax, europe, zoom_geom=None):
    europe.plot(ax=ax, color=LAND_COLOR, edgecolor=BORDER_COLOR, linewidth=0.5, zorder=1)

    minx, miny, maxx, maxy = get_zoom_bounds(zoom_geom)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    ax.set_aspect("equal", adjustable="box")
    style_ax(ax)

def save(fig, path):
    fig.patch.set_facecolor(FIG_BG)
    fig.tight_layout()
    fig.savefig(path, dpi=240, facecolor=FIG_BG, bbox_inches="tight")
    plt.close(fig)

def add_colorbar(fig, ax, mappable, label, ticks=None):
    cb = fig.colorbar(mappable, ax=ax, ticks=ticks, shrink=0.82, pad=0.02)
    cb.ax.tick_params(colors=TEXT_COLOR)
    cb.set_label(label, color=TEXT_COLOR)
    for spine in cb.ax.spines.values():
        spine.set_color(TEXT_COLOR)
    return cb

def add_branding(ax):
    ax.text(
        0.012, 0.015,
        "@CESTOF",
        transform=ax.transAxes,
        color="white",
        fontsize=16,
        fontweight="bold",
        fontfamily="Courier New",
        ha="left",
        va="bottom",
        zorder=50
    )

# -----------------------------
# COLOR SETUP
# -----------------------------
def get_discrete_hour_cmap():
    base = plt.cm.turbo
    colors = base(np.linspace(0, 1, 24))
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, 24.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def get_region_cmap():
    colors = [
        "#0b1e3c",
        "#1c3f66",
        "#2f5f8f",
        "#4a7fa8",
        "#6b9fb5",
        "#8faea0",
        "#b89b7a",
        "#c9785c",
        "#c04f3f",
        "#a92e2e",
        "#7a0f1f",
    ]
    return ListedColormap(colors)

def get_density_palette_levels(scale_factor=1.0):
    base_levels = [
        1, 2, 3, 4, 5, 7, 9, 11, 13, 15,
        18, 21, 24, 27, 30, 35, 40, 45,
        50, 55, 60, 65, 70, 80, 90, 100
    ]

    colors = [
        "#a7a0d6",
        "#8f82d4",
        "#6466dc",
        "#4b8fd7",
        "#4fc146",
        "#52cb44",
        "#6ddd43",
        "#b4dd43",
        "#dbdb43",
        "#f0c24b",
        "#f2aa4b",
        "#f88e49",
        "#ff4a4a",
        "#d84a4a",
        "#b64a4a",
        "#b64ac8",
        "#df5be3",
        "#d79adf",
        "#cec5e4",
        "#bdb1e8",
        "#a78cf0",
        "#8d6df2",
        "#724cf0",
        "#5b35df",
        "#4322ba",
        "#2b117f",
    ]

    levels = [float(v) * float(scale_factor) for v in base_levels]
    cmap = ListedColormap(colors)
    bounds = levels + [1e9]
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, levels, base_levels

def get_relative_density_cmap_and_norm():
    cmap, norm, levels, _ = get_density_palette_levels(scale_factor=1.0)
    return cmap, norm, levels

def marker_sizes(series, min_size=30, max_size=140):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return np.full(len(series), 50.0)

    s = s.fillna(s.median())
    smin = s.min()
    smax = s.max()

    if smin == smax:
        return np.full(len(series), 60.0)

    scaled = (s - smin) / (smax - smin)
    return min_size + scaled * (max_size - min_size)

def get_region_bins(values, n_classes=11):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.array([0, 1])

    max_val = float(values.max())

    if max_val <= 10:
        step = 1
    elif max_val <= 40:
        step = 5
    elif max_val <= 80:
        step = 10
    elif max_val <= 160:
        step = 20
    elif max_val <= 400:
        step = 50
    elif max_val <= 800:
        step = 100
    elif max_val <= 2000:
        step = 250
    elif max_val <= 4000:
        step = 500
    else:
        step = 1000

    upper = int(np.ceil(max_val / step) * step)
    bins = np.arange(0, upper + step, step)

    if len(bins) < 2:
        bins = np.array([0, step])

    if len(bins) - 1 > n_classes:
        idx = np.linspace(0, len(bins) - 1, n_classes + 1).round().astype(int)
        bins = bins[idx]
        bins = np.unique(bins)
        if len(bins) < 2:
            bins = np.array([0, upper if upper > 0 else 1])

    return bins

# -----------------------------
# FORECAST GEOJSON
# -----------------------------
def load_forecast_geojson(geojson_path):
    if geojson_path is None:
        return None

    path = Path(geojson_path)
    if not path.exists():
        print(f"Forecast GeoJSON not found: {geojson_path}")
        return None

    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        print(f"Could not read forecast GeoJSON: {e}")
        return None

    if gdf.empty:
        return None

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    return gdf.to_crs(epsg=3857)

def normalize_forecast_category(value):
    if pd.isna(value):
        return None

    s = str(value).strip().lower()

    if s in ["t-storm risk", "tstorm risk", "t-storm", "tstorm", "<5%", "lt5", "lt5%", "thunderstorm risk"]:
        return "tstorm"
    if s in ["general risk", "general", "5%", "5"]:
        return "general"
    if s in ["slight risk", "slight", "15%", "15"]:
        return "slight"
    if s in ["enhanced risk", "enhanced", "30%", "30"]:
        return "enhanced"
    if s in ["moderate risk", "moderate", "45%", "45"]:
        return "moderate"
    if s in ["severe risk", "severe", ">50%", "50%", "50", "over 50%"]:
        return "severe"

    return None

def normalize_forecast_prob(value):
    try:
        v = float(value)
    except Exception:
        return None

    if v < 5:
        return "tstorm"
    if v == 5:
        return "general"
    if v == 15:
        return "slight"
    if v == 30:
        return "enhanced"
    if v == 45:
        return "moderate"
    if v > 45:
        return "severe"

    return None

def get_forecast_category_column(gdf):
    for col in ["risk", "category", "label", "level", "probability", "prob"]:
        if col in gdf.columns:
            return col
    return None

def prepare_forecast_gdf(forecast_gdf):
    if forecast_gdf is None or forecast_gdf.empty:
        return None

    gdf = forecast_gdf.copy()
    cat_col = get_forecast_category_column(gdf)

    if cat_col is None:
        print("Forecast GeoJSON has no supported property column (risk/category/label/level/probability/prob).")
        return None

    if cat_col in ["prob", "probability"]:
        gdf["risk_key"] = gdf[cat_col].apply(normalize_forecast_prob)
    else:
        gdf["risk_key"] = gdf[cat_col].apply(normalize_forecast_category)

    gdf = gdf.dropna(subset=["risk_key"]).copy()

    if gdf.empty:
        print("Forecast GeoJSON loaded, but no valid forecast categories were found.")
        return None

    gdf["risk_rank"] = gdf["risk_key"].map(lambda x: FORECAST_STYLE[x]["rank"])
    return gdf

def clip_forecast_to_country(forecast_gdf, country_geom):
    if forecast_gdf is None or forecast_gdf.empty or country_geom is None:
        return forecast_gdf
    return forecast_gdf[forecast_gdf.geometry.intersects(country_geom)].copy()

def get_forecast_union_geometry(forecast_gdf):
    if forecast_gdf is None or forecast_gdf.empty:
        return None

    temp = forecast_gdf.copy()
    temp = temp[temp.geometry.notna()].copy()
    temp = temp[~temp.geometry.is_empty].copy()
    if temp.empty:
        return None

    try:
        temp["geometry"] = temp.geometry.buffer(0)
    except Exception:
        pass

    try:
        geom = temp.geometry.union_all()
    except Exception:
        geom = temp.unary_union

    if geom is None or geom.is_empty:
        return None

    return geom

def get_forecast_union_by_risk(forecast_gdf):
    unions = {}
    if forecast_gdf is None or forecast_gdf.empty:
        return unions

    for key in RISK_ORDER:
        part = forecast_gdf[forecast_gdf["risk_key"] == key].copy()
        if part.empty:
            continue
        part = part[part.geometry.notna()].copy()
        part = part[~part.geometry.is_empty].copy()
        if part.empty:
            continue
        try:
            part["geometry"] = part.geometry.buffer(0)
        except Exception:
            pass
        try:
            geom = part.geometry.union_all()
        except Exception:
            geom = part.unary_union
        if geom is not None and not geom.is_empty:
            unions[key] = geom

    return unions

def plot_forecast_overlay(ax, forecast_gdf, fill_alpha=0.22, boundary_alpha=1.0, z_fill=5, z_line=6):
    if forecast_gdf is None or forecast_gdf.empty:
        return

    temp = forecast_gdf.copy()
    temp = temp[temp.geometry.notna()].copy()
    temp = temp[~temp.geometry.is_empty].copy()
    if temp.empty:
        return

    try:
        temp["geometry"] = temp.geometry.buffer(0)
    except Exception:
        pass

    for key in RISK_ORDER:
        part = temp[temp["risk_key"] == key]
        if part.empty:
            continue

        color = FORECAST_STYLE[key]["color"]

        try:
            part.plot(
                ax=ax,
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
                alpha=fill_alpha,
                zorder=z_fill
            )
            part.boundary.plot(
                ax=ax,
                color=color,
                linewidth=1.3,
                alpha=boundary_alpha,
                zorder=z_line
            )
        except Exception:
            continue

def add_forecast_legend(ax, forecast_gdf):
    if forecast_gdf is None or forecast_gdf.empty:
        return

    present = []
    for key in RISK_ORDER:
        if key in forecast_gdf["risk_key"].unique():
            present.append(
                Patch(
                    facecolor=FORECAST_STYLE[key]["color"],
                    edgecolor=FORECAST_STYLE[key]["color"],
                    label=FORECAST_STYLE[key]["label"],
                    alpha=0.35
                )
            )

    if present:
        leg = ax.legend(
            handles=present,
            loc="upper left",
            frameon=True,
            facecolor="black",
            edgecolor="white",
            fontsize=9
        )
        for text in leg.get_texts():
            text.set_color("white")

# -----------------------------
# GENERIC REGION HELPERS
# -----------------------------
def get_region_name_column(gdf, preferred=None):
    if preferred is None:
        preferred = ["NAME", "ADMIN", "name", "admin"]

    for col in preferred:
        if col in gdf.columns:
            return col

    for col in gdf.columns:
        if str(gdf[col].dtype) == "object":
            return col

    raise ValueError("No valid region name column found")

# -----------------------------
# GENERIC REGION STATS
# -----------------------------
def get_region_stats(lightning_gdf, regions_gdf, source_col, stats_col_name):
    if regions_gdf is None or regions_gdf.empty or source_col is None:
        return pd.DataFrame()

    regions = regions_gdf[[source_col, "geometry"]].copy().rename(columns={source_col: stats_col_name})
    joined = gpd.sjoin(lightning_gdf, regions, how="left", predicate="intersects")
    joined = joined.dropna(subset=[stats_col_name]).copy()
    if joined.empty:
        return pd.DataFrame()

    joined["local_time"] = to_local_time(joined["strike_time"])
    joined["hour"] = joined["local_time"].dt.hour

    stats = joined.groupby(stats_col_name).agg(
        strikes=("geometry", "count"),
        peak_hour=("hour", lambda s: int(s.mode().iloc[0]) if not s.mode().empty else np.nan)
    ).reset_index()

    stats = stats[stats["strikes"] > 0].sort_values("strikes", ascending=False).reset_index(drop=True)
    return stats

def plot_region_map(regions_gdf, source_col, region_stats, stats_col, title, colorbar_label, output_file, out, europe, forecast_gdf=None, zoom_geom=None):
    if regions_gdf is None or regions_gdf.empty or region_stats.empty or source_col is None:
        return

    map_df = regions_gdf[[source_col, "geometry"]].copy()
    map_df = map_df.merge(region_stats, left_on=source_col, right_on=stats_col, how="inner")
    if map_df.empty:
        return

    cmap = get_region_cmap()
    bins = get_region_bins(map_df["strikes"], n_classes=cmap.N)
    norm = BoundaryNorm(bins, cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    draw_base(ax, europe, zoom_geom=zoom_geom)

    map_df.plot(
        ax=ax,
        column="strikes",
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.6,
        legend=False,
        zorder=10
    )

    plot_forecast_overlay(
        ax,
        forecast_gdf,
        fill_alpha=0.08,
        boundary_alpha=0.45,
        z_fill=18,
        z_line=19
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []

    tick_labels = []
    tick_positions = []
    for i in range(len(bins) - 1):
        tick_positions.append((bins[i] + bins[i + 1]) / 2)
        tick_labels.append(f"{int(bins[i])}-{int(bins[i + 1])}")

    cb = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
    cb.ax.tick_params(colors=TEXT_COLOR)
    cb.set_label(colorbar_label, color=TEXT_COLOR)
    for spine in cb.ax.spines.values():
        spine.set_color(TEXT_COLOR)
    cb.set_ticks(tick_positions)
    cb.set_ticklabels(tick_labels)

    minx, miny, maxx, maxy = get_zoom_bounds(zoom_geom)

    for _, row in map_df.iterrows():
        rep_point = row.geometry.representative_point()

        if not (minx <= rep_point.x <= maxx and miny <= rep_point.y <= maxy):
            continue

        ax.text(
            rep_point.x,
            rep_point.y,
            f"{int(row['strikes'])}",
            color="white",
            fontsize=8,
            fontweight="bold",
            fontfamily="Courier New",
            ha="center",
            va="center",
            zorder=30,
            clip_on=True
        )

    add_forecast_legend(ax, forecast_gdf)

    minx, miny, maxx, maxy = get_zoom_bounds(zoom_geom)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(title, color=TEXT_COLOR, fontsize=14)
    add_branding(ax)
    save(fig, out / output_file)

def plot_region_stats(region_stats, stats_col, title, output_file, out):
    if region_stats.empty:
        return

    plot_df = region_stats.sort_values("strikes", ascending=True).reset_index(drop=True)

    fig_height = max(6.5, 0.5 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    cmap = get_region_cmap()
    vmin = plot_df["strikes"].min()
    vmax = plot_df["strikes"].max()

    if vmax == vmin:
        colors = [cmap(0.6)] * len(plot_df)
    else:
        scaled = np.linspace(0, 1, len(plot_df))
        colors = [cmap(v) for v in scaled]

    bars = ax.barh(
        plot_df[stats_col],
        plot_df["strikes"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.7
    )

    ax.set_title(title, color=TEXT_COLOR, fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Strike count", color=TEXT_COLOR, fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.grid(True, axis="x", color=GRID_COLOR, alpha=0.28, linewidth=0.6)
    ax.grid(False, axis="y")
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

    max_val = plot_df["strikes"].max()
    ax.set_xlim(0, max_val * 1.16 if max_val > 0 else 1)

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        y = bar.get_y() + bar.get_height() / 2
        x = bar.get_width()

        ax.text(
            x + max(0.6, max_val * 0.012),
            y,
            f"{int(row['strikes']):,}",
            va="center",
            ha="left",
            color=TEXT_COLOR,
            fontsize=10,
            fontweight="bold"
        )

        if pd.notna(row.get("peak_hour", np.nan)):
            ax.text(
                x * 0.02 if x > 0 else 0.1,
                y,
                f"peak {int(row['peak_hour']):02d}:00",
                va="center",
                ha="left",
                color="#dddddd",
                fontsize=9
            )

    save(fig, out / output_file)

def write_region_stats(region_stats, output_file, out):
    if region_stats.empty:
        return
    region_stats.to_csv(out / output_file, index=False)

# -----------------------------
# RISK AREA STATS
# -----------------------------
def get_risk_area_stats(lightning_gdf, forecast_gdf):
    total_all = len(lightning_gdf)

    if forecast_gdf is None or forecast_gdf.empty:
        stats = pd.DataFrame([{
            "risk_key": "nonrisk",
            "label": FORECAST_STYLE["nonrisk"]["label"],
            "strikes": total_all,
            "percent_of_all_strikes": 100.0 if total_all > 0 else 0.0
        }])
        return stats, 0, 0.0

    left = lightning_gdf.copy()
    polys = forecast_gdf[["risk_key", "risk_rank", "geometry"]].copy()

    joined = gpd.sjoin(left, polys, how="left", predicate="intersects")

    in_risk = joined.dropna(subset=["risk_key"]).copy()
    if not in_risk.empty:
        in_risk = in_risk.sort_values(["strike_id", "risk_rank"], ascending=[True, False])
        in_risk = in_risk.drop_duplicates(subset=["strike_id"], keep="first")

    counted_ids = set(in_risk["strike_id"].tolist()) if not in_risk.empty else set()
    nonrisk_count = total_all - len(counted_ids)

    risk_counts = {key: 0 for key in RISK_ORDER}
    if not in_risk.empty:
        grouped = in_risk.groupby("risk_key").size()
        for key, value in grouped.items():
            if key in risk_counts:
                risk_counts[key] = int(value)

    rows = []
    for key in RISK_ORDER:
        rows.append({
            "risk_key": key,
            "label": FORECAST_STYLE[key]["label"],
            "strikes": risk_counts[key]
        })

    rows.append({
        "risk_key": "nonrisk",
        "label": FORECAST_STYLE["nonrisk"]["label"],
        "strikes": int(nonrisk_count)
    })

    stats = pd.DataFrame(rows)
    stats["percent_of_all_strikes"] = (stats["strikes"] / total_all * 100.0) if total_all > 0 else 0.0
    stats = stats[stats["strikes"] > 0].copy().reset_index(drop=True)

    total_in_risk = int(total_all - nonrisk_count)
    percent_in_risk = (total_in_risk / total_all * 100.0) if total_all > 0 else 0.0

    return stats, total_in_risk, percent_in_risk

def plot_risk_area_stats(risk_stats, total_in_risk, percent_in_risk, total_all, out):
    if risk_stats.empty:
        return

    colors = [FORECAST_STYLE[key]["color"] for key in risk_stats["risk_key"]]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    bars = ax.bar(risk_stats["label"], risk_stats["strikes"], color=colors, edgecolor="white", linewidth=0.8)

    ax.set_title("Lightning Strikes by Forecast Risk Area", color=TEXT_COLOR, fontsize=15, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Strike count", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

    ymax = max(risk_stats["strikes"].max(), 1)
    for bar, pct in zip(bars, risk_stats["percent_of_all_strikes"]):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.text(
            x, y + max(0.5, ymax * 0.01),
            f"{int(y)}\n{pct:.1f}%",
            ha="center", va="bottom",
            color=TEXT_COLOR, fontsize=9, fontweight="bold"
        )

    ax.text(
        0.99, 0.98,
        f"In any risk area: {total_in_risk:,} / {total_all:,} ({percent_in_risk:.1f}%)",
        transform=ax.transAxes,
        ha="right", va="top",
        color="white", fontsize=11, fontweight="bold"
    )

    save(fig, out / "15_risk_area_stats.png")

def write_risk_area_stats(risk_stats, out):
    if risk_stats.empty:
        return
    risk_stats.to_csv(out / "risk_area_stats.csv", index=False)

# -----------------------------
# GRID ANALYSIS / VERIFICATION
# -----------------------------
def build_analysis_grid(zoom_geom=None, cell_km=100):
    minx, miny, maxx, maxy = get_zoom_bounds(zoom_geom)
    cell_m = cell_km * 1000.0

    x_starts = np.arange(minx, maxx, cell_m)
    y_starts = np.arange(miny, maxy, cell_m)

    rows = []
    for r, y0 in enumerate(y_starts, start=1):
        for c, x0 in enumerate(x_starts, start=1):
            x1 = x0 + cell_m
            y1 = y0 + cell_m
            rows.append({
                "grid_id": f"R{r:02d}C{c:02d}",
                "row": r,
                "col": c,
                "geometry": box(x0, y0, x1, y1),
            })

    grid = gpd.GeoDataFrame(rows, crs="EPSG:3857")
    grid["grid_area_km2"] = grid.geometry.area / 1_000_000.0
    return grid

def get_grid_verification_stats(lightning_gdf, forecast_gdf=None, zoom_geom=None, cell_km=100, min_forecast_fraction=0.5):
    grid = build_analysis_grid(zoom_geom=zoom_geom, cell_km=cell_km)

    pts = lightning_gdf[["strike_id", "geometry"]].copy()
    joined = gpd.sjoin(pts, grid[["grid_id", "geometry"]], how="left", predicate="intersects")
    counts = joined.groupby("grid_id").size().rename("strikes").reset_index()
    grid = grid.merge(counts, on="grid_id", how="left")
    grid["strikes"] = grid["strikes"].fillna(0).astype(int)

    grid["any_forecast_overlap_percent"] = 0.0
    grid["assigned_risk"] = "nonrisk"
    grid["assigned_risk_label"] = FORECAST_STYLE["nonrisk"]["label"]
    grid["assigned_risk_rank"] = 0
    grid["assigned_risk_overlap_percent"] = 0.0
    grid["forecast_ge_threshold"] = False

    forecast_union = get_forecast_union_geometry(forecast_gdf)
    risk_unions = get_forecast_union_by_risk(forecast_gdf)

    any_overlaps = []
    assigned_keys = []
    assigned_percents = []
    assigned_ranks = []
    assigned_flags = []

    for geom in grid.geometry:
        if forecast_union is None or forecast_union.is_empty:
            any_overlap_pct = 0.0
        else:
            try:
                any_overlap_pct = (geom.intersection(forecast_union).area / geom.area) * 100.0 if geom.area > 0 else 0.0
            except Exception:
                any_overlap_pct = 0.0

        per_risk = []
        for key in RISK_ORDER:
            rgeom = risk_unions.get(key)
            if rgeom is None or rgeom.is_empty:
                pct = 0.0
            else:
                try:
                    pct = (geom.intersection(rgeom).area / geom.area) * 100.0 if geom.area > 0 else 0.0
                except Exception:
                    pct = 0.0
            per_risk.append((key, pct, FORECAST_STYLE[key]["rank"]))

        qualifying = [x for x in per_risk if x[1] >= (min_forecast_fraction * 100.0)]

        if qualifying:
            qualifying.sort(key=lambda x: (x[2], x[1]), reverse=True)
            chosen_key, chosen_pct, chosen_rank = qualifying[0]
            assigned_flag = True
        else:
            chosen_key, chosen_pct, chosen_rank = "nonrisk", 0.0, 0
            assigned_flag = False

        any_overlaps.append(any_overlap_pct)
        assigned_keys.append(chosen_key)
        assigned_percents.append(chosen_pct)
        assigned_ranks.append(chosen_rank)
        assigned_flags.append(assigned_flag)

    grid["any_forecast_overlap_percent"] = any_overlaps
    grid["assigned_risk"] = assigned_keys
    grid["assigned_risk_label"] = grid["assigned_risk"].map(lambda k: FORECAST_STYLE[k]["label"])
    grid["assigned_risk_rank"] = assigned_ranks
    grid["assigned_risk_overlap_percent"] = assigned_percents
    grid["forecast_ge_threshold"] = assigned_flags

    grid["hit"] = grid["forecast_ge_threshold"] & (grid["strikes"] > 0)
    grid["false_alarm"] = grid["forecast_ge_threshold"] & (grid["strikes"] == 0)
    grid["miss"] = (~grid["forecast_ge_threshold"]) & (grid["strikes"] > 0)
    grid["correct_null"] = (~grid["forecast_ge_threshold"]) & (grid["strikes"] == 0)

    risk_rows = []
    total_strikes = int(grid["strikes"].sum())

    for key in RISK_ORDER:
        sub = grid[grid["assigned_risk"] == key].copy()
        grids_total = int(len(sub))
        hits = int(sub["hit"].sum()) if not sub.empty else 0
        false_alarms = int(sub["false_alarm"].sum()) if not sub.empty else 0
        strikes_in_grids = int(sub["strikes"].sum()) if not sub.empty else 0
        struck_grids = int((sub["strikes"] > 0).sum()) if not sub.empty else 0

        risk_rows.append({
            "risk_key": key,
            "label": FORECAST_STYLE[key]["label"],
            "grids_total": grids_total,
            "grids_with_strikes": struck_grids,
            "hits": hits,
            "false_alarms": false_alarms,
            "strikes_in_assigned_grids": strikes_in_grids,
            "percent_of_all_strikes": (strikes_in_grids / total_strikes * 100.0) if total_strikes > 0 else 0.0,
        })

    sub = grid[grid["assigned_risk"] == "nonrisk"].copy()
    risk_rows.append({
        "risk_key": "nonrisk",
        "label": FORECAST_STYLE["nonrisk"]["label"],
        "grids_total": int(len(sub)),
        "grids_with_strikes": int((sub["strikes"] > 0).sum()) if not sub.empty else 0,
        "hits": 0,
        "false_alarms": 0,
        "strikes_in_assigned_grids": int(sub["strikes"].sum()) if not sub.empty else 0,
        "percent_of_all_strikes": (int(sub["strikes"].sum()) / total_strikes * 100.0) if total_strikes > 0 else 0.0,
    })

    risk_summary = pd.DataFrame(risk_rows)

    verification_total = int(grid["hit"].sum() + grid["false_alarm"].sum() + grid["miss"].sum())

    hits_n = int(grid["hit"].sum())
    false_n = int(grid["false_alarm"].sum())
    miss_n = int(grid["miss"].sum())
    correct_null_n = int(grid["correct_null"].sum())

    hit_pct = (hits_n / verification_total * 100.0) if verification_total > 0 else 0.0
    miss_pct = (miss_n / verification_total * 100.0) if verification_total > 0 else 0.0
    false_pct = (false_n / verification_total * 100.0) if verification_total > 0 else 0.0

    summary = pd.DataFrame([{
        "grid_cell_km": int(cell_km),
        "forecast_threshold_percent": float(min_forecast_fraction * 100.0),
        "total_grids": int(len(grid)),
        "grids_forecast": int(grid["forecast_ge_threshold"].sum()),
        "grids_with_strikes": int((grid["strikes"] > 0).sum()),
        "hits": hits_n,
        "false_alarms": false_n,
        "misses": miss_n,
        "correct_nulls": correct_null_n,
        "verification_total": verification_total,
        "hit_percent": hit_pct,
        "miss_percent": miss_pct,
        "false_alarm_percent": false_pct,
        "hits_majority": bool(hit_pct >= 50.0),
        "total_strikes": total_strikes,
        "strikes_in_forecast_grids": int(grid.loc[grid["forecast_ge_threshold"], "strikes"].sum()),
        "percent_strikes_in_forecast_grids": (
            grid.loc[grid["forecast_ge_threshold"], "strikes"].sum() / total_strikes * 100.0
            if total_strikes > 0 else 0.0
        ),
    }])

    return grid, risk_summary, summary

def write_grid_stats(grid_gdf, out):
    if grid_gdf is None or grid_gdf.empty:
        return
    export_df = grid_gdf.copy()
    export_df["geometry"] = export_df.geometry.to_wkt()
    export_df.to_csv(out / "100km_grid_stats.csv", index=False)

def write_grid_risk_summary(risk_summary_df, out):
    if risk_summary_df is None or risk_summary_df.empty:
        return
    risk_summary_df.to_csv(out / "100km_grid_risk_stats.csv", index=False)

def write_summary_csv(summary_df, out):
    if summary_df is None or summary_df.empty:
        return
    summary_df.to_csv(out / "summary.csv", index=False)

def plot_grid_map(grid_gdf, europe, out, forecast_gdf=None, zoom_geom=None, title_suffix=""):
    if grid_gdf is None or grid_gdf.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 9))
    draw_base(ax, europe, zoom_geom=zoom_geom)

    grid_gdf.boundary.plot(ax=ax, color="#777777", linewidth=0.35, alpha=0.65, zorder=8)

    active = grid_gdf[grid_gdf["strikes"] > 0].copy()
    if not active.empty:
        cmap = get_region_cmap()
        bins = get_region_bins(active["strikes"], n_classes=cmap.N)
        norm = BoundaryNorm(bins, cmap.N, clip=True)

        active.plot(
            ax=ax,
            column="strikes",
            cmap=cmap,
            norm=norm,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.80,
            zorder=11
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []

        tick_labels = []
        tick_positions = []
        for i in range(len(bins) - 1):
            tick_positions.append((bins[i] + bins[i + 1]) / 2)
            tick_labels.append(f"{int(bins[i])}-{int(bins[i + 1])}")

        cb = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
        cb.ax.tick_params(colors=TEXT_COLOR)
        cb.set_label("Strikes per 100 km grid", color=TEXT_COLOR)
        for spine in cb.ax.spines.values():
            spine.set_color(TEXT_COLOR)
        cb.set_ticks(tick_positions)
        cb.set_ticklabels(tick_labels)

    qualified = grid_gdf[grid_gdf["forecast_ge_threshold"]].copy()
    if not qualified.empty:
        qualified.boundary.plot(
            ax=ax,
            color="#00FFFF",
            linewidth=1.2,
            alpha=0.95,
            zorder=20
        )

    plot_forecast_overlay(
        ax,
        forecast_gdf,
        fill_alpha=0.05,
        boundary_alpha=0.50,
        z_fill=15,
        z_line=16
    )

    for _, row in active.iterrows():
        rp = row.geometry.representative_point()
        ax.text(
            rp.x,
            rp.y,
            f"{int(row['strikes'])}",
            color="white",
            fontsize=8,
            fontweight="bold",
            fontfamily="Courier New",
            ha="center",
            va="center",
            zorder=30
        )

    info_text = (
        f"100 km grid\n"
        f"Forecast grids: {int(grid_gdf['forecast_ge_threshold'].sum())}\n"
        f"Hit grids: {int(grid_gdf['hit'].sum())}\n"
        f"False alarm grids: {int(grid_gdf['false_alarm'].sum())}\n"
        f"Miss grids: {int(grid_gdf['miss'].sum())}"
    )

    ax.text(
        0.99, 0.01,
        info_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", edgecolor="white", alpha=0.75, boxstyle="round,pad=0.3"),
        zorder=40
    )

    add_forecast_legend(ax, forecast_gdf)

    minx, miny, maxx, maxy = get_zoom_bounds(zoom_geom)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(f"100 km Grid Verification Map{title_suffix}", color=TEXT_COLOR, fontsize=14)
    add_branding(ax)
    save(fig, out / "17_grid_100km_map.png")

def plot_grid_risk_map(grid_gdf, europe, out, forecast_gdf=None, zoom_geom=None, title_suffix=""):
    if grid_gdf is None or grid_gdf.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 9))
    draw_base(ax, europe, zoom_geom=zoom_geom)

    grid_gdf.boundary.plot(ax=ax, color="#555555", linewidth=0.30, alpha=0.60, zorder=8)

    assigned = grid_gdf[grid_gdf["forecast_ge_threshold"]].copy()
    for key in RISK_ORDER:
        part = assigned[assigned["assigned_risk"] == key]
        if part.empty:
            continue
        part.plot(
            ax=ax,
            facecolor=FORECAST_STYLE[key]["color"],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.55,
            zorder=12
        )

    plot_forecast_overlay(
        ax,
        forecast_gdf,
        fill_alpha=0.03,
        boundary_alpha=0.45,
        z_fill=15,
        z_line=16
    )

    for _, row in assigned.iterrows():
        rp = row.geometry.representative_point()
        ax.text(
            rp.x,
            rp.y,
            f"{int(row['strikes'])}",
            color="white",
            fontsize=8,
            fontweight="bold",
            fontfamily="Courier New",
            ha="center",
            va="center",
            zorder=30
        )

    handles = [
        Patch(facecolor=FORECAST_STYLE[k]["color"], edgecolor="white", label=FORECAST_STYLE[k]["label"], alpha=0.55)
        for k in RISK_ORDER if (assigned["assigned_risk"] == k).any()
    ]
    if handles:
        leg = ax.legend(handles=handles, loc="lower left", frameon=True, facecolor="black", edgecolor="white", fontsize=9)
        for text in leg.get_texts():
            text.set_color("white")

    minx, miny, maxx, maxy = get_zoom_bounds(zoom_geom)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(f"100 km Grid Assigned Risk Map{title_suffix}", color=TEXT_COLOR, fontsize=14)
    add_branding(ax)
    save(fig, out / "17b_grid_assigned_risk_map.png")

def plot_summary_graphics(summary_df, risk_summary_df, out):
    if summary_df is None or summary_df.empty:
        return

    row = summary_df.iloc[0]

    hits = int(row["hits"])
    false_alarms = int(row["false_alarms"])
    misses = int(row["misses"])
    correct_nulls = int(row["correct_nulls"])

    verification_total = hits + misses + false_alarms
    if verification_total > 0:
        hit_pct = hits / verification_total * 100.0
        miss_pct = misses / verification_total * 100.0
        false_alarm_pct = false_alarms / verification_total * 100.0
    else:
        hit_pct = 0.0
        miss_pct = 0.0
        false_alarm_pct = 0.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.patch.set_facecolor(FIG_BG)

    ax = axes[0]
    ax.set_facecolor(AX_BG)

    labels = ["Hits", "False alarms", "Misses", "Correct nulls"]
    values = [hits, false_alarms, misses, correct_nulls]
    colors = ["#2E8B57", "#FF8C00", "#8B0000", "#555555"]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Grid Verification Overview", color=TEXT_COLOR, fontsize=15, fontweight="bold")
    ax.set_ylabel("Grid count", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

    ymax = max(values) if values else 0
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(0.5, ymax * 0.02 if ymax > 0 else 0.5),
            str(int(val)),
            ha="center",
            va="bottom",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

    ax.text(
        0.99, 0.98,
        f"Strikes in forecast grids: {int(row['strikes_in_forecast_grids'])} / {int(row['total_strikes'])} ({row['percent_strikes_in_forecast_grids']:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="white",
        fontsize=10,
        fontweight="bold"
    )

    ax = axes[1]
    ax.set_facecolor(AX_BG)

    pct_labels = ["Hits", "Misses", "False alarms"]
    pct_values = [hit_pct, miss_pct, false_alarm_pct]
    pct_colors = ["#2E8B57", "#8B0000", "#FF8C00"]

    bars = ax.bar(pct_labels, pct_values, color=pct_colors, edgecolor="white", linewidth=0.8)

    ax.set_title("Forecast Success Percentages", color=TEXT_COLOR, fontsize=15, fontweight="bold")
    ax.set_ylabel("% of verification cases", color=TEXT_COLOR)
    ax.set_ylim(0, 100)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

    for bar, val in zip(bars, pct_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            color="white",
            fontsize=11,
            fontweight="bold"
        )

    if hit_pct >= 50:
        verdict = "Hits are the majority ✅"
        verdict_color = "#90EE90"
    else:
        verdict = "Hits are not the majority ❌"
        verdict_color = "#FF6666"

    ax.text(
        0.99, 0.97,
        verdict,
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=verdict_color,
        fontsize=12,
        fontweight="bold"
    )

    ax = axes[2]
    ax.set_facecolor(AX_BG)

    if risk_summary_df is not None and not risk_summary_df.empty:
        temp = risk_summary_df[risk_summary_df["risk_key"] != "nonrisk"].copy()
        if not temp.empty:
            x = np.arange(len(temp))
            hits_r = temp["hits"].astype(int).values
            false_r = temp["false_alarms"].astype(int).values
            colors_r = [FORECAST_STYLE[k]["color"] for k in temp["risk_key"]]

            ax.bar(x, hits_r, color=colors_r, edgecolor="white", linewidth=0.8, label="Hits")
            ax.bar(x, false_r, bottom=hits_r, color="none", edgecolor="white", linewidth=1.0, hatch="///", label="False alarms")

            ax.set_xticks(x)
            ax.set_xticklabels(temp["label"], rotation=20, color="white")
            ax.set_ylabel("Grid count", color=TEXT_COLOR)
            ax.set_title("Verification by Assigned Risk", color=TEXT_COLOR, fontsize=15, fontweight="bold")
            ax.tick_params(colors=TEXT_COLOR)
            ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.3)

            for spine in ax.spines.values():
                spine.set_color(TEXT_COLOR)

            for i, (_, r) in enumerate(temp.iterrows()):
                total = int(r["grids_total"])
                ax.text(
                    i,
                    hits_r[i] + false_r[i] + max(0.2, (hits_r[i] + false_r[i]) * 0.03 + 0.2),
                    f"{total}",
                    ha="center",
                    va="bottom",
                    color="white",
                    fontsize=9,
                    fontweight="bold"
                )

            leg = ax.legend(facecolor="black", edgecolor="white")
            for text in leg.get_texts():
                text.set_color("white")

    save(fig, out / "18_summary_verification_graphics.png")

# -----------------------------
# DENSITY
# -----------------------------
def compute_density_grid(gdf, zoom_geom=None, cell_km=5, sigma_cells=2.0):
    if gdf.empty:
        return None, None, None

    minx, miny, maxx, maxy = get_zoom_bounds(zoom_geom)

    cell_m = cell_km * 1000.0
    x_edges = np.arange(minx, maxx + cell_m, cell_m)
    y_edges = np.arange(miny, maxy + cell_m, cell_m)

    if len(x_edges) < 2 or len(y_edges) < 2:
        return None, None, None

    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values

    counts, xedges, yedges = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    smooth_counts = gaussian_filter(counts, sigma=sigma_cells, mode="constant")

    return smooth_counts.T, xedges, yedges

def plot_density_map(gdf, europe, out, forecast_gdf=None, zoom_geom=None, title_suffix=""):
    fig, ax = plt.subplots(figsize=(12, 9))
    draw_base(ax, europe, zoom_geom=zoom_geom)
    plot_forecast_overlay(ax, forecast_gdf)

    hb = ax.hexbin(
        gdf.geometry.x,
        gdf.geometry.y,
        gridsize=130,
        cmap="inferno",
        bins="log",
        mincnt=1,
        zorder=15
    )

    add_colorbar(fig, ax, hb, "Strike density")
    add_forecast_legend(ax, forecast_gdf)

    minx, miny, maxx, maxy = get_zoom_bounds(zoom_geom)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(f"Lightning Density Hexbin Map{title_suffix}", color=TEXT_COLOR, fontsize=14)
    add_branding(ax)
    save(fig, out / "03_density_map.png")

def plot_density_grid_map(gdf, europe, out, forecast_gdf=None, zoom_geom=None, title_suffix="", cell_km=5, sigma_cells=2.0):
    density_counts, xedges, yedges = compute_density_grid(
        gdf,
        zoom_geom=zoom_geom,
        cell_km=cell_km,
        sigma_cells=sigma_cells
    )
    if density_counts is None:
        return

    max_val = float(np.nanmax(density_counts)) if density_counts.size else 0.0
    if max_val <= 0:
        return

    relative_density = (density_counts / max_val) * 100.0

    cmap, norm, rel_levels = get_relative_density_cmap_and_norm()
    density_masked = np.ma.masked_where(relative_density < rel_levels[0], relative_density)

    fig, ax = plt.subplots(figsize=(12, 9))
    draw_base(ax, europe, zoom_geom=zoom_geom)

    mesh = ax.pcolormesh(
        xedges,
        yedges,
        density_masked,
        cmap=cmap,
        norm=norm,
        shading="auto",
        alpha=0.88,
        zorder=12
    )

    plot_forecast_overlay(
        ax,
        forecast_gdf,
        fill_alpha=0.06,
        boundary_alpha=0.5,
        z_fill=18,
        z_line=19
    )

    cb = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.02)
    cb.ax.tick_params(colors=TEXT_COLOR)
    cb.set_label("Relative density (% of current-event max)", color=TEXT_COLOR)
    for spine in cb.ax.spines.values():
        spine.set_color(TEXT_COLOR)

    shown_ticks = [v for v in rel_levels if v <= 100]
    cb.set_ticks(shown_ticks)
    cb.set_ticklabels([str(v) for v in shown_ticks])

    add_forecast_legend(ax, forecast_gdf)

    minx, miny, maxx, maxy = get_zoom_bounds(zoom_geom)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"Relative Lightning Density Grid Map ({cell_km} km cells, smoothed){title_suffix}",
        color=TEXT_COLOR,
        fontsize=14
    )
    add_branding(ax)
    save(fig, out / "03b_density_grid_map.png")


def plot_absolute_density_grid_map(
    gdf,
    europe,
    out,
    forecast_gdf=None,
    zoom_geom=None,
    title_suffix="",
    cell_km=5,
    sigma_cells=2.0,
    unit_mode="km2"
):
    density_counts, xedges, yedges = compute_density_grid(
        gdf,
        zoom_geom=zoom_geom,
        cell_km=cell_km,
        sigma_cells=sigma_cells
    )
    if density_counts is None:
        return

    if density_counts.size == 0:
        return

    cell_area_km2 = float(cell_km * cell_km)

    if unit_mode == "km2":
        density_vals = density_counts / cell_area_km2
        cbar_label = "Lightning density (strikes / km²)"
        title_unit = "strikes / km²"
        out_name = "03c_density_absolute_km2.png"
        cmap, norm, levels, base_levels = get_density_palette_levels(scale_factor=1.0 / cell_area_km2)
        density_masked = np.ma.masked_where(density_vals < levels[0], density_vals)
        shown_ticks = levels
        shown_ticklabels = [f"{v:g}" for v in base_levels]
    else:
        density_vals = density_counts
        cbar_label = f"Lightning density (strikes / {int(cell_area_km2)} km²)"
        title_unit = f"strikes / {int(cell_area_km2)} km²"
        out_name = f"03d_density_absolute_{int(cell_area_km2)}km2.png"
        cmap, norm, levels, base_levels = get_density_palette_levels(scale_factor=1.0)
        density_masked = np.ma.masked_where(density_vals < levels[0], density_vals)
        shown_ticks = levels
        shown_ticklabels = [str(v) for v in base_levels]

    max_val = float(np.nanmax(density_vals)) if density_vals.size else 0.0
    if max_val <= 0:
        return

    fig, ax = plt.subplots(figsize=(12, 9))
    draw_base(ax, europe, zoom_geom=zoom_geom)

    mesh = ax.pcolormesh(
        xedges,
        yedges,
        density_masked,
        cmap=cmap,
        norm=norm,
        shading="auto",
        alpha=0.88,
        zorder=12
    )

    plot_forecast_overlay(
        ax,
        forecast_gdf,
        fill_alpha=0.06,
        boundary_alpha=0.5,
        z_fill=18,
        z_line=19
    )

    cb = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.02)
    cb.ax.tick_params(colors=TEXT_COLOR)
    cb.set_label(cbar_label, color=TEXT_COLOR)
    for spine in cb.ax.spines.values():
        spine.set_color(TEXT_COLOR)
    cb.set_ticks(shown_ticks)
    cb.set_ticklabels(shown_ticklabels)

    add_forecast_legend(ax, forecast_gdf)

    minx, miny, maxx, maxy = get_zoom_bounds(zoom_geom)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"Absolute Lightning Density Grid Map ({cell_km} km cells, smoothed, {title_unit}){title_suffix}",
        color=TEXT_COLOR,
        fontsize=14
    )
    add_branding(ax)
    save(fig, out / out_name)

# -----------------------------
# CORE MAPS
# -----------------------------
def plot_hour_map(gdf, europe, out, forecast_gdf=None, zoom_geom=None, title_suffix=""):
    temp = gdf.dropna(subset=["strike_time"]).copy()
    if temp.empty:
        return

    temp["local_time"] = to_local_time(temp["strike_time"])
    temp["hour"] = temp["local_time"].dt.hour
    cmap, norm = get_discrete_hour_cmap()

    fig, ax = plt.subplots(figsize=(12, 9))
    draw_base(ax, europe, zoom_geom=zoom_geom)
    plot_forecast_overlay(ax, forecast_gdf)

    sc = ax.scatter(
        temp.geometry.x,
        temp.geometry.y,
        c=temp["hour"],
        cmap=cmap,
        norm=norm,
        marker="+",
        s=55,
        linewidths=1.0,
        alpha=0.95,
        zorder=20
    )

    add_colorbar(fig, ax, sc, "Hour (CEST/CET)", ticks=range(24))
    add_forecast_legend(ax, forecast_gdf)

    minx, miny, maxx, maxy = get_zoom_bounds(zoom_geom)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(f"Lightning Strikes by Hour{title_suffix}", color=TEXT_COLOR, fontsize=14)
    add_branding(ax)
    save(fig, out / "01_hour_map_discrete.png")

def plot_intensity_map(gdf, europe, out, forecast_gdf=None, zoom_geom=None, title_suffix=""):
    if INTENSITY_COLUMN not in gdf.columns:
        return

    temp = gdf.copy()
    temp["intensity"] = pd.to_numeric(temp[INTENSITY_COLUMN], errors="coerce")
    temp = temp.dropna(subset=["intensity"])
    if temp.empty:
        return

    sizes = marker_sizes(temp["intensity"])

    fig, ax = plt.subplots(figsize=(12, 9))
    draw_base(ax, europe, zoom_geom=zoom_geom)
    plot_forecast_overlay(ax, forecast_gdf)

    sc = ax.scatter(
        temp.geometry.x,
        temp.geometry.y,
        c=temp["intensity"],
        cmap="plasma",
        marker="+",
        s=sizes,
        linewidths=1.1,
        alpha=0.9,
        zorder=20
    )

    add_colorbar(fig, ax, sc, f"Intensity ({INTENSITY_COLUMN})")
    add_forecast_legend(ax, forecast_gdf)

    minx, miny, maxx, maxy = get_zoom_bounds(zoom_geom)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(f"Lightning Intensity Map{title_suffix}", color=TEXT_COLOR, fontsize=14)
    add_branding(ax)
    save(fig, out / "02_intensity_map.png")

def plot_intensity_hexbin_map(gdf, europe, out, forecast_gdf=None, zoom_geom=None, title_suffix=""):
    if INTENSITY_COLUMN not in gdf.columns:
        return

    temp = gdf.copy()
    temp["intensity"] = pd.to_numeric(temp[INTENSITY_COLUMN], errors="coerce")
    temp = temp.dropna(subset=["intensity"])
    if temp.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 9))
    draw_base(ax, europe, zoom_geom=zoom_geom)
    plot_forecast_overlay(ax, forecast_gdf)

    hb = ax.hexbin(
        temp.geometry.x,
        temp.geometry.y,
        C=temp["intensity"],
        reduce_C_function=np.mean,
        gridsize=120,
        cmap="plasma",
        mincnt=1,
        zorder=15
    )

    add_colorbar(fig, ax, hb, f"Mean intensity ({INTENSITY_COLUMN})")
    add_forecast_legend(ax, forecast_gdf)

    minx, miny, maxx, maxy = get_zoom_bounds(zoom_geom)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(f"Mean Intensity Hexbin Map{title_suffix}", color=TEXT_COLOR, fontsize=14)
    add_branding(ax)
    save(fig, out / "04_intensity_hexbin_map.png")

# -----------------------------
# GIF
# -----------------------------
def floor_to_interval(ts, minutes=30):
    return ts.floor(f"{minutes}min")

def ceil_to_interval(ts, minutes=30):
    floored = ts.floor(f"{minutes}min")
    if floored == ts:
        return ts
    return floored + pd.Timedelta(minutes=minutes)

def build_gif_intervals(local_times, gif_mode):
    minutes = 60 if gif_mode == "1h" else 30
    start_ts = floor_to_interval(local_times.min(), minutes=minutes)
    end_ts = ceil_to_interval(local_times.max(), minutes=minutes)
    intervals = pd.date_range(start=start_ts, end=end_ts, freq=f"{minutes}min")
    return intervals, minutes

def plot_gif_frame(
    temp,
    europe,
    forecast_gdf,
    zoom_geom,
    frame_end,
    frame_path,
    title_suffix="",
    cumulative=True
):
    cmap, norm = get_discrete_hour_cmap()

    fig, ax = plt.subplots(figsize=(12, 9))
    draw_base(ax, europe, zoom_geom=zoom_geom)
    plot_forecast_overlay(ax, forecast_gdf)

    if not temp.empty:
        sc = ax.scatter(
            temp.geometry.x,
            temp.geometry.y,
            c=temp["hour"],
            cmap=cmap,
            norm=norm,
            marker="+",
            s=55,
            linewidths=1.0,
            alpha=0.95,
            zorder=20
        )
        add_colorbar(fig, ax, sc, "Hour (CEST/CET)", ticks=range(24))

    add_forecast_legend(ax, forecast_gdf)

    minx, miny, maxx, maxy = get_zoom_bounds(zoom_geom)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")

    stamp = frame_end.strftime("%Y-%m-%d %H:%M")
    count = len(temp)

    if cumulative:
        ax.set_title(
            f"Lightning Animation{title_suffix}\nUp to {stamp} CEST/CET — {count} strikes",
            color=TEXT_COLOR,
            fontsize=14
        )
    else:
        ax.set_title(
            f"Lightning Animation{title_suffix}\n{stamp} CEST/CET — {count} strikes",
            color=TEXT_COLOR,
            fontsize=14
        )

    add_branding(ax)
    save(fig, frame_path)

def create_lightning_gif(gdf, europe, out, forecast_gdf=None, zoom_geom=None, title_suffix="", gif_mode="30min"):
    if gdf.empty:
        return

    temp = gdf.dropna(subset=["strike_time"]).copy()
    if temp.empty:
        return

    temp["local_time"] = to_local_time(temp["strike_time"])
    temp["hour"] = temp["local_time"].dt.hour

    intervals, minutes = build_gif_intervals(temp["local_time"], gif_mode)

    frames_dir = out / "_gif_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_files = []

    for i, frame_end in enumerate(intervals):
        frame_df = temp[temp["local_time"] <= frame_end].copy()
        frame_file = frames_dir / f"frame_{i:03d}.png"

        plot_gif_frame(
            frame_df,
            europe,
            forecast_gdf,
            zoom_geom,
            frame_end,
            frame_file,
            title_suffix=title_suffix,
            cumulative=True
        )
        frame_files.append(frame_file)

    if not frame_files:
        print("No GIF frames created.")
        return

    images = [Image.open(fp).convert("P", palette=Image.ADAPTIVE) for fp in frame_files]
    gif_name = "16_lightning_animation_1h.gif" if minutes == 60 else "16_lightning_animation_30min.gif"
    gif_path = out / gif_name

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=700,
        loop=0
    )

    for img in images:
        img.close()

    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"GIF saved: {gif_path}")

# -----------------------------
# ARCHIVE HELPERS
# -----------------------------
def get_archive_day_folder():
    today = pd.Timestamp.now(tz=LOCAL_TIMEZONE).strftime("%Y-%m-%d")
    archive_day = Path("output") / "archive" / today
    archive_day.mkdir(parents=True, exist_ok=True)
    return archive_day

def save_filtered_strikes_geojson(gdf):
    if gdf is None or gdf.empty:
        return

    archive_day = get_archive_day_folder()
    out_file = archive_day / "filtered_strikes.geojson"

    export_gdf = gdf.copy()
    try:
        export_gdf = export_gdf.to_crs(epsg=4326)
    except Exception:
        pass

    export_gdf.to_file(out_file, driver="GeoJSON")
    print(f"Saved filtered strikes GeoJSON: {out_file}")

# -----------------------------
# TIME / BASIC STATS GRAPHS
# -----------------------------
def plot_hourly_counts(df, out):
    d = df.dropna(subset=["strike_time"]).copy()
    if d.empty:
        return

    d["local_time"] = to_local_time(d["strike_time"])
    d["hour"] = d["local_time"].dt.hour
    counts = d["hour"].value_counts().sort_index().reindex(range(24), fill_value=0)

    cmap, norm = get_discrete_hour_cmap()
    bar_colors = [cmap(norm(h)) for h in counts.index]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    ax.bar(counts.index, counts.values, color=bar_colors)
    ax.set_title("Hourly Lightning Counts (Central European Time)", color=TEXT_COLOR, fontsize=14)
    ax.set_xlabel("Hour (CEST/CET)", color=TEXT_COLOR)
    ax.set_ylabel("Strike count", color=TEXT_COLOR)
    ax.set_xticks(range(24))
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

    ymax = counts.max() if len(counts) else 0
    for x, y in zip(counts.index, counts.values):
        ax.text(
            x, y + max(0.5, ymax * 0.01), str(int(y)),
            ha="center", va="bottom", color=TEXT_COLOR, fontsize=8
        )

    save(fig, out / "05_hourly_counts.png")

def plot_hourly_intensity(df, out):
    if INTENSITY_COLUMN not in df.columns:
        return

    d = df.dropna(subset=["strike_time"]).copy()
    if d.empty:
        return

    d["intensity"] = pd.to_numeric(d[INTENSITY_COLUMN], errors="coerce")
    d = d.dropna(subset=["intensity"])
    if d.empty:
        return

    d["local_time"] = to_local_time(d["strike_time"])
    d["hour"] = d["local_time"].dt.hour
    mean_intensity = d.groupby("hour")["intensity"].mean().reindex(range(24), fill_value=0)

    cmap, norm = get_discrete_hour_cmap()
    colors = [cmap(norm(h)) for h in mean_intensity.index]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    ax.bar(mean_intensity.index, mean_intensity.values, color=colors)
    ax.set_title(f"Mean Intensity by Hour ({INTENSITY_COLUMN})", color=TEXT_COLOR, fontsize=14)
    ax.set_xlabel("Hour (CEST/CET)", color=TEXT_COLOR)
    ax.set_ylabel("Mean intensity", color=TEXT_COLOR)
    ax.set_xticks(range(24))
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

    save(fig, out / "06_hourly_mean_intensity.png")

def plot_time_detail(df, out):
    d = df.dropna(subset=["strike_time"]).copy()
    if d.empty:
        return

    d["local_time"] = to_local_time(d["strike_time"])
    d = d.sort_values("local_time").reset_index(drop=True)
    d["hour_float"] = (
        d["local_time"].dt.hour +
        d["local_time"].dt.minute / 60.0 +
        d["local_time"].dt.second / 3600.0 +
        d["local_time"].dt.microsecond / 3_600_000_000.0
    )
    d["hour"] = d["local_time"].dt.hour

    cmap, norm = get_discrete_hour_cmap()

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    ax.scatter(
        range(len(d)),
        d["hour_float"],
        c=d["hour"],
        cmap=cmap,
        norm=norm,
        marker="+",
        s=40,
        linewidths=1.0,
        alpha=0.9
    )

    ax.set_title("Strike Timing Detail", color=TEXT_COLOR, fontsize=14)
    ax.set_xlabel("Strike index (time ordered)", color=TEXT_COLOR)
    ax.set_ylabel("Hour of day (CEST/CET)", color=TEXT_COLOR)
    ax.set_yticks(range(24))
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

    save(fig, out / "07_time_detail.png")

def plot_daily_counts(df, out):
    d = df.dropna(subset=["strike_time"]).copy()
    if d.empty:
        return

    d["local_time"] = to_local_time(d["strike_time"])
    d["day"] = d["local_time"].dt.date
    counts = d["day"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    ax.plot(counts.index, counts.values, marker="o", linewidth=2)
    ax.set_title("Daily Lightning Counts", color=TEXT_COLOR, fontsize=14)
    ax.set_xlabel("Day", color=TEXT_COLOR)
    ax.set_ylabel("Strike count", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

    save(fig, out / "08_daily_counts.png")

def plot_monthly(df, out):
    d = df.dropna(subset=["strike_time"]).copy()
    if d.empty:
        return

    d["local_time"] = to_local_time(d["strike_time"])
    d["month"] = d["local_time"].dt.month
    counts = d["month"].value_counts().sort_index().reindex(range(1, 13), fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    ax.bar(counts.index, counts.values)
    ax.set_title("Monthly Lightning Counts", color=TEXT_COLOR, fontsize=14)
    ax.set_xlabel("Month", color=TEXT_COLOR)
    ax.set_ylabel("Strike count", color=TEXT_COLOR)
    ax.set_xticks(range(1, 13))
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

    save(fig, out / "08_monthly_counts.png")

# -----------------------------
# SUMMARY TXT
# -----------------------------
def write_summary(
    df,
    out,
    country_stats=None,
    province_stats=None,
    risk_stats=None,
    grid_stats=None,
    grid_risk_summary=None,
    summary_df=None,
    total_in_risk=0,
    percent_in_risk=0.0,
    time_filter=None,
    country_filter_display=None
):
    lines = []
    lines.append("LIGHTNING ANALYSIS SUMMARY")
    lines.append("==========================")
    lines.append(f"Total strikes in selection: {len(df):,}")

    if country_filter_display:
        lines.append(f"Country filter applied: {country_filter_display}")

    if time_filter:
        lines.append(f"Time filter applied (local time): {time_filter}")

    if not df.empty:
        lines.append(f"Latitude range: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
        lines.append(f"Longitude range: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")

        local_times = to_local_time(df["strike_time"].dropna())
        if not local_times.empty:
            lines.append(f"First strike time: {local_times.min()}")
            lines.append(f"Last strike time : {local_times.max()}")

            hourly = (
                df.dropna(subset=["strike_time"])
                .assign(local_time=lambda x: to_local_time(x["strike_time"]))
                .assign(hour=lambda x: x["local_time"].dt.hour)
                .groupby("hour")
                .size()
                .reindex(range(24), fill_value=0)
            )
            lines.append("")
            lines.append("Hourly counts (CEST/CET):")
            for hour, count in hourly.items():
                lines.append(f"{hour:02d}:00 -> {int(count)}")

        if INTENSITY_COLUMN in df.columns:
            s = pd.to_numeric(df[INTENSITY_COLUMN], errors="coerce").dropna()
            if not s.empty:
                lines.append("")
                lines.append(f"Intensity column used: {INTENSITY_COLUMN}")
                lines.append(f"Min intensity: {s.min()}")
                lines.append(f"Max intensity: {s.max()}")
                lines.append(f"Mean intensity: {s.mean():.2f}")
                lines.append(f"Median intensity: {s.median():.2f}")

    if country_stats is not None and not country_stats.empty:
        lines.append("")
        lines.append("Countries struck:")
        for _, row in country_stats.iterrows():
            line = f"{row['country']}: {int(row['strikes'])} strikes"
            if pd.notna(row.get("peak_hour", np.nan)):
                line += f", peak hour {int(row['peak_hour']):02d}:00"
            lines.append(line)

    if province_stats is not None and not province_stats.empty:
        lines.append("")
        lines.append("Provinces/regions struck:")
        for _, row in province_stats.iterrows():
            line = f"{row['province']}: {int(row['strikes'])} strikes"
            if pd.notna(row.get("peak_hour", np.nan)):
                line += f", peak hour {int(row['peak_hour']):02d}:00"
            lines.append(line)

    if risk_stats is not None and not risk_stats.empty:
        lines.append("")
        lines.append("Forecast risk area verification:")
        lines.append(f"Strikes in any risk area: {total_in_risk:,} / {len(df):,} ({percent_in_risk:.1f}%)")
        for _, row in risk_stats.iterrows():
            lines.append(
                f"{row['label']}: {int(row['strikes'])} strikes ({row['percent_of_all_strikes']:.1f}% of all strikes)"
            )

    if summary_df is not None and not summary_df.empty:
        row = summary_df.iloc[0]
        lines.append("")
        lines.append(f"{int(row['grid_cell_km'])} km grid verification:")
        lines.append(f"Forecast threshold: {row['forecast_threshold_percent']:.1f}%")
        lines.append(f"Total grids: {int(row['total_grids'])}")
        lines.append(f"Forecast grids: {int(row['grids_forecast'])}")
        lines.append(f"Grids with strikes: {int(row['grids_with_strikes'])}")
        lines.append(f"Hits: {int(row['hits'])}")
        lines.append(f"False alarms: {int(row['false_alarms'])}")
        lines.append(f"Misses: {int(row['misses'])}")
        lines.append(f"Correct nulls: {int(row['correct_nulls'])}")
        lines.append(
            f"Success percentages -> Hits: {row['hit_percent']:.1f}% | "
            f"Misses: {row['miss_percent']:.1f}% | "
            f"False alarms: {row['false_alarm_percent']:.1f}%"
        )
        lines.append(f"Hits majority: {'YES' if bool(row['hits_majority']) else 'NO'}")
        lines.append(
            f"Strikes in forecast grids: {int(row['strikes_in_forecast_grids'])} / {int(row['total_strikes'])} "
            f"({row['percent_strikes_in_forecast_grids']:.1f}%)"
        )

    if grid_risk_summary is not None and not grid_risk_summary.empty:
        lines.append("")
        lines.append("Verification by assigned grid risk:")
        for _, row in grid_risk_summary.iterrows():
            lines.append(
                f"{row['label']}: grids={int(row['grids_total'])}, "
                f"grids with strikes={int(row['grids_with_strikes'])}, "
                f"hits={int(row['hits'])}, false alarms={int(row['false_alarms'])}, "
                f"strikes={int(row['strikes_in_assigned_grids'])} "
                f"({row['percent_of_all_strikes']:.1f}% of all strikes)"
            )

    if grid_stats is not None and not grid_stats.empty:
        top_grids = grid_stats.sort_values("strikes", ascending=False).head(10)
        lines.append("")
        lines.append("Top strike grids:")
        for _, row in top_grids.iterrows():
            if int(row["strikes"]) <= 0:
                continue
            lines.append(
                f"{row['grid_id']}: {int(row['strikes'])} strikes, "
                f"assigned risk={row['assigned_risk_label']}, "
                f"assigned overlap={row['assigned_risk_overlap_percent']:.1f}%"
            )

    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# -----------------------------
# MAIN
# -----------------------------
def main():
    csv_path, forecast_path, time_filter, country_filter, zoom_mode, gif_mode, region_mode = parse_args()

    out = Path("output")
    out.mkdir(exist_ok=True)

    print("Reading CSV...")
    df = load_csv(csv_path)

    print("Normalizing data...")
    df = normalize_dataframe(df)

    file_local_date = extract_date_from_filename(csv_path)
    if file_local_date is not None:
        print(f"Keeping only strikes for local date from filename: {file_local_date}")
        df = filter_to_local_day_from_filename(df, csv_path)

    europe_mode = False
    europe_zoom_name = None
    country_filter_for_selection = country_filter

    if country_filter:
        parts = str(country_filter).strip().split()
        if parts and parts[0].lower() == "europe":
            europe_mode = True
            country_filter_for_selection = None
            if len(parts) >= 2:
                europe_zoom_name = " ".join(parts[1:])

    if europe_mode:
        print("Filtering Europe...")
        df = filter_europe(df)
        active_bbox_4326 = EUROPE_BBOX_4326
    else:
        print("Filtering Central Europe...")
        df = filter_ce(df)
        active_bbox_4326 = CENTRAL_EUROPE_BBOX_4326

    if time_filter:
        print(f"Applying time filter: {time_filter} (local time)")
        df = apply_time_filter(df, time_filter)

    print("Loading map and projecting to EPSG:3857...")
    bbox_geom_4326 = box(
        active_bbox_4326["lon_min"],
        active_bbox_4326["lat_min"],
        active_bbox_4326["lon_max"],
        active_bbox_4326["lat_max"],
    )
    europe = load_map(bbox_geom_4326)
    countries_for_join = load_map(bbox_geom_4326)
    provinces = load_provinces(bbox_geom_4326)
    if region_mode == "orp" and normalize_country_name(country_filter_for_selection) == "czechia":
        provinces = load_orp("orp.geojson", bbox_geom_4326)

    gdf = make_gdf(df)

    forecast_gdf = None
    if forecast_path is not None:
        print("Loading forecast GeoJSON...")
        raw_forecast = load_forecast_geojson(forecast_path)
        forecast_gdf = prepare_forecast_gdf(raw_forecast)

    zoom_geom = get_bbox_zoom_geometry(active_bbox_4326)
    country_filter_display = None
    region_focus_geom = None
    region_focus_name = None

    if country_filter_for_selection:
        print(f"Applying country filter: {country_filter_for_selection}")
        country_geom, display_name = find_country_geometry(countries_for_join, country_filter_for_selection)
        if country_geom is None:
            print(f"Country not found in Natural Earth: {country_filter_for_selection}")
            raise SystemExit(1)

        country_filter_display = display_name
        zoom_geom = country_geom

        gdf = filter_to_country(gdf, country_geom)
        if gdf.empty:
            print(f"No strikes found in {display_name} after filtering.")
            raise SystemExit(0)

        countries_for_join = clip_regions_to_country(countries_for_join, country_geom, use_representative_point=True)
        provinces = clip_regions_to_country(provinces, country_geom, use_representative_point=True)
        forecast_gdf = clip_forecast_to_country(forecast_gdf, country_geom)
        region_focus_geom = country_geom
        region_focus_name = display_name
    elif europe_zoom_name:
        print(f"Zooming Europe view to: {europe_zoom_name}")
        country_geom, display_name = find_country_geometry(countries_for_join, europe_zoom_name)
        if country_geom is None:
            print(f"Country not found in Natural Earth: {europe_zoom_name}")
            raise SystemExit(1)
        zoom_geom = country_geom
        country_filter_display = f"{display_name} (zoom)"
        region_focus_geom = country_geom
        region_focus_name = display_name

    if zoom_mode == "risk":
        if forecast_gdf is None or forecast_gdf.empty:
            print("Risk zoom requested, but no valid forecast polygons were loaded.")
            raise SystemExit(1)

        risk_zoom_geom = get_risk_zoom_geometry(forecast_gdf)
        if risk_zoom_geom is not None and not risk_zoom_geom.is_empty:
            zoom_geom = risk_zoom_geom
            print("Zooming to forecast risk area...")
        else:
            print("Risk zoom requested, but forecast polygons have no usable extent.")
            raise SystemExit(1)

    print(f"Strikes in selection: {len(gdf)}")
    if gdf.empty:
        print("No strikes found after filtering.")
        return

    print("Saving filtered strikes to archive...")
    save_filtered_strikes_geojson(gdf)

    region_stats_gdf = gdf
    region_countries_for_join = countries_for_join
    region_provinces_for_join = provinces

    if region_focus_geom is not None:
        region_stats_gdf = filter_to_country(gdf, region_focus_geom)
        region_countries_for_join = clip_regions_to_country(
            countries_for_join,
            region_focus_geom,
            use_representative_point=True
        )
        region_provinces_for_join = clip_regions_to_country(
            provinces,
            region_focus_geom,
            use_representative_point=True
        )

    print("Computing country statistics...")
    country_source_col = get_region_name_column(region_countries_for_join, preferred=["NAME", "ADMIN"])
    country_stats = get_region_stats(region_stats_gdf, region_countries_for_join, country_source_col, "country")

    print("Computing province statistics...")
    if region_mode == "orp":
        province_source_col = get_region_name_column(region_provinces_for_join, preferred=["NAZEV", "NAZEV_ORP", "ORP", "name", "NAME"])
    else:
        province_source_col = get_region_name_column(region_provinces_for_join, preferred=["name", "name_en", "gn_name"])
    province_stats = get_region_stats(region_stats_gdf, region_provinces_for_join, province_source_col, "province")

    print("Computing risk area statistics...")
    risk_stats, total_in_risk, percent_in_risk = get_risk_area_stats(gdf, forecast_gdf)

    print("Computing 100 km grid verification...")
    grid_stats, grid_risk_summary, summary_df = get_grid_verification_stats(
        gdf,
        forecast_gdf=forecast_gdf,
        zoom_geom=zoom_geom,
        cell_km=GRID_CELL_KM,
        min_forecast_fraction=GRID_FORECAST_MIN_FRACTION
    )

    if country_filter_display:
        title_suffix = f" — {country_filter_display}"
    elif zoom_mode == "risk":
        title_suffix = " — Risk Area"
    elif europe_mode:
        title_suffix = " (Europe)"
    else:
        title_suffix = " (Central Europe)"

    print("Creating maps...")
    plot_hour_map(gdf, europe, out, forecast_gdf, zoom_geom=zoom_geom, title_suffix=title_suffix)
    plot_intensity_map(gdf, europe, out, forecast_gdf, zoom_geom=zoom_geom, title_suffix=title_suffix)
    plot_density_map(gdf, europe, out, forecast_gdf, zoom_geom=zoom_geom, title_suffix=title_suffix)
    plot_density_grid_map(
        gdf,
        europe,
        out,
        forecast_gdf,
        zoom_geom=zoom_geom,
        title_suffix=title_suffix,
        cell_km=5,
        sigma_cells=2.0
    )
    plot_absolute_density_grid_map(
        gdf,
        europe,
        out,
        forecast_gdf,
        zoom_geom=zoom_geom,
        title_suffix=title_suffix,
        cell_km=5,
        sigma_cells=1.2,
        unit_mode="km2"
    )
    plot_absolute_density_grid_map(
        gdf,
        europe,
        out,
        forecast_gdf,
        zoom_geom=zoom_geom,
        title_suffix=title_suffix,
        cell_km=5,
        sigma_cells=1.2,
        unit_mode="cell"
    )
    plot_intensity_hexbin_map(gdf, europe, out, forecast_gdf, zoom_geom=zoom_geom, title_suffix=title_suffix)

    plot_region_map(
        region_countries_for_join, country_source_col, country_stats, "country",
        f"Countries Struck by Lightning{title_suffix}",
        "Strikes per country",
        "10_country_strike_map.png",
        out, europe, forecast_gdf, zoom_geom=zoom_geom
    )

    plot_region_map(
        region_provinces_for_join, province_source_col, province_stats, "province",
        f"Provinces / Regions Struck by Lightning{title_suffix}",
        "Strikes per province",
        "12_province_strike_map.png",
        out, europe, forecast_gdf, zoom_geom=zoom_geom
    )

    plot_grid_map(
        grid_stats,
        europe,
        out,
        forecast_gdf=forecast_gdf,
        zoom_geom=zoom_geom,
        title_suffix=title_suffix
    )

    plot_grid_risk_map(
        grid_stats,
        europe,
        out,
        forecast_gdf=forecast_gdf,
        zoom_geom=zoom_geom,
        title_suffix=title_suffix
    )

    if gif_mode is not None:
        print(f"Creating GIF animation ({gif_mode})...")
        create_lightning_gif(
            gdf,
            europe,
            out,
            forecast_gdf=forecast_gdf,
            zoom_geom=zoom_geom,
            title_suffix=title_suffix,
            gif_mode=gif_mode
        )

    print("Creating statistics...")
    plot_hourly_counts(gdf, out)
    plot_hourly_intensity(gdf, out)
    plot_time_detail(gdf, out)
    plot_daily_counts(gdf, out)
    plot_monthly(gdf, out)
    plot_summary_graphics(summary_df, grid_risk_summary, out)

    plot_region_stats(country_stats, "country", f"Lightning Strikes by Country{title_suffix}", "11_country_stats.png", out)
    write_region_stats(country_stats, "country_stats.csv", out)

    plot_region_stats(province_stats, "province", f"Lightning Strikes by Province / Region{title_suffix}", "13_province_stats.png", out)
    write_region_stats(province_stats, "province_stats.csv", out)

    if not risk_stats.empty:
        plot_risk_area_stats(risk_stats, total_in_risk, percent_in_risk, len(gdf), out)
        write_risk_area_stats(risk_stats, out)

    write_grid_stats(grid_stats, out)
    write_grid_risk_summary(grid_risk_summary, out)
    write_summary_csv(summary_df, out)

    print("Writing summary...")
    write_summary(
        gdf,
        out,
        country_stats=country_stats,
        province_stats=province_stats,
        risk_stats=risk_stats,
        grid_stats=grid_stats,
        grid_risk_summary=grid_risk_summary,
        summary_df=summary_df,
        total_in_risk=total_in_risk,
        percent_in_risk=percent_in_risk,
        time_filter=time_filter,
        country_filter_display=country_filter_display
    )

    print(f"Done. Files saved in: {out.resolve()}")

if __name__ == "__main__":
    main()