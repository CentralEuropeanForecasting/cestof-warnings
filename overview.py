#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_ROOT = BASE_DIR / "output" / "archive"
OVERVIEW_ROOT = BASE_DIR / "output" / "overviews"
LOCAL_TIMEZONE = "Europe/Prague"  # Central Europe time for hourly maps

# -----------------------------
# STYLE
# -----------------------------
FIG_BG = "black"
AX_BG = "black"
TEXT_COLOR = "white"
GRID_COLOR = "#666666"
LAND_COLOR = "black"
BORDER_COLOR = "white"

CENTRAL_EUROPE_BBOX_4326 = {
    "lon_min": 2.0,
    "lon_max": 28.5,
    "lat_min": 42.0,
    "lat_max": 56.5,
}

ACTIVE_BOUNDS_3857 = None
MIN_ZOOM_SIZE_M = 50000
MIN_ZOOM_PAD_M = 15000

NATURAL_EARTH_COUNTRIES_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
NATURAL_EARTH_PROVINCES_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"

ORP_GEOJSON_PATH = BASE_DIR / "orp.geojson"

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

RISK_COLORS = {
    "T-storm Risk (<5%)": "#90EE90",
    "General Risk (5%)": "#2E8B57",
    "Slight Risk (15%)": "#FFD700",
    "Enhanced Risk (30%)": "#FF8C00",
    "Moderate Risk (45%)": "#8B0000",
    "Severe Risk (>50%)": "#FF00FF",
    "Non-risk": "#666666",
}

MIN_STORM_DAY_STRIKES_COUNTRY = 10
MIN_STORM_DAY_STRIKES_PROVINCE = 5

# -----------------------------
# COLOR MAPS
# -----------------------------
def get_cmap():
    return ListedColormap([
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
    ])

def get_storm_day_discrete_cmap(max_days):
    max_days = max(int(max_days), 1)
    base = plt.cm.nipy_spectral
    colors = base(np.linspace(0.12, 0.92, max_days))
    cmap = ListedColormap(colors)
    bounds = np.arange(0.5, max_days + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def get_bins(values, n_classes=11):
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

def get_relative_density_cmap_and_norm():
    levels = [
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

    cmap = ListedColormap(colors)
    bounds = levels + [1e9]
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, levels

# -----------------------------
# HELPERS
# -----------------------------
def get_absolute_density_cmap_and_norm(scale_factor=1.0):
    base_levels = [
        1, 2, 3, 4, 5, 7, 9, 11, 13, 15,
        18, 21, 24, 27, 30, 35, 40, 45,
        50, 55, 60, 65, 70, 80, 90, 100
    ]
    levels = [v / float(scale_factor) for v in base_levels]

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

    cmap = ListedColormap(colors)
    bounds = levels + [1e9]
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, levels

def style(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.3)
    for s in ax.spines.values():
        s.set_color(TEXT_COLOR)

def save(fig, path):
    fig.patch.set_facecolor(FIG_BG)
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor=FIG_BG, bbox_inches="tight")
    plt.close(fig)

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

def safe_float(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None

def safe_int(v):
    try:
        if pd.isna(v):
            return None
        return int(v)
    except Exception:
        return None

def parse_any_datetime(value):
    if value is None:
        return pd.NaT
    try:
        s = str(value).strip()
        if not s:
            return pd.NaT
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(dt):
            dt = pd.to_datetime(s, errors="coerce")
        return dt
    except Exception:
        return pd.NaT

def get_points_with_time(gdf):
    if gdf is None or gdf.empty:
        return pd.DataFrame(columns=["hour"])

    time_col = None
    for col in ["strike_time", "time", "datetime", "date", "timestamp", "utc_time"]:
        if col in gdf.columns:
            time_col = col
            break

    if time_col is None:
        return pd.DataFrame(columns=["hour"])

    times = gdf[time_col].apply(parse_any_datetime)
    out = pd.DataFrame({"dt": times})
    out = out.dropna(subset=["dt"]).copy()
    if out.empty:
        return pd.DataFrame(columns=["hour"])

    try:
        if getattr(out["dt"].dt, "tz", None) is not None:
            out["dt"] = out["dt"].dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)
    except Exception:
        pass

    out["hour"] = out["dt"].dt.hour
    return out

def combine_hourly(records):
    frames = []
    for r in records:
        gdf = r.get("points")
        hourly = get_points_with_time(gdf)
        if hourly is not None and not hourly.empty:
            frames.append(hourly[["hour"]])

    base = pd.DataFrame({"hour": list(range(24))})

    if not frames:
        base["count"] = 0
        base["percent"] = 0.0
        return base

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["hour"]).copy()
    if out.empty:
        base["count"] = 0
        base["percent"] = 0.0
        return base

    out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    out = out.dropna(subset=["hour"]).copy()
    out["hour"] = out["hour"].astype(int)
    out = out[(out["hour"] >= 0) & (out["hour"] <= 23)].copy()

    counts = out.groupby("hour").size().reset_index(name="count")
    merged = base.merge(counts, on="hour", how="left")
    merged["count"] = merged["count"].fillna(0).astype(int)

    total = int(merged["count"].sum())
    merged["percent"] = (merged["count"] / total * 100.0) if total > 0 else 0.0
    return merged

def count_thunderstorm_days(records):
    return int(sum(1 for r in records if int(r.get("total", 0)) > 0))

def compute_period_counts(records):
    total_strikes = int(sum(int(r.get("total", 0)) for r in records))
    thunderstorm_days = count_thunderstorm_days(records)
    day_count = int(len(records))
    total_hours = int(day_count * 24)
    thunderstorms_per_hour = (thunderstorm_days / total_hours) if total_hours > 0 else 0.0
    thunderstorm_percentage = (thunderstorm_days / day_count * 100.0) if day_count > 0 else 0.0

    return pd.DataFrame([{
        "days_in_period": day_count,
        "total_strikes_count": total_strikes,
        "thunderstorm_days_count": thunderstorm_days,
        "total_hours_in_period": total_hours,
        "thunderstorms_per_hour": thunderstorms_per_hour,
        "thunderstorm_percentage": thunderstorm_percentage,
    }])

def _week_number_from_key(key):
    m = re.match(r"^(\d{4})-W(\d{2})$", str(key))
    return int(m.group(2)) if m else None

def _month_number_from_key(key):
    m = re.match(r"^(\d{4})-(\d{2})$", str(key))
    return int(m.group(2)) if m else None

def compute_anomaly_summary(name, key, current_total, current_storm_days, groups):
    ref_groups = get_reference_groups(name, key, groups)

    ref_totals = [int(sum(int(r.get("total", 0)) for r in recs)) for _, recs in ref_groups]
    ref_storm_days = [count_thunderstorm_days(recs) for _, recs in ref_groups]

    mean_total = float(np.mean(ref_totals)) if ref_totals else 0.0
    mean_storm_days = float(np.mean(ref_storm_days)) if ref_storm_days else 0.0

    strike_anomaly = float(current_total) - mean_total
    strike_anomaly_percent = ((float(current_total) - mean_total) / mean_total * 100.0) if mean_total != 0 else None

    storm_day_anomaly = float(current_storm_days) - mean_storm_days
    storm_day_anomaly_percent = ((float(current_storm_days) - mean_storm_days) / mean_storm_days * 100.0) if mean_storm_days != 0 else None

    return pd.DataFrame([{
        "period_type": name,
        "period_key": key,
        "baseline_mode": "all_previous_periods_same_type",
        "current_total_strikes": int(current_total),
        "historical_mean_total_strikes": safe_float(mean_total),
        "strike_anomaly": safe_float(strike_anomaly),
        "strike_anomaly_percent": safe_float(strike_anomaly_percent),
        "current_thunderstorm_days": int(current_storm_days),
        "historical_mean_thunderstorm_days": safe_float(mean_storm_days),
        "thunderstorm_day_anomaly": safe_float(storm_day_anomaly),
        "thunderstorm_day_anomaly_percent": safe_float(storm_day_anomaly_percent),
        "reference_periods_used": int(len(ref_totals)),
    }])



def get_reference_groups(name, key, groups):
    ordered = sorted(groups.items(), key=lambda x: str(x[0]))
    refs = []

    for gk, recs in ordered:
        if str(gk) >= str(key):
            break
        refs.append((gk, recs))

    return refs


def compute_region_anomaly_table(current_df, ref_dfs, region_col):
    cols = [region_col, "strikes"]
    cur = current_df[cols].copy() if not current_df.empty and region_col in current_df.columns and "strikes" in current_df.columns else pd.DataFrame(columns=cols)
    if cur.empty:
        cur = pd.DataFrame(columns=cols)
    cur = cur.rename(columns={"strikes": "current_strikes"})
    if not cur.empty:
        cur["current_strikes"] = pd.to_numeric(cur["current_strikes"], errors="coerce").fillna(0).astype(float)

    ref_frames = []
    for ref in ref_dfs:
        if ref.empty or region_col not in ref.columns or "strikes" not in ref.columns:
            continue
        tmp = ref[[region_col, "strikes"]].copy()
        tmp["strikes"] = pd.to_numeric(tmp["strikes"], errors="coerce").fillna(0).astype(float)
        ref_frames.append(tmp)

    if ref_frames:
        ref_all = pd.concat(ref_frames, ignore_index=True)
        ref_mean = ref_all.groupby(region_col)["strikes"].mean().reset_index()
        ref_mean = ref_mean.rename(columns={"strikes": "historical_mean_strikes"})
    else:
        ref_mean = pd.DataFrame(columns=[region_col, "historical_mean_strikes"])

    out = cur.merge(ref_mean, on=region_col, how="outer")
    if out.empty:
        return pd.DataFrame(columns=[
            region_col,
            "current_strikes",
            "historical_mean_strikes",
            "strike_anomaly",
            "strike_anomaly_percent",
        ])

    out[region_col] = out[region_col].astype(str)
    out["current_strikes"] = pd.to_numeric(out.get("current_strikes"), errors="coerce").fillna(0.0)
    out["historical_mean_strikes"] = pd.to_numeric(out.get("historical_mean_strikes"), errors="coerce").fillna(0.0)
    out["strike_anomaly"] = out["current_strikes"] - out["historical_mean_strikes"]
    out["strike_anomaly_percent"] = np.where(
        out["historical_mean_strikes"] != 0,
        (out["strike_anomaly"] / out["historical_mean_strikes"]) * 100.0,
        np.where(out["current_strikes"] > 0, 100.0, 0.0),
    )
    out = out.sort_values("strike_anomaly", ascending=False).reset_index(drop=True)
    return out


def add_area_density_columns(anomaly_df, base_regions, region_key, out_region_col):
    if anomaly_df.empty or region_key is None or out_region_col not in anomaly_df.columns:
        return anomaly_df

    areas = base_regions[[region_key, "geometry"]].copy()
    areas["area_km2"] = areas.geometry.area / 1_000_000.0
    areas = areas[[region_key, "area_km2"]].rename(columns={region_key: out_region_col})

    out = anomaly_df.merge(areas, on=out_region_col, how="left")
    out["current_density_km2"] = np.where(out["area_km2"] > 0, out["current_strikes"] / out["area_km2"], np.nan)
    out["historical_mean_density_km2"] = np.where(
        out["area_km2"] > 0,
        out["historical_mean_strikes"] / out["area_km2"],
        np.nan,
    )
    out["density_anomaly_km2"] = out["current_density_km2"] - out["historical_mean_density_km2"]
    out["density_anomaly_percent"] = np.where(
        out["historical_mean_density_km2"].notna() & (out["historical_mean_density_km2"] != 0),
        (out["density_anomaly_km2"] / out["historical_mean_density_km2"]) * 100.0,
        np.nan,
    )
    return out


def get_diverging_bins(values, n_classes=11):
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return np.array([-1.0, 0.0, 1.0])

    max_abs = float(np.max(np.abs(vals)))
    if max_abs <= 0:
        max_abs = 1.0
    return np.linspace(-max_abs, max_abs, n_classes + 1)


def plot_anomaly_map(base_regions, region_key, df, df_col, value_col, path, title, background_countries, colorbar_label):
    if df.empty or region_key is None or df_col not in df.columns or value_col not in df.columns:
        return

    temp_base = base_regions[[region_key, "geometry"]].copy()
    g = temp_base.merge(df, left_on=region_key, right_on=df_col, how="inner")
    g = g[pd.to_numeric(g[value_col], errors="coerce").notna()].copy()

    if g.empty:
        return

    bins = get_diverging_bins(g[value_col], n_classes=10)
    cmap = plt.cm.RdBu_r
    norm = BoundaryNorm(bins, cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(FIG_BG)

    background_countries.plot(ax=ax, color=LAND_COLOR, edgecolor=BORDER_COLOR, linewidth=0.5, zorder=1)
    g.plot(
        ax=ax,
        column=value_col,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.6,
        zorder=10
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cb = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
    cb.ax.tick_params(colors=TEXT_COLOR)
    cb.set_label(colorbar_label, color=TEXT_COLOR)
    for spine in cb.ax.spines.values():
        spine.set_color(TEXT_COLOR)

    tick_positions = []
    tick_labels = []
    for i in range(len(bins) - 1):
        tick_positions.append((bins[i] + bins[i + 1]) / 2)
        tick_labels.append(f"{bins[i]:.2g} to {bins[i + 1]:.2g}")
    cb.set_ticks(tick_positions)
    cb.set_ticklabels(tick_labels)

    minx, miny, maxx, maxy = get_bounds_3857()
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    style(ax)
    ax.set_title(title, color=TEXT_COLOR, fontsize=14)

    for _, row in g.iterrows():
        rp = row.geometry.representative_point()
        try:
            label = f"{float(row[value_col]):.2f}" if "density" in value_col else f"{int(round(float(row[value_col])))}"
        except Exception:
            label = str(row[value_col])
        ax.text(
            rp.x,
            rp.y,
            label,
            color="white",
            fontsize=7,
            fontweight="bold",
            fontfamily="Courier New",
            ha="center",
            va="center",
            zorder=30
        )

    add_branding(ax)
    save(fig, path)


def parse_args():
    country_filter = None
    if len(sys.argv) >= 2:
        country_filter = " ".join(sys.argv[1:]).strip() or None
    return country_filter

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
            if normalize_country_name(v) == requested:
                return row.geometry, values[0]

    return None, None

def filter_points_to_country(gdf, country_geom):
    if gdf is None or gdf.empty or country_geom is None:
        return gdf
    return gdf[gdf.geometry.intersects(country_geom)].copy().reset_index(drop=True)

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

def load_orp():
    if not ORP_GEOJSON_PATH.exists():
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

    gdf = gpd.read_file(ORP_GEOJSON_PATH).copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    return gdf.to_crs(epsg=3857)

def get_orp_name_column(gdf):
    for col in ["NAZEV", "NAZEV_ORP", "ORP", "name", "NAME"]:
        if col in gdf.columns:
            return col
    return None

def compute_region_table_from_points(points_gdf, regions_gdf, region_key, out_col):
    if points_gdf is None or points_gdf.empty or regions_gdf is None or regions_gdf.empty or region_key is None:
        return pd.DataFrame(columns=[out_col, "strikes"])

    regions = regions_gdf[[region_key, "geometry"]].copy().rename(columns={region_key: out_col})
    joined = gpd.sjoin(points_gdf, regions, how="left", predicate="intersects")
    joined = joined.dropna(subset=[out_col]).copy()

    if joined.empty:
        return pd.DataFrame(columns=[out_col, "strikes"])

    out = joined.groupby(out_col).size().reset_index(name="strikes")
    out["strikes"] = out["strikes"].astype(int)
    out = out[out["strikes"] > 0].copy()
    return out.sort_values("strikes", ascending=False).reset_index(drop=True)

# -----------------------------
# FIND DAILY FOLDERS
# -----------------------------
def find_days(root):
    if not root.exists():
        return []
    return sorted([
        p for p in root.iterdir()
        if p.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", p.name)
    ])

# -----------------------------
# READ DAILY DATA
# -----------------------------
def try_read_points(path):
    geojson_candidates = [
        path / "filtered_strikes.geojson",
        path / "strikes.geojson",
        path / "lightning.geojson",
        path / "points.geojson",
    ]

    for fp in geojson_candidates:
        if fp.exists():
            try:
                gdf = gpd.read_file(fp)
                if gdf.empty:
                    continue
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=4326)
                return gdf.to_crs(epsg=3857)
            except Exception:
                continue

    csv_candidates = [
        path / f"{path.name}.csv",
        path / "strikes.csv",
        path / "lightning.csv",
        path / "filtered_strikes.csv",
    ]

    for fp in csv_candidates:
        if fp.exists():
            try:
                df = pd.read_csv(
                    fp,
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
                df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
                df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
                df = df.dropna(subset=["latitude", "longitude"]).copy()
                if df.empty:
                    continue
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                    crs="EPSG:4326"
                )
                return gdf.to_crs(epsg=3857)
            except Exception:
                continue

    root_csv = BASE_DIR / f"{path.name}.csv"
    if root_csv.exists():
        try:
            df = pd.read_csv(
                root_csv,
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
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
            df = df.dropna(subset=["latitude", "longitude"]).copy()
            if not df.empty:
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                    crs="EPSG:4326"
                )
                return gdf.to_crs(epsg=3857)
        except Exception:
            pass

    for fp in sorted(path.glob("*.geojson")):
        try:
            gdf = gpd.read_file(fp)
            if gdf.empty:
                continue
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            gdf = gdf.to_crs(epsg=3857)
            geom_types = set(gdf.geometry.geom_type.dropna().unique())
            if geom_types and geom_types.issubset({"Point", "MultiPoint"}):
                return gdf
        except Exception:
            continue

    return gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

def read_day(path):
    summary = path / "summary.txt"
    summary_csv = path / "summary.csv"
    country = path / "country_stats.csv"
    province = path / "province_stats.csv"
    risk = path / "risk_area_stats.csv"
    orp = path / "orp_stats.csv"

    total = 0
    first = None
    last = None

    hits = 0
    misses = 0
    false_alarms = 0
    correct_nulls = 0
    verification_total = 0
    hit_percent = 0.0
    miss_percent = 0.0
    false_alarm_percent = 0.0
    hits_majority = False
    strikes_in_forecast_grids = 0
    total_strikes = 0
    percent_strikes_in_forecast_grids = 0.0
    grids_forecast = 0
    grids_with_strikes = 0

    if summary.exists():
        txt = summary.read_text(encoding="utf-8", errors="ignore")

        m = re.search(r"Total.*?:\s*([\d,]+)", txt)
        if m:
            total = int(m.group(1).replace(",", ""))

        m = re.search(r"First strike time:\s*(.+)", txt)
        if m:
            first = m.group(1).strip()

        m = re.search(r"Last strike time\s*:\s*(.+)", txt)
        if m:
            last = m.group(1).strip()

    if summary_csv.exists():
        try:
            sdf = pd.read_csv(summary_csv)
            if not sdf.empty:
                row = sdf.iloc[0]
                hits = int(row.get("hits", 0))
                misses = int(row.get("misses", 0))
                false_alarms = int(row.get("false_alarms", 0))
                correct_nulls = int(row.get("correct_nulls", 0))
                verification_total = int(row.get("verification_total", 0))
                hit_percent = float(row.get("hit_percent", 0.0))
                miss_percent = float(row.get("miss_percent", 0.0))
                false_alarm_percent = float(row.get("false_alarm_percent", 0.0))
                hits_majority = bool(row.get("hits_majority", False))
                strikes_in_forecast_grids = int(row.get("strikes_in_forecast_grids", 0))
                total_strikes = int(row.get("total_strikes", 0))
                percent_strikes_in_forecast_grids = float(row.get("percent_strikes_in_forecast_grids", 0.0))
                grids_forecast = int(row.get("grids_forecast", 0))
                grids_with_strikes = int(row.get("grids_with_strikes", 0))
        except Exception:
            pass

    return {
        "date": path.name,
        "total": total,
        "first": first,
        "last": last,
        "country": pd.read_csv(country) if country.exists() else pd.DataFrame(),
        "province": pd.read_csv(province) if province.exists() else pd.DataFrame(),
        "risk": pd.read_csv(risk) if risk.exists() else pd.DataFrame(),
        "orp": pd.read_csv(orp) if orp.exists() else pd.DataFrame(),
        "points": try_read_points(path),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_nulls": correct_nulls,
        "verification_total": verification_total,
        "hit_percent": hit_percent,
        "miss_percent": miss_percent,
        "false_alarm_percent": false_alarm_percent,
        "hits_majority": hits_majority,
        "strikes_in_forecast_grids": strikes_in_forecast_grids,
        "total_strikes": total_strikes,
        "percent_strikes_in_forecast_grids": percent_strikes_in_forecast_grids,
        "grids_forecast": grids_forecast,
        "grids_with_strikes": grids_with_strikes,
    }

# -----------------------------
# COMBINE TABLES
# -----------------------------
def combine(records, key, col):
    frames = []
    for r in records:
        df = r[key]
        if not df.empty and col in df.columns and "strikes" in df.columns:
            tmp = df[[col, "strikes"]].copy()
            tmp["strikes"] = pd.to_numeric(tmp["strikes"], errors="coerce").fillna(0)
            frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=[col, "strikes"])

    out = pd.concat(frames, ignore_index=True)
    out = out.groupby(col)["strikes"].sum().reset_index()
    out["strikes"] = out["strikes"].astype(int)
    out = out[out["strikes"] > 0].copy()
    return out.sort_values("strikes", ascending=False).reset_index(drop=True)

def combine_risk(records):
    frames = []
    for r in records:
        df = r["risk"]
        if df.empty:
            continue

        label_col = None
        for c in ["label", "risk_key"]:
            if c in df.columns:
                label_col = c
                break

        if label_col is None or "strikes" not in df.columns:
            continue

        tmp = df[[label_col, "strikes"]].copy()
        tmp = tmp.rename(columns={label_col: "label"})
        tmp["strikes"] = pd.to_numeric(tmp["strikes"], errors="coerce").fillna(0)
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["label", "strikes"])

    out = pd.concat(frames, ignore_index=True)
    out = out.groupby("label")["strikes"].sum().reset_index()
    out["strikes"] = out["strikes"].astype(int)
    out = out[out["strikes"] > 0].copy()
    return out.sort_values("strikes", ascending=False).reset_index(drop=True)

def combine_storm_days(records, key, col, min_strikes=10):
    frames = []

    for r in records:
        df = r[key]
        if df.empty or col not in df.columns or "strikes" not in df.columns:
            continue

        tmp = df[[col, "strikes"]].copy()
        tmp["strikes"] = pd.to_numeric(tmp["strikes"], errors="coerce").fillna(0)
        tmp = tmp[tmp["strikes"] >= min_strikes].copy()

        if tmp.empty:
            continue

        tmp["storm_days"] = 1
        frames.append(tmp[[col, "storm_days"]])

    if not frames:
        return pd.DataFrame(columns=[col, "storm_days"])

    out = pd.concat(frames, ignore_index=True)
    out = out.groupby(col)["storm_days"].sum().reset_index()
    out["storm_days"] = out["storm_days"].astype(int)
    out = out[out["storm_days"] > 0].copy()
    return out.sort_values("storm_days", ascending=False).reset_index(drop=True)

def combine_verification(records):
    hits = sum(int(r.get("hits", 0)) for r in records)
    misses = sum(int(r.get("misses", 0)) for r in records)
    false_alarms = sum(int(r.get("false_alarms", 0)) for r in records)
    correct_nulls = sum(int(r.get("correct_nulls", 0)) for r in records)

    verification_total = hits + misses + false_alarms

    hit_percent = (hits / verification_total * 100.0) if verification_total > 0 else 0.0
    miss_percent = (misses / verification_total * 100.0) if verification_total > 0 else 0.0
    false_alarm_percent = (false_alarms / verification_total * 100.0) if verification_total > 0 else 0.0
    hits_majority = hit_percent >= 50.0

    strikes_in_forecast_grids = sum(int(r.get("strikes_in_forecast_grids", 0)) for r in records)
    total_strikes = sum(int(r.get("total_strikes", 0)) for r in records)
    percent_strikes_in_forecast_grids = (
        strikes_in_forecast_grids / total_strikes * 100.0
        if total_strikes > 0 else 0.0
    )

    grids_forecast = sum(int(r.get("grids_forecast", 0)) for r in records)
    grids_with_strikes = sum(int(r.get("grids_with_strikes", 0)) for r in records)

    return pd.DataFrame([{
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_nulls": correct_nulls,
        "verification_total": verification_total,
        "hit_percent": hit_percent,
        "miss_percent": miss_percent,
        "false_alarm_percent": false_alarm_percent,
        "hits_majority": hits_majority,
        "strikes_in_forecast_grids": strikes_in_forecast_grids,
        "total_strikes": total_strikes,
        "percent_strikes_in_forecast_grids": percent_strikes_in_forecast_grids,
        "grids_forecast": grids_forecast,
        "grids_with_strikes": grids_with_strikes,
    }])

def combine_points(records):
    frames = []
    for r in records:
        gdf = r.get("points")
        if gdf is not None and not gdf.empty:
            frames.append(gdf)

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:3857")

# -----------------------------
# LOAD MAPS
# -----------------------------
def load_countries():
    world = gpd.read_file(NATURAL_EARTH_COUNTRIES_URL)
    if "CONTINENT" in world.columns:
        europe = world[world["CONTINENT"] == "Europe"].copy()
    elif "continent" in world.columns:
        europe = world[world["continent"] == "Europe"].copy()
    else:
        europe = world.copy()
    return europe.to_crs(epsg=3857)

def load_provinces():
    provinces = gpd.read_file(NATURAL_EARTH_PROVINCES_URL).copy()
    if provinces.crs is None:
        provinces = provinces.set_crs(epsg=4326)

    bbox = CENTRAL_EUROPE_BBOX_4326
    clip_geom = box(bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"])

    provinces = provinces.to_crs(epsg=4326)
    provinces = provinces[provinces.geometry.intersects(clip_geom)].copy()
    return provinces.to_crs(epsg=3857)

def get_country_name_column(gdf):
    for col in ["NAME", "ADMIN", "name", "admin"]:
        if col in gdf.columns:
            return col
    return None

def get_province_name_column(gdf):
    for col in ["name", "name_en", "gn_name", "woe_name", "NAME"]:
        if col in gdf.columns:
            return col
    return None

def get_bounds_3857():
    global ACTIVE_BOUNDS_3857
    if ACTIVE_BOUNDS_3857 is not None:
        return ACTIVE_BOUNDS_3857

    bbox = CENTRAL_EUROPE_BBOX_4326
    g = gpd.GeoSeries(
        [box(bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"])],
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    return g.total_bounds

def set_active_bounds_from_geometry(geom, min_size_m=MIN_ZOOM_SIZE_M, min_pad_m=MIN_ZOOM_PAD_M):
    global ACTIVE_BOUNDS_3857
    if geom is None:
        ACTIVE_BOUNDS_3857 = None
        return

    minx, miny, maxx, maxy = geom.bounds
    dx = max(maxx - minx, float(min_size_m))
    dy = max(maxy - miny, float(min_size_m))
    padx = max(dx * 0.08, float(min_pad_m))
    pady = max(dy * 0.08, float(min_pad_m))
    ACTIVE_BOUNDS_3857 = (minx - padx, miny - pady, maxx + padx, maxy + pady)

# -----------------------------
# DENSITY
# -----------------------------
def compute_density_grid(gdf, cell_km=5, sigma_cells=2.0):
    if gdf is None or gdf.empty:
        return None, None, None

    minx, miny, maxx, maxy = get_bounds_3857()

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

def plot_density_grid_map(gdf, europe, path, title, cell_km=5, sigma_cells=2.0):
    density_counts, xedges, yedges = compute_density_grid(
        gdf,
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

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(FIG_BG)

    europe.plot(ax=ax, color=LAND_COLOR, edgecolor=BORDER_COLOR, linewidth=0.5, zorder=1)

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

    cb = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.02)
    cb.ax.tick_params(colors=TEXT_COLOR)
    cb.set_label("Relative density (% of overview max)", color=TEXT_COLOR)
    for spine in cb.ax.spines.values():
        spine.set_color(TEXT_COLOR)

    shown_ticks = [v for v in rel_levels if v <= 100]
    cb.set_ticks(shown_ticks)
    cb.set_ticklabels([str(v) for v in shown_ticks])

    minx, miny, maxx, maxy = get_bounds_3857()
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    style(ax)
    ax.set_title(title, color=TEXT_COLOR, fontsize=14)
    add_branding(ax)
    save(fig, path)

def plot_absolute_density_grid_map(gdf, europe, path, title, cell_km=5, sigma_cells=1.2, unit_mode="km2"):
    density_counts, xedges, yedges = compute_density_grid(
        gdf,
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
        scale_factor = cell_area_km2
        cbar_label = "Lightning density (strikes / km²)"
    else:
        density_vals = density_counts
        scale_factor = 1.0
        cbar_label = f"Lightning density (strikes / {int(cell_area_km2)} km²)"

    max_val = float(np.nanmax(density_vals)) if density_vals.size else 0.0
    if max_val <= 0:
        return

    cmap, norm, levels = get_absolute_density_cmap_and_norm(scale_factor=scale_factor)
    density_masked = np.ma.masked_where(density_vals < levels[0], density_vals)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(FIG_BG)

    europe.plot(ax=ax, color=LAND_COLOR, edgecolor=BORDER_COLOR, linewidth=0.5, zorder=1)

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

    cb = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.02)
    cb.ax.tick_params(colors=TEXT_COLOR)
    cb.set_label(cbar_label, color=TEXT_COLOR)
    for spine in cb.ax.spines.values():
        spine.set_color(TEXT_COLOR)

    shown_ticks = [v for v in levels if v <= max_val]
    if not shown_ticks:
        shown_ticks = [levels[0]]
    cb.set_ticks(shown_ticks)
    cb.set_ticklabels([f"{v:g}" for v in shown_ticks])

    minx, miny, maxx, maxy = get_bounds_3857()
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    style(ax)
    ax.set_title(title, color=TEXT_COLOR, fontsize=14)
    add_branding(ax)
    save(fig, path)

# -----------------------------
# PLOTS
# -----------------------------
def plot_daily(df, path, title):
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    ax.plot(df["date"], df["total"], marker="o", linewidth=2)

    ax.set_title(title, color=TEXT_COLOR, fontsize=15, fontweight="bold")
    ax.set_xlabel("Date", color=TEXT_COLOR)
    ax.set_ylabel("Strike count", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, rotation=45)
    ax.grid(True, color=GRID_COLOR, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

    ymax = df["total"].max() if len(df) else 0
    for x, y in zip(df["date"], df["total"]):
        ax.text(
            x, y + max(0.5, ymax * 0.01),
            str(int(y)),
            color="white",
            fontsize=8,
            ha="center",
            va="bottom"
        )

    save(fig, path)

def plot_bar(df, col, path, title):
    if df.empty:
        return

    plot_df = df.head(25).sort_values("strikes", ascending=True).reset_index(drop=True)

    fig_height = max(6.5, 0.45 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    cmap = get_cmap()
    if len(plot_df) == 1:
        colors = [cmap(0.7)]
    else:
        colors = [cmap(v) for v in np.linspace(0, 1, len(plot_df))]

    bars = ax.barh(
        plot_df[col],
        plot_df["strikes"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.72
    )

    ax.set_title(title, color=TEXT_COLOR, fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Strike count", color=TEXT_COLOR, fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.grid(True, axis="x", color=GRID_COLOR, alpha=0.28)
    ax.grid(False, axis="y")

    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

    max_val = plot_df["strikes"].max()
    ax.set_xlim(0, max_val * 1.16 if max_val > 0 else 1)

    for bar, val in zip(bars, plot_df["strikes"]):
        ax.text(
            bar.get_width() + max(0.6, max_val * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{int(val):,}",
            va="center",
            ha="left",
            color=TEXT_COLOR,
            fontsize=10,
            fontweight="bold"
        )

    save(fig, path)

def plot_storm_days_bar(df, col, path, title):
    if df.empty:
        return

    plot_df = df.head(25).sort_values("storm_days", ascending=True).reset_index(drop=True)

    fig_height = max(6.5, 0.45 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    cmap, norm = get_storm_day_discrete_cmap(max(plot_df["storm_days"].max(), 1))
    colors = [cmap(norm(v)) for v in plot_df["storm_days"]]

    bars = ax.barh(
        plot_df[col],
        plot_df["storm_days"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.72
    )

    ax.set_title(title, color=TEXT_COLOR, fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Storm days", color=TEXT_COLOR, fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.grid(True, axis="x", color=GRID_COLOR, alpha=0.28)
    ax.grid(False, axis="y")

    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

    max_val = max(plot_df["storm_days"].max(), 1)
    ax.set_xlim(0, max_val * 1.16 if max_val > 0 else 1)

    for bar, val in zip(bars, plot_df["storm_days"]):
        ax.text(
            bar.get_width() + max(0.2, max_val * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{int(val)}",
            va="center",
            ha="left",
            color=TEXT_COLOR,
            fontsize=10,
            fontweight="bold"
        )

    save(fig, path)

def plot_risk_bar(df, path, title):
    if df.empty:
        return

    temp = df.copy()
    colors = [RISK_COLORS.get(str(label), "#666666") for label in temp["label"]]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    bars = ax.bar(temp["label"], temp["strikes"], color=colors, edgecolor="white", linewidth=0.8)

    total = temp["strikes"].sum()

    ax.set_title(title, color=TEXT_COLOR, fontsize=15, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Strike count", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)

    ymax = max(temp["strikes"].max(), 1)
    for bar, val in zip(bars, temp["strikes"]):
        pct = (val / total * 100.0) if total > 0 else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(0.5, ymax * 0.01),
            f"{int(val)}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontsize=9,
            fontweight="bold"
        )

    save(fig, path)

def plot_map(base_regions, region_key, df, df_col, value_col, path, title, background_countries, colorbar_label, storm_days_mode=False):
    if df.empty or region_key is None:
        return

    temp_base = base_regions[[region_key, "geometry"]].copy()
    g = temp_base.merge(df, left_on=region_key, right_on=df_col, how="inner")
    g = g[g[value_col] > 0].copy()

    if g.empty:
        print(f"Map join failed for {title}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(FIG_BG)

    background_countries.plot(ax=ax, color=LAND_COLOR, edgecolor=BORDER_COLOR, linewidth=0.5, zorder=1)

    if storm_days_mode:
        max_days = int(max(g[value_col].max(), 1))
        cmap, norm = get_storm_day_discrete_cmap(max_days)

        g.plot(
            ax=ax,
            column=value_col,
            cmap=cmap,
            norm=norm,
            edgecolor="white",
            linewidth=0.6,
            zorder=10
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []

        cb = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
        cb.ax.tick_params(colors=TEXT_COLOR)
        cb.set_label(colorbar_label, color=TEXT_COLOR)
        for spine in cb.ax.spines.values():
            spine.set_color(TEXT_COLOR)

        if max_days <= 25:
            ticks = list(range(1, max_days + 1))
        elif max_days <= 60:
            ticks = list(range(1, max_days + 1, 2))
        elif max_days <= 120:
            ticks = list(range(1, max_days + 1, 5))
        elif max_days <= 240:
            ticks = list(range(1, max_days + 1, 10))
        else:
            ticks = list(range(1, max_days + 1, 20))

        if max_days not in ticks:
            ticks.append(max_days)

        cb.set_ticks(ticks)
        cb.set_ticklabels([str(t) for t in ticks])
    else:
        cmap = get_cmap()
        bins = get_bins(g[value_col], n_classes=cmap.N)
        norm = BoundaryNorm(bins, cmap.N, clip=True)

        g.plot(
            ax=ax,
            column=value_col,
            cmap=cmap,
            norm=norm,
            edgecolor="white",
            linewidth=0.6,
            zorder=10
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

    minx, miny, maxx, maxy = get_bounds_3857()
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    style(ax)
    ax.set_title(title, color=TEXT_COLOR, fontsize=14)

    for _, row in g.iterrows():
        rp = row.geometry.representative_point()
        ax.text(
            rp.x,
            rp.y,
            f"{int(row[value_col])}",
            color="white",
            fontsize=8,
            fontweight="bold",
            fontfamily="Courier New",
            ha="center",
            va="center",
            zorder=30
        )

    add_branding(ax)
    save(fig, path)


def get_hour_map_discrete_cmap_and_norm():
    """24-bin discrete cmap for hour maps, matching the uploaded hour map style."""
    colors = plt.cm.turbo(np.linspace(0.02, 0.98, 24))
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, 24.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def plot_hour_map_discrete(points_gdf, europe, path, title):
    """Plot lightning points coloured by local hour and save hour_map_discrete.png."""
    if points_gdf is None or points_gdf.empty:
        return

    gdf = points_gdf.copy()

    time_col = None
    for col in ["strike_time", "time", "datetime", "date", "timestamp", "utc_time"]:
        if col in gdf.columns:
            time_col = col
            break

    if time_col is None:
        print(f"No time column found for {path}")
        return

    times = gdf[time_col].apply(parse_any_datetime)
    valid = times.notna()
    gdf = gdf.loc[valid].copy()
    times = times.loc[valid]

    if gdf.empty:
        return

    try:
        if getattr(times.dt, "tz", None) is not None:
            times = times.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)
    except Exception:
        pass

    gdf["hour_local"] = times.dt.hour.astype(int)
    gdf = gdf[(gdf["hour_local"] >= 0) & (gdf["hour_local"] <= 23)].copy()

    if gdf.empty:
        return

    cmap, norm = get_hour_map_discrete_cmap_and_norm()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    europe.plot(ax=ax, color=LAND_COLOR, edgecolor=BORDER_COLOR, linewidth=0.45, zorder=1)

    ax.scatter(
        gdf.geometry.x,
        gdf.geometry.y,
        c=gdf["hour_local"],
        cmap=cmap,
        norm=norm,
        s=18,
        marker="+",
        linewidths=1.0,
        alpha=0.95,
        zorder=20,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cb = fig.colorbar(sm, ax=ax, shrink=0.86, pad=0.025)
    cb.ax.tick_params(colors=TEXT_COLOR)
    cb.set_label("Hour (CEST/CET)", color=TEXT_COLOR)
    cb.set_ticks(list(range(24)))
    cb.set_ticklabels([str(i) for i in range(24)])
    for spine in cb.ax.spines.values():
        spine.set_color(TEXT_COLOR)

    minx, miny, maxx, maxy = get_bounds_3857()
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    style(ax)
    ax.set_title(title, color=TEXT_COLOR, fontsize=14)
    add_branding(ax)
    save(fig, path)

def plot_verification_graphics(summary_df, path, title):
    if summary_df.empty:
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

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.patch.set_facecolor(FIG_BG)

    ax = axes[0]
    ax.set_facecolor(AX_BG)

    labels = ["Hits", "False alarms", "Misses", "Correct nulls"]
    values = [hits, false_alarms, misses, correct_nulls]
    colors = ["#2E8B57", "#FF8C00", "#8B0000", "#555555"]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(f"{title} — Verification Overview", color=TEXT_COLOR, fontsize=15, fontweight="bold")
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

    ax.set_title(f"{title} — Forecast Success Percentages", color=TEXT_COLOR, fontsize=15, fontweight="bold")
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

    verdict = "Hits are the majority ✅" if hit_pct >= 50 else "Hits are not the majority ❌"
    verdict_color = "#90EE90" if hit_pct >= 50 else "#FF6666"

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

    save(fig, path)

# -----------------------------
# GROUPING
# -----------------------------
def group_days(records):
    g = {}
    for r in records:
        key = r["date"]
        g.setdefault(key, []).append(r)
    return g

def group_week(records):
    g = {}
    for r in records:
        d = datetime.strptime(r["date"], "%Y-%m-%d")
        key = f"{d.year}-W{d.isocalendar().week:02d}"
        g.setdefault(key, []).append(r)
    return g

def group_month(records):
    g = {}
    for r in records:
        key = r["date"][:7]
        g.setdefault(key, []).append(r)
    return g

def group_year(records):
    g = {}
    for r in records:
        key = r["date"][:4]
        g.setdefault(key, []).append(r)
    return g

# -----------------------------
# BUILD
# -----------------------------
def build(name, groups, countries, provinces, orps=None, overview_root=OVERVIEW_ROOT):
    country_key = get_country_name_column(countries)
    province_key = get_province_name_column(provinces)
    orp_key = get_orp_name_column(orps) if orps is not None and not orps.empty else None

    for key, recs in groups.items():
        out = overview_root / name / key
        out.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([{
            "date": r["date"],
            "total": r["total"],
            "first": r["first"],
            "last": r["last"],
        } for r in recs]).sort_values("date").reset_index(drop=True)

        c = combine(recs, "country", "country")
        p = combine(recs, "province", "province")
        r = combine_risk(recs)
        o = combine(recs, "orp", "orp") if orp_key is not None else pd.DataFrame(columns=["orp", "strikes"])

        c_storm = combine_storm_days(recs, "country", "country", MIN_STORM_DAY_STRIKES_COUNTRY)
        p_storm = combine_storm_days(recs, "province", "province", MIN_STORM_DAY_STRIKES_PROVINCE)
        o_storm = combine_storm_days(recs, "orp", "orp", MIN_STORM_DAY_STRIKES_PROVINCE) if orp_key is not None else pd.DataFrame(columns=["orp", "storm_days"])

        verification = combine_verification(recs)
        points = combine_points(recs)
        hourly = combine_hourly(recs)
        period_counts = compute_period_counts(recs)
        anomaly = compute_anomaly_summary(
            name,
            key,
            int(df["total"].sum()) if not df.empty else 0,
            int(period_counts.iloc[0]["thunderstorm_days_count"]) if not period_counts.empty else 0,
            groups
        )

        ref_groups = get_reference_groups(name, key, groups)
        ref_country_tables = [combine(ref_recs, "country", "country") for _, ref_recs in ref_groups]
        ref_province_tables = [combine(ref_recs, "province", "province") for _, ref_recs in ref_groups]
        ref_orp_tables = [combine(ref_recs, "orp", "orp") for _, ref_recs in ref_groups] if orp_key is not None else []

        country_anomaly = compute_region_anomaly_table(c, ref_country_tables, "country")
        province_anomaly = compute_region_anomaly_table(p, ref_province_tables, "province")
        orp_anomaly = compute_region_anomaly_table(o, ref_orp_tables, "orp") if orp_key is not None else pd.DataFrame(columns=["orp", "current_strikes", "historical_mean_strikes", "strike_anomaly", "strike_anomaly_percent"])

        country_anomaly = add_area_density_columns(country_anomaly, countries, country_key, "country")
        province_anomaly = add_area_density_columns(province_anomaly, provinces, province_key, "province")
        if orp_key is not None:
            orp_anomaly = add_area_density_columns(orp_anomaly, orps, orp_key, "orp")

        df.to_csv(out / "overview.csv", index=False)
        c.to_csv(out / "country_totals.csv", index=False)
        p.to_csv(out / "province_totals.csv", index=False)
        o.to_csv(out / "orp_totals.csv", index=False)
        r.to_csv(out / "risk_totals.csv", index=False)
        c_storm.to_csv(out / "country_storm_days.csv", index=False)
        p_storm.to_csv(out / "province_storm_days.csv", index=False)
        o_storm.to_csv(out / "orp_storm_days.csv", index=False)
        verification.to_csv(out / "verification_summary.csv", index=False)
        hourly.to_csv(out / "thunderstorms_per_hour.csv", index=False)
        period_counts.to_csv(out / "thunderstorm_count_summary.csv", index=False)
        anomaly.to_csv(out / "anomaly_summary.csv", index=False)
        country_anomaly.to_csv(out / "country_anomaly.csv", index=False)
        province_anomaly.to_csv(out / "province_anomaly.csv", index=False)
        orp_anomaly.to_csv(out / "orp_anomaly.csv", index=False)

        plot_daily(df, out / "daily.png", f"{name.title()} Overview — {key}")
        plot_bar(c, "country", out / "countries.png", f"Countries — {key}")
        plot_bar(p, "province", out / "provinces.png", f"Provinces — {key}")
        if orp_key is not None:
            plot_bar(o, "orp", out / "orps.png", f"ORP — {key}")
        plot_risk_bar(r, out / "risk.png", f"Risk Areas — {key}")

        plot_storm_days_bar(
            c_storm,
            "country",
            out / "country_storm_days.png",
            f"Country Storm Days — {key} (min {MIN_STORM_DAY_STRIKES_COUNTRY} strikes/day)"
        )
        plot_storm_days_bar(
            p_storm,
            "province",
            out / "province_storm_days.png",
            f"Province Storm Days — {key} (min {MIN_STORM_DAY_STRIKES_PROVINCE} strikes/day)"
        )
        if orp_key is not None:
            plot_storm_days_bar(
                o_storm,
                "orp",
                out / "orp_storm_days.png",
                f"ORP Storm Days — {key} (min {MIN_STORM_DAY_STRIKES_PROVINCE} strikes/day)"
            )

        plot_map(
            countries,
            country_key,
            c,
            "country",
            "strikes",
            out / "country_map.png",
            f"Country Map — {key}",
            countries,
            "Strike count",
            storm_days_mode=False
        )
        plot_map(
            provinces,
            province_key,
            p,
            "province",
            "strikes",
            out / "province_map.png",
            f"Province Map — {key}",
            countries,
            "Strike count",
            storm_days_mode=False
        )
        if orp_key is not None:
            plot_map(
                orps,
                orp_key,
                o,
                "orp",
                "strikes",
                out / "orp_map.png",
                f"ORP Map — {key}",
                countries,
                "Strike count",
                storm_days_mode=False
            )

        plot_map(
            countries,
            country_key,
            c_storm,
            "country",
            "storm_days",
            out / "country_storm_days_map.png",
            f"Country Storm Days Map — {key}",
            countries,
            "Storm days",
            storm_days_mode=True
        )
        plot_map(
            provinces,
            province_key,
            p_storm,
            "province",
            "storm_days",
            out / "province_storm_days_map.png",
            f"Province Storm Days Map — {key}",
            countries,
            "Storm days",
            storm_days_mode=True
        )
        if orp_key is not None:
            plot_map(
                orps,
                orp_key,
                o_storm,
                "orp",
                "storm_days",
                out / "orp_storm_days_map.png",
                f"ORP Storm Days Map — {key}",
                countries,
                "Storm days",
                storm_days_mode=True
            )

        plot_anomaly_map(
            countries,
            country_key,
            country_anomaly,
            "country",
            "strike_anomaly",
            out / "country_anomaly_map.png",
            f"Country Strike Anomaly Map — {key}",
            countries,
            "Strike anomaly"
        )
        plot_anomaly_map(
            provinces,
            province_key,
            province_anomaly,
            "province",
            "strike_anomaly",
            out / "province_anomaly_map.png",
            f"Province Strike Anomaly Map — {key}",
            countries,
            "Strike anomaly"
        )
        plot_anomaly_map(
            countries,
            country_key,
            country_anomaly,
            "country",
            "density_anomaly_km2",
            out / "country_density_anomaly_km2_map.png",
            f"Country Density Anomaly Map — {key} (strikes / km²)",
            countries,
            "Density anomaly (strikes / km²)"
        )
        plot_anomaly_map(
            provinces,
            province_key,
            province_anomaly,
            "province",
            "density_anomaly_km2",
            out / "province_density_anomaly_km2_map.png",
            f"Province Density Anomaly Map — {key} (strikes / km²)",
            countries,
            "Density anomaly (strikes / km²)"
        )
        if orp_key is not None:
            plot_anomaly_map(
                orps,
                orp_key,
                orp_anomaly,
                "orp",
                "strike_anomaly",
                out / "orp_anomaly_map.png",
                f"ORP Strike Anomaly Map — {key}",
                countries,
                "Strike anomaly"
            )
            plot_anomaly_map(
                orps,
                orp_key,
                orp_anomaly,
                "orp",
                "density_anomaly_km2",
                out / "orp_density_anomaly_km2_map.png",
                f"ORP Density Anomaly Map — {key} (strikes / km²)",
                countries,
                "Density anomaly (strikes / km²)"
            )

        plot_verification_graphics(
            verification,
            out / "verification_graphics.png",
            f"{name.title()} Verification — {key}"
        )

        if points is not None and not points.empty:
            plot_hour_map_discrete(
                points,
                countries,
                out / "hour_map_discrete.png",
                f"Lightning Strikes by Hour ({key})"
            )

            plot_density_grid_map(
                points,
                countries,
                out / "density_grid_map.png",
                f"{name.title()} Density Overview — {key}",
                cell_km=5,
                sigma_cells=2.0
            )
            plot_absolute_density_grid_map(
                points,
                countries,
                out / "density_absolute_km2.png",
                f"{name.title()} Absolute Density Overview — {key} (strikes / km²)",
                cell_km=5,
                sigma_cells=1.2,
                unit_mode="km2"
            )
            plot_absolute_density_grid_map(
                points,
                countries,
                out / "density_absolute_25km2.png",
                f"{name.title()} Absolute Density Overview — {key} (strikes / 25 km²)",
                cell_km=5,
                sigma_cells=1.2,
                unit_mode="cell"
            )

        vrow = verification.iloc[0] if not verification.empty else None

        meta = {
            "total": int(df["total"].sum()) if not df.empty else 0,
            "days": df["date"].tolist(),
            "storm_day_threshold_country": MIN_STORM_DAY_STRIKES_COUNTRY,
            "storm_day_threshold_province": MIN_STORM_DAY_STRIKES_PROVINCE,
            "top_country": None if c.empty else str(c.iloc[0]["country"]),
            "top_country_strikes": None if c.empty else int(c.iloc[0]["strikes"]),
            "top_province": None if p.empty else str(p.iloc[0]["province"]),
            "top_province_strikes": None if p.empty else int(p.iloc[0]["strikes"]),
            "top_country_storm_region": None if c_storm.empty else str(c_storm.iloc[0]["country"]),
            "top_country_storm_days": None if c_storm.empty else int(c_storm.iloc[0]["storm_days"]),
            "top_province_storm_region": None if p_storm.empty else str(p_storm.iloc[0]["province"]),
            "top_province_storm_days": None if p_storm.empty else int(p_storm.iloc[0]["storm_days"]),
            "hits": None if vrow is None else int(vrow["hits"]),
            "misses": None if vrow is None else int(vrow["misses"]),
            "false_alarms": None if vrow is None else int(vrow["false_alarms"]),
            "correct_nulls": None if vrow is None else int(vrow["correct_nulls"]),
            "verification_total": None if vrow is None else int(vrow["verification_total"]),
            "hit_percent": None if vrow is None else float(vrow["hit_percent"]),
            "miss_percent": None if vrow is None else float(vrow["miss_percent"]),
            "false_alarm_percent": None if vrow is None else float(vrow["false_alarm_percent"]),
            "hits_majority": None if vrow is None else bool(vrow["hits_majority"]),
            "strikes_in_forecast_grids": None if vrow is None else int(vrow["strikes_in_forecast_grids"]),
            "total_strikes_verification": None if vrow is None else int(vrow["total_strikes"]),
            "percent_strikes_in_forecast_grids": None if vrow is None else float(vrow["percent_strikes_in_forecast_grids"]),
            "grids_forecast": None if vrow is None else int(vrow["grids_forecast"]),
            "grids_with_strikes": None if vrow is None else int(vrow["grids_with_strikes"]),
            "density_points_available": bool(points is not None and not points.empty),
            "density_point_count": 0 if points is None else int(len(points)),
            "days_in_period": None if period_counts.empty else int(period_counts.iloc[0]["days_in_period"]),
            "total_strikes_count": None if period_counts.empty else int(period_counts.iloc[0]["total_strikes_count"]),
            "thunderstorm_days_count": None if period_counts.empty else int(period_counts.iloc[0]["thunderstorm_days_count"]),
            "total_hours_in_period": None if period_counts.empty else int(period_counts.iloc[0]["total_hours_in_period"]),
            "thunderstorms_per_hour": None if period_counts.empty else float(period_counts.iloc[0]["thunderstorms_per_hour"]),
            "thunderstorm_percentage": None if period_counts.empty else float(period_counts.iloc[0]["thunderstorm_percentage"]),
            "anomaly_period_type": None if anomaly.empty else str(anomaly.iloc[0]["period_type"]),
            "anomaly_period_key": None if anomaly.empty else str(anomaly.iloc[0]["period_key"]),
            "historical_mean_total_strikes": None if anomaly.empty else safe_float(anomaly.iloc[0]["historical_mean_total_strikes"]),
            "strike_anomaly": None if anomaly.empty else safe_float(anomaly.iloc[0]["strike_anomaly"]),
            "strike_anomaly_percent": None if anomaly.empty else safe_float(anomaly.iloc[0]["strike_anomaly_percent"]),
            "historical_mean_thunderstorm_days": None if anomaly.empty else safe_float(anomaly.iloc[0]["historical_mean_thunderstorm_days"]),
            "thunderstorm_day_anomaly": None if anomaly.empty else safe_float(anomaly.iloc[0]["thunderstorm_day_anomaly"]),
            "thunderstorm_day_anomaly_percent": None if anomaly.empty else safe_float(anomaly.iloc[0]["thunderstorm_day_anomaly_percent"]),
            "anomaly_reference_periods_used": None if anomaly.empty else safe_int(anomaly.iloc[0]["reference_periods_used"]),
            "country_anomaly_regions": 0 if country_anomaly.empty else int(len(country_anomaly)),
            "province_anomaly_regions": 0 if province_anomaly.empty else int(len(province_anomaly)),
            "top_country_anomaly_region": None if country_anomaly.empty else str(country_anomaly.iloc[0]["country"]),
            "top_country_anomaly_value": None if country_anomaly.empty else safe_float(country_anomaly.iloc[0]["strike_anomaly"]),
            "top_province_anomaly_region": None if province_anomaly.empty else str(province_anomaly.iloc[0]["province"]),
            "top_province_anomaly_value": None if province_anomaly.empty else safe_float(province_anomaly.iloc[0]["strike_anomaly"]),
            "top_orp": None if o.empty else str(o.iloc[0]["orp"]),
            "top_orp_strikes": None if o.empty else int(o.iloc[0]["strikes"]),
            "top_orp_storm_region": None if o_storm.empty else str(o_storm.iloc[0]["orp"]),
            "top_orp_storm_days": None if o_storm.empty else int(o_storm.iloc[0]["storm_days"]),
            "orp_anomaly_regions": 0 if orp_anomaly.empty else int(len(orp_anomaly)),
            "top_orp_anomaly_region": None if orp_anomaly.empty else str(orp_anomaly.iloc[0]["orp"]),
            "top_orp_anomaly_value": None if orp_anomaly.empty else safe_float(orp_anomaly.iloc[0]["strike_anomaly"]),
        }

        with open(out / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

# -----------------------------
# MAIN
# -----------------------------
def main():
    global ACTIVE_BOUNDS_3857
    ACTIVE_BOUNDS_3857 = None

    country_filter = parse_args()
    normalized_country = normalize_country_name(country_filter) if country_filter else None

    target_overview_root = OVERVIEW_ROOT / normalized_country if normalized_country else OVERVIEW_ROOT

    print("Archive root:", ARCHIVE_ROOT)
    print("Overview root:", target_overview_root)
    print("Country storm-day threshold:", MIN_STORM_DAY_STRIKES_COUNTRY)
    print("Province storm-day threshold:", MIN_STORM_DAY_STRIKES_PROVINCE)
    print("Storm-day colors: exact discrete color per day count, 0 hidden")
    print("Verification: enabled from summary.csv")
    print("Density overview: enabled from GeoJSON or daily CSV points")
    print("Anomaly summary: enabled for week/month/year")
    print("Thunderstorm counts/hour/%: enabled for all overview periods")
    if normalized_country:
        print("Country filter:", normalized_country)

    days = find_days(ARCHIVE_ROOT)

    if not days:
        print("❌ No daily archive folders found")
        return

    print(f"Found {len(days)} days")

    records = [read_day(d) for d in days]

    countries = load_countries()
    provinces = load_provinces()
    orps = load_orp()

    if normalized_country:
        country_geom, country_display_name = find_country_geometry(countries, normalized_country)
        if country_geom is None:
            print(f"❌ Country not found: {country_filter}")
            return

        print(f"Applying country filter: {country_display_name}")
        set_active_bounds_from_geometry(country_geom)

        countries = clip_regions_to_country(countries, country_geom, use_representative_point=True, crop=True)
        provinces = clip_regions_to_country(provinces, country_geom, use_representative_point=True, crop=True)
        if orps is not None and not orps.empty:
            orps = clip_regions_to_country(orps, country_geom, use_representative_point=True, crop=True)

        country_key = get_country_name_column(countries)
        province_key = get_province_name_column(provinces)
        orp_key = get_orp_name_column(orps) if orps is not None and not orps.empty else None

        for rec in records:
            rec["points"] = filter_points_to_country(rec.get("points"), country_geom)
            rec["country"] = compute_region_table_from_points(rec["points"], countries, country_key, "country")
            rec["province"] = compute_region_table_from_points(rec["points"], provinces, province_key, "province")
            rec["orp"] = compute_region_table_from_points(rec["points"], orps, orp_key, "orp") if orp_key is not None else pd.DataFrame(columns=["orp", "strikes"])
            rec["total"] = int(len(rec["points"])) if rec.get("points") is not None else 0

    print("Building daily overviews...")
    build("days", group_days(records), countries, provinces, orps=orps, overview_root=target_overview_root)

    print("Building week overview...")
    build("week", group_week(records), countries, provinces, orps=orps, overview_root=target_overview_root)

    print("Building month overview...")
    build("month", group_month(records), countries, provinces, orps=orps, overview_root=target_overview_root)

    print("Building year overview...")
    build("year", group_year(records), countries, provinces, orps=orps, overview_root=target_overview_root)

    print("✅ DONE")

if __name__ == "__main__":
    main()
