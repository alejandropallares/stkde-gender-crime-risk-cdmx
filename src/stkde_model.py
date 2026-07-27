"""
stkde_model.py
--------------
Estimación de densidad espacio-temporal (ST-KDE) para el prototipo académico
de riesgo de delitos con perspectiva de género en CDMX.

Metodología
-----------
Densidad en un punto (s*, t*):

    f(s*, t*) = (1/n) Σ K_spatial(d_km / h_s) · K_temporal(Δt_horas / h_t)

- Espacial: distancia Haversine (km) entre celda e incidente.
- Temporal: diferencia en horas entre instante de consulta e instante del incidente.

Selección de kernel y bandwidths (LOO-CV)
-----------------------------------------
Se evalúan tres kernels compactos estándar en KDE:
  - Gaussiano:    K(u) = exp(-u²/2)
  - Epanechnikov: K(u) = 0.75(1-u²)  si |u|≤1, else 0
  - Cuartico:     K(u) = (15/16)(1-u²)² si |u|≤1, else 0

Se elige el trío (kernel, h_s, h_t) con mayor log-verosimilitud leave-one-out
(LOO-CV) sobre una muestra estratificada de incidentes (máx. 400).

Rejillas de búsqueda:
  h_s ∈ {0.5, 1.0, 1.5, 2.0, 3.0} km
  h_t ∈ {24, 72, 168, 336, 720} horas (1 d – 30 d)

El ajuste (fit_stkde) se ejecuta una sola vez sobre el dataset, se serializa
en stkde_config.json y se reutiliza en cada consulta (sin recalcular LOO-CV).

Parámetros seleccionados por LOO-CV (muestra n=400, estabilizado vs 800/1500):
  kernel = gaussian, h_s = 3.0 km, h_t = 720 h

Clasificación de riesgo
-----------------------
Los valores de densidad de la grilla se particionan en tres tertiles
(cuantiles 33 % y 66 %) — umbrales derivados exclusivamente de la
distribución ST-KDE, sin constantes arbitrarias de corte.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# Colores estándar del prototipo (Bajo / Medio / Alto)
RISK_COLORS = {
    "Low": "#16a34a",
    "Medium": "#ca8a04",
    "High": "#dc2626",
}
RISK_LABELS_ES = {
    "Low": "Bajo",
    "Medium": "Medio",
    "High": "Alto",
}

# Parámetros legacy para estimación puntual (/api/estimate)
SPATIAL_BANDWIDTH = 0.015
TEMPORAL_BANDWIDTH_H = 3.0
TEMPORAL_BANDWIDTH_D = 14
SEARCH_RADIUS_KM = 3.0

CV_SAMPLE_SIZE = 400
SPATIAL_BW_CANDIDATES_KM = [0.5, 1.0, 1.5, 2.0, 3.0]
TEMPORAL_BW_CANDIDATES_H = [24.0, 72.0, 168.0, 336.0, 720.0]

# Valores por defecto = selección LOO-CV validada (gaussian, h_s=3, h_t=720)
DEFAULT_KERNEL = "gaussian"
DEFAULT_SPATIAL_BW_KM = 3.0
DEFAULT_TEMPORAL_BW_H = 720.0

# Compatibilidad con código/notebooks que referencían el nombre anterior
FIXED_SPATIAL_BW_KM = DEFAULT_SPATIAL_BW_KM

CONFIG_PATH = Path(__file__).parent / "stkde_config.json"

# Nombres de columna del dataset real (parquet unificado)
COL_LON = "COORD. X"
COL_LAT = "COORD. Y"
COL_DATE = "FECHA DE LOS HECHOS"
COL_HOUR = "HORA DE LOS HECHOS"


@dataclass
class STKDEConfig:
    """Parámetros seleccionados automáticamente para el ST-KDE."""

    kernel_name: str
    kernel_rationale: str
    h_spatial_km: float
    h_temporal_hours: float
    loo_log_likelihood: float
    cv_sample_size: int
    spatial_candidates_km: list[float] = field(
        default_factory=lambda: list(SPATIAL_BW_CANDIDATES_KM)
    )
    temporal_candidates_hours: list[float] = field(
        default_factory=lambda: list(TEMPORAL_BW_CANDIDATES_H)
    )


_cached_config: Optional[STKDEConfig] = None


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def _kernel_gaussian(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u * u)


def _kernel_epanechnikov(u: np.ndarray) -> np.ndarray:
    return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)


def _kernel_quartic(u: np.ndarray) -> np.ndarray:
    t = 1.0 - u * u
    return np.where(np.abs(u) <= 1.0, (15.0 / 16.0) * t * t, 0.0)


KERNELS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], str]] = {
    "gaussian": (
        _kernel_gaussian,
        "Colas suaves; LOO-CV favorece gaussiano cuando los hotspots son difusos.",
    ),
    "epanechnikov": (
        _kernel_epanechnikov,
        "Compacto y eficiente en MISE; LOO-CV lo prefiere con clusters bien delimitados.",
    ),
    "quartic": (
        _kernel_quartic,
        "Similar a Epanechnikov pero más suave en el borde; útil con dispersión intermedia.",
    ),
}


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distancia Haversine vectorizada (km)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _gaussian_kernel(x: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-0.5 * (x / bandwidth) ** 2)


def _incident_datetimes(df: pd.DataFrame) -> np.ndarray:
    """Convierte incidentes a timestamps numpy (datetime64[h])."""
    return pd.to_datetime(
        df[COL_DATE].astype(str) + " " + df[COL_HOUR].astype(str) + ":00:00"
    ).values


def _temporal_diff_hours(incident_times: np.ndarray, target: datetime) -> np.ndarray:
    target_np = np.datetime64(target, "h")
    delta = np.abs(incident_times.astype("datetime64[h]") - target_np)
    return delta.astype("timedelta64[h]").astype(float)


def _apply_kernel(u: np.ndarray, kernel_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    return kernel_fn(u)


def _loo_log_likelihood(
    dist_km: np.ndarray,
    temp_hours: np.ndarray,
    kernel_fn: Callable[[np.ndarray], np.ndarray],
    h_s: float,
    h_t: float,
) -> float:
    """
    Log-verosimilitud LOO-CV para ST-KDE con kernel producto espacio-temporal.
    dist_km, temp_hours: matrices (n, n) de pares i,j.
    """
    n = dist_km.shape[0]
    u_s = dist_km / h_s
    u_t = temp_hours / h_t
    k = _apply_kernel(u_s, kernel_fn) * _apply_kernel(u_t, kernel_fn)
    np.fill_diagonal(k, 0.0)
    densities = k.sum(axis=1) / ((n - 1) * h_s * h_t + 1e-12)
    densities = np.clip(densities, 1e-18, None)
    return float(np.mean(np.log(densities)))


def _build_pairwise_distances(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    n = len(lats)
    lat1 = np.repeat(lats, n).reshape(n, n)
    lon1 = np.repeat(lons, n).reshape(n, n)
    lat2 = np.tile(lats, n).reshape(n, n)
    lon2 = np.tile(lons, n).reshape(n, n)
    return _haversine_km(lat1, lon1, lat2, lon2)


def _build_pairwise_temporal(incident_times: np.ndarray) -> np.ndarray:
    n = len(incident_times)
    t = incident_times.astype("datetime64[h]").astype(float)
    diff = np.abs(t[:, None] - t[None, :])
    return diff


def _cv_sample(df: pd.DataFrame, max_size: int = CV_SAMPLE_SIZE) -> pd.DataFrame:
    if len(df) <= max_size:
        return df.copy()

    df = df.copy()
    df["_lat_bin"] = pd.cut(df[COL_LAT], bins=8, labels=False)
    df["_lon_bin"] = pd.cut(df[COL_LON], bins=8, labels=False)
    df["_stratum"] = df["_lat_bin"].astype(str) + "_" + df["_lon_bin"].astype(str)

    per_stratum = max(1, max_size // max(df["_stratum"].nunique(), 1))
    parts = []
    for _, group in df.groupby("_stratum"):
        parts.append(group.sample(n=min(len(group), per_stratum), random_state=42))
    sampled = pd.concat(parts).drop(columns=["_lat_bin", "_lon_bin", "_stratum"])

    if len(sampled) > max_size:
        sampled = sampled.sample(n=max_size, random_state=42)
    return sampled.reset_index(drop=True)


def fit_stkde(df: pd.DataFrame, target_datetime: datetime | None = None) -> STKDEConfig:
    """
    Selecciona kernel, h_s y h_t mediante LOO-CV + grid search conjunto.
    Usa una muestra estratificada de hasta CV_SAMPLE_SIZE incidentes.
    """
    del target_datetime  # reservado por compatibilidad de firma; el LOO no depende de t*

    if df.empty:
        return STKDEConfig(
            kernel_name=DEFAULT_KERNEL,
            kernel_rationale="Sin datos; valores por defecto (LOO-CV validado).",
            h_spatial_km=DEFAULT_SPATIAL_BW_KM,
            h_temporal_hours=DEFAULT_TEMPORAL_BW_H,
            loo_log_likelihood=float("-inf"),
            cv_sample_size=0,
            spatial_candidates_km=list(SPATIAL_BW_CANDIDATES_KM),
            temporal_candidates_hours=list(TEMPORAL_BW_CANDIDATES_H),
        )

    cv_df = _cv_sample(df)
    lats = cv_df[COL_LAT].values
    lons = cv_df[COL_LON].values
    times = _incident_datetimes(cv_df)
    dist_km = _build_pairwise_distances(lats, lons)
    temp_hours = _build_pairwise_temporal(times)

    best_kernel = DEFAULT_KERNEL
    best_ll = float("-inf")
    best_rationale = KERNELS[DEFAULT_KERNEL][1]
    best_hs = DEFAULT_SPATIAL_BW_KM
    best_ht = DEFAULT_TEMPORAL_BW_H

    for kernel_name, (kernel_fn, rationale) in KERNELS.items():
        for h_s in SPATIAL_BW_CANDIDATES_KM:
            for h_t in TEMPORAL_BW_CANDIDATES_H:
                ll = _loo_log_likelihood(dist_km, temp_hours, kernel_fn, h_s, h_t)
                if ll > best_ll:
                    best_ll = ll
                    best_kernel = kernel_name
                    best_rationale = rationale
                    best_hs = h_s
                    best_ht = h_t

    return STKDEConfig(
        kernel_name=best_kernel,
        kernel_rationale=(
            f"Seleccionado '{best_kernel}' por mayor LOO log-likelihood ({best_ll:.4f}). "
            f"{best_rationale} "
            f"Bandwidth espacial h_s={best_hs} km y temporal h_t={best_ht} h "
            f"seleccionados por grid search conjunto LOO-CV "
            f"(muestra estratificada n={len(cv_df)})."
        ),
        h_spatial_km=best_hs,
        h_temporal_hours=best_ht,
        loo_log_likelihood=best_ll,
        cv_sample_size=len(cv_df),
        spatial_candidates_km=list(SPATIAL_BW_CANDIDATES_KM),
        temporal_candidates_hours=list(TEMPORAL_BW_CANDIDATES_H),
    )


def save_stkde_config(config: STKDEConfig, path: Path | None = None) -> Path:
    """Serializa la configuración ST-KDE a JSON."""
    out = path or CONFIG_PATH
    payload = asdict(config)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_stkde_config(path: Path | None = None) -> STKDEConfig:
    """Carga la configuración ST-KDE desde JSON."""
    src = path or CONFIG_PATH
    data = json.loads(src.read_text(encoding="utf-8"))
    return STKDEConfig(**data)


def get_stkde_config(
    df: pd.DataFrame | None = None,
    *,
    force_refit: bool = False,
    path: Path | None = None,
) -> STKDEConfig:
    """
    Devuelve la configuración ST-KDE cacheada.

    - Si existe stkde_config.json (y no force_refit), la carga una vez.
    - Si no existe, ejecuta fit_stkde(df) una sola vez, serializa y cachea.
    """
    global _cached_config
    config_path = path or CONFIG_PATH

    if not force_refit:
        if _cached_config is not None:
            return _cached_config
        if config_path.exists():
            _cached_config = load_stkde_config(config_path)
            return _cached_config

    if df is None or df.empty:
        raise ValueError(
            "No hay stkde_config.json y se requiere un DataFrame no vacío para ejecutar fit_stkde()."
        )

    config = fit_stkde(df)
    save_stkde_config(config, config_path)
    _cached_config = config
    return config


def ensure_stkde_config(df: pd.DataFrame, path: Path | None = None) -> STKDEConfig:
    """
    Garantiza que exista configuración serializada.
    Ejecuta fit_stkde una sola vez si el JSON aún no existe.
    """
    config_path = path or CONFIG_PATH
    if config_path.exists():
        return get_stkde_config(path=config_path)

    print(f"[ST-KDE] Ajustando parámetros LOO-CV (una vez) → {config_path.name}")
    config = get_stkde_config(df, force_refit=True, path=config_path)
    print(
        f"[ST-KDE] Config guardada: kernel={config.kernel_name}, "
        f"h_s={config.h_spatial_km} km, h_t={config.h_temporal_hours} h, "
        f"LL={config.loo_log_likelihood:.4f}"
    )
    return config


def _stkde_weights(
    cell_lats: np.ndarray,
    cell_lons: np.ndarray,
    df: pd.DataFrame,
    target_datetime: datetime,
    config: STKDEConfig,
) -> np.ndarray:
    """Densidad ST-KDE en cada celda (vector length = n_cells)."""
    if df.empty:
        return np.zeros(len(cell_lats))

    kernel_fn = KERNELS[config.kernel_name][0]
    inc_lats = df[COL_LAT].values
    inc_lons = df[COL_LON].values
    inc_times = _incident_datetimes(df)
    temp_diff = _temporal_diff_hours(inc_times, target_datetime)

    n_cells = len(cell_lats)
    n_inc = len(df)
    densities = np.zeros(n_cells)

    # Bloques para limitar memoria (6090 × 1500 ≈ 9 M elementos por bloque)
    chunk = 500
    for start in range(0, n_cells, chunk):
        end = min(start + chunk, n_cells)
        clat = cell_lats[start:end]
        clon = cell_lons[start:end]

        lat1 = np.repeat(clat, n_inc).reshape(end - start, n_inc)
        lon1 = np.repeat(clon, n_inc).reshape(end - start, n_inc)
        lat2 = np.tile(inc_lats, end - start).reshape(end - start, n_inc)
        lon2 = np.tile(inc_lons, end - start).reshape(end - start, n_inc)

        dist = _haversine_km(lat1, lon1, lat2, lon2)
        u_s = dist / config.h_spatial_km
        u_t = temp_diff[None, :] / config.h_temporal_hours

        k = _apply_kernel(u_s, kernel_fn) * _apply_kernel(u_t, kernel_fn)
        densities[start:end] = k.sum(axis=1) / (n_inc * config.h_spatial_km * config.h_temporal_hours)

    return densities


def classify_densities(densities: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Clasifica densidades en Low / Medium / High usando tertiles (Q33, Q66).
    Retorna niveles, colores y metadatos de umbrales.
    """
    positive = densities[densities > 0]
    if len(positive) == 0:
        levels = np.full(len(densities), "Low")
        colors = np.array([RISK_COLORS["Low"]] * len(densities))
        return levels, colors, {"q33": 0.0, "q66": 0.0, "method": "tertiles (sin densidad positiva)"}

    q33, q66 = np.percentile(positive, [33.33, 66.67])
    levels = np.where(
        densities <= q33,
        "Low",
        np.where(densities <= q66, "Medium", "High"),
    )
    # Celdas con densidad exactamente 0 → Bajo
    levels = np.where(densities <= 0, "Low", levels)
    colors = np.array([RISK_COLORS[l] for l in levels])
    meta = {
        "method": "tertiles (cuantiles 33.33 % y 66.67 % sobre densidades > 0)",
        "q33": round(float(q33), 8),
        "q66": round(float(q66), 8),
    }
    return levels, colors, meta


def compute_grid_stkde(
    df: pd.DataFrame,
    cells: list[dict],
    target_datetime: datetime,
    config: STKDEConfig | None = None,
) -> dict:
    """
    Calcula ST-KDE para cada celda de la grilla.

    cells: lista con {cell_id, row, col, centroid_lat, centroid_lon, geometry}
    Usa la configuración precomputada (stkde_config.json) salvo que se pase `config`.
    """
    config = config or get_stkde_config(df)

    lats = np.array([c["centroid_lat"] for c in cells])
    lons = np.array([c["centroid_lon"] for c in cells])
    densities = _stkde_weights(lats, lons, df, target_datetime, config)
    levels, colors, threshold_meta = classify_densities(densities)

    # Alto solo si contiene al menos un incidente dentro de la celda (500 m)
    from cdmx_geo import cell_ids_with_incidents

    occupied = cell_ids_with_incidents(df)
    has_incident_in_cell = np.array([cell["cell_id"] in occupied for cell in cells])
    no_incident = (levels == "High") & ~has_incident_in_cell
    levels = np.where(no_incident, "Medium", levels)
    colors = np.array([RISK_COLORS[l] for l in levels])
    threshold_meta["high_requires_incident_in_cell"] = True

    features = []
    for i, cell in enumerate(cells):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "cell_id": cell["cell_id"],
                    "row": cell["row"],
                    "col": cell["col"],
                    "density": round(float(densities[i]), 8),
                    "risk_level": levels[i],
                    "risk_label": RISK_LABELS_ES[levels[i]],
                    "color": colors[i],
                },
                "geometry": cell["geometry"],
            }
        )

    return {
        "type": "FeatureCollection",
        "metadata": {
            "kernel": config.kernel_name,
            "kernel_rationale": config.kernel_rationale,
            "bandwidth_spatial_km": config.h_spatial_km,
            "bandwidth_temporal_hours": config.h_temporal_hours,
            "bandwidth_selection": "LOO-CV precomputado (stkde_config.json)",
            "loo_log_likelihood": round(config.loo_log_likelihood, 6),
            "cv_sample_size": config.cv_sample_size,
            "spatial_candidates_km": config.spatial_candidates_km,
            "temporal_candidates_hours": config.temporal_candidates_hours,
            "classification": threshold_meta,
            "reference_datetime": target_datetime.strftime("%Y-%m-%d %H:00"),
            "incident_count": len(df),
            "cell_count": len(cells),
        },
        "features": features,
    }


# ---------------------------------------------------------------------------
# API puntual legacy (/api/estimate)
# ---------------------------------------------------------------------------
def estimate_risk(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    date_str: str,
    hour: int,
    crime_type: Optional[str] = None,
) -> dict:
    """Estimación ST-KDE en un punto (lat, lon, fecha, hora)."""
    if df.empty:
        return _empty_result()

    target_date = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour)
    work_df = df.copy()
    if crime_type and crime_type != "All" and "crime_type" in work_df.columns:
        work_df = work_df[work_df["crime_type"] == crime_type]
    if work_df.empty:
        work_df = df.copy()

    config = get_stkde_config(work_df)
    density_score = float(
        _stkde_weights(
            np.array([lat]),
            np.array([lon]),
            work_df,
            target_date,
            config,
        )[0]
    )

    lats = work_df[COL_LAT].values
    lons = work_df[COL_LON].values
    dist_km = _haversine_km(lat, lon, lats, lons)
    nearby_mask = dist_km <= config.h_spatial_km
    nearby_df = work_df[nearby_mask].copy()
    nearby_count = int(nearby_mask.sum())

    positive = _stkde_weights(
        work_df[COL_LAT].values[: min(200, len(work_df))],
        work_df[COL_LON].values[: min(200, len(work_df))],
        work_df,
        target_date,
        config,
    )
    positive = positive[positive > 0]
    if len(positive) > 0:
        q33, q66 = np.percentile(positive, [33.33, 66.67])
        if density_score <= q33:
            risk_level = "Low"
        elif density_score <= q66:
            risk_level = "Medium"
        else:
            risk_level = "High"
    else:
        risk_level = "Low"

    probability = round(min(max(density_score / (density_score + 0.01), 0.01), 0.99), 4)

    if not nearby_df.empty and "crime_type" in nearby_df.columns:
        dominant_crime = nearby_df["crime_type"].mode().iloc[0]
    else:
        dominant_crime = crime_type or "N/A"

    # Incidentes cercanos (dentro de h_s), ordenados por distancia
    contributing: list[dict] = []
    if nearby_count > 0:
        nearby_view = nearby_df.copy()
        nearby_view["_dist_km"] = dist_km[nearby_mask]
        nearby_view = nearby_view.sort_values("_dist_km")
        # Limitar puntos en mapa para no saturar el cliente
        max_points = 200
        for _, row in nearby_view.head(max_points).iterrows():
            contributing.append(
                {
                    COL_LAT: float(row[COL_LAT]),
                    COL_LON: float(row[COL_LON]),
                    COL_DATE: str(row[COL_DATE]),
                    COL_HOUR: int(row[COL_HOUR]),
                    "distance_km": round(float(row["_dist_km"]), 3),
                }
            )

    return {
        "risk_level": risk_level,
        "probability": probability,
        "density_score": round(density_score, 8),
        "nearby_count": nearby_count,
        "dominant_crime": dominant_crime,
        "contributing_incidents": contributing,
        "search_radius_km": config.h_spatial_km,
        "method": f"ST-KDE ({config.kernel_name}, h_s={config.h_spatial_km} km, h_t={config.h_temporal_hours} h)",
        "kernel": config.kernel_name,
        "bandwidth_spatial_km": config.h_spatial_km,
        "bandwidth_temporal_hours": config.h_temporal_hours,
        "reference_datetime": target_date.strftime("%Y-%m-%d %H:00"),
    }


def compute_risk_grid(
    df: pd.DataFrame,
    grid_step: float = 0.025,
    crime_type: Optional[str] = None,
) -> list[dict]:
    """Calcula malla de riesgo (legacy, submuestreada)."""
    if df.empty:
        return []

    work_df = df.copy()
    if crime_type and crime_type != "All" and "crime_type" in work_df.columns:
        work_df = work_df[work_df["crime_type"] == crime_type]

    lat_min, lat_max = work_df[COL_LAT].min(), work_df[COL_LAT].max()
    lon_min, lon_max = work_df[COL_LON].min(), work_df[COL_LON].max()
    lat_points = np.arange(lat_min, lat_max, grid_step)
    lon_points = np.arange(lon_min, lon_max, grid_step)

    ref_date = datetime.strptime(
        work_df[COL_DATE].mode().iloc[0] if not work_df.empty else "2024-06-01",
        "%Y-%m-%d",
    ).replace(hour=int(work_df[COL_HOUR].median()))

    zones = []
    for la in lat_points[::2]:
        for lo in lon_points[::2]:
            result = estimate_risk(work_df, la, lo, ref_date.strftime("%Y-%m-%d"), ref_date.hour)
            zones.append(
                {
                    "lat": round(float(la), 4),
                    "lng": round(float(lo), 4),
                    "risk_level": result["risk_level"],
                    "probability": result["probability"],
                }
            )
    return zones


def _empty_result() -> dict:
    return {
        "risk_level": "Low",
        "probability": 0.05,
        "density_score": 0.0,
        "nearby_count": 0,
        "dominant_crime": "N/A",
        "contributing_incidents": [],
        "search_radius_km": SEARCH_RADIUS_KM,
        "method": "ST-KDE (no data)",
    }