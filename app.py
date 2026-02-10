import io
import os
import time
import hashlib
import re
from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from shapely.geometry import shape, Point


# -------------------------
# Configuración (QUEBRADANEGRA)
# -------------------------
st.set_page_config(page_title="Georreferenciación - Quebradanegra (Cundinamarca)", layout="wide")

MUNICIPIO_NOMBRE = "Quebradanegra"
DEPARTAMENTO_NOMBRE = "Cundinamarca"

QUEBRADANEGRA_CENTER = (5.1175, -74.4793)
DEFAULT_ZOOM = 13

CACHE_PATH = os.path.join(".cache", "geocode_cache.parquet")

# User-Agent obligatorio (Nominatim lo exige)
UA = "georreferenciacion-quebradanegra/1.0 (contacto: soporte@example.com)"


# -------------------------
# Helpers
# -------------------------
def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _hash_key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ensure_cache_df(cache_path: str) -> pd.DataFrame:
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass
    return pd.DataFrame(columns=["addr_key", "address", "lat", "lon", "provider", "ts", "raw"])


def save_cache_df(df_cache: pd.DataFrame, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_cache.to_parquet(cache_path, index=False)


def http_get_json(url: str, params: dict | None = None, timeout: int = 60) -> dict:
    headers = {"User-Agent": UA}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def nominatim_get_with_retry(url: str, params: dict, retries: int = 3) -> dict:
    headers = {"User-Agent": UA}
    last = None
    for i in range(retries):
        r = requests.get(url, params=params, headers=headers, timeout=60)
        last = r
        if r.status_code == 429:  # rate limited
            time.sleep(2 + i * 2)
            continue
        r.raise_for_status()
        return r.json()
    if last is not None:
        last.raise_for_status()
    return {}


@st.cache_data(show_spinner=False)
def municipio_geojson(municipio: str, departamento: str) -> dict:
    """
    Polígono del municipio vía Nominatim (OSM).
    Retorna FeatureCollection con 1 feature o vacío.
    """
    q = f"{municipio}, {departamento}, Colombia"
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": 1, "polygon_geojson": 1}

    data = nominatim_get_with_retry(url, params=params, retries=3)
    if not data:
        return {"type": "FeatureCollection", "features": []}

    item = data[0]
    geo = item.get("geojson")
    if not geo:
        return {"type": "FeatureCollection", "features": []}

    feature = {
        "type": "Feature",
        "properties": {
            "name": municipio,
            "state": departamento,
            "display_name": item.get("display_name", ""),
        },
        "geometry": geo,
    }
    return {"type": "FeatureCollection", "features": [feature]}


# -------------------------
# Normalización direcciones (solo si geocodificas)
# -------------------------
def normalize_address(raw: str) -> str:
    s = _safe_str(raw).upper()
    s = re.sub(r"\s+", " ", s).strip()

    s = s.replace(" N ", " # ").replace(" NO ", " # ").replace(" NUM ", " # ")
    replacements = {
        "CLL": "CALLE",
        "CL": "CALLE",
        "CRA": "CARRERA",
        "CR": "CARRERA",
        "KR": "CARRERA",
        "AK": "AVENIDA CARRERA",
        "AC": "AVENIDA CALLE",
        "AV": "AVENIDA",
        "DG": "DIAGONAL",
        "TV": "TRANSVERSAL",
        "VDA": "VEREDA",
        "VRD": "VEREDA",
    }
    for k, v in replacements.items():
        s = re.sub(rf"\b{k}\b", v, s)

    s = re.sub(r"\s*#\s*", " # ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_address(row: pd.Series, address_col: str, extra_col: Optional[str]) -> str:
    base = normalize_address(row.get(address_col, ""))
    extra = normalize_address(row.get(extra_col, "")) if extra_col else ""
    parts = []
    if base and base not in {"NO APLICA", "N/A", "NA"}:
        parts.append(base)
    if extra and extra not in {"NO APLICA", "N/A", "NA"}:
        parts.append(extra)

    parts.extend([MUNICIPIO_NOMBRE.upper(), DEPARTAMENTO_NOMBRE.upper(), "COLOMBIA"])
    return ", ".join([p for p in parts if p])


# -------------------------
# Geocodificación (opcional)
# -------------------------
def google_geocode(address: str, api_key: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": address, "key": api_key, "region": "co", "language": "es"}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK":
            return None, None, None
        result = data["results"][0]
        loc = result["geometry"]["location"]
        return float(loc["lat"]), float(loc["lng"]), result.get("formatted_address")
    except Exception:
        return None, None, None


def nominatim_geocode(address: str, geocode_fn) -> Tuple[Optional[float], Optional[float]]:
    try:
        loc = geocode_fn(address)
        if loc:
            return float(loc.latitude), float(loc.longitude)
    except Exception:
        return None, None
    return None, None


def geocode_dataframe_within_polygon(
    df: pd.DataFrame,
    address_col: str,
    extra_col: Optional[str],
    max_rows: int,
    cache_path: str,
    polygon,
    provider: str,
    google_api_key: str,
    fallback_to_nominatim: bool,
) -> pd.DataFrame:
    df = df.copy()
    if "lat" not in df.columns:
        df["lat"] = np.nan
    if "lon" not in df.columns:
        df["lon"] = np.nan

    df_cache = ensure_cache_df(cache_path)
    cached = {row["addr_key"]: (row["lat"], row["lon"]) for _, row in df_cache.iterrows()}

    geocode_fn = None
    if provider == "nominatim" or fallback_to_nominatim:
        geolocator = Nominatim(user_agent=UA)
        geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    pending = df[df["lat"].isna()].head(max_rows)
    prog = st.progress(0, text=f"Geocodificando (forzando dentro de {MUNICIPIO_NOMBRE})...")
    total = len(pending) if len(pending) > 0 else 1

    for i, (idx, row) in enumerate(pending.iterrows(), start=1):
        addr1 = build_address(row, address_col, extra_col)
        if not addr1.strip():
            continue

        key1 = _hash_key(addr1)
        lat, lon, raw = None, None, None

        if key1 in cached:
            lat, lon = cached[key1]
        else:
            if provider == "google":
                if google_api_key:
                    lat, lon, raw = google_geocode(addr1, google_api_key)
            else:
                lat, lon = nominatim_geocode(addr1, geocode_fn)

            if (lat is None or lon is None) and fallback_to_nominatim and geocode_fn:
                lat2, lon2 = nominatim_geocode(addr1, geocode_fn)
                if lat2 is not None and lon2 is not None:
                    lat, lon = lat2, lon2
                    if raw is None:
                        raw = "fallback_nominatim"

            df_cache = pd.concat(
                [df_cache, pd.DataFrame([{
                    "addr_key": key1,
                    "address": addr1,
                    "lat": lat,
                    "lon": lon,
                    "provider": provider if raw is None else str(raw),
                    "ts": int(time.time()),
                    "raw": raw,
                }])],
                ignore_index=True
            )
            cached[key1] = (lat, lon)

        inside = False
        if lat is not None and lon is not None and not (pd.isna(lat) or pd.isna(lon)):
            inside = polygon.contains(Point(float(lon), float(lat)))

        if inside:
            df.at[idx, "lat"] = lat
            df.at[idx, "lon"] = lon
        else:
            df.at[idx, "lat"] = np.nan
            df.at[idx, "lon"] = np.nan

        prog.progress(i / total, text=f"Geocodificando {i}/{len(pending)}")

    save_cache_df(df_cache, cache_path)
    return df


def keep_only_inside_polygon(df_in: pd.DataFrame, poly) -> pd.DataFrame:
    df = df_in.copy()
    mask = []
    for _, r in df.iterrows():
        lat, lon = r.get("lat"), r.get("lon")
        ok = False
        if lat is not None and lon is not None and not (pd.isna(lat) or pd.isna(lon)):
            ok = poly.contains(Point(float(lon), float(lat)))
        mask.append(ok)
    df.loc[~pd.Series(mask, index=df.index), ["lat", "lon"]] = np.nan
    return df


# -------------------------
# Servicios e iconos
# -------------------------
SERVICE_STYLES = {
    "SALUD": {"icon": "plus-square", "color": "red"},
    "EDUCACIÓN": {"icon": "graduation-cap", "color": "blue"},
    "EMPLEO": {"icon": "briefcase", "color": "green"},
    "DESARROLLO": {"icon": "chart-line", "color": "purple"},
    "BIENESTAR SOCIAL": {"icon": "hands-helping", "color": "orange"},
    "ENTORNO SALUDABLE": {"icon": "leaf", "color": "darkgreen"},
    "REDES DE APOYO": {"icon": "users", "color": "cadetblue"},
    "OTRO": {"icon": "map-marker", "color": "gray"},
}


def normalize_service(x: str) -> str:
    s = _safe_str(x).upper()
    s = re.sub(r"\s+", " ", s).strip()

    # match exacto
    if s in SERVICE_STYLES:
        return s

    # tolerancia
    if "SALUD" in s:
        return "SALUD"
    if "EDU" in s:
        return "EDUCACIÓN"
    if "EMPLE" in s or "TRABA" in s:
        return "EMPLEO"
    if "DESAR" in s:
        return "DESARROLLO"
    if "BIENEST" in s:
        return "BIENESTAR SOCIAL"
    if "ENTORNO" in s or "AMBIEN" in s:
        return "ENTORNO SALUDABLE"
    if "RED" in s or "APOYO" in s:
        return "REDES DE APOYO"
    return "OTRO"


def add_legend(m: folium.Map) -> None:
    rows = []
    for k, v in SERVICE_STYLES.items():
        rows.append(
            f"""
            <div style="display:flex;align-items:center;margin-bottom:4px;">
              <div style="width:10px;height:10px;background:{v['color']};border-radius:50%;margin-right:8px;"></div>
              <span style="font-size:12px;">{k}</span>
            </div>
            """
        )
    html = f"""
    <div style="
        position: fixed;
        bottom: 30px; left: 30px; z-index: 9999;
        background: white; padding: 10px 12px; border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,.15);
        ">
      <div style="font-weight:700;margin-bottom:6px;">Tipos de servicio</div>
      {''.join(rows)}
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


# -------------------------
# Mapa
# -------------------------
def make_map(
    df_points: pd.DataFrame,
    municipio_gj: dict,
    service_col: str,
    icon_png_map: Dict[str, Optional[bytes]],
) -> folium.Map:
    pts = df_points.dropna(subset=["lat", "lon"]).copy()

    if len(pts) > 0:
        center = (float(pts["lat"].astype(float).mean()), float(pts["lon"].astype(float).mean()))
        zoom = 13
    else:
        center = QUEBRADANEGRA_CENTER
        zoom = DEFAULT_ZOOM

    m = folium.Map(location=center, zoom_start=zoom, control_scale=True)

    folium.GeoJson(
        municipio_gj,
        name=f"Municipio {MUNICIPIO_NOMBRE}",
        style_function=lambda feature: {
            "fillColor": "#2E7D32",
            "color": "#1B5E20",
            "weight": 2,
            "fillOpacity": 0.15,
        },
    ).add_to(m)

    cluster = MarkerCluster(name="Puntos").add_to(m)

    if service_col and service_col in pts.columns:
        pts["_SERVICIO"] = pts[service_col].apply(normalize_service)
    else:
        pts["_SERVICIO"] = "OTRO"

    for _, r in pts.iterrows():
        servicio = _safe_str(r.get("_SERVICIO", "OTRO")) or "OTRO"
        style = SERVICE_STYLES.get(servicio, SERVICE_STYLES["OTRO"])

        label = (
            _safe_str(r.get("NOMBRE", ""))
            or _safe_str(r.get("DIRECCION", ""))
            or _safe_str(r.get("address_full", ""))
            or "Sitio"
        )

        popup = f"""
        <b>{label}</b><br>
        <b>Servicio:</b> {servicio}<br>
        <b>Lat:</b> {float(r["lat"]):.6f}<br>
        <b>Lon:</b> {float(r["lon"]):.6f}
        """

        if servicio in icon_png_map and icon_png_map[servicio] is not None:
            icon = folium.CustomIcon(
                icon_image=icon_png_map[servicio],
                icon_size=(28, 28),
                icon_anchor=(14, 14),
            )
            folium.Marker(
                location=(float(r["lat"]), float(r["lon"])),
                popup=folium.Popup(popup, max_width=350),
                icon=icon,
            ).add_to(cluster)
        else:
            folium.Marker(
                location=(float(r["lat"]), float(r["lon"])),
                popup=folium.Popup(popup, max_width=350),
                icon=folium.Icon(color=style["color"], icon=style["icon"], prefix="fa"),
            ).add_to(cluster)

    add_legend(m)
    folium.LayerControl().add_to(m)
    return m


# -------------------------
# UI
# -------------------------
st.title("Georreferenciar servicios sociales (solo dentro de Quebradanegra, Cundinamarca)")

with st.sidebar:
    st.header("Entrada")
    uploaded = st.file_uploader("Sube tu Excel (.xlsx)", type=["xlsx"])

    st.header("Proveedor de geocodificación (solo si NO tienes lat/lon)")
    provider = st.selectbox("Proveedor", options=["google", "nominatim"], index=0)
    google_api_key = st.text_input("Google API Key (si usas Google)", type="password")
    fallback_to_nominatim = st.checkbox("Fallback a Nominatim si falla Google", value=True)

    st.header("Ejecución")
    max_rows = st.number_input("Máximo de filas a geocodificar por ejecución", min_value=10, max_value=5000, value=300, step=50)
    run_geocode = st.checkbox("Ejecutar geocodificación (si NO tienes lat/lon)", value=False)

    st.header("Logos PNG por tipo (opcional)")
    st.caption("Si subes un PNG, se usará como ícono para ese tipo de servicio.")
    icon_png_map: Dict[str, Optional[bytes]] = {}
    for k in SERVICE_STYLES.keys():
        if k == "OTRO":
            continue
        f = st.file_uploader(f"Logo PNG para {k}", type=["png"], key=f"logo_{k}")
        icon_png_map[k] = f.getvalue() if f is not None else None

    st.header("Salida")
    enable_download = st.checkbox("Permitir descarga del Excel", value=True)

if not uploaded:
    st.info("Sube un Excel para comenzar.")
    st.stop()

df = pd.read_excel(uploaded)
st.subheader("Vista previa")
st.dataframe(df.head(25), use_container_width=True)

cols = df.columns.tolist()

# Detectar lat/lon si vienen (variantes comunes)
def find_col(candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None

lat_col = find_col(["lat", "latitude", "y"])
lon_col = find_col(["lon", "lng", "longitude", "x"])

st.subheader("Configuración de columnas")
address_col = st.selectbox("Columna de dirección (si aplica)", options=cols, index=0)

extra_col_opt = ["(ninguna)"] + cols
extra_col_choice = st.selectbox("Columna opcional de complemento (vereda/barrio/etc.)", options=extra_col_opt, index=0)
extra_col = None if extra_col_choice == "(ninguna)" else extra_col_choice

service_col_opt = ["(ninguna)"] + cols
service_col_choice = st.selectbox("Columna de TIPO DE SERVICIO", options=service_col_opt, index=0)
service_col = None if service_col_choice == "(ninguna)" else service_col_choice

df = df.copy()

# Coordenadas directas
if lat_col and lon_col:
    df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    st.success(f"Detecté coordenadas: lat='{lat_col}', lon='{lon_col}'. No es necesario geocodificar.")
else:
    df["lat"] = np.nan
    df["lon"] = np.nan
    st.warning("No detecté columnas lat/lon. Si quieres coordenadas, activa geocodificación con dirección.")

# Limpieza básica: evita coordenadas imposibles
df.loc[(df["lat"] < -5) | (df["lat"] > 15), ["lat", "lon"]] = np.nan
df.loc[(df["lon"] < -82) | (df["lon"] > -66), ["lat", "lon"]] = np.nan

# Construir address_full (solo útil si geocodificas)
df["address_full"] = df.apply(lambda r: build_address(r, address_col, extra_col), axis=1)

# Polígono del municipio
with st.spinner("Cargando polígono oficial del municipio (Nominatim/OSM)..."):
    municipio_gj = municipio_geojson(MUNICIPIO_NOMBRE, DEPARTAMENTO_NOMBRE)

if not municipio_gj.get("features"):
    st.error("No se pudo obtener el polígono de Quebradanegra desde Nominatim.")
    st.stop()

municipio_poly = shape(municipio_gj["features"][0]["geometry"])

# Geocodificación (solo si usuario lo pide y no hay lat/lon)
if run_geocode:
    if lat_col and lon_col:
        st.info("Tu Excel ya trae lat/lon; no se ejecutó geocodificación.")
    else:
        if provider == "google" and not google_api_key:
            st.error("Seleccionaste Google pero no ingresaste API Key.")
            st.stop()

        with st.spinner(f"Geocodificando y forzando dentro de {MUNICIPIO_NOMBRE}..."):
            df = geocode_dataframe_within_polygon(
                df=df,
                address_col=address_col,
                extra_col=extra_col,
                max_rows=int(max_rows),
                cache_path=CACHE_PATH,
                polygon=municipio_poly,
                provider=provider,
                google_api_key=google_api_key,
                fallback_to_nominatim=fallback_to_nominatim,
            )
        st.success("Geocodificación completada (o parcial según el límite).")

# Invalidar puntos fuera del municipio
df = keep_only_inside_polygon(df, municipio_poly)

# Tipo normalizado
if service_col and service_col in df.columns:
    df["_SERVICIO"] = df[service_col].apply(normalize_service)
else:
    df["_SERVICIO"] = "OTRO"

total = len(df)
with_coords = df.dropna(subset=["lat", "lon"]).shape[0]
st.write(f"Registros: **{total:,}** | Dentro de {MUNICIPIO_NOMBRE} (con coords): **{with_coords:,}**")

# -------------------------
# Filtros
# -------------------------
st.subheader("Filtros")

colA, colB, colC = st.columns([1, 1, 1.2])

servicios_disponibles = sorted(df["_SERVICIO"].dropna().unique().tolist())
with colA:
    filtro_servicios = st.multiselect("Tipos de servicio", options=servicios_disponibles, default=servicios_disponibles)

vereda_col = None
for cand in ["VEREDA", "Vereda", "vereda"]:
    if cand in df.columns:
        vereda_col = cand
        break

with colB:
    if vereda_col:
        veredas = sorted(df[vereda_col].dropna().astype(str).unique().tolist())
        filtro_veredas = st.multiselect("Veredas", options=veredas, default=veredas)
    else:
        filtro_veredas = None
        st.caption("No se detectó columna VEREDA (opcional).")

with colC:
    q = st.text_input("Buscar (nombre/dirección)", value="").strip().lower()

df_f = df.copy()
df_f = df_f[df_f["_SERVICIO"].isin(filtro_servicios)]

if vereda_col and filtro_veredas is not None:
    df_f = df_f[df_f[vereda_col].astype(str).isin(filtro_veredas)]

if q:
    cols_busqueda = [c for c in ["NOMBRE", "DIRECCION", "address_full"] if c in df_f.columns]
    if cols_busqueda:
        mask = np.zeros(len(df_f), dtype=bool)
        for c in cols_busqueda:
            mask |= df_f[c].astype(str).str.lower().str.contains(q, na=False)
        df_f = df_f[mask]

# -------------------------
# Conteo y mapa
# -------------------------
st.subheader("Conteo por tipo (según filtros, solo puntos válidos)")
conteo_serv = (
    df_f.dropna(subset=["lat", "lon"])
    .groupby("_SERVICIO")
    .size()
    .reset_index(name="sitios")
    .sort_values("sitios", ascending=False)
)
st.dataframe(conteo_serv, use_container_width=True)

st.subheader("Mapa: polígono municipal + puntos por tipo (iconos/logos)")
m = make_map(df_f, municipio_gj, service_col=service_col or "", icon_png_map=icon_png_map)
st_folium(m, width=1100, height=650)

st.subheader("Resultados (muestra - según filtros)")
out_cols = []
for c in ["NOMBRE", "DIRECCION"]:
    if c in df_f.columns:
        out_cols.append(c)
if address_col in df_f.columns and address_col not in out_cols:
    out_cols.append(address_col)
if extra_col and extra_col in df_f.columns:
    out_cols.append(extra_col)
if vereda_col and vereda_col in df_f.columns:
    out_cols.append(vereda_col)
if service_col and service_col in df_f.columns:
    out_cols.append(service_col)
out_cols += ["_SERVICIO", "lat", "lon", "address_full"]
out_cols = [c for c in out_cols if c in df_f.columns]
out_cols = list(dict.fromkeys(out_cols))

st.dataframe(df_f[out_cols].head(300), use_container_width=True)

# -------------------------
# Descarga
# -------------------------
if enable_download:
    c1, c2 = st.columns(2)

    with c1:
        output_all = io.BytesIO()
        with pd.ExcelWriter(output_all, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="todo")
        st.download_button(
            "Descargar TODO",
            data=output_all.getvalue(),
            file_name="quebradanegra_servicios_todo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with c2:
        output_f = io.BytesIO()
        with pd.ExcelWriter(output_f, engine="openpyxl") as writer:
            df_f.to_excel(writer, index=False, sheet_name="filtrado")
        st.download_button(
            "Descargar FILTRADO",
            data=output_f.getvalue(),
            file_name="quebradanegra_servicios_filtrado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
