import io
import os
import json
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

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from shapely.geometry import shape, Point


# -------------------------
# Configuración
# -------------------------
st.set_page_config(page_title="Georreferenciación Bogotá - Kennedy (UPZ)", layout="wide")

BOGOTA_CENTER = (4.6486259, -74.247894)
DEFAULT_ZOOM = 11

ARCGIS_BASE = "https://sinu.sdp.gov.co/serverp/rest/services/Consultas/MapServer"
UPZ_LAYER = 4
LOCALIDADES_LAYER = 5
KENNEDY_CODIGO_LOCALIDAD = "8"

CACHE_PATH = os.path.join(".cache", "geocode_cache.parquet")


# -------------------------
# Helpers generales
# -------------------------
def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def _hash_key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

@st.cache_data(show_spinner=False)
def fetch_geojson(url: str) -> dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def kennedy_geojson() -> dict:
    url = (
        f"{ARCGIS_BASE}/{LOCALIDADES_LAYER}/query"
        f"?where=CODIGO_LOCALIDAD%3D%27{KENNEDY_CODIGO_LOCALIDAD}%27"
        f"&outFields=*&f=geojson&outSR=4326"
    )
    return fetch_geojson(url)

def upz_kennedy_geojson() -> dict:
    url = (
        f"{ARCGIS_BASE}/{UPZ_LAYER}/query"
        f"?where=CODIGO_LOCALIDAD%3D%27{KENNEDY_CODIGO_LOCALIDAD}%27"
        f"&outFields=CODIGO_UPZ,NOMBRE,CODIGO_LOCALIDAD&f=geojson&outSR=4326"
    )
    return fetch_geojson(url)

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


# -------------------------
# Normalización de direcciones
# -------------------------
def normalize_bogota_address(raw: str) -> str:
    s = _safe_str(raw).upper()
    s = re.sub(r"\s+", " ", s).strip()

    s = s.replace(" N ", " # ")
    s = s.replace(" NO ", " # ")
    s = s.replace(" NUM ", " # ")

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
    }
    for k, v in replacements.items():
        s = re.sub(rf"\b{k}\b", v, s)

    s = re.sub(r"\s*#\s*", " # ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_address(row: pd.Series, address_col: str, extra_col: Optional[str]) -> str:
    base = normalize_bogota_address(row.get(address_col, ""))
    extra = normalize_bogota_address(row.get(extra_col, "")) if extra_col else ""

    parts = []
    if base and base not in {"NO APLICA", "N/A", "NA"}:
        parts.append(base)
    if extra and extra not in {"NO APLICA", "N/A", "NA"}:
        parts.append(extra)

    parts.extend(["LOCALIDAD KENNEDY", "BOGOTÁ", "COLOMBIA"])
    return ", ".join([p for p in parts if p])


# -------------------------
# Geocodificación Google / Nominatim
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
        lat = float(loc["lat"])
        lon = float(loc["lng"])
        formatted = result.get("formatted_address")
        return lat, lon, formatted
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

def geocode_dataframe_within_kennedy(
    df: pd.DataFrame,
    address_col: str,
    extra_col: Optional[str],
    max_rows: int,
    cache_path: str,
    kennedy_polygon,
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
        geolocator = Nominatim(user_agent="bogota_kennedy_geocoder_streamlit")
        geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    pending = df[df["lat"].isna()].head(max_rows)

    prog = st.progress(0, text="Geocodificando (forzando dentro de Kennedy)...")
    total = len(pending) if len(pending) > 0 else 1

    for i, (idx, row) in enumerate(pending.iterrows(), start=1):
        addr1 = build_address(row, address_col, extra_col)
        if not addr1.strip():
            continue

        key1 = _hash_key(addr1)

        lat, lon = None, None
        raw = None

        if key1 in cached:
            lat, lon = cached[key1]
        else:
            if provider == "google":
                if google_api_key:
                    lat, lon, raw = google_geocode(addr1, google_api_key)
                else:
                    lat, lon, raw = None, None, None
            else:
                lat, lon = nominatim_geocode(addr1, geocode_fn)

            if (lat is None or lon is None or pd.isna(lat) or pd.isna(lon)) and fallback_to_nominatim and geocode_fn:
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
            inside = kennedy_polygon.contains(Point(float(lon), float(lat)))

        if not inside:
            base_only = normalize_bogota_address(row.get(address_col, ""))
            addr2 = ", ".join([p for p in [base_only, "KENNEDY", "BOGOTÁ", "COLOMBIA"] if p])
            key2 = _hash_key(addr2)

            if key2 in cached:
                lat2, lon2 = cached[key2]
            else:
                lat2, lon2 = None, None
                if provider == "google" and google_api_key:
                    lat2, lon2, _ = google_geocode(addr2, google_api_key)
                elif geocode_fn:
                    lat2, lon2 = nominatim_geocode(addr2, geocode_fn)

                if (lat2 is None or lon2 is None) and fallback_to_nominatim and geocode_fn and provider == "google":
                    lat3, lon3 = nominatim_geocode(addr2, geocode_fn)
                    if lat3 is not None and lon3 is not None:
                        lat2, lon2 = lat3, lon3

                df_cache = pd.concat(
                    [df_cache, pd.DataFrame([{
                        "addr_key": key2,
                        "address": addr2,
                        "lat": lat2,
                        "lon": lon2,
                        "provider": f"{provider}_retry",
                        "ts": int(time.time()),
                        "raw": None,
                    }])],
                    ignore_index=True
                )
                cached[key2] = (lat2, lon2)

            if lat2 is not None and lon2 is not None and not (pd.isna(lat2) or pd.isna(lon2)):
                if kennedy_polygon.contains(Point(float(lon2), float(lat2))):
                    lat, lon = lat2, lon2
                    inside = True

        if inside:
            df.at[idx, "lat"] = lat
            df.at[idx, "lon"] = lon
        else:
            df.at[idx, "lat"] = np.nan
            df.at[idx, "lon"] = np.nan

        prog.progress(i / total, text=f"Geocodificando {i}/{len(pending)}")

    save_cache_df(df_cache, cache_path)
    return df


# -------------------------
# Asignación punto -> UPZ (Kennedy)
# -------------------------
def points_to_upz(df_points: pd.DataFrame, upz_geojson: dict) -> pd.DataFrame:
    df = df_points.copy()
    df["UPZ_CODIGO"] = None
    df["UPZ_NOMBRE"] = None

    upz_polys = []
    for ft in upz_geojson.get("features", []):
        geom = shape(ft["geometry"])
        props = ft.get("properties", {})
        upz_polys.append((geom, props.get("CODIGO_UPZ"), props.get("NOMBRE")))

    for idx, r in df.dropna(subset=["lat", "lon"]).iterrows():
        p = Point(float(r["lon"]), float(r["lat"]))
        for geom, cod, nom in upz_polys:
            if geom.contains(p):
                df.at[idx, "UPZ_CODIGO"] = cod
                df.at[idx, "UPZ_NOMBRE"] = nom
                break

    return df


# -------------------------
# Iconografía por tipo de servicio
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
    if s in SERVICE_STYLES:
        return s
    # tolerancia básica
    if "SALUD" in s:
        return "SALUD"
    if "EDU" in s:
        return "EDUCACIÓN"
    if "EMPLE" in s or "TRABAJO" in s:
        return "EMPLEO"
    if "BIENEST" in s:
        return "BIENESTAR SOCIAL"
    if "ENTORNO" in s or "AMBIEN" in s:
        return "ENTORNO SALUDABLE"
    if "RED" in s or "APOYO" in s:
        return "REDES DE APOYO"
    if "DESAR" in s:
        return "DESARROLLO"
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
# Mapa (Kennedy + UPZ hover + puntos con iconos)
# -------------------------
def make_map(df_points: pd.DataFrame, kennedy_gj: dict, upz_gj: dict, service_col: str, icon_png_map: Dict[str, bytes]) -> folium.Map:
    pts = df_points.dropna(subset=["lat", "lon"]).copy()

    if len(pts) > 0:
        center = (float(pts["lat"].astype(float).mean()), float(pts["lon"].astype(float).mean()))
        zoom = 12
    else:
        center = BOGOTA_CENTER
        zoom = DEFAULT_ZOOM

    m = folium.Map(location=center, zoom_start=zoom, control_scale=True)

    folium.GeoJson(
        kennedy_gj,
        name="Localidad Kennedy",
        style_function=lambda feature: {
            "fillColor": "#FF0000",
            "color": "#CC0000",
            "weight": 2,
            "fillOpacity": 0.12,
        },
    ).add_to(m)

    conteo_upz = (
        pts.dropna(subset=["UPZ_CODIGO"])
        .groupby(["UPZ_CODIGO", "UPZ_NOMBRE"])
        .size()
        .reset_index(name="BENEFICIARIOS")
    )
    conteo_dict: Dict[str, int] = {str(r["UPZ_CODIGO"]): int(r["BENEFICIARIOS"]) for _, r in conteo_upz.iterrows()}

    upz_gj_aug = json.loads(json.dumps(upz_gj))
    for ft in upz_gj_aug.get("features", []):
        cod = str(ft.get("properties", {}).get("CODIGO_UPZ", ""))
        ft["properties"]["BENEFICIARIOS"] = conteo_dict.get(cod, 0)

    def upz_style(_feature):
        return {"fillColor": "#FFCC00", "color": "#666666", "weight": 1, "fillOpacity": 0.10}

    def upz_highlight(_feature):
        return {"weight": 3, "color": "#FF0000", "fillOpacity": 0.18}

    folium.GeoJson(
        upz_gj_aug,
        name="UPZ Kennedy",
        style_function=upz_style,
        highlight_function=upz_highlight,
        tooltip=folium.GeoJsonTooltip(
            fields=["NOMBRE", "CODIGO_UPZ", "BENEFICIARIOS"],
            aliases=["UPZ:", "Código UPZ:", "Beneficiarios:"],
            localize=True
        ),
    ).add_to(m)

    # Normalizar servicio
    if service_col and service_col in pts.columns:
        pts["_SERVICIO"] = pts[service_col].apply(normalize_service)
    else:
        pts["_SERVICIO"] = "OTRO"

    # Marcadores con icono por servicio
    for _, r in pts.iterrows():
        label = _safe_str(r.get("address_full", "")) or _safe_str(r.get("NOMBRE", "")) or "Sitio"
        upz_name = _safe_str(r.get("UPZ_NOMBRE", "")) or "N/D"
        servicio = _safe_str(r.get("_SERVICIO", "OTRO")) or "OTRO"

        popup = f"""
        <b>{label}</b><br>
        <b>Servicio:</b> {servicio}<br>
        <b>UPZ:</b> {upz_name}
        """

        style = SERVICE_STYLES.get(servicio, SERVICE_STYLES["OTRO"])

        # Si cargaron PNG para ese servicio, úsalo como logo
        if servicio in icon_png_map and icon_png_map[servicio] is not None:
            icon = folium.CustomIcon(
                icon_image=icon_png_map[servicio],
                icon_size=(28, 28),
                icon_anchor=(14, 14),
            )
            folium.Marker(
                location=(float(r["lat"]), float(r["lon"])),
                popup=folium.Popup(popup, max_width=350),
                icon=icon
            ).add_to(m)
        else:
            folium.Marker(
                location=(float(r["lat"]), float(r["lon"])),
                popup=folium.Popup(popup, max_width=350),
                icon=folium.Icon(color=style["color"], icon=style["icon"], prefix="fa")
            ).add_to(m)

    add_legend(m)
    folium.LayerControl().add_to(m)
    return m


# -------------------------
# UI
# -------------------------
st.title("Georreferenciar servicios sociales en Bogotá (Kennedy) + UPZ + iconos por tipo")

with st.sidebar:
    st.header("Entrada")
    uploaded = st.file_uploader("Sube tu Excel (.xlsx)", type=["xlsx"])

    st.header("Proveedor de geocodificación (solo si NO tienes lat/lon)")
    provider = st.selectbox("Proveedor", options=["google", "nominatim"], index=0)
    google_api_key = st.text_input("Google API Key (si usas Google)", type="password")
    fallback_to_nominatim = st.checkbox("Fallback a Nominatim si falla Google", value=True)

    st.header("Ejecución")
    max_rows = st.number_input("Máximo de filas a geocodificar por ejecución", min_value=10, max_value=5000, value=300, step=50)
    run_geocode = st.checkbox("Ejecutar geocodificación ahora (solo si NO tienes lat/lon)", value=False)

    st.header("Logos PNG por tipo (opcional)")
    st.caption("Si subes un PNG, se usará como ícono en el mapa para ese servicio.")
    icon_png_map = {}
    for k in SERVICE_STYLES.keys():
        if k == "OTRO":
            continue
        f = st.file_uploader(f"Logo PNG para {k}", type=["png"], key=f"logo_{k}")
        icon_png_map[k] = f.getvalue() if f is not None else None

    st.header("Salida")
    enable_download = st.checkbox("Permitir descarga del Excel enriquecido", value=True)

if not uploaded:
    st.info("Sube un Excel para comenzar.")
    st.stop()

df = pd.read_excel(uploaded)
st.subheader("Vista previa")
st.dataframe(df.head(25), use_container_width=True)

cols = df.columns.tolist()

# Detectar lat/lon si vienen en el Excel (variantes comunes)
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
extra_col_choice = st.selectbox("Columna opcional de complemento (barrio/torre/etc.)", options=extra_col_opt, index=0)
extra_col = None if extra_col_choice == "(ninguna)" else extra_col_choice

service_col_opt = ["(ninguna)"] + cols
service_col_choice = st.selectbox("Columna de TIPO DE SERVICIO (SALUD/EDUCACIÓN/…)", options=service_col_opt, index=0)
service_col = None if service_col_choice == "(ninguna)" else service_col_choice

df = df.copy()

# Si ya hay lat/lon en el Excel, úsalos (y renómbralos internamente a lat/lon)
if lat_col and lon_col:
    df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    st.success(f"Detecté coordenadas en tu Excel: lat = '{lat_col}', lon = '{lon_col}'. No es necesario geocodificar.")
else:
    df["lat"] = np.nan
    df["lon"] = np.nan
    st.warning("No detecté columnas lat/lon. Si quieres coordenadas, activa geocodificación.")

df["address_full"] = df.apply(lambda r: build_address(r, address_col, extra_col), axis=1)

with st.spinner("Cargando polígonos oficiales (Kennedy + UPZ de Kennedy)..."):
    kennedy_gj = kennedy_geojson()
    upz_gj = upz_kennedy_geojson()

if not kennedy_gj.get("features"):
    st.error("No se pudo cargar el polígono de Kennedy.")
    st.stop()

kennedy_poly = shape(kennedy_gj["features"][0]["geometry"])

# Ejecutar geocodificación SOLO si el usuario lo pidió y no hay lat/lon
if run_geocode:
    if lat_col and lon_col:
        st.info("Tu Excel ya trae lat/lon; no se ejecutó geocodificación.")
    else:
        if provider == "google" and not google_api_key:
            st.error("Seleccionaste Google pero no ingresaste API Key.")
            st.stop()

        with st.spinner("Geocodificando y forzando que todo quede dentro de Kennedy..."):
            df = geocode_dataframe_within_kennedy(
                df=df,
                address_col=address_col,
                extra_col=extra_col,
                max_rows=int(max_rows),
                cache_path=CACHE_PATH,
                kennedy_polygon=kennedy_poly,
                provider=provider,
                google_api_key=google_api_key,
                fallback_to_nominatim=fallback_to_nominatim,
            )
        st.success("Geocodificación completada (o parcial según el límite).")

# Mantener solo puntos dentro de Kennedy (si hay coords)
def keep_only_inside_kennedy(df_in: pd.DataFrame, poly) -> pd.DataFrame:
    df2 = df_in.copy()
    mask = []
    for _, r in df2.iterrows():
        lat, lon = r.get("lat"), r.get("lon")
        ok = False
        if lat is not None and lon is not None and not (pd.isna(lat) or pd.isna(lon)):
            ok = poly.contains(Point(float(lon), float(lat)))
        mask.append(ok)
    # si no tiene coords, queda False
    df2.loc[~pd.Series(mask, index=df2.index), ["lat", "lon"]] = np.nan
    return df2

df = keep_only_inside_kennedy(df, kennedy_poly)

# Asignar UPZ
df = points_to_upz(df, upz_gj)

total = len(df)
with_coords = df.dropna(subset=["lat", "lon"]).shape[0]
st.write(f"Registros: **{total:,}** | Dentro de Kennedy (con coords): **{with_coords:,}**")

st.subheader("Mapa: Kennedy + UPZ (hover) + puntos por tipo (iconos/logos)")
m = make_map(df, kennedy_gj, upz_gj, service_col=service_col or "", icon_png_map=icon_png_map)
st_folium(m, width=1100, height=650)

# Conteo por UPZ (solo puntos dentro de Kennedy)
st.subheader("Conteo por UPZ (solo puntos dentro de Kennedy)")
pts = df.dropna(subset=["lat", "lon"]).copy()
conteo = (
    pts.dropna(subset=["UPZ_CODIGO"])
    .groupby(["UPZ_CODIGO", "UPZ_NOMBRE"])
    .size()
    .reset_index(name="beneficiarios")
    .sort_values("beneficiarios", ascending=False)
)
st.dataframe(conteo, use_container_width=True)

# Conteo por tipo de servicio
if service_col and service_col in df.columns:
    st.subheader("Conteo por tipo de servicio (solo puntos dentro de Kennedy)")
    df["_SERVICIO"] = df[service_col].apply(normalize_service)
    conteo_serv = (
        df.dropna(subset=["lat", "lon"])
        .groupby("_SERVICIO")
        .size()
        .reset_index(name="sitios")
        .sort_values("sitios", ascending=False)
    )
    st.dataframe(conteo_serv, use_container_width=True)

st.subheader("Resultados (muestra)")
out_cols = [address_col] + ([extra_col] if extra_col else []) + ([(service_col or "")] if service_col else []) + ["address_full", "lat", "lon", "UPZ_CODIGO", "UPZ_NOMBRE"]
out_cols = [c for c in out_cols if c and c in df.columns] + ["lat", "lon", "UPZ_CODIGO", "UPZ_NOMBRE"]
out_cols = list(dict.fromkeys(out_cols))  # unique preserve order
st.dataframe(df[out_cols].head(200), use_container_width=True)

if enable_download:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="geocodificado")
    st.download_button(
        "Descargar Excel enriquecido (coords + UPZ + tipo)",
        data=output.getvalue(),
        file_name="kennedy_servicios_georreferenciados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
