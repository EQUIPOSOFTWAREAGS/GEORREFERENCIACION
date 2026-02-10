# app.py
import io
import json
import re
import pathlib
from typing import Optional, Dict

import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

from shapely.geometry import shape, Point


# -------------------------
# Configuración
# -------------------------
st.set_page_config(page_title="Georreferenciación - Quebradanegra (Cundinamarca)", layout="wide")

MUNICIPIO_NOMBRE = "Quebradanegra"
DEPARTAMENTO_NOMBRE = "Cundinamarca"

QUEBRADANEGRA_CENTER = (5.1175, -74.4793)
DEFAULT_ZOOM = 13

# ✅ Excel predeterminado en la carpeta principal (junto a app.py)
DEFAULT_EXCEL_PATH = pathlib.Path("UBICACIONES.xlsx")

# ✅ GeoJSON opcional en la carpeta principal (junto a app.py)
# Si NO lo tienes, el mapa igual funciona pero no valida "dentro del municipio".
LOCAL_GEOJSON_PATH = pathlib.Path("quebradanegra.geojson")


# -------------------------
# Helpers
# -------------------------
def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_geojson_local(path: pathlib.Path) -> dict:
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}
    try:
        gj = json.loads(path.read_text(encoding="utf-8"))
        if gj.get("type") == "Feature":
            gj = {"type": "FeatureCollection", "features": [gj]}
        return gj
    except Exception:
        return {"type": "FeatureCollection", "features": []}


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

    if s in SERVICE_STYLES:
        return s
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


def keep_only_inside_polygon(df_in: pd.DataFrame, poly) -> pd.DataFrame:
    df = df_in.copy()
    ok = []
    for _, r in df.iterrows():
        lat, lon = r.get("lat"), r.get("lon")
        inside = False
        if lat is not None and lon is not None and not (pd.isna(lat) or pd.isna(lon)):
            inside = poly.contains(Point(float(lon), float(lat)))
        ok.append(inside)
    df.loc[~pd.Series(ok, index=df.index), ["lat", "lon"]] = np.nan
    return df


def make_map(df_points: pd.DataFrame, municipio_gj: dict) -> folium.Map:
    pts = df_points.dropna(subset=["lat", "lon"]).copy()

    if len(pts) > 0:
        center = (float(pts["lat"].mean()), float(pts["lon"].mean()))
        zoom = 13
    else:
        center = QUEBRADANEGRA_CENTER
        zoom = DEFAULT_ZOOM

    m = folium.Map(location=center, zoom_start=zoom, control_scale=True)

    # Polígono (si existe)
    if municipio_gj.get("features"):
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

    for _, r in pts.iterrows():
        servicio = normalize_service(r.get("TIPO_SERVICIO", "OTRO"))
        style = SERVICE_STYLES.get(servicio, SERVICE_STYLES["OTRO"])

        label = _safe_str(r.get("SITIO", "")) or _safe_str(r.get("Direccion", "")) or "Sitio"

        popup = f"""
        <b>{label}</b><br>
        <b>Servicio:</b> {servicio}<br>
        <b>Lat:</b> {float(r["lat"]):.6f}<br>
        <b>Lon:</b> {float(r["lon"]):.6f}
        """

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
st.title("Servicios sociales georreferenciados - Quebradanegra (Cundinamarca)")

with st.sidebar:
    st.header("Datos")
    uploaded = st.file_uploader("Sube otro Excel (opcional)", type=["xlsx"])
    st.caption("Si NO subes nada, se usa UBICACIONES.xlsx en la carpeta principal.")

    st.header("Polígono municipal (opcional)")
    uploaded_geojson = st.file_uploader("Subir GeoJSON (Feature/FeatureCollection)", type=["geojson", "json"])

    st.header("Salida")
    enable_download = st.checkbox("Permitir descarga", value=True)

# -------------------------
# 1) Cargar Excel: upload o default (SIN carpeta data/)
# -------------------------
if uploaded is not None:
    df = pd.read_excel(uploaded)
    st.info("Usando Excel cargado por el usuario.")
else:
    if not DEFAULT_EXCEL_PATH.exists():
        st.error("No se encontró UBICACIONES.xlsx en la carpeta principal (junto a app.py).")
        st.stop()
    df = pd.read_excel(DEFAULT_EXCEL_PATH)
    st.success("Usando Excel predeterminado del aplicativo (UBICACIONES.xlsx).")

df = df.copy()

# -------------------------
# 2) Normalizar columnas (tu archivo real)
# -------------------------
# En tu archivo el tipo está en "Unnamed: 2"
if "TIPO_SERVICIO" not in df.columns:
    if "Unnamed: 2" in df.columns:
        df = df.rename(columns={"Unnamed: 2": "TIPO_SERVICIO"})
    elif "TIPO" in df.columns:
        df = df.rename(columns={"TIPO": "TIPO_SERVICIO"})

# Asegurar lat/lon
df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")

# Limpiar coordenadas imposibles (Colombia)
df.loc[(df["lat"] < -5) | (df["lat"] > 15), ["lat", "lon"]] = np.nan
df.loc[(df["lon"] < -82) | (df["lon"] > -66), ["lat", "lon"]] = np.nan

# Servicio normalizado
df["TIPO_SERVICIO"] = df.get("TIPO_SERVICIO", "OTRO").apply(normalize_service)

# -------------------------
# ✅ 3) MOSTRAR PRIMERO LOS DATOS (lo que pediste)
# -------------------------
st.subheader("Datos cargados (primero)")
st.dataframe(df, use_container_width=True)

# -------------------------
# 4) Cargar polígono: upload -> local (ambos en raíz)
# -------------------------
municipio_gj = {"type": "FeatureCollection", "features": []}

if uploaded_geojson is not None:
    try:
        municipio_gj = json.loads(uploaded_geojson.getvalue().decode("utf-8"))
        if municipio_gj.get("type") == "Feature":
            municipio_gj = {"type": "FeatureCollection", "features": [municipio_gj]}
        st.success("Polígono cargado desde archivo subido.")
    except Exception:
        st.error("GeoJSON subido inválido.")
        st.stop()
else:
    municipio_gj = load_geojson_local(LOCAL_GEOJSON_PATH)
    if municipio_gj.get("features"):
        st.info("Usando polígono local: quebradanegra.geojson (en la carpeta principal).")
    else:
        st.warning("No hay polígono local en raíz. El mapa funcionará, pero NO validará 'dentro del municipio'.")

# -------------------------
# 5) Validación dentro del polígono (si existe)
# -------------------------
if municipio_gj.get("features"):
    municipio_poly = shape(municipio_gj["features"][0]["geometry"])
    df = keep_only_inside_polygon(df, municipio_poly)

total = len(df)
with_coords = df.dropna(subset=["lat", "lon"]).shape[0]
st.write(f"Registros: **{total:,}** | Con coordenadas válidas: **{with_coords:,}**")

# -------------------------
# 6) Filtros rápidos
# -------------------------
st.subheader("Filtros")
col1, col2 = st.columns([1, 1])

with col1:
    servicios = sorted(df["TIPO_SERVICIO"].dropna().unique().tolist())
    sel_serv = st.multiselect("Tipos de servicio", servicios, default=servicios)

with col2:
    q = st.text_input("Buscar por SITIO", value="").strip().lower()

df_f = df[df["TIPO_SERVICIO"].isin(sel_serv)].copy()
if q and "SITIO" in df_f.columns:
    df_f = df_f[df_f["SITIO"].astype(str).str.lower().str.contains(q, na=False)]

# -------------------------
# 7) Conteo y mapa
# -------------------------
st.subheader("Conteo por tipo (según filtros)")
conteo = (
    df_f.dropna(subset=["lat", "lon"])
    .groupby("TIPO_SERVICIO")
    .size()
    .reset_index(name="sitios")
    .sort_values("sitios", ascending=False)
)
st.dataframe(conteo, use_container_width=True)

st.subheader("Mapa")
m = make_map(df_f, municipio_gj)
st_folium(m, width=1100, height=650)

# -------------------------
# 8) Descarga
# -------------------------
if enable_download:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_f.to_excel(writer, index=False, sheet_name="filtrado")
    st.download_button(
        "Descargar Excel (filtrado)",
        data=output.getvalue(),
        file_name="quebradanegra_servicios_filtrado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
