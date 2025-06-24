import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely.geometry import MultiPolygon, Point
from pyproj import Transformer
from sklearn.cluster import DBSCAN
import alphashape

# -------------------------------------------------
# Streamlit page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Field Boundary Detection & Activity Classification",
    layout="wide",
)

st.title("Field Boundary Detection & Activity Classification from GPS + ERPM Data")

# -------------------------------------------------
# Constants – tweak here for different datasets
# -------------------------------------------------
DBSCAN_EPS_METERS = 3.1          # Radius for clustering (meters)
DBSCAN_MIN_SAMPLES = 18          # Minimum points per cluster
ALPHA_SHAPE_PARAM = 0.1          # Tightness of hull fit (0 → convex hull)
ACRE_TO_SQ_METERS = 4046.86      # 1 Acre in m²
GUNTHA_TO_SQ_METERS = ACRE_TO_SQ_METERS / 40  # 1 Guntha in m² (Indian land unit)

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def load_and_clean_data(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """Read CSV and apply basic cleaning/validation."""
    df = pd.read_csv(uploaded_file, low_memory=False)
    required_cols = {"timestamp", "latitude", "longitude", "ERPM"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must contain columns: {required_cols}")
        return pd.DataFrame()

    # keep only required columns in correct order
    df = df[list(required_cols)].copy()

    # drop NaNs & duplicates
    df = df.dropna().drop_duplicates()

    # Cast ERPM to numeric, timestamp to datetime
    df["ERPM"] = pd.to_numeric(df["ERPM"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna()

    # sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def transform_to_xy(df: pd.DataFrame) -> None:
    """Add projected X/Y columns (EPSG:24378) for distance/area math."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:24378", always_xy=True)
    df["x"], df["y"] = transformer.transform(df["longitude"].values, df["latitude"].values)


def classify_activity(erpm_series: pd.Series) -> tuple[str, int, int]:
    """Return activity label and counts of positive/negative ERPM readings."""
    num_pos = (erpm_series > 0).sum()
    num_neg = (erpm_series < 0).sum()
    delta = num_pos - num_neg
    if delta > 0:
        label = "Earthing"
    elif delta < 0:
        label = "Weeding"
    else:
        label = "Undetermined / Balanced"
    return label, int(num_pos), int(num_neg)


def polygon_to_latlon_coords(polygon, transformer_inv):
    """Convert polygon exterior coords from projected → lat/lon for mapping."""
    return [transformer_inv.transform(x, y)[::-1] for x, y in polygon.exterior.coords]

# -------------------------------------------------
# Main App Flow
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV containing columns: 'timestamp', 'latitude', 'longitude', 'ERPM'",
    type=["csv"],
)

if uploaded_file:
    df = load_and_clean_data(uploaded_file)

    if df.empty:
        st.stop()

    st.success(f"Loaded {len(df):,} valid GPS points.")

    # Coordinate transform
    transform_to_xy(df)

    coords = df[["x", "y"]].values

    if len(coords) < DBSCAN_MIN_SAMPLES:
        st.warning("Not enough points for clustering – please upload more data.")
        st.stop()

    # -------------------------------------------------
    # DBSCAN clustering
    # -------------------------------------------------
    db = DBSCAN(eps=DBSCAN_EPS_METERS, min_samples=DBSCAN_MIN_SAMPLES, metric="euclidean")
    df["cluster"] = db.fit_predict(coords)

    cluster_ids = sorted(c for c in set(df["cluster"]) if c != -1)
    if not cluster_ids:
        st.info("No distinct field clusters detected.")
        st.stop()

    st.write(f"Detected {len(cluster_ids)} potential field(s). Building boundaries…")

    # Prepare map
    map_center = [df["latitude"].mean(), df["longitude"].mean()]
    fmap = folium.Map(location=map_center, zoom_start=17, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Satellite",
        name="Esri Satellite",
        overlay=True,
        control=True,
    ).add_to(fmap)

    transformer_inv = Transformer.from_crs("EPSG:24378", "EPSG:4326", always_xy=True)

    results = []
    hull_counter = 0

    for cid in cluster_ids:
        cluster_df = df[df["cluster"] == cid]
        if len(cluster_df) < 4:
            # Need at least 4 to form an area
            continue

        # Build alphashape boundary
        hull = alphashape.alphashape(list(zip(cluster_df["x"], cluster_df["y"])), ALPHA_SHAPE_PARAM)
        if hull.is_empty or hull.geom_type not in ["Polygon", "MultiPolygon"]:
            continue

        # Every hull may be MultiPolygon (rare) – iterate
        polygons = hull.geoms if isinstance(hull, MultiPolygon) else [hull]

        for poly in polygons:
            hull_counter += 1
            area_sq_m = poly.area
            area_gunthas = area_sq_m / GUNTHA_TO_SQ_METERS

            activity, n_pos, n_neg = classify_activity(cluster_df["ERPM"])

            # Lat/lon coords for mapping
            latlon_coords = polygon_to_latlon_coords(poly, transformer_inv)
            folium.Polygon(
                locations=latlon_coords,
                color="green",
                fill=True,
                fill_opacity=0.35,
                tooltip=f"Field {hull_counter}: {area_gunthas:.2f} gunthas (\n{activity})",
            ).add_to(fmap)

            # Centroid marker
            centroid_latlon = transformer_inv.transform(*poly.centroid.coords[0])[::-1]
            folium.Marker(centroid_latlon, popup=f"Field {hull_counter}").add_to(fmap)

            # Timestamp span
            ts_first = cluster_df["timestamp"].iloc[0]
            ts_last = cluster_df["timestamp"].iloc[-1]

            results.append({
                "Field #": hull_counter,
                "Area (gunthas)": round(area_gunthas, 2),
                "Activity": activity,
                "+ ERPM": n_pos,
                "- ERPM": n_neg,
                "First ts": ts_first,
                "Last ts": ts_last,
            })

    # -------------------------------------------------
    # Map & Results output
    # -------------------------------------------------
    with st.container():
        st.subheader("Field Map")
        st_folium(fmap, height=600, width=900)

    if results:
        st.subheader("Detected Fields – Summary")
        results_df = pd.DataFrame(results).sort_values("Area (gunthas)", ascending=False)
        st.dataframe(results_df, use_container_width=True)
    else:
        st.info("No valid field boundaries could be constructed from the clusters.")
else:
    st.info("📂 Please upload a CSV file to begin analysis.")
