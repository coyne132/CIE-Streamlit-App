# ===================================
# Imports
# GIS, visualization, and Streamlit utilities required for spatial
# ===================================

import geopandas as gpd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely import wkb
from shapely.affinity import scale
from shapely.geometry import Point
from math import radians, cos, sin, asin, sqrt
from contextlib import contextmanager
from textwrap import dedent
import json
import matplotlib.pyplot as plt
import contextily as ctx

# ===================================
# Page Setup
# Configures Streamlit page layout and displays a high-level disclaimer clarifying the dashboard’s decision-support role.
# ===================================

st.set_page_config(
	page_title = "Community Information Exchange (CIE) Navigation Center Siting Dashboard",
	layout="wide"
)

st.markdown(
	"""
	This dashboard supports prioritization of candidate areas for Community Information Exchange (CIE) Navigation Center siting. It does not determine final site selection. 
	"""
)

# ===================================
# Theme Colors (CIE-Inspired)
# Centralized color palette used consistently across maps, charts, tables, and UI components for clarity and visual hierarchy.
# ===================================

COLORS = {
	#Neutrals:
	"white": "#FFFFFF",
	"page_bg":"#EE52F5",
	"panel_bg": "#FFFFFF",
	"border": "#D6DEE4",
	"text_primary": "#2B2F36",
	"text_secondary": "#5E6A73",

	#Primary accents:
	"teal": "#1CBABD",
	"teal_dark": "#1F5855",

	#Highlights:
	"amber": "#E3AB4C",
	"jasmine": "#F6D98A",

	#Soft fills:
	"azure_mist": "#D4E7E7",
	"lavender_blush": "#F8EAEA",

	#Restricted (charts only)
	"grapefruit": "#FF7681",
	"amaranth": "#8A1232",
}

# ===================================
# Page Style
# Custom CSS to improve typography, spacing, card layout, and visual consistency across dashboard component
# ===================================

st.markdown(
	"""
	<style>

	/* Base Text*/
	html, body, .stApp {
		font-size: 14px;
		color: #2B2F36
	}

	/* Page title*/
	h1 {
		font-size: 1.6rem;
		font-weight: 600;
		color: #1F2A44;
		margin-bottom: 0.5rem;
	}

	/* section headers */
	h2 { 
		font-size: 1.2rem;
		font-weight: 600;
		color: #1F2A44;
		margin-top: 1rem;
		margin-bottom: 0.25rem;	
	}

	/* subsection headers */
	strong {
		font-weight: 600;
		color: #1F2A44;
	}

	/* body text */ 
	p, li, span, div {
	}

	/* charts */
	div[data-testid="stPlotlyChart"],
	iframe{
		border-radius: 6px;
		background: white;
		border-color: #000000;
	}

	/* app background */
	.stApp {
		background-color: #FBF1DA; 
		color: #2B2F36;
	}

	/* header */
	header[data-testid="stHeader"] {
		background-color: #F2F4F7;
		border-bottom: 1px solid #E1E5E8;
	}
	
	/* Sidebar */
	section[data-testid="stSidebar"] {
		background-color: #FFFFFF;
	}
	
	section[data-testid="stSidebar"] h1,
	section[data-testid="stSidebar"] h2,
	section[data-testid="stSidebar"] h3 {
		color: #000000;
	}
	
	
	/* Dataframe */ 
	div[data-testid="stVerticalBlock"] > div:has(> h2):has(.stDataFrame) {
		background-color: #FFFBF4;
		padding: 1rem 1.2rem;
		border-radius: 10px;
		box-shadow: 0 1px 8px rgba(15, 23, 42, 0.06);
		margin-bottom: 1.25rem;
		box-sizing: border-box;
		max-width: 100%;
	}
	div[data-testid="stDataFrame"] tbody tr {
		height: 32px;
	}
	
	div[data-testid="stDataFrame"] {
		border-radius: 8px;
		border: 1px solid #E5E7EB;
		background: #FFFFFF;
	}

	.element-container:has(.stDataFrame) > div {
		background: #FFFBF4;
		padding: 1rem;
		border-radius: 10px;
		box-shadow: 0 1px 8px rgba(15, 23, 50, 0.06);
		margin-left: auto;
		margin-right: auto;
	}

	/* Charts */
	div[data-testid="stVerticalBlock"] > div:has(> h2):has(.stPlotlyChart) {
		background-color: #FFFBF4;
		padding: 1.2rem 1.4rem;
		border-radius: 10px;
		box-shadow: 0 1px 10px rgba(15, 23, 42, 0.07);
		margin-bottom: 1.25rem;
		box-sizing: border-box;
		max-width: 100%;
	}

	.element-container:has(.stPlotlyChart) > div {
		background: #FFFBF4;
		padding: 1rem;
		border-radius: 10px;
		box-shadow: 0 1px 8px rgba(15, 23, 50, 0.06);			
		max-width: 850px;  
	}
	/* Maps */
	div[data-testid="stVerticalBlock"] > div:has(> iframe) {
		background-color: #FFFFFF;
		padding: 1rem;
		border-radius: 10px;
		box-shadow: 0 3px 12px rgba(15, 23, 42, 0.10);
		margin-bottom: 1.25rem;	
		box-sizing: border-box;
		max-width: 100%;
	}
		
	.map-card {
		background-color: #ffffff;
		border-radius: 10px;
		padding: 0;
		box-shadow: 0 3px 12px rgba(15, 23, 42, 0.10);
	}
	
	/* App */ 
	.app-card {
		background-color: #FFFBF4;
		border-radius: 10px;
		padding: 1.1rem 1.3rem;
		margin-bottom: 1.2rem;
		border: 1px solid rgba(0,0,0,0.03);
		box-shadow: 0 1px 8px rgba(15, 23, 42, 0.06);
	}

	.app-card-title {
		font-weight: 500;
		font-size: 1.5rem;
		margin-bottom: 0.5rem;
		color: #1F2A44;
	}
	
	.app-card.primary {		
		border-left: 6px solid #E3AB4C; /* your amber */
		box-shadow: 0 3px 12px rgba(0,0,0,0.08);
		max-width: 2285px;
		margin-left: 0;
		margin-right: auto;
	}
	.app-root {
		max-width: 1200px;
		margin: 0 auto;
		padding: 0 1rem;
	}
	
	/* Summary Overview */
	.summary-grid {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 1.05rem;
		margin-top: 0.75rem;
	}

	.summary-label {
		font-size: 0.65rem;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: #6B7280; 
	}

	.summary-value {
		font-weight: 600;
		font-size: 0.95rem;
		color: #111827;
	}

	
	.section-divider {
		height: 1px;
		background: linear-gradient(
		to right,
		transparent,
		#E5E7EB,
		transparent
		);
  		margin: 1.5rem 0;
	}
	
	/* Select Box */
	.stSelectbox {
		margin-top: 0.5rem;
		margin-bottom: 1.25rem;
	}

	div[data-baseweb="select"] > div:hover {
		border-color: #C9973A;
	}

	div[data-baseweb="select"] > div:focus-within {
		outline: none;
		border-color: #E3AB4C;
		box-shadow: 0 0 0 2px rgba(227, 171, 76, 0.25);
	}

 	div[data-baseweb="select"] > div {
		background-color: #FFFBF4 !important;
   	}

	div[role="option"]:hover {
		background-color: #FFFBF4 !important;
	}

	div[aria-selected="true"] {
		background-color: #FFF1D6 !important;
		font-weight: 500;
	}

	<style>
	""",
	unsafe_allow_html=True
)

st.markdown('<div class="app-root">', unsafe_allow_html=True)
# ===================================
# Load Data (cached)
# Loads all required datasets using Streamlit caching to improve performance and avoid redundant disk reads.
# ===================================

#Define function to load acs_pca data:
@st.cache_data
def load_acs_pca():
	return gpd.read_parquet("data/acs_pca.parquet")

#Define function to load tract_dsh data:
@st.cache_data
def load_tract_dsh_lookup():
	return pd.read_parquet("data/tract_dsh_lookup.parquet")

#Define function to load acs_prioritization data:
@st.cache_data
def load_acs_prioritization():
	return pd.read_parquet("data/acs_prioritization.parquet")

#Define function to load public_transit:
@st.cache_data
def load_public_transit():
	return pd.read_parquet("data/public_transit.parquet")

#Define function to load public_libraries:
@st.cache_data
def load_public_libraries():
	return pd.read_parquet("data/libraries_export.parquet")

#Define function to load zipcodes:
@st.cache_data
def load_zipcodes():
	return gpd.read_file("data/franklin_zcta.geojson").to_crs(epsg=4326)

#Load acs_pca data:
acs_pca = load_acs_pca()

#Load tract_dsh data:
tract_dsh_lookup = load_tract_dsh_lookup()

#Load acs_prioritization data:
acs_prioritization = load_acs_prioritization()

#Load public_transit:
public_transit = load_public_transit()

#Load public_libraries:
public_libraries = load_public_libraries()

#Load zipcodes:
zips = load_zipcodes()

# ===================================
# Working Dataframe 
# Assembles the core analysis dataframe by merging structural access scores with prioritization variables and geographic context.
# ===================================
df = acs_pca.copy()

#merge prioritization variables:
df = df.merge(
	acs_prioritization,
	on="GEOID",
	how="left",
	validate="one_to_one"
)

#Add DSH flag:
df = df.merge(
	tract_dsh_lookup[["GEOID"]].drop_duplicates().assign(dsh_flag=1),
	on="GEOID",
	how="left"
)

df["dsh_flag"] = df["dsh_flag"].fillna(0).astype(int)

df = df.merge(
	public_transit[["GEOID"]].drop_duplicates().assign(has_bus_stop=1),
	on="GEOID",
	how="left"
)

df["has_bus_stop"] = df["has_bus_stop"].fillna(0).astype(int)

def make_tract_short(geoid):
	return str(geoid).replace("39049", "", 1)

df["tract_short"] = (	
	df["GEOID"]
	.apply(make_tract_short)
)

tract_dsh_lookup["tract_short"] = (	
	tract_dsh_lookup["GEOID"]
	.astype(str)
	.str.replace("^39049", "", regex=True)
)

# ===================================
# Sidebar Controls
# User inputs for scenario exploration, including number of results, priority focus, alpha weighting, and optional anti-clustering.
# ===================================
if "sidebar_open" not in st.session_state:
	st.session_state.sidebar_open = True
if "has_run" not in st.session_state:
	st.session_state.has_run = False
if not st.session_state.sidebar_open:
	if st.button("Adjust Filters"):
		st.session_state.sidebar_open = True
		
#Add sidebar control title:
st.sidebar.header("Filter and Prioritization Options")

#Add option to adjust number of results:
with st.sidebar.form("scenario_controls"):
	top_n_input = st.sidebar.selectbox(
		"Number of tracts to display",
		[5, 10, 20],
		index=0,
		help="Controls how many of the highest-ranked census tracts are displayed in the results and map."
	)

	with st.sidebar.expander("Advanced Prioritization (Optional)"):
		priority_focus_input = st.radio(
			"Select a priority focus",
			options=[
				"None (Structural Need Only)",
				"Families & Children",
				"Early Childhood Vulnerability",
				"Uninsured Youth",
				"Aging Population",
				"Maternal & Infant Health",
				"Public Transportation Access"
			],
			help="Applies additional weighting to emphasize specific populations or access needs. Select 'None' to prioritize overall structural barriers only."
		)
		alpha_input = st.slider(
			"Balance data-driven need vs stakeholder priorities",
			min_value = 0.0,
			max_value = 1.0,
			value = 0.5,
			step = 0.1,
			help=(
				"α = 0 → Structural Access Disparity Score reflects structural access only (PCA) | "
				"α = 1 → Structural Access Disparity Score reflects stakeholder priorities only | "
				"Values in between blend both."
			)
		)
		enforce_min_distance_input = st.checkbox(
			"Enforce minimum distance between results (1.0 miles)",
			value = False
		)
	
	run_model = st.form_submit_button("Apply Changes")
	
	if run_model:
		st.session_state.top_n = top_n_input
		st.session_state.priority_focus = priority_focus_input
		st.session_state.alpha = alpha_input
		st.session_state.enforce_min_distance = enforce_min_distance_input
		st.session_state.has_run = True
		st.session_state.sidebar_open = False
	
	if not st.session_state.has_run:
		st.info("Adjust options and click **Apply Changes** to update results.")
		st.stop()
if not st.session_state.sidebar_open:
	st.markdown(
		"""
		<style>
		section[data-testid="stSidebar"] {
			display: none;
		}
		</style>
		""",
		unsafe_allow_html = True
	)

# ===================================
# Prioritization and Boosting Logic
# Computes a hybrid score blending PCA-based structural need with stakeholder priorities, then applies optional ranking-only boosts.
# ===================================

df["hybrid_score"] = (
	st.session_state.alpha * df["stakeholder_barrier_score_norm"]
	+ (1 - st.session_state.alpha) * df["pca_barrier_score_norm"] 
)

df["baseline_rank"] = (
	df["hybrid_score"]
	.rank(method="dense", ascending=False)
	.astype(int)
)
df["adjusted_score"] = df["hybrid_score"]
BOOST = .10

if st.session_state.priority_focus == "Families & Children":
	threshold = df["pct_hh_under18"].median()
	df.loc[
		df["pct_hh_under18"] > threshold,
		"adjusted_score"
	] *= (1 + BOOST)

elif st.session_state.priority_focus == "Early Childhood Vulnerability":
	threshold = df["pct_families_poverty_under5"].median()
	df.loc[
		df["pct_families_poverty_under5"] > threshold,
		"adjusted_score"
	] *= (1 + BOOST)

elif st.session_state.priority_focus == "Uninsured Youth":
	threshold = df["pct_no_insurance_under19"].median()
	df.loc[
		df["pct_no_insurance_under19"] > threshold,
		"adjusted_score"
	] *= (1 + BOOST)

elif st.session_state.priority_focus == "Aging Population":
	threshold = df["pct_hh_over65"].median()
	df.loc[
		df["pct_hh_over65"] > threshold,
		"adjusted_score"
	] *= (1 + BOOST)

elif st.session_state.priority_focus == "Maternal & Infant Health":
	threshold = df["pct_births_women_below200poverty"].median()
	df.loc[
		df["pct_births_women_below200poverty"] > threshold,
		"adjusted_score"
	] *= (1 + BOOST)

elif st.session_state.priority_focus == "Public Transportation Access":
	df.loc[
		(df["has_bus_stop"] == 0),
		"adjusted_score"
	] *= (1 + BOOST)

# ===================================
# Scenario Ranking Logic
# Produces final scenario ranks based on adjusted scores using dense ranking to avoid gaps.
# ===================================
df = df.sort_values("adjusted_score", ascending = False)
df["priority_rank"] = (
	df["adjusted_score"]
	.rank(method="dense", ascending=False)
	.astype(int)
)
	
# ===================================
# Anticlustering Logic
# Optional geographic constraint enforcing minimum distance between selected tracts to encourage spatial coverage without altering scores.
# ===================================

def apply_anti_clustering(
	gdf,
	rank_col,
	top_n,
	min_distance_miles = 1.0
):
	"""
	Selects top-ranked tracts while enforcing a minimum distance between selected tract centroids.
	"""
	gdf = gdf.sort_values(rank_col)
	gdf_proj = gdf.to_crs(epsg = 3857)
	
	min_distance_m = min_distance_miles * 1609.34
	
	selected = []
	for _, row in gdf_proj.iterrows():
		if len(selected) >= st.session_state.top_n:
			break
			
		if all(
			row.geometry.distance(sel.geometry) >= min_distance_m
			for sel in selected
		):
			selected.append(row)
	result = gpd.GeoDataFrame(
		selected, crs=gdf_proj.crs
	)
	return result.to_crs(gdf.crs)

if st.session_state.enforce_min_distance:
	top_df = apply_anti_clustering(
		df,
		rank_col = "priority_rank",
		top_n = st.session_state.top_n,
		min_distance_miles=1.0
	)
else:
	top_df = df.head(st.session_state.top_n)

top_df["tract_short"] = top_df["GEOID"].apply(make_tract_short)

st.markdown('<div class="app-root">', unsafe_allow_html=True)

# ===================================
# Row 1 - Scenario Summary
# Displays current scenario settings (focus, alpha, top N, anti-clustering) to ensure transparency and reproducibility
# ===================================
st.markdown(
"""<div class="app-card primary">
  <div class="app-card-title">Scenario Overview</div>
  <div class="summary-grid">
    <div>
      <div class="summary-label">FOCUS</div>
      <div class="summary-value">{priority}</div>
    </div>
    <div>
      <div class="summary-label">TOP TRACTS</div>
      <div class="summary-value">{top_n}</div>
    </div>
    <div>
      <div class="summary-label">ALPHA</div>
      <div class="summary-value">{alpha}</div>
    </div>
    <div>
      <div class="summary-label">ANTI‑CLUSTERING</div>
      <div class="summary-value">{anti}</div>
    </div>
  </div>
</div>""".format(
    priority=st.session_state.priority_focus,
    top_n=st.session_state.top_n,
    alpha=st.session_state.alpha,
    anti="Enabled" if st.session_state.enforce_min_distance else "Disabled"
),
unsafe_allow_html=True
)

# ===================================
# Row 2 - Main Layout (3 Columns)
# Three-column layout combining tract inspection, geographic context, and ranked results for side-by-side interpretatio
# ===================================
left_col, center_col, right_col = st.columns([3, 5, 5.5])

# ===================================
# Left Column - DSH partners
# Enables selection of a top-ranked census tract and displays existing DSH partners and nearby public libraries.
# ===================================

def parse_coord(coord):
	if pd.isna(coord):
		return None
	coord = coord.strip()
	sign = -1 if any (d in coord for d in ["S", "W"]) else 1
	
	coord = (
		coord
		.replace("°", "")
		.replace("N", "")
		.replace("S", "")
		.replace("E", "")
		.replace("W", "")
		.strip()
	)

	return sign * float(coord)

tract_dsh_lookup["LAT"] = tract_dsh_lookup["LAT"].apply(parse_coord)
tract_dsh_lookup["LNG"] = tract_dsh_lookup["LNG"].apply(parse_coord)

with left_col:
	st.markdown("### Inspect a Top‑Ranked Census Tract")
		
	tract_label_map = {
		f"Rank {row.priority_rank} | {row.tract_short}": row.GEOID
		for _, row in top_df.sort_values("priority_rank").iterrows()
 	}
	
	selected_option = st.selectbox(
		"Select a tract to view partner availability.",
		options= ["Select a tract"] + list(tract_label_map.keys())
	)
	
	st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
	
	if selected_option == "Select a tract":
		selected_geoid = None
		selected_row = None
		selected_geom = gpd.GeoSeries(
			[Point(-83.0, 39.97)],
			crs="EPSG:4326"
		).iloc[0]
		clat, clon = None, None
	else:
		selected_geoid = tract_label_map.get(selected_option)
	
	selected_match = top_df[
		top_df["GEOID"].astype(str) == selected_geoid
	]

	if selected_match.empty:
		selected_row = None
		
	else:
		selected_row = selected_match.iloc[0]
		selected_geom = gpd.GeoSeries(
			[selected_row.geometry],
			crs=top_df.crs
		).to_crs(epsg=4326).iloc[0]
		
		centroid = selected_geom.centroid
		clat, clon = centroid.y, centroid.x

	libraries = public_libraries.copy()
	libraries["geometry"] = libraries["geometry"].apply(
		lambda g: wkb.loads(g) if isinstance(g, (bytes, bytearray)) else g
	)
	
	libraries = gpd.GeoDataFrame(
		libraries,
		geometry="geometry",
		crs="EPSG:4326"
	)

	libraries["GEOID"] = libraries["GEOID"].astype(str)

	libraries["is_dsh_site"] = (
		libraries["Address"]
		.str.lower()
		.str.replace(".", "", regex=False)
		.str.strip()
		.isin(
			tract_dsh_lookup["Location Address"]
			.str.lower()
			.str.replace(".", "", regex=False)
			.str.strip()
		)
	)

	libraries_in_tract = gpd.sjoin(
		libraries,
    		top_df.loc[
        		top_df["GEOID"].astype(str) == selected_geoid,
        		["GEOID", "geometry"]
    		],
    		how="inner",
    		predicate="within"
	)

	libraries_in_tract = libraries_in_tract.drop(columns=["index_right"])

	dsh_in_tract = tract_dsh_lookup[
		tract_dsh_lookup["GEOID"] == selected_geoid
	]

	library_only = libraries[
		libraries["is_dsh_site"] == False
	]
	
	def haversine(lat1, lon1, lat2, lon2):
		lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
		dlon = lon2 - lon1
		dlat = lat2 - lat1
		a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
		return 2 * asin(sqrt(a)) * 3958.8
	
	fallback_rad_miles = 1.2
	
	dsh_nearby = pd.DataFrame()
	libraries_nearby = gpd.GeoDataFrame()

	use_fallback = (
		(selected_option != "Select a tract") and
		dsh_in_tract.empty and 
		libraries_in_tract.empty and
		selected_row is not None
	)

# ===================================
# Fallback Logic:
# If no sites exist within the tract, nearby partners are shown within a fixed radius to support practical siting discussion.
# ==================================

	if use_fallback:
		dsh_nearby = tract_dsh_lookup[
			tract_dsh_lookup.apply(
				lambda r: haversine(
					clat, clon,
					r["LAT"], r["LNG"]
				) <= fallback_rad_miles,
				axis=1
			)
		]

		libraries_nearby = library_only[
			library_only.apply(
				lambda r: haversine(
					clat, clon,
					r.geometry.y, r.geometry.x
				) <= fallback_rad_miles,
				axis=1
			)
		]
		
		libraries_nearby = libraries_nearby[
			libraries_nearby["is_dsh_site"] == False
		]

	if not selected_option == "Select a tract":
		st.subheader("Potential Site Locations within Top-Priority Census Tracts")

	if selected_option == "Select a tract":
		st.markdown(" ")
	
	elif not dsh_in_tract.empty:
		st.markdown("**DSH Partners in This Tract**")
		for _, row in dsh_in_tract.iterrows():
			st.markdown(
				f"- **{row['Location Name']}:** \n"
				f" {row['Location Address']}"
			)
	else:
		st.markdown("**No DSH partners located in this tract.**")

	if (
		(selected_option != "Select a tract") 
		and dsh_in_tract.empty
		and not libraries_in_tract.empty
	):
		st.markdown("**Public Libraries in This Tract (Potential Partners)**")
		for _, row in libraries_in_tract.iterrows():
			st.markdown(
				f"- **{row['Name']}:** \n "
				f" {row['Address']}"
			)

	if dsh_in_tract.empty and libraries_in_tract.empty and (selected_option != "Select a tract"):
		st.markdown(
			f"**No DSH partners or libraries located in this tract.** \n"
			f"\n Showing nearby sites within {fallback_rad_miles} miles: "
		)

		if not dsh_nearby.empty:
			st.markdown("**Nearby DSH Partners**")
			for _, row in dsh_nearby.iterrows():
				st.markdown(
					f"- **{row['Location Name']}:** \n"
					f" {row['Location Address']}"
				)

		if not libraries_nearby.empty:
			st.markdown("**Nearby Public Libraries (Potential Partners)**")
			for _, row in libraries_nearby.iterrows():
				st.markdown(
					f"- **{row['Name']}:** \n "
					f" {row['Address']}"
				)	
		
	st.markdown(
		'</div>',
		unsafe_allow_html=True
	)

# ===================================
# Center Column - Choropleth
# Displays all census tracts for context, highlighting only the top-ranked tracts. Selecting a tract zooms and reveals nearby sites.
# ===================================

with center_col:
	st.subheader("Geographic Distribution")
	if selected_geoid == None:
		zoom_location = [39.97, -83.0]
		zoom_start = 11
		selected_row = None
	else:
		zoom_location = [clat, clon]
		zoom_start = 12
	
	m = folium.Map(
		location=zoom_location,
		zoom_start=zoom_start,
		tiles="CartoDB positron"
	)

	#Base tracts
	folium.GeoJson(
		df,
		style_function = lambda x: {
        		"fillColor": COLORS["azure_mist"],        		
			"color": COLORS["teal_dark"],
       			"fillOpacity": 0.35,
        		"weight": 0.2
		}
	).add_to(m)
	
	#Highlight top N
	if selected_row is None:
		folium.GeoJson(
			top_df,
			style_function = lambda x: {
        			"fillColor": COLORS["amber"],
      	  			"color": COLORS["teal_dark"],
        			"fillOpacity": 1.0,
        			"weight": 0.6
			},
			tooltip = folium.GeoJsonTooltip(
				fields=["tract_short", "priority_rank"],
				aliases=["Census Tract", "Priority Rank"]
			)
		).add_to(m)
		
	else:	
		folium.GeoJson(
			top_df[top_df["GEOID"].astype(str) == selected_geoid],
			style_function = lambda x: {
        			"fillColor": COLORS["amber"],
      	  			"color": COLORS["teal_dark"],
        			"fillOpacity": 0.60,
        			"weight": 0.6
			},
			tooltip = folium.GeoJsonTooltip(
				fields=["tract_short", "priority_rank"],
				aliases=["Census Tract", "Priority Rank"]
			)
		).add_to(m)

		expanded_geom = scale(
   			selected_geom,
    			xfact=1.05,
    			yfact=1.08,
    			origin="center"
		)

		bounds = expanded_geom.bounds

		m.fit_bounds([
			[bounds[1], bounds[0]],
			[bounds[3], bounds[2]]
		]
		)
		for _, row in tract_dsh_lookup.iterrows():
			folium.CircleMarker(
        			location = [row["LAT"], row["LNG"]],
        			radius = 5,
        			color = COLORS["teal"],
       				fill=True,
				fill_opacity = 1,
				tooltip=f"""
				<b>{row['Partner']}</b><br>
				{row['Location Name']}<br>
				{row['Location Address']}
				"""
			).add_to(m)
		
		
		for _, row in library_only.iterrows():
			folium.CircleMarker(
        			location = [row.geometry.y, row.geometry.x],
        			radius = 5,
        			color = COLORS["grapefruit"],
       				fill=True,
				fill_opacity = 1,
				tooltip=f"""
				<b>{row['Name']}</b><br>
				{row['Address']}
				"""
			).add_to(m)

	#Add zip code overlay:
	zip_layer = folium.FeatureGroup(name="ZIP Codes", show=False)
	folium.GeoJson(
		zips,
		style_function=lambda x: {
			"fillOpacity": 0,
			"color": COLORS["jasmine"],
			"weight": 0.9
		},
		tooltip=folium.GeoJsonTooltip(
			fields=["ZIP"],
			aliases=["ZIP Code"]
		)
	).add_to(zip_layer)

	zip_layer.add_to(m)
	folium.LayerControl(collapsed=True).add_to(m)
	

	st_folium(m, use_container_width=True, height=800)
	
# ===================================
# Right Column - Ranks & Scoring Breakdown
# ===================================

with right_col:
# ===================================
# Rankings Table:
# Displays baseline and scenario ranks, access disparity scores, and population context for top-priority tracts.
# ===================================
	
	if st.session_state.priority_focus == "None (Structural Need Only)":
		
		display_cols = [
			"baseline_rank",
			"tract_short",
			"hybrid_score",
			"total_population"
		]
		expand_column = "  	 	 																	"
		col_rename = {
			"baseline_rank": expand_column + "Baseline Rank" + expand_column,
			"tract_short": expand_column + "GEOID"+ expand_column,
			"hybrid_score": expand_column + "	Access Disparity Score		" + expand_column,
			"total_population": expand_column + "Total Population" + expand_column
		}
		table_title = "Top Census Tracts (Structural Need Only)"

		with st.container():
		
			st.subheader("Ranked Census Tracts")
			st.markdown(
			"<div style='font-weight:400; margin-top:-6px; margin-bottom: 6px; color:#000000;'>"
			"Ranking reflects baseline structural access disparity only.",
			unsafe_allow_html=True
			)
			
			st.dataframe(
				top_df[display_cols].rename(columns =col_rename),
				hide_index=True,
				use_container_width=False,
				column_config={
					"Baseline Rank": st.column_config.NumberColumn(width="200"),
					"GEOID": st.column_config.TextColumn(width="200"),
					"Access Disparity Score": st.column_config.NumberColumn(width="200"),
					"Total Population": st.column_config.NumberColumn(width="200"),
				},	
				
			)

	else:

		display_cols = [
			"priority_rank",
			"tract_short",
			"baseline_rank",
			"hybrid_score",
			"total_population"
		]
		expand_column = "  	 	 																	"

		col_rename = {
			"priority_rank": ("  Scenario Rank  "),
			"tract_short": (expand_column + "GEOID" + expand_column),
			"baseline_rank": (expand_column + "Baseline Rank" + expand_column),
			"hybrid_score": (expand_column + "Access Disparity Score" + expand_column),
			"total_population": ("     Total Population     ")	
		}
	
		with st.container():
		
			st.subheader("Ranked Census Tracts")
			st.markdown(
				"<div style='font-weight:600; margin-top:-6px; margin-bottom: 6px;'>"	
				"Baseline rank reflects structural need only. Scenario rank reflects prioritization focus.",
				unsafe_allow_html=True
			)
			st.dataframe(
				top_df[display_cols].rename(columns =col_rename),
				hide_index=True,
				use_container_width=False,
				column_config={
        				"Scenario Rank": st.column_config.NumberColumn(width="80"),
					"Baseline Rank": st.column_config.NumberColumn(width="80"),
					"GEOID": st.column_config.TextColumn(width="80"),
					"Access Disparity Score": st.column_config.NumberColumn(width="80"),
					"Total Population": st.column_config.NumberColumn(width="80"),
				}
			)

	st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ===================================
# Score Breakdown:
# Calculates percent contribution of each barrier category to the overall access disparity score for interpretability.
# ===================================
	category_cols = [
		"Economic_score",
		"Transportation_score",
		"Digital_score",
		"Education_score",
		"Health Access_score",
		"Functional_score",
		"Communication_score"
	]

	hybrid_weights = {
		"Economic_score": 0.1638,
		"Transportation_score": 0.1814,
		"Digital_score": 0.1425,
		"Education_score": 0.1088,
		"Health Access_score": 0.1056,
		"Functional_score": 0.1238,
		"Communication_score": 0.1737
	}

	def compute_score_breakdown(df, weights, categories):
		breakdown_df = df.copy()

		for cat in categories:
			breakdown_df[cat] = (df[cat] * weights[cat]).abs()

		total = breakdown_df[categories].sum(axis=1)

		breakdown_df[categories] = breakdown_df[categories].div(total, axis=0) * 100
	
		keep_cols = ["GEOID", "tract_short", "priority_rank", "hybrid_score"] + categories 

		return breakdown_df[keep_cols]

	breakdown = compute_score_breakdown(
		top_df,
		categories = category_cols,
		weights = hybrid_weights
	)
	
	breakdown = (
		breakdown
		.dropna()
		.loc[
			breakdown[category_cols].sum(axis=1) > 0
		]
	)

	#rename for readability:
	breakdown_display = breakdown.rename(columns ={
		"Economic_score": "Economic",
		"Transportation_score": "Transportation",
		"Digital_score": "Digital",
		"Education_score": "Education",
		"Health Access_score": "Health Access",
		"Functional_score": "Functional",
		"Communication_score": "Communication"
	})	

# ===================================
# Category Contribution Visualization:
# Stacked bar chart showing relative influence of each barrier category on the access disparity score.
# ===================================

	def stacked_category_bar(breakdown_df):
		category_colors = {
			"Economic": "#2A5F5B",
			"Transportation": "#2F7F7A",
			"Digital": "#6FAFB4",
			"Communication": "#93D5D3",
			"Education": COLORS["jasmine"],
			"Functional": COLORS["amber"],
			"Health Access": "#CE8E2D"	
		}
		category_columns = [
			"Economic",
			"Transportation",
			"Digital",
			"Communication",
			"Education",
			"Functional",
			"Health Access"	
		]
		geoids = breakdown_df["tract_short"].astype(str).tolist()

		fig = go.Figure()

		for cat in category_columns: 
			fig.add_bar(
				y = breakdown_df["priority_rank"].astype(str) + " | " + geoids,
				x = breakdown_df[cat],
				name = cat,
				orientation="h",
				marker_color = category_colors.get(cat)
			)

		fig.update_layout(
			barmode="stack",
			title = "Barrier Category Contributions to Access Disparity Scores of Top-Priority Tracts",
			xaxis_title = "Percent Contribution (sums to 100%)",
			yaxis_title = "Census Tract (GEOID)",
			height=300,
			legend_title = "Barrier Category",
			legend=dict(
				x=1,
				y=1,
				xanchor="left",
				yanchor="top"
			),
			margin = dict(
				l=40, 
				r=150, 
				t=40, 
				b=10
			),
			xaxis = dict(
				range=[0, 100],
				showgrid=False
			),
			yaxis = dict(
				type="category",
				categoryorder="array",
				categoryarray=geoids,
				autorange = "reversed"
			)
		)

		fig.update_traces(
			hovertemplate="%{x:.1f}%<extra></extra>"
		)
	
		return fig
	fig = stacked_category_bar(breakdown_display)
	
	with st.container():
		
		st.subheader("What Drives the Baseline Scores?")

		st.plotly_chart(fig, use_container_width=True)
		st.markdown(
			"Bars are ordered by priority rank."
			" Segment sizes show the percent contribution of each category to the Access Disparity score."
		)

# ===================================
# Create Tabs
# Additional views for methodological transparency and guidance on interpretation.
# ===================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ===================================
# Interpretation Guidance:
# Explains limitations and appropriate use of results, emphasizing that outputs support discussion rather than final siting decision
# ===================================

st.info("Open the guide below to learn how to use this tool.")

with st.expander("How to Use This Tool", expanded=False):	
	st.markdown("### Overview")

	st.write("""
	This tool helps identify census tracts in Franklin County with the greatest barriers to accessing health and social services. 
	It combines data-driven analysis with stakeholder input to support decision-making for CIE Navigation Center siting.
	""")


	st.markdown("### Step 1: Set Your Scenario")

	st.write("""
	Use the sidebar controls to adjust how results are generated:

	- **Number of tracts to display:** Choose how many high-priority tracts to show  
	- **Prioritization focus:** Emphasize specific populations (optional)  
	- **α (alpha):** Adjust balance between data-driven results and stakeholder priorities  
	- **Distance constraint:** Ensure results are geographically spread out  
	""")

	st.markdown("### Step 2: Interpret Results")

	st.write("""
	The dashboard provides several outputs:

	- **Map:** Highlights top-ranked census tracts  
	- **Ranking table:** Shows tract rankings and scores  
	- **Category breakdown:** Shows what barriers are driving each tract’s score  

	Higher scores indicate greater structural barriers to access.
	""")


	st.markdown("### Understanding the Score")

	st.write("""
	The Access Disparity Score reflects multiple barriers, including economic, transportation, digital, education, and health access factors.

	- Higher values → more barriers to access  
	- Lower values → fewer barriers  

	
	The final score blends two components: a data-driven structural score (PCA) and a stakeholder-informed score. Adjust α to control how much influence each has. By default, α is set to 0.5.

	""")


	st.markdown("### Important Notes")

	st.write("""
	- This tool identifies high-need areas and recommends existing Digital Skills Hub (DSH) partner sites or non-DSH partner public libraries as potential CIE Navigation Center locations.**  
	- Results should be used alongside local knowledge and operational considerations  
	- Scores do not reflect service availability or capacity  
	""")

st.markdown('</div>', unsafe_allow_html=True)