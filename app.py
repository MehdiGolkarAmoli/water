"""
Water Turbidity Time Series Analysis
=====================================
Sentinel-2 based NDTI (Normalized Difference Turbidity Index) calculation

ALGORITHM:
1. Link S2_SR with S2_CLOUD_PROBABILITY
2. Cloud masking: probability < 20
3. Water body detection: NDWI = (B3 - B12) / (B3 + B12) > 0.1
4. Turbidity: NDTI = (B4 - B3) / (B4 + B3)
5. Monthly composites with mean turbidity statistics
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import from_bounds
import datetime
import math
import ee
import tempfile
import requests
import time
import warnings
import base64
import json
from datetime import date
from PIL import Image
from io import BytesIO

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Water Turbidity Analysis",
    page_icon="🌊"
)

import folium
from folium import plugins
from streamlit_folium import st_folium

# =============================================================================
# CONSTANTS
# =============================================================================
# Bands needed for RGB and turbidity calculation
RGB_BANDS = ['B4', 'B3', 'B2']  # Red, Green, Blue
CALC_BANDS = ['B3', 'B4', 'B12']  # For NDWI and NDTI

# Cloud masking threshold
CLOUD_PROB_THRESHOLD = 20

# Water body detection threshold (NDWI)
NDWI_THRESHOLD = 0.1

# Download settings
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
DOWNLOAD_TIMEOUT = 120
CHUNK_SIZE = 8192
MIN_FILE_SIZE = 10000

# Status constants
STATUS_NO_DATA = "no_data"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# =============================================================================
# Session State Initialization
# =============================================================================
if 'drawn_polygons' not in st.session_state:
    st.session_state.drawn_polygons = []
if 'last_drawn_polygon' not in st.session_state:
    st.session_state.last_drawn_polygon = None
if 'ee_initialized' not in st.session_state:
    st.session_state.ee_initialized = False
if 'current_temp_dir' not in st.session_state:
    st.session_state.current_temp_dir = None
if 'downloaded_months' not in st.session_state:
    st.session_state.downloaded_months = {}
if 'month_statuses' not in st.session_state:
    st.session_state.month_statuses = {}
if 'turbidity_results' not in st.session_state:
    st.session_state.turbidity_results = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'selected_region_index' not in st.session_state:
    st.session_state.selected_region_index = 0
if 'processing_in_progress' not in st.session_state:
    st.session_state.processing_in_progress = False
if 'processing_config' not in st.session_state:
    st.session_state.processing_config = None
if 'mean_turbidity_data' not in st.session_state:
    st.session_state.mean_turbidity_data = {}


# =============================================================================
# Earth Engine Authentication
# =============================================================================
@st.cache_resource
def initialize_earth_engine():
    """Initialize Earth Engine"""
    try:
        ee.Initialize()
        return True, "Earth Engine initialized"
    except Exception:
        try:
            base64_key = os.environ.get('GOOGLE_EARTH_ENGINE_KEY_BASE64')
            
            if base64_key:
                key_json = base64.b64decode(base64_key).decode()
                key_data = json.loads(key_json)
                
                key_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
                with open(key_file.name, 'w') as f:
                    json.dump(key_data, f)
                
                credentials = ee.ServiceAccountCredentials(key_data['client_email'], key_file.name)
                ee.Initialize(credentials)
                os.unlink(key_file.name)
                return True, "Authenticated with Service Account"
            else:
                ee.Authenticate()
                ee.Initialize()
                return True, "Authenticated"
        except Exception as auth_error:
            return False, f"Auth failed: {str(auth_error)}"


# =============================================================================
# Helper Functions
# =============================================================================
def get_utm_zone(longitude):
    return math.floor((longitude + 180) / 6) + 1


def validate_geotiff_file(file_path, expected_bands=1):
    """Validate that a GeoTIFF file is complete and readable."""
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(file_path)
        if file_size < MIN_FILE_SIZE:
            return False, f"File too small ({file_size} bytes)"
        
        with rasterio.open(file_path) as src:
            if src.count < expected_bands:
                return False, f"Wrong band count ({src.count}, expected {expected_bands})"
        
        return True, "File is valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# =============================================================================
# Turbidity Calculation (GEE Server-Side)
# =============================================================================
def create_turbidity_collection(aoi, start_date, end_date, cloudy_pixel_percentage=30):
    """
    Create turbidity collection matching the notebook algorithm.
    
    Steps:
    1. Link S2_SR with S2_CLOUD_PROBABILITY
    2. Apply cloud mask (probability < 20)
    3. Calculate NDWI for water body detection
    4. Calculate NDTI (turbidity index)
    5. Mask with cloud-free and water body
    """
    # Get S2 SR collection
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(aoi)
             .filterDate(start_date, end_date)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage)))
    
    # Get cloud probability collection
    s2_cloud_prob = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date))
    
    # Join collections
    join_filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
    joined = ee.Join.saveFirst('cloud_probability').apply(
        primary=s2_sr, secondary=s2_cloud_prob, condition=join_filter
    )
    
    def add_cloud_band(feature):
        img = ee.Image(feature)
        cloud_prob_img = ee.Image(img.get('cloud_probability'))
        return img.addBands(cloud_prob_img.select('probability'))
    
    s2_joined = ee.ImageCollection(joined.map(add_cloud_band))
    
    # Apply turbidity calculation
    def calculate_turbidity(img):
        # Cloud mask: probability < 20
        cloud = img.select('probability')
        cloud_free = cloud.lt(CLOUD_PROB_THRESHOLD)
        
        # Scale reflectance bands
        sr = img.select(['B2', 'B3', 'B4', 'B12']).multiply(0.0001)
        
        # NDWI for water body detection: (B3 - B12) / (B3 + B12)
        ndwi = sr.normalizedDifference(['B3', 'B12']).rename('ndwi')
        water_body = ndwi.gt(NDWI_THRESHOLD)
        
        # NDTI (turbidity): (B4 - B3) / (B4 + B3)
        ndti = sr.normalizedDifference(['B4', 'B3']).rename('ndti')
        
        # Apply masks
        ndti_masked = ndti.updateMask(cloud_free).updateMask(water_body)
        
        # Keep RGB bands for visualization (scaled)
        rgb = sr.select(['B4', 'B3', 'B2'])
        
        # Combine NDTI with RGB
        combined = ndti_masked.addBands(rgb).addBands(water_body.rename('water_mask'))
        
        return combined.clip(aoi).copyProperties(img, ['system:time_start'])
    
    return s2_joined.map(calculate_turbidity)


def get_monthly_composite(turbidity_collection, aoi, year, month):
    """
    Create monthly composite from turbidity collection.
    Returns composite with NDTI and RGB bands.
    """
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')
    
    monthly = turbidity_collection.filterDate(start, end)
    
    # Check image count
    count = monthly.size().getInfo()
    
    if count == 0:
        return None, 0, "No images"
    
    # Create composite - use median for NDTI, median for RGB
    composite = monthly.median()
    
    # Calculate statistics
    stats = composite.select('ndti').reduceRegion(
        reducer=ee.Reducer.mean().combine(
            ee.Reducer.count(), sharedInputs=True
        ).combine(
            ee.Reducer.minMax(), sharedInputs=True
        ),
        geometry=aoi,
        scale=10,
        maxPixels=1e13
    )
    
    return composite, count, stats


# =============================================================================
# Download Functions
# =============================================================================
def download_band_with_retry(image, band, aoi, output_path, scale=10):
    """Download a single band with retry mechanism."""
    try:
        region = aoi.bounds().getInfo()['coordinates']
    except Exception as e:
        return False, f"AOI bounds error: {e}"
    
    temp_path = output_path + '.tmp'
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    if os.path.exists(output_path):
        is_valid, msg = validate_geotiff_file(output_path, expected_bands=1)
        if is_valid:
            return True, "cached"
        os.remove(output_path)
    
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            url = image.select(band).getDownloadURL({
                'scale': scale, 'region': region, 'format': 'GEO_TIFF', 'bands': [band]
            })
            
            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    last_error = "GEE rate limit"
                    raise Exception(last_error)
                
                downloaded_size = 0
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                
                if downloaded_size < MIN_FILE_SIZE:
                    last_error = f"File too small ({downloaded_size} bytes)"
                    raise Exception(last_error)
                
                is_valid, msg = validate_geotiff_file(temp_path, expected_bands=1)
                if is_valid:
                    os.replace(temp_path, output_path)
                    return True, "success"
                else:
                    last_error = f"Validation failed: {msg}"
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise Exception(last_error)
            else:
                last_error = f"HTTP {response.status_code}"
                raise Exception(last_error)
                
        except requests.exceptions.Timeout:
            last_error = "Timeout"
        except requests.exceptions.ConnectionError:
            last_error = "Connection error"
        except Exception as e:
            if last_error is None:
                last_error = str(e)
        
        for f in [output_path, temp_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        
        if attempt < MAX_RETRIES - 1:
            wait_time = RETRY_DELAY_BASE ** (attempt + 1)
            time.sleep(wait_time)
    
    return False, last_error


def download_monthly_data(composite, aoi, temp_dir, month_name, scale=10, status_placeholder=None):
    """
    Download monthly composite (NDTI + RGB bands).
    Returns paths to downloaded files.
    """
    ndti_path = os.path.join(temp_dir, f"ndti_{month_name}.tif")
    rgb_path = os.path.join(temp_dir, f"rgb_{month_name}.tif")
    
    # Check cache
    ndti_valid, _ = validate_geotiff_file(ndti_path, expected_bands=1)
    rgb_valid, _ = validate_geotiff_file(rgb_path, expected_bands=3)
    
    if ndti_valid and rgb_valid:
        if status_placeholder:
            status_placeholder.info(f"✅ {month_name} using cache")
        return ndti_path, rgb_path, STATUS_COMPLETE, "Cached"
    
    try:
        # Download NDTI
        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Downloading NDTI...")
        
        success, msg = download_band_with_retry(composite, 'ndti', aoi, ndti_path, scale)
        if not success:
            return None, None, STATUS_FAILED, f"NDTI download failed: {msg}"
        
        # Download RGB bands
        bands_dir = os.path.join(temp_dir, f"bands_{month_name}")
        os.makedirs(bands_dir, exist_ok=True)
        
        rgb_bands = ['B4', 'B3', 'B2']
        band_files = []
        
        for i, band in enumerate(rgb_bands):
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: RGB {band} ({i+1}/3)...")
            
            band_file = os.path.join(bands_dir, f"{band}.tif")
            success, msg = download_band_with_retry(composite, band, aoi, band_file, scale)
            
            if not success:
                return None, None, STATUS_FAILED, f"RGB {band} download failed: {msg}"
            
            band_files.append(band_file)
        
        # Merge RGB bands
        if status_placeholder:
            status_placeholder.text(f"📦 {month_name}: Merging RGB...")
        
        with rasterio.open(band_files[0]) as src:
            meta = src.meta.copy()
        meta.update(count=3)
        
        with rasterio.open(rgb_path, 'w', **meta) as dst:
            for i, band_file in enumerate(band_files):
                with rasterio.open(band_file) as src:
                    dst.write(src.read(1), i+1)
        
        return ndti_path, rgb_path, STATUS_COMPLETE, "Downloaded"
        
    except Exception as e:
        return None, None, STATUS_FAILED, f"Error: {str(e)}"


# =============================================================================
# Visualization Functions
# =============================================================================
def create_turbidity_colormap():
    """Create custom colormap for turbidity visualization."""
    # Blue (clear) -> Green -> Yellow -> Red (turbid)
    colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
    return LinearSegmentedColormap.from_list('turbidity', colors, N=256)


def generate_thumbnails(ndti_path, rgb_path, month_name, max_size=300):
    """Generate RGB and turbidity thumbnails."""
    try:
        # Read NDTI
        with rasterio.open(ndti_path) as src:
            ndti_data = src.read(1)
        
        # Read RGB
        with rasterio.open(rgb_path) as src:
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)
        
        # Process RGB
        rgb = np.stack([red, green, blue], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0.0)
        
        def percentile_stretch(band, lower=2, upper=98):
            valid = band[band > 0]
            if len(valid) == 0:
                return np.zeros_like(band, dtype=np.uint8)
            p_low = np.percentile(valid, lower)
            p_high = np.percentile(valid, upper)
            if p_high <= p_low:
                p_high = p_low + 0.001
            stretched = np.clip((band - p_low) / (p_high - p_low), 0, 1)
            return (stretched * 255).astype(np.uint8)
        
        rgb_uint8 = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(3):
            rgb_uint8[:, :, i] = percentile_stretch(rgb[:, :, i])
        
        # Create turbidity visualization
        ndti_valid = np.nan_to_num(ndti_data, nan=np.nan)
        
        # Calculate mean turbidity (excluding NaN)
        valid_ndti = ndti_valid[~np.isnan(ndti_valid) & (ndti_valid != 0)]
        mean_turbidity = np.nanmean(valid_ndti) if len(valid_ndti) > 0 else np.nan
        valid_pixel_count = len(valid_ndti)
        total_pixels = ndti_data.size
        water_coverage = (valid_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Create turbidity image with colormap
        cmap = create_turbidity_colormap()
        
        # Normalize NDTI to 0-1 range for colormap (typical range -0.3 to 0.3)
        ndti_normalized = np.clip((ndti_valid + 0.3) / 0.6, 0, 1)
        ndti_normalized = np.nan_to_num(ndti_normalized, nan=0)
        
        # Apply colormap
        turbidity_colored = cmap(ndti_normalized)[:, :, :3]  # RGB only
        turbidity_uint8 = (turbidity_colored * 255).astype(np.uint8)
        
        # Mask non-water areas (where NDTI is NaN or 0)
        water_mask = (~np.isnan(ndti_valid)) & (ndti_valid != 0)
        for i in range(3):
            turbidity_uint8[:, :, i] = np.where(water_mask, turbidity_uint8[:, :, i], 50)  # Gray for non-water
        
        # Convert to PIL
        pil_rgb = Image.fromarray(rgb_uint8, mode='RGB')
        pil_turbidity = Image.fromarray(turbidity_uint8, mode='RGB')
        
        # Resize
        h, w = pil_rgb.size[1], pil_rgb.size[0]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_rgb = pil_rgb.resize((new_w, new_h), Image.LANCZOS)
            pil_turbidity = pil_turbidity.resize((new_w, new_h), Image.LANCZOS)
        
        return {
            'rgb_image': pil_rgb,
            'turbidity_image': pil_turbidity,
            'month_name': month_name,
            'mean_turbidity': mean_turbidity,
            'water_coverage': water_coverage,
            'valid_pixels': valid_pixel_count
        }
        
    except Exception as e:
        st.warning(f"Error generating thumbnails for {month_name}: {e}")
        return None


# =============================================================================
# Main Processing Pipeline
# =============================================================================
def process_turbidity_timeseries(aoi, start_date, end_date, cloudy_pixel_percentage=30, 
                                  scale=10, resume=False):
    """Main processing pipeline for turbidity analysis."""
    try:
        if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
            st.session_state.current_temp_dir = tempfile.mkdtemp()
        temp_dir = st.session_state.current_temp_dir
        
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
        
        st.info(f"📅 Processing {total_months} months | 📁 {temp_dir}")
        
        # =====================================================================
        # PHASE 1: Create turbidity collection
        # =====================================================================
        st.header("Phase 1: Create Turbidity Collection")
        st.info(f"☁️ Cloud threshold: < {CLOUD_PROB_THRESHOLD}% | 💧 Water: NDWI > {NDWI_THRESHOLD}")
        
        with st.spinner("Creating turbidity collection..."):
            turbidity_collection = create_turbidity_collection(
                aoi, start_date, end_date, cloudy_pixel_percentage
            )
            collection_size = turbidity_collection.size().getInfo()
        
        st.success(f"✅ Collection created with {collection_size} images")
        
        # =====================================================================
        # PHASE 2: Download monthly composites
        # =====================================================================
        st.header("Phase 2: Download Monthly Data")
        
        downloaded_months = {}
        month_statuses = {}
        mean_turbidity_data = {}
        
        # Prepare month list
        month_infos = []
        for month_index in range(total_months):
            year = start_dt.year + (start_dt.month - 1 + month_index) // 12
            month = (start_dt.month - 1 + month_index) % 12 + 1
            month_name = f"{year}-{month:02d}"
            month_infos.append({
                'month_name': month_name,
                'year': year,
                'month': month
            })
        
        st.info(f"📅 Months: {month_infos[0]['month_name']} to {month_infos[-1]['month_name']}")
        
        # Check cache
        if resume and st.session_state.downloaded_months:
            for month_name, paths in st.session_state.downloaded_months.items():
                if paths.get('ndti') and paths.get('rgb'):
                    ndti_valid, _ = validate_geotiff_file(paths['ndti'], 1)
                    rgb_valid, _ = validate_geotiff_file(paths['rgb'], 3)
                    if ndti_valid and rgb_valid:
                        downloaded_months[month_name] = paths
                        month_statuses[month_name] = {'status': STATUS_COMPLETE, 'message': 'Cached'}
            
            if downloaded_months:
                st.info(f"🔄 Found {len(downloaded_months)} cached months")
        
        # Restore previous statuses
        if resume and st.session_state.month_statuses:
            for month_name, status_info in st.session_state.month_statuses.items():
                if month_name not in month_statuses:
                    month_statuses[month_name] = status_info
        
        # Process remaining months
        months_to_process = [m for m in month_infos 
                            if m['month_name'] not in downloaded_months 
                            and m['month_name'] not in month_statuses]
        
        if months_to_process:
            st.info(f"📥 {len(months_to_process)} months to process")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, month_info in enumerate(months_to_process):
                month_name = month_info['month_name']
                
                status_text.text(f"🔍 {month_name}: Getting composite...")
                
                # Get monthly composite
                composite, count, stats = get_monthly_composite(
                    turbidity_collection, aoi, 
                    month_info['year'], month_info['month']
                )
                
                if composite is None or count == 0:
                    month_statuses[month_name] = {'status': STATUS_NO_DATA, 'message': 'No images'}
                    st.session_state.month_statuses[month_name] = month_statuses[month_name]
                    st.write(f"⚫ **{month_name}**: No data")
                    progress_bar.progress((idx + 1) / len(months_to_process))
                    continue
                
                # Download data
                ndti_path, rgb_path, status, message = download_monthly_data(
                    composite, aoi, temp_dir, month_name, scale, status_text
                )
                
                month_statuses[month_name] = {'status': status, 'message': message}
                st.session_state.month_statuses[month_name] = month_statuses[month_name]
                
                if status == STATUS_COMPLETE:
                    downloaded_months[month_name] = {'ndti': ndti_path, 'rgb': rgb_path}
                    st.session_state.downloaded_months[month_name] = downloaded_months[month_name]
                    st.write(f"🟢 **{month_name}**: {message} ({count} images)")
                else:
                    st.write(f"🔴 **{month_name}**: {message}")
                
                progress_bar.progress((idx + 1) / len(months_to_process))
            
            progress_bar.empty()
            status_text.empty()
        
        # Summary
        st.divider()
        status_counts = {s: sum(1 for ms in month_statuses.values() if ms['status'] == s) 
                        for s in [STATUS_NO_DATA, STATUS_COMPLETE, STATUS_FAILED]}
        
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Complete", status_counts.get(STATUS_COMPLETE, 0))
        col2.metric("🔴 Failed", status_counts.get(STATUS_FAILED, 0))
        col3.metric("⚫ No Data", status_counts.get(STATUS_NO_DATA, 0))
        
        if not downloaded_months:
            st.error("❌ No months downloaded!")
            return []
        
        st.success(f"✅ Downloaded {len(downloaded_months)}/{total_months} months")
        
        # =====================================================================
        # PHASE 3: Generate thumbnails and calculate statistics
        # =====================================================================
        st.header("Phase 3: Generate Visualizations")
        
        results = []
        month_names = sorted(downloaded_months.keys())
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, month_name in enumerate(month_names):
            status_text.text(f"🎨 {month_name}: Generating visualization...")
            
            paths = downloaded_months[month_name]
            thumb = generate_thumbnails(paths['ndti'], paths['rgb'], month_name)
            
            if thumb:
                results.append(thumb)
                mean_turbidity_data[month_name] = {
                    'mean': thumb['mean_turbidity'],
                    'coverage': thumb['water_coverage']
                }
            
            progress_bar.progress((idx + 1) / len(month_names))
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.mean_turbidity_data = mean_turbidity_data
        st.success(f"✅ Generated {len(results)} visualizations!")
        
        return results
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []


# =============================================================================
# Display Functions
# =============================================================================
def display_results(results):
    """Display turbidity results with RGB and turbidity images side by side."""
    if not results:
        return
    
    st.subheader("🌊 Monthly Turbidity Results")
    
    # Display mode selection
    mode = st.radio("Display:", ["Side by Side", "Turbidity Only", "RGB Only"], horizontal=True)
    
    # Color legend
    with st.expander("🎨 Turbidity Color Legend"):
        fig, ax = plt.subplots(figsize=(8, 0.5))
        cmap = create_turbidity_colormap()
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_xticks([0, 128, 255])
        ax.set_xticklabels(['-0.3 (Clear)', '0 (Moderate)', '0.3 (Turbid)'])
        ax.set_yticks([])
        ax.set_title('NDTI (Normalized Difference Turbidity Index)')
        st.pyplot(fig)
        plt.close()
    
    st.divider()
    
    if mode == "Side by Side":
        for i in range(0, len(results), 2):
            cols = st.columns(4)
            for j in range(2):
                idx = i + j
                if idx < len(results):
                    r = results[idx]
                    mean_str = f"{r['mean_turbidity']:.4f}" if not np.isnan(r['mean_turbidity']) else "N/A"
                    cols[j*2].image(r['rgb_image'], caption=f"{r['month_name']} RGB")
                    cols[j*2+1].image(r['turbidity_image'], 
                                     caption=f"{r['month_name']} NDTI (mean: {mean_str})")
    else:
        key = 'turbidity_image' if mode == "Turbidity Only" else 'rgb_image'
        for row in range((len(results) + 3) // 4):
            cols = st.columns(4)
            for c in range(4):
                idx = row * 4 + c
                if idx < len(results):
                    r = results[idx]
                    if mode == "Turbidity Only":
                        mean_str = f"{r['mean_turbidity']:.4f}" if not np.isnan(r['mean_turbidity']) else "N/A"
                        cap = f"{r['month_name']} (NDTI: {mean_str})"
                    else:
                        cap = r['month_name']
                    cols[c].image(r[key], caption=cap)


def display_turbidity_chart(results):
    """Display time series chart of mean turbidity values."""
    if not results:
        return
    
    st.subheader("📊 Mean Turbidity Time Series")
    
    # Prepare data
    months = []
    mean_values = []
    coverage_values = []
    
    for r in results:
        if not np.isnan(r['mean_turbidity']):
            months.append(r['month_name'])
            mean_values.append(r['mean_turbidity'])
            coverage_values.append(r['water_coverage'])
    
    if not months:
        st.warning("No valid turbidity data to plot")
        return
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    # Plot mean turbidity
    color1 = '#1f77b4'
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Mean NDTI', color=color1)
    line1 = ax1.plot(months, mean_values, 'o-', color=color1, linewidth=2, markersize=8, label='Mean NDTI')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(min(mean_values) - 0.02, max(mean_values) + 0.02)
    
    # Add horizontal reference lines
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral (NDTI=0)')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Second y-axis for water coverage
    ax2 = ax1.twinx()
    color2 = '#2ca02c'
    ax2.set_ylabel('Water Coverage (%)', color=color2)
    line2 = ax2.bar(months, coverage_values, alpha=0.3, color=color2, label='Water Coverage')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(coverage_values) * 1.2 if coverage_values else 100)
    
    # Title
    plt.title('Water Turbidity Analysis Over Time', fontsize=14, fontweight='bold')
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper left')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Statistics table
    st.subheader("📈 Statistics Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean NDTI", f"{np.mean(mean_values):.4f}")
    col2.metric("Max NDTI", f"{np.max(mean_values):.4f}")
    col3.metric("Min NDTI", f"{np.min(mean_values):.4f}")
    col4.metric("Std Dev", f"{np.std(mean_values):.4f}")
    
    # Data table
    with st.expander("📋 Monthly Data Table"):
        import pandas as pd
        df = pd.DataFrame({
            'Month': months,
            'Mean NDTI': [f"{v:.4f}" for v in mean_values],
            'Water Coverage (%)': [f"{v:.1f}" for v in coverage_values]
        })
        st.dataframe(df, use_container_width=True)


# =============================================================================
# Main Application
# =============================================================================
def main():
    st.title("🌊 Water Turbidity Analysis")
    st.markdown("""
    **NDTI (Normalized Difference Turbidity Index)** analysis using Sentinel-2 imagery.
    
    | Parameter | Formula | Threshold |
    |-----------|---------|-----------|
    | Cloud Mask | probability | < 20% |
    | Water Body | NDWI = (B3-B12)/(B3+B12) | > 0.1 |
    | Turbidity | NDTI = (B4-B3)/(B4+B3) | -0.3 to 0.3 |
    """)
    
    # Initialize Earth Engine
    ee_ok, ee_msg = initialize_earth_engine()
    if not ee_ok:
        st.error(ee_msg)
        st.stop()
    st.sidebar.success(ee_msg)
    
    # Parameters
    st.sidebar.header("⚙️ Parameters")
    cloudy_pct = st.sidebar.slider(
        "Max Cloud % (metadata)", 0, 50, 30, 5,
        help="Filter images by CLOUDY_PIXEL_PERCENTAGE metadata",
        disabled=st.session_state.processing_in_progress
    )
    
    # Cache Status
    st.sidebar.header("🗂️ Cache Status")
    
    cache_info = []
    if st.session_state.downloaded_months:
        cache_info.append(f"📥 {len(st.session_state.downloaded_months)} months downloaded")
    if st.session_state.turbidity_results:
        cache_info.append(f"🌊 {len(st.session_state.turbidity_results)} results")
    
    if cache_info:
        for info in cache_info:
            st.sidebar.success(info)
    else:
        st.sidebar.info("No cached data")
    
    if st.session_state.processing_in_progress:
        st.sidebar.error("⏳ Processing in progress...")
    
    if st.sidebar.button("🗑️ Clear All Cache", disabled=st.session_state.processing_in_progress):
        for key in ['downloaded_months', 'month_statuses', 'turbidity_results',
                    'mean_turbidity_data', 'current_temp_dir', 'processing_config']:
            if key in st.session_state:
                if isinstance(st.session_state[key], dict):
                    st.session_state[key] = {}
                elif isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
        st.session_state.processing_complete = False
        st.session_state.processing_in_progress = False
        st.rerun()
    
    # Stop processing button
    if st.session_state.processing_in_progress:
        if st.sidebar.button("🛑 Stop Processing", type="primary"):
            st.session_state.processing_in_progress = False
            st.session_state.processing_config = None
            st.warning("⚠️ Processing stopped. You can resume later.")
            st.rerun()
    
    # Region Selection
    st.header("1️⃣ Select Water Body Region")
    
    if not st.session_state.processing_in_progress:
        m = folium.Map(location=[35.6892, 51.3890], zoom_start=8)
        plugins.Draw(export=True, position='topleft', draw_options={
            'polyline': False, 'rectangle': True, 'polygon': True,
            'circle': False, 'marker': False, 'circlemarker': False
        }).add_to(m)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                        attr='Google', name='Satellite').add_to(m)
        folium.LayerControl().add_to(m)
        
        map_data = st_folium(m, width=800, height=500)
        
        if map_data and map_data.get('last_active_drawing'):
            geom = map_data['last_active_drawing'].get('geometry', {})
            if geom.get('type') == 'Polygon':
                st.session_state.last_drawn_polygon = Polygon(geom['coordinates'][0])
                st.success(f"✅ Region selected")
        
        if st.button("💾 Save Region"):
            if st.session_state.last_drawn_polygon:
                is_duplicate = False
                for existing in st.session_state.drawn_polygons:
                    if existing.equals(st.session_state.last_drawn_polygon):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                    st.success("✅ Region saved!")
                    st.rerun()
                else:
                    st.warning("⚠️ This region is already saved")
            else:
                st.warning("⚠️ Draw a region first")
    else:
        st.info("🔒 Map is locked during processing")
    
    # Saved Regions
    if st.session_state.drawn_polygons:
        st.subheader("📍 Saved Regions")
        for i, p in enumerate(st.session_state.drawn_polygons):
            c1, c2, c3 = st.columns([3, 1, 1])
            centroid = p.centroid
            c1.write(f"**Region {i+1}**: ~{p.area * 111 * 111:.2f} km²")
            c2.write(f"Center: ({centroid.y:.4f}, {centroid.x:.4f})")
            if c3.button("🗑️", key=f"del_{i}", disabled=st.session_state.processing_in_progress):
                st.session_state.drawn_polygons.pop(i)
                if st.session_state.selected_region_index >= len(st.session_state.drawn_polygons):
                    st.session_state.selected_region_index = max(0, len(st.session_state.drawn_polygons) - 1)
                st.rerun()
    
    # Time Period
    st.header("2️⃣ Time Period")
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", value=date(2024, 1, 1), disabled=st.session_state.processing_in_progress)
    end = c2.date_input("End (exclusive)", value=date(2025, 1, 1), disabled=st.session_state.processing_in_progress)
    
    if start >= end:
        st.error("Invalid dates")
        st.stop()
    
    months = (end.year - start.year) * 12 + (end.month - start.month)
    first_month = f"{start.year}-{start.month:02d}"
    last_year = start.year + (start.month - 1 + months - 1) // 12
    last_month_num = (start.month - 1 + months - 1) % 12 + 1
    last_month = f"{last_year}-{last_month_num:02d}"
    
    st.info(f"📅 **{months} months**: {first_month} → {last_month}")
    
    # Process
    st.header("3️⃣ Process")
    
    selected_polygon = None
    
    if st.session_state.drawn_polygons:
        region_options = []
        for i, p in enumerate(st.session_state.drawn_polygons):
            area = p.area * 111 * 111
            region_options.append(f"Region {i+1} (~{area:.2f} km²)")
        
        if st.session_state.selected_region_index >= len(st.session_state.drawn_polygons):
            st.session_state.selected_region_index = 0
        
        selected_idx = st.selectbox(
            "🎯 Select Region",
            range(len(region_options)),
            format_func=lambda i: region_options[i],
            index=st.session_state.selected_region_index,
            disabled=st.session_state.processing_in_progress
        )
        
        st.session_state.selected_region_index = selected_idx
        selected_polygon = st.session_state.drawn_polygons[selected_idx]
        st.success(f"✅ Selected: Region {selected_idx + 1}")
        
    elif st.session_state.last_drawn_polygon is not None:
        selected_polygon = st.session_state.last_drawn_polygon
        st.info("ℹ️ Using unsaved drawn region")
    else:
        st.warning("⚠️ Draw a region on the map first")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_new = st.button(
            "🚀 Start Analysis",
            type="primary",
            disabled=st.session_state.processing_in_progress or selected_polygon is None
        )
    
    with col2:
        has_cache = bool(st.session_state.downloaded_months)
        resume_btn = st.button(
            "🔄 Resume",
            disabled=not has_cache or st.session_state.processing_in_progress
        )
    
    # Processing logic
    should_process = False
    resume_mode = False
    
    if start_new:
        should_process = True
        resume_mode = False
        st.session_state.month_statuses = {}
        st.session_state.downloaded_months = {}
        st.session_state.turbidity_results = []
        st.session_state.mean_turbidity_data = {}
        st.session_state.processing_complete = False
        
        st.session_state.processing_config = {
            'polygon_coords': list(selected_polygon.exterior.coords),
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'cloudy_pct': cloudy_pct
        }
        st.session_state.processing_in_progress = True
    
    elif resume_btn:
        should_process = True
        resume_mode = True
        st.session_state.processing_in_progress = True
        
        if st.session_state.processing_config is None:
            st.session_state.processing_config = {
                'polygon_coords': list(selected_polygon.exterior.coords),
                'start_date': start.strftime('%Y-%m-%d'),
                'end_date': end.strftime('%Y-%m-%d'),
                'cloudy_pct': cloudy_pct
            }
    
    elif st.session_state.processing_in_progress and st.session_state.processing_config is not None:
        should_process = True
        resume_mode = True
        st.info("🔄 Auto-continuing processing...")
    
    if should_process:
        config = st.session_state.processing_config
        
        if config is None:
            st.error("❌ No processing configuration found!")
            st.session_state.processing_in_progress = False
            st.stop()
        
        aoi = ee.Geometry.Polygon([config['polygon_coords']])
        
        try:
            results = process_turbidity_timeseries(
                aoi,
                config['start_date'],
                config['end_date'],
                config['cloudy_pct'],
                10,
                resume=resume_mode
            )
            
            if results:
                st.session_state.turbidity_results = results
                st.session_state.processing_complete = True
            
            st.session_state.processing_in_progress = False
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state.processing_in_progress = False
    
    # Display Results
    if st.session_state.processing_complete and st.session_state.turbidity_results:
        st.divider()
        st.header("📊 Results")
        
        display_results(st.session_state.turbidity_results)
        
        st.divider()
        
        display_turbidity_chart(st.session_state.turbidity_results)


if __name__ == "__main__":
    main()
