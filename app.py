"""
Water Turbidity Time Series Analysis
=====================================
Sentinel-2 based NDTI (Normalized Difference Turbidity Index) calculation

ALGORITHM:
1. Link S2_SR with S2_CLOUD_PROBABILITY
2. Cloud masking: probability < 15
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
# Bands needed for RGB and calculations
RGB_BANDS = ['B4', 'B3', 'B2']  # Red, Green, Blue
CALC_BANDS_TURBIDITY = ['B3', 'B4', 'B11', 'B12']  # For NDWI, NDTI, and NDSI (snow)
CALC_BANDS_CHLOROPHYLL = ['B1', 'B3', 'B8', 'B11']  # For Chlorophyll, NDWI, NDSI

# Cloud masking threshold
CLOUD_PROB_THRESHOLD = 15

# Water body detection thresholds
NDWI_THRESHOLD_TURBIDITY = 0.1  # NDWI using B3, B12
NDWI_THRESHOLD_CHLOROPHYLL = 0.05  # NDWI using B3, B8

# Snow detection thresholds
# NDSI (Normalized Difference Snow Index) = (B3 - B11) / (B3 + B11)
NDSI_THRESHOLD = 0.3
SNOW_B11_THRESHOLD = 0.1  # Snow has high SWIR reflectance, water has low

# Water Quality Parameter Options
PARAM_TURBIDITY = "Turbidity (NDTI)"
PARAM_CHLOROPHYLL = "Chlorophyll Index"

# Chlorophyll visualization range
CHL_VMIN = 2
CHL_VMAX = 100

# Download settings
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
DOWNLOAD_TIMEOUT = 120
CHUNK_SIZE = 8192
MIN_FILE_SIZE = 500

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
if 'selected_parameter' not in st.session_state:
    st.session_state.selected_parameter = PARAM_TURBIDITY


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
# Water Quality Calculation (GEE Server-Side)
# =============================================================================
def create_water_quality_collection(aoi, start_date, end_date, parameter_type, cloudy_pixel_percentage=30):
    """
    Create water quality collection for either Turbidity or Chlorophyll.
    
    For TURBIDITY (NDTI):
    1. Link S2_SR with S2_CLOUD_PROBABILITY
    2. Apply cloud mask (probability < 20)
    3. Calculate NDSI for snow detection: (B3 - B11) / (B3 + B11)
    4. Create snow mask: NDSI > 0.4 AND B11 > 0.1
    5. Calculate NDWI for water body detection: (B3 - B12) / (B3 + B12) > 0.1
    6. Calculate NDTI (turbidity index): (B4 - B3) / (B4 + B3)
    
    For CHLOROPHYLL:
    1. Link S2_SR with S2_CLOUD_PROBABILITY
    2. Apply cloud mask (probability < 20)
    3. Calculate NDSI for snow detection: (B3 - B11) / (B3 + B11)
    4. Create snow mask: NDSI > 0.4 AND B11 > 0.1
    5. Calculate NDWI for water body detection: (B3 - B8) / (B3 + B8) >= 0.05
    6. Calculate Chlorophyll Index: 4.26 * (B3/B1)^3.94
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
    
    if parameter_type == PARAM_TURBIDITY:
        # Apply turbidity calculation with snow detection
        def calculate_turbidity(img):
            # Cloud mask: probability < 20
            cloud = img.select('probability')
            cloud_free = cloud.lt(CLOUD_PROB_THRESHOLD)
            
            # Scale reflectance bands (B2, B3, B4, B11, B12)
            sr = img.select(['B2', 'B3', 'B4', 'B11', 'B12']).multiply(0.0001)
            
            # SNOW DETECTION: NDSI = (B3 - B11) / (B3 + B11)
            ndsi = sr.normalizedDifference(['B3', 'B11']).rename('ndsi')
            is_snow = ndsi.gt(NDSI_THRESHOLD).And(sr.select('B11').gt(SNOW_B11_THRESHOLD))
            
            # WATER DETECTION: NDWI = (B3 - B12) / (B3 + B12) > 0.1, excluding snow
            ndwi = sr.normalizedDifference(['B3', 'B12']).rename('ndwi')
            water_body = ndwi.gt(NDWI_THRESHOLD_TURBIDITY).And(is_snow.Not())
            
            # TURBIDITY: NDTI = (B4 - B3) / (B4 + B3)
            ndti = sr.normalizedDifference(['B4', 'B3']).rename('wq_index')
            
            # Apply all masks
            wq_masked = ndti.updateMask(cloud_free).updateMask(water_body)
            
            # RGB bands for visualization
            rgb = sr.select(['B4', 'B3', 'B2'])
            
            combined = (wq_masked
                       .addBands(rgb)
                       .addBands(water_body.rename('water_mask'))
                       .addBands(is_snow.rename('snow_mask'))
                       .addBands(ndsi))
            
            return combined.clip(aoi).copyProperties(img, ['system:time_start'])
        
        return s2_joined.map(calculate_turbidity)
    
    else:  # CHLOROPHYLL
        # Apply chlorophyll calculation with snow detection
        def calculate_chlorophyll(img):
            # Cloud mask: probability < 20
            cloud = img.select('probability')
            cloud_free = cloud.lt(CLOUD_PROB_THRESHOLD)
            
            # Scale reflectance bands (B1, B2, B3, B4, B8, B11)
            sr = img.select(['B1', 'B2', 'B3', 'B4', 'B8', 'B11']).multiply(0.0001)
            
            # SNOW DETECTION: NDSI = (B3 - B11) / (B3 + B11)
            ndsi = sr.normalizedDifference(['B3', 'B11']).rename('ndsi')
            is_snow = ndsi.gt(NDSI_THRESHOLD).And(sr.select('B11').gt(SNOW_B11_THRESHOLD))
            
            # WATER DETECTION: NDWI = (B3 - B8) / (B3 + B8) >= 0.05, excluding snow
            ndwi = sr.normalizedDifference(['B3', 'B8']).rename('ndwi')
            water_body = ndwi.gte(NDWI_THRESHOLD_CHLOROPHYLL).And(is_snow.Not())
            
            # CHLOROPHYLL INDEX: 4.26 * (B3/B1)^3.94
            chl_index = sr.expression(
                "4.26 * pow((B03 / B01), 3.94)",
                {
                    "B03": sr.select('B3'),
                    "B01": sr.select('B1')
                }
            ).rename('wq_index')
            
            # Apply all masks
            wq_masked = chl_index.updateMask(cloud_free).updateMask(water_body)
            
            # RGB bands for visualization
            rgb = sr.select(['B4', 'B3', 'B2'])
            
            combined = (wq_masked
                       .addBands(rgb)
                       .addBands(water_body.rename('water_mask'))
                       .addBands(is_snow.rename('snow_mask'))
                       .addBands(ndsi))
            
            return combined.clip(aoi).copyProperties(img, ['system:time_start'])
        
        return s2_joined.map(calculate_chlorophyll)


def get_monthly_composite(wq_collection, aoi, year, month):
    """
    Create monthly composite from water quality collection.
    Returns composite with water quality index and RGB bands.
    """
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')
    
    monthly = wq_collection.filterDate(start, end)
    
    # Check image count
    count = monthly.size().getInfo()
    
    if count == 0:
        return None, 0, "No images"
    
    # Create composite - use median
    composite = monthly.median()
    
    # Calculate statistics
    stats = composite.select('wq_index').reduceRegion(
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
    Download monthly composite (Water Quality Index + RGB + Snow mask bands).
    Returns paths to downloaded files.
    """
    wq_path = os.path.join(temp_dir, f"wq_index_{month_name}.tif")
    rgb_path = os.path.join(temp_dir, f"rgb_{month_name}.tif")
    snow_path = os.path.join(temp_dir, f"snow_{month_name}.tif")
    
    # Check cache
    wq_valid, _ = validate_geotiff_file(wq_path, expected_bands=1)
    rgb_valid, _ = validate_geotiff_file(rgb_path, expected_bands=3)
    snow_valid, _ = validate_geotiff_file(snow_path, expected_bands=1)
    
    if wq_valid and rgb_valid and snow_valid:
        if status_placeholder:
            status_placeholder.info(f"✅ {month_name} using cache")
        return wq_path, rgb_path, snow_path, STATUS_COMPLETE, "Cached"
    
    try:
        # Download Water Quality Index
        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Downloading water quality index...")
        
        success, msg = download_band_with_retry(composite, 'wq_index', aoi, wq_path, scale)
        if not success:
            return None, None, None, STATUS_FAILED, f"WQ Index download failed: {msg}"
        
        # Download Snow mask
        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Downloading snow mask...")
        
        success, msg = download_band_with_retry(composite, 'snow_mask', aoi, snow_path, scale)
        if not success:
            return None, None, None, STATUS_FAILED, f"Snow mask download failed: {msg}"
        
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
                return None, None, None, STATUS_FAILED, f"RGB {band} download failed: {msg}"
            
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
        
        return wq_path, rgb_path, snow_path, STATUS_COMPLETE, "Downloaded"
        
    except Exception as e:
        return None, None, None, STATUS_FAILED, f"Error: {str(e)}"


# =============================================================================
# Visualization Functions
# =============================================================================
def create_turbidity_colormap():
    """Create custom colormap for turbidity visualization."""
    # Blue (clear) -> Green -> Yellow -> Red (turbid)
    colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
    return LinearSegmentedColormap.from_list('turbidity', colors, N=256)


def create_chlorophyll_colormap():
    """Create rainbow colormap for chlorophyll visualization (matching notebook)."""
    # Rainbow colormap: violet -> blue -> cyan -> green -> yellow -> orange -> red
    colors = ['#9400D3', '#4B0082', '#0000FF', '#00FF00', '#FFFF00', '#FF7F00', '#FF0000']
    return LinearSegmentedColormap.from_list('chlorophyll', colors, N=256)


def generate_thumbnails(wq_path, rgb_path, snow_path, month_name, parameter_type, max_size=300):
    """Generate RGB, water quality index, and snow mask thumbnails."""
    try:
        # Read Water Quality Index
        with rasterio.open(wq_path) as src:
            wq_data = src.read(1)
        
        # Read RGB
        with rasterio.open(rgb_path) as src:
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)
        
        # Read Snow mask
        with rasterio.open(snow_path) as src:
            snow_data = src.read(1)
        
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
        
        # Create water quality visualization
        wq_valid = np.nan_to_num(wq_data, nan=np.nan)
        
        # Calculate mean value (excluding NaN)
        valid_wq = wq_valid[~np.isnan(wq_valid) & (wq_valid != 0)]
        mean_value = np.nanmean(valid_wq) if len(valid_wq) > 0 else np.nan
        valid_pixel_count = len(valid_wq)
        total_pixels = wq_data.size
        water_coverage = (valid_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Calculate snow coverage
        snow_valid = np.nan_to_num(snow_data, nan=0)
        snow_pixels = np.sum(snow_valid > 0)
        snow_coverage = (snow_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Choose colormap and normalization based on parameter type
        if parameter_type == PARAM_TURBIDITY:
            cmap = create_turbidity_colormap()
            # Normalize NDTI to 0-1 range (typical range -0.3 to 0.3)
            wq_normalized = np.clip((wq_valid + 0.3) / 0.6, 0, 1)
        else:  # CHLOROPHYLL
            cmap = create_chlorophyll_colormap()
            # Normalize chlorophyll to 0-1 range (typical range 2 to 100)
            wq_normalized = np.clip((wq_valid - CHL_VMIN) / (CHL_VMAX - CHL_VMIN), 0, 1)
        
        wq_normalized = np.nan_to_num(wq_normalized, nan=0)
        
        # Apply colormap
        wq_colored = cmap(wq_normalized)[:, :, :3]  # RGB only
        wq_uint8 = (wq_colored * 255).astype(np.uint8)
        
        # Mask non-water areas (where WQ index is NaN or 0)
        water_mask = (~np.isnan(wq_valid)) & (wq_valid != 0)
        for i in range(3):
            wq_uint8[:, :, i] = np.where(water_mask, wq_uint8[:, :, i], 50)  # Gray for non-water
        
        # Create snow visualization (cyan/white for snow)
        snow_rgb = np.zeros((*snow_data.shape, 3), dtype=np.uint8)
        snow_rgb[:, :, 0] = np.where(snow_valid > 0, 200, 50)  # R - light for snow
        snow_rgb[:, :, 1] = np.where(snow_valid > 0, 230, 50)  # G - light for snow
        snow_rgb[:, :, 2] = np.where(snow_valid > 0, 255, 50)  # B - bright for snow
        
        # Create combined RGB with snow highlighted (cyan overlay)
        rgb_with_snow = rgb_uint8.copy()
        snow_mask_bool = snow_valid > 0
        rgb_with_snow[:, :, 0] = np.where(snow_mask_bool, 
                                          np.clip(rgb_uint8[:, :, 0] * 0.5 + 100, 0, 255).astype(np.uint8),
                                          rgb_uint8[:, :, 0])
        rgb_with_snow[:, :, 1] = np.where(snow_mask_bool,
                                          np.clip(rgb_uint8[:, :, 1] * 0.5 + 200, 0, 255).astype(np.uint8),
                                          rgb_uint8[:, :, 1])
        rgb_with_snow[:, :, 2] = np.where(snow_mask_bool,
                                          np.clip(rgb_uint8[:, :, 2] * 0.5 + 255, 0, 255).astype(np.uint8),
                                          rgb_uint8[:, :, 2])
        
        # Convert to PIL
        pil_rgb = Image.fromarray(rgb_uint8, mode='RGB')
        pil_wq = Image.fromarray(wq_uint8, mode='RGB')
        pil_snow = Image.fromarray(snow_rgb, mode='RGB')
        pil_rgb_snow = Image.fromarray(rgb_with_snow, mode='RGB')
        
        # Resize
        h, w = pil_rgb.size[1], pil_rgb.size[0]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_rgb = pil_rgb.resize((new_w, new_h), Image.LANCZOS)
            pil_wq = pil_wq.resize((new_w, new_h), Image.LANCZOS)
            pil_snow = pil_snow.resize((new_w, new_h), Image.NEAREST)
            pil_rgb_snow = pil_rgb_snow.resize((new_w, new_h), Image.LANCZOS)
        
        return {
            'rgb_image': pil_rgb,
            'wq_image': pil_wq,
            'snow_image': pil_snow,
            'rgb_snow_image': pil_rgb_snow,
            'month_name': month_name,
            'mean_value': mean_value,
            'water_coverage': water_coverage,
            'snow_coverage': snow_coverage,
            'valid_pixels': valid_pixel_count,
            'snow_pixels': snow_pixels,
            'parameter_type': parameter_type
        }
        
    except Exception as e:
        st.warning(f"Error generating thumbnails for {month_name}: {e}")
        return None


# =============================================================================
# Main Processing Pipeline
# =============================================================================
def process_water_quality_timeseries(aoi, start_date, end_date, parameter_type,
                                      cloudy_pixel_percentage=30, scale=10, resume=False):
    """Main processing pipeline for water quality analysis (Turbidity or Chlorophyll)."""
    try:
        if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
            st.session_state.current_temp_dir = tempfile.mkdtemp()
        temp_dir = st.session_state.current_temp_dir
        
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
        
        # Parameter-specific labels
        param_short = "NDTI" if parameter_type == PARAM_TURBIDITY else "Chl-a"
        param_icon = "🌊" if parameter_type == PARAM_TURBIDITY else "🌿"
        
        st.info(f"📅 Processing {total_months} months | {param_icon} {parameter_type} | 📁 {temp_dir}")
        
        # =====================================================================
        # PHASE 1: Create water quality collection
        # =====================================================================
        st.header(f"Phase 1: Create {parameter_type} Collection")
        
        if parameter_type == PARAM_TURBIDITY:
            st.info(f"""
            ☁️ Cloud: < {CLOUD_PROB_THRESHOLD}% | 
            ❄️ Snow: NDSI > {NDSI_THRESHOLD}, B11 > {SNOW_B11_THRESHOLD} | 
            💧 Water: NDWI(B3,B12) > {NDWI_THRESHOLD_TURBIDITY} |
            📊 NDTI = (B4-B3)/(B4+B3)
            """)
        else:
            st.info(f"""
            ☁️ Cloud: < {CLOUD_PROB_THRESHOLD}% | 
            ❄️ Snow: NDSI > {NDSI_THRESHOLD}, B11 > {SNOW_B11_THRESHOLD} | 
            💧 Water: NDWI(B3,B8) ≥ {NDWI_THRESHOLD_CHLOROPHYLL} |
            📊 Chl = 4.26 × (B3/B1)^3.94
            """)
        
        with st.spinner(f"Creating {parameter_type} collection..."):
            wq_collection = create_water_quality_collection(
                aoi, start_date, end_date, parameter_type, cloudy_pixel_percentage
            )
            collection_size = wq_collection.size().getInfo()
        
        st.success(f"✅ Collection created with {collection_size} images")
        
        # =====================================================================
        # PHASE 2: Download monthly composites
        # =====================================================================
        st.header("Phase 2: Download Monthly Data")
        
        downloaded_months = {}
        month_statuses = {}
        mean_wq_data = {}
        
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
                if paths.get('wq_index') and paths.get('rgb') and paths.get('snow'):
                    wq_valid, _ = validate_geotiff_file(paths['wq_index'], 1)
                    rgb_valid, _ = validate_geotiff_file(paths['rgb'], 3)
                    snow_valid, _ = validate_geotiff_file(paths['snow'], 1)
                    if wq_valid and rgb_valid and snow_valid:
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
                    wq_collection, aoi, 
                    month_info['year'], month_info['month']
                )
                
                if composite is None or count == 0:
                    month_statuses[month_name] = {'status': STATUS_NO_DATA, 'message': 'No images'}
                    st.session_state.month_statuses[month_name] = month_statuses[month_name]
                    st.write(f"⚫ **{month_name}**: No data")
                    progress_bar.progress((idx + 1) / len(months_to_process))
                    continue
                
                # Download data
                wq_path, rgb_path, snow_path, status, message = download_monthly_data(
                    composite, aoi, temp_dir, month_name, scale, status_text
                )
                
                month_statuses[month_name] = {'status': status, 'message': message}
                st.session_state.month_statuses[month_name] = month_statuses[month_name]
                
                if status == STATUS_COMPLETE:
                    downloaded_months[month_name] = {'wq_index': wq_path, 'rgb': rgb_path, 'snow': snow_path}
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
            thumb = generate_thumbnails(
                paths['wq_index'], paths['rgb'], paths['snow'], 
                month_name, parameter_type
            )
            
            if thumb:
                results.append(thumb)
                mean_wq_data[month_name] = {
                    'mean': thumb['mean_value'],
                    'coverage': thumb['water_coverage'],
                    'snow': thumb['snow_coverage']
                }
            
            progress_bar.progress((idx + 1) / len(month_names))
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.mean_turbidity_data = mean_wq_data
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
    """Display water quality results with RGB, index, and snow images."""
    if not results:
        return
    
    # Get parameter type from first result
    parameter_type = results[0].get('parameter_type', PARAM_TURBIDITY)
    param_short = "NDTI" if parameter_type == PARAM_TURBIDITY else "Chl-a"
    param_icon = "🌊" if parameter_type == PARAM_TURBIDITY else "🌿"
    
    st.subheader(f"{param_icon} Monthly {parameter_type} Results")
    
    # Display mode selection
    mode = st.radio("Display:", ["Side by Side", f"{param_short} Only", "RGB Only", "Snow Detection", "RGB + Snow Overlay"], horizontal=True)
    
    # Color legend
    with st.expander("🎨 Color Legends"):
        col1, col2 = st.columns(2)
        
        with col1:
            if parameter_type == PARAM_TURBIDITY:
                st.write("**Turbidity (NDTI)**")
                fig, ax = plt.subplots(figsize=(6, 0.5))
                cmap = create_turbidity_colormap()
                gradient = np.linspace(0, 1, 256).reshape(1, -1)
                ax.imshow(gradient, aspect='auto', cmap=cmap)
                ax.set_xticks([0, 128, 255])
                ax.set_xticklabels(['-0.3 (Clear)', '0 (Moderate)', '0.3 (Turbid)'])
                ax.set_yticks([])
                st.pyplot(fig)
                plt.close()
            else:
                st.write("**Chlorophyll Index**")
                fig, ax = plt.subplots(figsize=(6, 0.5))
                cmap = create_chlorophyll_colormap()
                gradient = np.linspace(0, 1, 256).reshape(1, -1)
                ax.imshow(gradient, aspect='auto', cmap=cmap)
                ax.set_xticks([0, 128, 255])
                ax.set_xticklabels([f'{CHL_VMIN} (Low)', f'{(CHL_VMIN+CHL_VMAX)/2:.0f}', f'{CHL_VMAX} (High)'])
                ax.set_yticks([])
                st.pyplot(fig)
                plt.close()
        
        with col2:
            st.write("**Snow Detection**")
            st.markdown("""
            - 🔵 **Cyan/White**: Snow/Ice detected (NDSI > 0.4, B11 > 0.1)
            - ⬛ **Gray**: No snow
            - Water quality is calculated **excluding** snow pixels
            """)
    
    st.divider()
    
    if mode == "Side by Side":
        for i in range(0, len(results), 2):
            cols = st.columns(4)
            for j in range(2):
                idx = i + j
                if idx < len(results):
                    r = results[idx]
                    if parameter_type == PARAM_TURBIDITY:
                        mean_str = f"{r['mean_value']:.4f}" if not np.isnan(r['mean_value']) else "N/A"
                    else:
                        mean_str = f"{r['mean_value']:.2f}" if not np.isnan(r['mean_value']) else "N/A"
                    snow_str = f" ❄️{r['snow_coverage']:.1f}%" if r['snow_coverage'] > 0 else ""
                    cols[j*2].image(r['rgb_image'], caption=f"{r['month_name']} RGB{snow_str}")
                    cols[j*2+1].image(r['wq_image'], 
                                     caption=f"{r['month_name']} {param_short}: {mean_str}")
    
    elif mode == "Snow Detection":
        for row in range((len(results) + 3) // 4):
            cols = st.columns(4)
            for c in range(4):
                idx = row * 4 + c
                if idx < len(results):
                    r = results[idx]
                    snow_pct = r['snow_coverage']
                    cap = f"{r['month_name']} ❄️ {snow_pct:.1f}%"
                    cols[c].image(r['snow_image'], caption=cap)
    
    elif mode == "RGB + Snow Overlay":
        for row in range((len(results) + 3) // 4):
            cols = st.columns(4)
            for c in range(4):
                idx = row * 4 + c
                if idx < len(results):
                    r = results[idx]
                    snow_pct = r['snow_coverage']
                    water_pct = r['water_coverage']
                    cap = f"{r['month_name']} 💧{water_pct:.1f}% ❄️{snow_pct:.1f}%"
                    cols[c].image(r['rgb_snow_image'], caption=cap)
    
    else:
        key = 'wq_image' if param_short in mode else 'rgb_image'
        for row in range((len(results) + 3) // 4):
            cols = st.columns(4)
            for c in range(4):
                idx = row * 4 + c
                if idx < len(results):
                    r = results[idx]
                    if param_short in mode:
                        if parameter_type == PARAM_TURBIDITY:
                            mean_str = f"{r['mean_value']:.4f}" if not np.isnan(r['mean_value']) else "N/A"
                        else:
                            mean_str = f"{r['mean_value']:.2f}" if not np.isnan(r['mean_value']) else "N/A"
                        cap = f"{r['month_name']} ({param_short}: {mean_str})"
                    else:
                        cap = r['month_name']
                    cols[c].image(r[key], caption=cap)


def display_water_quality_chart(results):
    """Display time series chart of mean water quality values with snow coverage."""
    if not results:
        return
    
    # Get parameter type from first result
    parameter_type = results[0].get('parameter_type', PARAM_TURBIDITY)
    param_short = "NDTI" if parameter_type == PARAM_TURBIDITY else "Chl-a"
    param_icon = "🌊" if parameter_type == PARAM_TURBIDITY else "🌿"
    param_unit = "" if parameter_type == PARAM_TURBIDITY else " (µg/L)"
    
    st.subheader(f"📊 Mean {parameter_type} Time Series")
    
    # Prepare data
    months = []
    mean_values = []
    coverage_values = []
    snow_values = []
    
    for r in results:
        months.append(r['month_name'])
        mean_values.append(r['mean_value'] if not np.isnan(r['mean_value']) else 0)
        coverage_values.append(r['water_coverage'])
        snow_values.append(r['snow_coverage'])
    
    if not months:
        st.warning("No data to plot")
        return
    
    # Check if any valid data
    valid_values = [m for m in mean_values if m != 0]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    
    # =========================================================================
    # Plot 1: Mean water quality index with water coverage
    # =========================================================================
    color1 = '#1f77b4' if parameter_type == PARAM_TURBIDITY else '#228B22'
    ax1.set_xlabel('Month')
    ax1.set_ylabel(f'Mean {param_short}{param_unit}', color=color1)
    
    if valid_values:
        line1 = ax1.plot(months, mean_values, 'o-', color=color1, linewidth=2, markersize=8, label=f'Mean {param_short}')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Set y-axis limits based on parameter type
        if parameter_type == PARAM_TURBIDITY:
            ax1.set_ylim(min(mean_values) - 0.02, max(mean_values) + 0.02)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral (NDTI=0)')
        else:
            ax1.set_ylim(0, max(mean_values) * 1.2)
    else:
        ax1.text(0.5, 0.5, f'No valid {param_short} data\n(water may be frozen/snow-covered)', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    
    # Rotate x-axis labels
    ax1.set_xticklabels(months, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Second y-axis for water coverage
    ax1_twin = ax1.twinx()
    color2 = '#2ca02c'
    ax1_twin.set_ylabel('Water Coverage (%)', color=color2)
    ax1_twin.bar(months, coverage_values, alpha=0.3, color=color2, label='Water Coverage')
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    ax1_twin.set_ylim(0, max(coverage_values) * 1.3 if max(coverage_values) > 0 else 100)
    
    ax1.set_title(f'{param_icon} {parameter_type} Analysis Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    
    # =========================================================================
    # Plot 2: Snow coverage time series
    # =========================================================================
    color_snow = '#00bfff'
    ax2.fill_between(months, snow_values, alpha=0.5, color=color_snow, label='Snow Coverage')
    ax2.plot(months, snow_values, 'o-', color='#0080ff', linewidth=2, markersize=6)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Snow Coverage (%)', color='#0080ff')
    ax2.tick_params(axis='y', labelcolor='#0080ff')
    ax2.set_xticklabels(months, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(snow_values) * 1.3 if max(snow_values) > 0 else 10)
    ax2.set_title('❄️ Snow/Ice Coverage Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Statistics table
    st.subheader("📈 Statistics Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if valid_values:
        if parameter_type == PARAM_TURBIDITY:
            col1.metric(f"Mean {param_short}", f"{np.mean(valid_values):.4f}")
            col2.metric(f"Max {param_short}", f"{np.max(valid_values):.4f}")
            col3.metric(f"Min {param_short}", f"{np.min(valid_values):.4f}")
        else:
            col1.metric(f"Mean {param_short}", f"{np.mean(valid_values):.2f}")
            col2.metric(f"Max {param_short}", f"{np.max(valid_values):.2f}")
            col3.metric(f"Min {param_short}", f"{np.min(valid_values):.2f}")
    else:
        col1.metric(f"Mean {param_short}", "N/A")
        col2.metric(f"Max {param_short}", "N/A")
        col3.metric(f"Min {param_short}", "N/A")
    
    col4.metric("Avg Water %", f"{np.mean(coverage_values):.1f}%")
    col5.metric("Avg Snow %", f"{np.mean(snow_values):.1f}%")
    
    # Data table
    with st.expander("📋 Monthly Data Table"):
        import pandas as pd
        
        if parameter_type == PARAM_TURBIDITY:
            value_col = [f"{v:.4f}" if v != 0 else "N/A" for v in mean_values]
        else:
            value_col = [f"{v:.2f}" if v != 0 else "N/A" for v in mean_values]
        
        df = pd.DataFrame({
            'Month': months,
            f'Mean {param_short}': value_col,
            'Water Coverage (%)': [f"{v:.1f}" for v in coverage_values],
            'Snow Coverage (%)': [f"{v:.1f}" for v in snow_values]
        })
        st.dataframe(df, use_container_width=True)


# =============================================================================
# Main Application
# =============================================================================
def main():
    st.title("🌊 Water Quality Analysis")
    st.markdown("""
    **Water Quality Parameter Analysis** using Sentinel-2 imagery.
    
    Choose between **Turbidity (NDTI)** or **Chlorophyll Index** analysis.
    """)
    
    # Initialize Earth Engine
    ee_ok, ee_msg = initialize_earth_engine()
    if not ee_ok:
        st.error(ee_msg)
        st.stop()
    st.sidebar.success(ee_msg)
    
    # ==========================================================================
    # PARAMETER SELECTION
    # ==========================================================================
    st.sidebar.header("🔬 Water Quality Parameter")
    
    parameter_type = st.sidebar.radio(
        "Select Parameter:",
        [PARAM_TURBIDITY, PARAM_CHLOROPHYLL],
        index=0 if st.session_state.selected_parameter == PARAM_TURBIDITY else 1,
        disabled=st.session_state.processing_in_progress,
        help="""
        **Turbidity (NDTI)**: Measures water clarity/suspended sediments
        **Chlorophyll**: Measures algae/phytoplankton concentration
        """
    )
    st.session_state.selected_parameter = parameter_type
    
    # Show parameter info
    if parameter_type == PARAM_TURBIDITY:
        st.sidebar.info("""
        **NDTI Formula:**
        `(B4 - B3) / (B4 + B3)`
        
        **Water Detection:**
        `NDWI(B3,B12) > 0.1`
        
        **Range:** -0.3 to 0.3
        """)
    else:
        st.sidebar.info("""
        **Chlorophyll Formula:**
        `4.26 × (B3/B1)^3.94`
        
        **Water Detection:**
        `NDWI(B3,B8) ≥ 0.05`
        
        **Range:** 2 to 100 µg/L
        """)
    
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
        cache_info.append(f"📊 {len(st.session_state.turbidity_results)} results")
    
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
    
    # ==========================================================================
    # ALGORITHM INFO TABLE
    # ==========================================================================
    with st.expander("📋 Algorithm Details", expanded=False):
        if parameter_type == PARAM_TURBIDITY:
            st.markdown("""
            | Step | Parameter | Formula/Threshold | Description |
            |------|-----------|-------------------|-------------|
            | 1 | Cloud Mask | probability < 20% | Remove cloudy pixels |
            | 2 | Snow Detection | NDSI > 0.4 & B11 > 0.1 | Identify snow/ice |
            | 3 | Water Body | NDWI(B3,B12) > 0.1 | Detect water (excluding snow) |
            | 4 | **Turbidity** | **(B4-B3)/(B4+B3)** | **Calculate NDTI** |
            """)
        else:
            st.markdown("""
            | Step | Parameter | Formula/Threshold | Description |
            |------|-----------|-------------------|-------------|
            | 1 | Cloud Mask | probability < 20% | Remove cloudy pixels |
            | 2 | Snow Detection | NDSI > 0.4 & B11 > 0.1 | Identify snow/ice |
            | 3 | Water Body | NDWI(B3,B8) ≥ 0.05 | Detect water (excluding snow) |
            | 4 | **Chlorophyll** | **4.26 × (B3/B1)^3.94** | **Calculate Chl-a index** |
            """)
    
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
    
    # Show selected parameter
    param_icon = "🌊" if parameter_type == PARAM_TURBIDITY else "🌿"
    st.info(f"{param_icon} **Selected Parameter:** {parameter_type}")
    
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
            'cloudy_pct': cloudy_pct,
            'parameter_type': parameter_type
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
                'cloudy_pct': cloudy_pct,
                'parameter_type': parameter_type
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
            results = process_water_quality_timeseries(
                aoi,
                config['start_date'],
                config['end_date'],
                config.get('parameter_type', PARAM_TURBIDITY),
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
        
        display_water_quality_chart(st.session_state.turbidity_results)


if __name__ == "__main__":
    main()
