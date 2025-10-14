import argparse
import os
import os.path
import shutil
import sys
import time
import glob
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
from pathlib import Path
import datetime
import calendar
from urllib.parse import urljoin
import re
import datetime
from urllib.parse import urljoin
from io import StringIO
import requests
import csv
import json
from bs4 import BeautifulSoup
from scipy.interpolate import griddata
from exactextract import exact_extract
from rasterio.features import geometry_mask
# import extractextract as ee

try:
    import gdal
except:
    from osgeo import gdal
import rasterio
from rasterio.merge import merge
from rasterio.plot import show_hist
from rasterstats import zonal_stats
import geopandas as gpd
from functools import partial
import pyproj
from shapely.ops import transform

from pathlib import Path
import pandas as pd

loc_dir = Path(os.getcwd())

angles  = ["OffNadir_Composite_Snow_Free", "NearNadir_Composite_Snow_Free", "AllAngle_Composite_Snow_Free"]

def file_exists(url, headers):
    """Check if a file exists at a given URL."""
    response = requests.head(url, headers=headers)
    return response.status_code == 200

def download_file(url, dest_path, token):
    """Download a file from a URL with proper chunking."""
    headers = {'Authorization': f'Bearer {token}'}
    try:
        with requests.get(url, headers=headers, stream=True) as response:
            response.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"Downloaded successfully: {dest_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

def downloadh5_dates(year, loc_dir, h5_folder, token, h5_list):
    """
    Download H5 files for the specified year from the Black Marble dataset,
    including December of the previous year for better time series imputation.
    
    Parameters:
        year (int): The main target year for which to download data.
        loc_dir (Path): The base directory where data will be saved.
        h5_folder (str): The folder name to store downloaded H5 files.
        token (str): Authorization token for the data request.
        h5_list (list): List of tile IDs to download.
    """
    def fetch_day_entries(base_year):
        url = urljoin("https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5000/VNP46A3/", str(base_year))
        try:
            csv_data = geturl(f'{url}.csv', token)
            days = [f for f in csv.DictReader(StringIO(csv_data), skipinitialspace=True)]
            print(f"✅ CSV data fetched for {base_year}")
        except Exception as e:
            print(f"⚠️ CSV fetch failed for {base_year}: {e}. Trying JSON instead.")
            json_data = geturl(f'{url}.json', token)
            days = json.loads(json_data)
            print(f"✅ JSON data fetched for {base_year}")
        return days

    # Fetch entries for target year and previous year
    all_days = {
        year - 1: fetch_day_entries(year - 1),
        year: fetch_day_entries(year)
        # year + 1: fetch_day_entries(year + 1)
    }

    # Filter only relevant days: Dec of previous year, all of target year, Jan of next year
    valid_days = set()
    for y, days in all_days.items():
        for d in days:
            day = int(d['name'])
            if y == year - 1 and day >= 335:
                valid_days.add((y, day))
            elif y == year:
                valid_days.add((y, day))
            # elif y == year + 1 and day <= 31:
            #     valid_days.add((y, day))

    # Prepare base destination folder
    dest_folder = loc_dir / h5_folder
    os.makedirs(dest_folder, exist_ok=True)

    for y, day in sorted(valid_days):
        day_string = f"{day:03d}"
        source_URL = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5000/VNP46A3/{y}/{day_string}/"
        dest_day_folder = dest_folder / f"{y}{day_string}"
        os.makedirs(dest_day_folder, exist_ok=True)

        for tile in h5_list:
            filename_pattern = f"VNP46A3.A{y}{day_string}.{tile}.001"
            try:
                response = requests.get(source_URL, headers={'Authorization': f'Bearer {token}'})
                soup = BeautifulSoup(response.content, "html.parser")
            except Exception as e:
                print(f"Error accessing {source_URL} for tile {tile}: {e}")
                continue

            for link in soup.find_all('a'):
                filename = link.get('href')
                if filename and filename_pattern in filename and filename.endswith('.h5'):
                    dest_path = dest_day_folder / os.path.basename(filename)
                    if dest_path.exists():
                        print(f"Already exists, skipping: {filename}")
                        break

                    file_URL = urljoin(source_URL, filename)
                    for attempt in range(11):
                        try:
                            # download_file(file_URL, dest_path, token)
                            # print(f"Downloaded: {filename} to {dest_path}")
                            # break
                            download_file(file_URL, dest_path, token)
                            # Check if file exists and has nonzero size
                            if dest_path.exists() and dest_path.stat().st_size > 0:
                                print(f"✅ Downloaded: {filename} to {dest_path}")
                                break
                            else:
                                print(f"❌ File appears missing or empty after download: {dest_path}")
                                dest_path.unlink(missing_ok=True)  # remove if corrupted
                                if attempt == 11:
                                    print(f"🚫 Failed after 3 attempts: {filename}")
                        except Exception as e:
                            print(f"Attempt {attempt + 1} failed for {filename}: {e}")
                            if attempt == 10:
                                print(f"Failed to download {filename} after 10 attempts.")
                    break  # Proceed to next tile


def geturl(url, token=None, out=None):
    headers = {'user-agent': 'my_downloader'}
    if token is not None:
        headers['Authorization'] = 'Bearer ' + token
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        if out:
            out.write(response.content)
        else:
            return response.content.decode('utf-8')
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None

def extract_offnadir_and_quality(src_h5, destination, angle):
    """Extract specified angle composite and quality subdatasets and save as GeoTIFFs."""
    rasterFilePre = src_h5.name
    fileExtension = ".tif"
    hdflayer = gdal.Open(str(src_h5), gdal.GA_ReadOnly)
    
    if hdflayer is not None:
        composite_subdataset = None
        quality_subdataset = None
        
        for i in hdflayer.GetSubDatasets():
            subhdflayer = i[0]
            a = re.split('/', subhdflayer)
            if a[-1] == angle:
                composite_subdataset = subhdflayer
            elif a[-1] == f"{angle}_Quality":
                quality_subdataset = subhdflayer
        
        if composite_subdataset and quality_subdataset:
            rlayer = gdal.Open(composite_subdataset, gdal.GA_ReadOnly)
            composite_data_scaled = rlayer.ReadAsArray() * 0.1
            composite_output = os.path.join(destination, f"{angle}_{rasterFilePre}{fileExtension}")
            
            # Set up geotransform and projection
            HorizontalTileNumber = int(rlayer.GetMetadata_Dict()["HorizontalTileNumber"])
            VerticalTileNumber = int(rlayer.GetMetadata_Dict()["VerticalTileNumber"])
            WestBoundCoord = (10 * HorizontalTileNumber) - 180
            NorthBoundCoord = 90 - (10 * VerticalTileNumber)
            EastBoundCoord = WestBoundCoord + 10
            SouthBoundCoord = NorthBoundCoord - 10
            EPSG = "-a_srs EPSG:4326"
            translateOptionText = f"{EPSG} -a_ullr {WestBoundCoord} {NorthBoundCoord} {EastBoundCoord} {SouthBoundCoord}"
            translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine(translateOptionText))

            # Create a memory dataset to hold the scaled data and save it as GeoTIFF
            driver = gdal.GetDriverByName('MEM')
            mem_dataset = driver.Create('', composite_data_scaled.shape[1], composite_data_scaled.shape[0], 1, gdal.GDT_Float32)
            mem_dataset.GetRasterBand(1).WriteArray(composite_data_scaled)
            mem_dataset.SetGeoTransform(rlayer.GetGeoTransform())  # Set the geotransform from the original layer
            mem_dataset.SetProjection(rlayer.GetProjection())      # Set the projection from the original layer

            # Write the extracted composite data to a GeoTIFF file
            try:
                gdal.Translate(str(composite_output), mem_dataset, options=translateoptions)
                print(f"Saved OffNadir composite to: {composite_output}")
            except Exception as e:
                print(f"Error writing composite GeoTIFF: {e}")

            # Extract and save the quality layer similarly
            qlayer = gdal.Open(quality_subdataset, gdal.GA_ReadOnly)
            quality_output = os.path.join(destination, f"{angle}_Quality_{rasterFilePre}{fileExtension}")
            try:
                gdal.Translate(str(quality_output), qlayer, options=translateoptions)
                print(f"Saved OffNadir quality to: {quality_output}")
            except Exception as e:
                print(f"Error writing quality GeoTIFF: {e}")

            return composite_output, quality_output
        else:
            print("Could not find required subdatasets for composite and quality.")
            return None, None
    else:
        print("HDF layer could not be opened.")
        return None, None

#old masking
def apply_quality_mask(composite_path, quality_path, destination):
    """Apply the quality mask to the OffNadir composite and save the filtered output."""
    with rasterio.open(composite_path) as composite, rasterio.open(quality_path) as quality:
        composite_data = composite.read(1).astype(float)  # Convert to float to handle NaNs
        quality_data = quality.read(1)

        # Mask out poor quality pixels (quality value `01`)
        composite_data[quality_data == 1] = np.nan
        # composite_data[np.isin(quality_data, [1, 2])] = np.nan

        # Define output path
        # filtered_output = Path(destination) / f"{Path(composite_path).stem}_Filtered.tif"
        filtered_output = destination / f"{composite_path.stem}_Filtered.tif"

        # Update metadata and save the filtered output
        profile = composite.profile
        profile.update(dtype='float32')

        with rasterio.open(filtered_output, 'w', **profile) as dst:
            dst.write(composite_data, 1)

        print(f"Filtered and saved to: {filtered_output}")
        return filtered_output

def decompress_gz(file_path):
    """Decompress a .gz file and return the path to the extracted .tif file."""
    file_path = Path(file_path)
    
    if file_path.suffix != ".gz":
        print(f"✅ File is already decompressed: {file_path}")
        return file_path  # Return the original path if it's already a .tif

    decompressed_path = file_path.with_suffix('')  # Remove .gz extension (e.g., .tif.gz → .tif)
    print(f"🗜️ Decompressing {file_path} to {decompressed_path}")

    try:
        with gzip.open(file_path, 'rb') as f_in, open(decompressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)  # Fully extract the .tif data

        print(f"✅ Successfully decompressed to {decompressed_path}")
        return decompressed_path  # Return the extracted .tif path
    except Exception as e:
        print(f"❌ Error decompressing {file_path}: {e}")
        return None  # Return None if extraction fails

def apply_ephemeral_mask(viirs_path, ephemeral_mask_path, output_path):
    """Apply ephemeral lights mask (1 = retain, 0 = drop) to VIIRS raster.
    
    - Decompresses .tif.gz files if needed before applying the mask.
    """
    # Decompress ephemeral mask if needed
    # ephemeral_mask_path = decompress_gz(ephemeral_mask_path)

    # if ephemeral_mask_path is None or not os.path.exists(ephemeral_mask_path):
    #     print(f"❌ Error: Ephemeral mask file not found after decompression: {ephemeral_mask_path}")
    #     return None

    # Ensure the file exists
    if not Path(ephemeral_mask_path).exists():
        print(f"❌ Error: Ephemeral mask file not found: {ephemeral_mask_path}")
        return None

    # Open VIIRS raster and ephemeral mask
    with rasterio.open(viirs_path) as viirs, rasterio.open(ephemeral_mask_path) as mask:
        viirs_data = viirs.read(1).astype(float)  # Convert VIIRS data to float for NaN handling
        mask_data = mask.read(1)  # Read ephemeral mask (1 = keep, 0 = drop)

        # Apply the mask (drop ephemeral lights)
        viirs_data[mask_data == 0] = 0  # Set ephemeral pixels to 0 (or np.nan if preferred)

        # Save the masked output
        profile = viirs.profile
        profile.update(dtype='float32')

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(viirs_data, 1)

    print(f"Step 2: Ephemeral mask applied and saved at {output_path}")
    return output_path

def apply_water_mask(viirs_path, water_mask_path, output_path):
    """
    Apply water mask to VIIRS raster.
    - Water mask: 1 = water (drop), NoData = land (keep)
    - VIIRS pixels over water are set to 0
    """

    if not Path(water_mask_path).exists():
        print(f"❌ Error: Water mask file not found: {water_mask_path}")
        return None

    with rasterio.open(viirs_path) as viirs, rasterio.open(water_mask_path) as mask:
        viirs_data = viirs.read(1).astype(float)
        mask_data = mask.read(1, masked=True)  # honors NoData in mask

        # Drop VIIRS pixels where water mask == 1
        viirs_data[mask_data == 1] = 0

        # Save the masked result
        profile = viirs.profile
        profile.update(dtype='float32')

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(viirs_data, 1)

    print(f"✅ Water mask applied and saved at: {output_path}")
    return output_path


def resample_to_black_marble(input_raster, black_marble_path, output_path):
    """Resample VIIRS raster to match the Black Marble grid (resolution, CRS)."""
    with rasterio.open(input_raster) as src, rasterio.open(black_marble_path) as bm:
        bm_transform = bm.transform
        bm_crs = bm.crs
        bm_width = bm.width
        bm_height = bm.height

        # Resampled array
        resampled_data = np.empty((bm_height, bm_width), dtype=np.float32)

        # Update metadata
        resampled_meta = src.meta.copy()
        resampled_meta.update({
            "transform": bm_transform,
            "width": bm_width,
            "height": bm_height,
            "crs": bm_crs,
            "dtype": "float32"
        })

        # Resample using bilinear interpolation
        from rasterio.warp import reproject, Resampling
        reproject(
            source=src.read(1),
            destination=resampled_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=bm_transform,
            dst_crs=bm_crs,
            resampling=Resampling.bilinear
        )

        # Save the resampled raster
        with rasterio.open(output_path, "w", **resampled_meta) as dst:
            dst.write(resampled_data, 1)

    print(f"Resampled to Black Marble and saved at {output_path}")
    return output_path

try:
    from StringIO import StringIO   # python2
except ImportError:
    from io import StringIO         # python3

#combines several geotiffs in the source folder into one mosaic file       
def mosaic(src, dest):
    src_files_to_mosaic = []
    for fp in src:
        src1 = rasterio.open(fp)
        src_files_to_mosaic.append(src1)

    #edits attributes of the single mosaic file
    out_mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({"driver": "GTiff",
                      "height": out_mosaic.shape[1],
                      "width": out_mosaic.shape[2],
                      "transform": out_trans,
                      }
                     )
    
    #writes the single mosaic file
    with rasterio.open(dest, "w", **out_meta) as dest:
        dest.write(out_mosaic)
 
# using exactextract package
def zonalStats(raster_src, zones, csv, field_id):
    """
    Compute zonal statistics using exact_extract and save results to CSV.

    Parameters:
    raster_src (str): Path to the raster file (GeoTIFF).
    zones (str): Path to the vector file (Shapefile or GeoJSON).
    csv (str): Path to the output CSV file.
    field_id (str): Column name in the vector file to use as an identifier.
    """

    # Load the vector data
    gdf = gpd.read_file(zones)

    # Load the raster to ensure CRS consistency
    with rasterio.open(raster_src) as src:
        raster_meta = src.meta

    # Ensure the vector data has the same CRS as the raster
    if gdf.crs != raster_meta["crs"]:
        gdf = gdf.to_crs(raster_meta["crs"])

    # Compute polygon area in square kilometers
    try:
        proj = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'), pyproj.Proj(init='esri:102025'))
    except:
        proj = partial(pyproj.transform, pyproj.Proj('epsg:4326'), pyproj.Proj('esri:102025'))

    gdf['area_km2'] = [transform(proj, geom).area / (1000 * 1000) for geom in gdf['geometry']]

    # Compute zonal statistics using exact_extract
    stats = exact_extract(raster_src, gdf, ["count", "sum", "mean", "stdev", "min", "max", "median"])

    # Extract statistics from 'properties' key
    def extract_stat(stats_list, stat_name):
        """ Extracts a specific statistic from the 'properties' field safely. """
        return [s["properties"].get(stat_name, None) if isinstance(s, dict) and "properties" in s else None for s in stats_list]

    # Create a DataFrame with selected columns
    df = pd.DataFrame({
        field_id: gdf[field_id],  # Unique identifier
        # "ADM3_EN": gdf["ADM3_EN"],
        # "ADM2_EN": gdf["ADM2_EN"],
        # "ADM2_PCODE": gdf["ADM2_PCODE"],
        "ADM1_EN": gdf["ADM1_EN"],
        "ADM1_PCODE": gdf["ADM1_PCODE"],
        "area_km2": gdf["area_km2"],  # Polygon area
        "min": extract_stat(stats, "min"),
        "max": extract_stat(stats, "max"),
        "mean": extract_stat(stats, "mean"),
        "count": extract_stat(stats, "count"),
        "sum": extract_stat(stats, "sum"),
        "std": extract_stat(stats, "stdev"),
        "median": extract_stat(stats, "median")
    })


    count_lit = []
    with rasterio.open(raster_src) as src:
        array = src.read(1).astype(float)
        nodata = src.nodata
        array[array == nodata] = np.nan

        for geom in gdf.geometry:
            mask = geometry_mask([geom], transform=src.transform, invert=True,
                                out_shape=(src.height, src.width))
            values = array[mask]
            count_lit.append(np.sum(values > 0))


    df["count_lit"] = count_lit

    # Save results to CSV
    df.to_csv(csv, index=False)

    print(f"Zonal statistics saved to {csv}")

#performs raster averaging
def average(src, dest):
    
    #initiates a list
    src_files_to_mosaic = []      
    
    for idx, val in enumerate(src):
        if idx == 0:
            meta = rasterio.open(val).meta #copies meta value from first raster input
        src_files_to_mosaic.append(rasterio.open(val).read(1)) #read rasters and save into a list
        
    array = np.array(src_files_to_mosaic, dtype = 'float32')  #converts list into a numpy array
    array[array < 0] = np.nan #converts masked pixels to NAN
    
    # Perform averaging, nanmean excludes all NAN values
    array_out = np.nanmean(array, axis=0)
    
    # Get metadata from one of the input files
    meta.update(dtype=rasterio.float32)

    #writes the single mosaic file
    with rasterio.open(dest, "w", **meta) as dest:
        dest.write(array_out.astype(rasterio.float32),1)

#creates masks based on preset criteria       
def mask_to_zero(src,value, dest):
    
    array = np.array(rasterio.open(src).read(1), dtype = 'float32')  #converts list into a numpy array
    array[array == value] = np.nan #converts masked pixels to NAN
       
    with rasterio.open(src) as raster:
        meta = raster.meta    
        
    meta.update(dtype=rasterio.float32)
    
    #writes the single mosaic file
    with rasterio.open(dest, "w", **meta) as dest:
        dest.write(array,1)

from scipy.interpolate import interp1d

def interpolate_ntl_1d(tif_paths, output_dir, target_year, masked_value=6553.5):
    """
    Performs per-pixel temporal interpolation of NTL rasters.
    Saves only those from the specified target year.

    Parameters:
        tif_paths (list of Path): List of GeoTIFFs from previous, current, and next year.
        output_dir (Path): Directory where interpolated rasters are saved.
        target_year (int): Year of interest for output.
        masked_value (float): The value to treat as missing (e.g., 6553.5).

    Returns:
        list of Path: Saved file paths from the target year.
    """

    def extract_date(f):
        parts = f.stem.split('.')
        yyyyddd = parts[1][1:]
        # return datetime.strptime(yyyyddd, "%Y%j")
        return datetime.datetime.strptime(yyyyddd, "%Y%j")

    tif_paths = sorted(tif_paths, key=extract_date)
    dates = [extract_date(p) for p in tif_paths]
    dates_pd = pd.to_datetime(dates)

    # Read all rasters into a 3D array (T, H, W)
    with rasterio.open(tif_paths[0]) as src:
        profile = src.profile
        height, width = src.height, src.width

    stack = np.empty((len(tif_paths), height, width), dtype=np.float32)
    for t, tif in enumerate(tif_paths):
        with rasterio.open(tif) as src:
            arr = src.read(1).astype(np.float32)
            # arr[arr == masked_value] = np.nan
            arr[(arr == masked_value) | (arr == 0.0)] = np.nan
            stack[t] = arr

    # Vectorized interpolation across time using pandas (over flattened spatial grid)
    reshaped = stack.reshape(stack.shape[0], -1)  # (T, H*W)
    df = pd.DataFrame(reshaped, index=dates_pd)
    df_interp = df.interpolate(method='linear', limit_direction='both')

    # Reshape back to (T, H, W)
    stack_interp = df_interp.to_numpy().reshape(stack.shape)

    # Save only layers from the target year
    output_paths = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for t, date in enumerate(dates):
        if date.year != target_year:
            continue

        out_path = output_dir / f"{tif_paths[t].stem.replace('Filtered', 'Interpolated')}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(stack_interp[t].astype(np.float32), 1)

        output_paths.append(out_path)

    return output_paths
