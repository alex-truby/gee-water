import ee
import geemap
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def add_ndvi_l30(image):
    """
    Adds an NDVI (Normalized Difference Vegetation Index) band to an HLS L30 image.

    The NDVI is calculated using the formula:
        NDVI = (NIR - Red) / (NIR + Red)

    This function uses the `B4` band (Red) and the `B5` band (Near Infrared)
    from the HLS L30 dataset to compute NDVI and appends it as a new band named `NDVI`.

    Args:
        image (ee.Image): A Google Earth Engine Image object from the HLS L30 dataset,
                          containing the `B4` (Red) and `B5` (NIR) bands.

    Returns:
        ee.Image: The input image with an additional `NDVI` band.
    """
    red = image.select("B4")
    nir = image.select("B5")
    ndvi = (nir.subtract(red)).divide(nir.add(red)).rename("NDVI")
    return image.addBands(ndvi)


def add_savi_l30(image, L):
    """
    Adds an SAVI (Soil Adjusted Vegetation Index) band to an HLS L30 image.

    The SAVI is calculated using the formula:
        SAVI = (NIR - Red) / (NIR + Red + L) × (1 + L)

    This function uses the `B4` band (Red) and the `B5` band (Near Infrared)
    from the HLS L30 dataset to compute NDVI and appends it as a new band named `SAVI`.

    Args:
        image (ee.Image): A Google Earth Engine Image object from the HLS L30 dataset,
                          containing the `B4` (Red) and `B5` (NIR) bands.
        L (float): Soil ajustment correction factor. Recommend adjustment factors are below:
            - 1.0:  Recommended for very sparse vegetation (e.g., desert or semi-arid regions
                with significant bare soil exposure).
            - 0.5: Used for intermediate vegetation cover (e.g., mixed shrubland or grassland with
              some bare soil exposure).
            - 0.25: Suitable for areas with higher vegetation cover, where soil influence is less
            significant.
            - 0.0 Equivalent to NDVI, used when vegetation is very dense, and soil influence is negligible.


    Returns:
        ee.Image: The input image with an additional `NDVI` band.
    """
    red = image.select("B4")
    nir = image.select("B5")
    ndvi = (
        ((nir.subtract(red)).divide(nir.add(red).add(L))).multiply(1 + L).rename("SAVI")
    )
    return image.addBands(ndvi)


def create_monthly_composites(collection, band_name=None):
    """
    Create a dictionary {(year, month): ee.Image} covering all months that
    exist in the given collection's date range. Each ee.Image has 'year' and
    'month' properties and is the mean of all images in that month.

    If a given month has no images, it creates a "dummy" image with the chosen
    band_name, masked out, to avoid errors in .select().
    """

    # 1. Find earliest and latest timestamps in the collection
    #    (these return milliseconds from epoch)
    earliest_millis = collection.aggregate_min("system:time_start")
    latest_millis = collection.aggregate_max("system:time_start")

    # 2. Convert to ee.Date objects
    earliest_date_ee = ee.Date(earliest_millis)
    latest_date_ee = ee.Date(latest_millis)

    # 3. Get the numeric year/month from these ee.Date objects
    earliest_year = earliest_date_ee.get("year").getInfo()
    earliest_month = earliest_date_ee.get("month").getInfo()
    latest_year = latest_date_ee.get("year").getInfo()
    latest_month = latest_date_ee.get("month").getInfo()

    # Dictionary to hold results
    monthly_dict = {}

    # 4. Loop over years from earliest_year to latest_year
    for year in range(earliest_year, latest_year + 1):

        # Determine the start month for this year
        if year == earliest_year:
            start_m = earliest_month
        else:
            start_m = 1

        # Determine the end month for this year
        if year == latest_year:
            end_m = latest_month
        else:
            end_m = 12

        # 5. Loop over months within that range
        for month in range(start_m, end_m + 1):

            # Construct the start/end dates for this (year, month)
            start_date = ee.Date.fromYMD(year, month, 1)
            end_date = start_date.advance(1, "month")

            # Filter collection to that date range
            monthly_coll = collection.filterDate(start_date, end_date)

            # Optionally select the band(s)
            if band_name:
                monthly_coll = monthly_coll.select([band_name])

            # Check how many images exist
            count = monthly_coll.size().getInfo()

            if count == 0:
                # Create a dummy image so .select() won't fail later
                if band_name:
                    empty_img = (
                        ee.Image(0).rename(band_name).updateMask(ee.Image(0).neq(0))
                    )  # fully masked
                else:
                    # If band_name is None, just create a 1-band image named "dummy"
                    empty_img = (
                        ee.Image(0).rename(["dummy"]).updateMask(ee.Image(0).neq(0))
                    )

                monthly_img = empty_img.set("year", year).set("month", month)

            else:
                # Mean composite
                monthly_img = monthly_coll.mean().set("year", year).set("month", month)

            monthly_dict[(year, month)] = monthly_img

    return monthly_dict


def get_time_series(monthly_dict, band_name, roi):
    """
    Given a dict {(year, month): ee.Image},
    compute mean over 'roi' for the specified 'band_name' in each image.
    Returns a pandas DataFrame with columns ['year', 'month', band_name, 'date'].
    """
    records = []

    # Loop over all (year, month) keys in the dictionary
    for (year, month), img in monthly_dict.items():
        try:
            # Select the band
            img_selected = img.select([band_name])

            # Compute mean over the ROI
            mean_dict = img_selected.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=10000, maxPixels=1e13
            )

            # Extract the mean value; could be None if dummy/empty
            mean_val = mean_dict.get(band_name).getInfo()
        except Exception as e:
            # If the band doesn't exist or another error occurs, set None
            print(f"Error for {year}-{month:02d}: {e}")
            mean_val = None

        # Add to records
        records.append({"year": year, "month": month, band_name: mean_val})

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(records)

    # Optional: create a 'date' column (use day=1 for monthly data)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1), errors="coerce")

    # Sort by date
    df = df.sort_values(by="date").reset_index(drop=True)

    return df


def get_monthly_image(collection, band, year, month):
    """
    Creates a monthly mean NDVI image from a collection for a specified year and month.

    This function filters the input collection by date, selects the specified band,
    calculates the mean for the given month, and returns the resulting image with
    a renamed band to indicate the year and month.

    Args:
        collection (ee.ImageCollection): The input ImageCollection containing NDVI or
                                         similar vegetation index data.
        band (str): The band name to select for NDVI or other vegetation index.
        year (int): The year for the monthly aggregation.
        month (int): The month for the monthly aggregation (1-12).

    Returns:
        ee.Image: A single image representing the monthly mean NDVI (or specified band),
                  with the band renamed to include the year and month in the format
                  `{band}_{year}_{month}`.
    """
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, "month")

    return (
        collection.filterDate(start_date, end_date)
        .select(band)
        .mean()
        .rename(f"{band}_{year}_{month}")
    )


def annual_agg_ic(image_collection, agg_type, start_year, end_year):
    """
    Aggregates an ImageCollection to annual totals or averages for a specified range of years.

    This function takes a Google Earth Engine ImageCollection and aggregates the images
    within each calendar year to produce a single image per year. The aggregated images
    are stored in a new ImageCollection with a 'year' property for each image indicating
    the corresponding year.

    Args:
        image_collection (ee.ImageCollection): The input ImageCollection to aggregate.
                                               Typically contains daily, monthly, or other
                                               frequent time-step data.
        agg_type (str): The aggregation type, either 'sum' for annual totals or 'mean' for annual averages.
        start_year (int): The starting year for the aggregation (inclusive).
        end_year (int): The ending year for the aggregation (inclusive).

    Returns:
        ee.ImageCollection: A new ImageCollection where each image represents the aggregated
                            value for a calendar year, with a 'year' property specifying the year.

    Raises:
        ValueError: If `agg_type` is not 'sum' or 'mean'.
    """

    def agg_for_year(year):
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(
            year + 1, 1, 1
        )  # Up to (but not including) Jan 1 of next year
        annual_coll = image_collection.filterDate(start, end)
        if agg_type == "sum":
            annual_img = annual_coll.sum().set({"year": year})
        elif agg_type == "mean":
            annual_img = annual_coll.mean().set({"year": year})
        else:
            raise ValueError("Invalid agg_type. Expected 'sum' or 'mean'.")
        return annual_img

    images = []
    for y in range(start_year, end_year + 1):
        images.append(agg_for_year(y))
    return ee.ImageCollection(images)


def get_annual_median_image(year, roi, gee_image_collection_id, selection_band):
    """
    Creates an annual median composite image from a specified GEE ImageCollection.

    This function filters the input ImageCollection by year and region of interest (ROI),
    selects the average radiance band, and calculates the median composite for the year.

    Args:
        year (int): The year for which the median composite is calculated.
        roi (ee.Geometry): The region of interest (ROI) to filter the ImageCollection.
        gee_image_collection_id (str): The Earth Engine ImageCollection ID
                                       (e.g., "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG").
        selection_band (str): Desired  band of which to take annual median.

    Returns:
        ee.Image: A single image representing the annual median composite of the
                  average radiance (`avg_rad`) band within the specified ROI and year.
    """
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = ee.Date.fromYMD(year, 12, 31)

    # Load Monthly Data
    ic = (
        ee.ImageCollection(gee_image_collection_id)
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .select([selection_band])
    )  # Select the average band

    # Create a median composite for the year
    median_image = ic.median()

    return median_image


def reduce_to_basin_means_annual(image, basins):
    """
    Reduces an image to annual mean values for specified basins.

    This function calculates the mean value of the input image for each basin
    in the provided FeatureCollection and associates the calculated values
    with the corresponding year metadata from the input image.

    Args:
        image (ee.Image): The input Earth Engine image containing annual data.
                          The image must have a 'year' property.
        basins (ee.FeatureCollection): A FeatureCollection of basins (e.g., polygons)
                                        for which mean values are calculated.

    Returns:
        ee.FeatureCollection: A FeatureCollection where each feature corresponds to a
                              basin with its mean value for the input image and the year
                              metadata added as a property.
    """
    # adjust 'scale' based on dataset's native resolution
    fc = image.reduceRegions(collection=basins, reducer=ee.Reducer.mean(), scale=10000)
    return fc.map(lambda f: f.set({"year": image.get("year")}))


def get_annual_pdf(
    feature_collection: ee.featurecollection.FeatureCollection,
    agg_type: str,
    output_col_name: str,
):
    """
    Converts a GEE FeatureCollection into a pandas DataFrame with annual aggregate values.

    This function processes a FeatureCollection, extracts relevant properties (basin ID,
    year, and an aggregation value), and compiles them into a pandas DataFrame. It is
    specifically designed to work with datasets containing annual aggregate statistics
    (e.g., WWF basin datasets).

    Args:
        feature_collection (ee.featurecollection.FeatureCollection): The input FeatureCollection
            containing annual aggregation data for basins or regions.
        agg_type (str): The property name in the FeatureCollection representing the
            aggregation type (e.g., mean, sum, etc.).
        output_col_name (str): The column name for the aggregation values in the resulting DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame with columns `HYBAS_ID`, `year`, and the specified
                          output column name, containing the basin ID, year, and the corresponding
                          aggregate values.
    """
    features_dict = feature_collection.getInfo()
    records = []

    for f in features_dict["features"]:
        props = f["properties"]
        # HYBAS is the basin ID as outlined by the WWF basin dataset
        basin_id = props.get("HYBAS_ID")

        aggregate = props.get(agg_type)

        # store year/month as integers, then create a date from them
        year = int(props["year"])

        records.append([basin_id, year, aggregate])

    pdf = pd.DataFrame(records, columns=["HYBAS_ID", "year", output_col_name])

    return pdf


def get_basin_geodataframe(basin_level_id, bounding_geom):
    """
    Generates a GeoDataFrame of basins clipped to a specified boundary geometry.

    This function loads a specified basin dataset, filters it to the provided bounding
    geometry, clips the basin geometries to the inland boundary, and converts the result
    into a GeoPandas GeoDataFrame. The output is reprojected to EPSG:4326.

    Args:
        basin_level_id (str): The Earth Engine FeatureCollection ID representing the
                              basin dataset (e.g., HydroBASINS dataset ID).
        bounding_geom (ee.Geometry): The bounding geometry used to filter and clip
                                      the basin features.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the clipped basin geometries
                                and associated attributes, reprojected to EPSG:4326.
    """
    # 1. Load and filter to bounding_geom
    basins = ee.FeatureCollection(basin_level_id).filterBounds(bounding_geom)

    # 2. Clip each basin geometry to the inland boundary
    basins_clipped = basins.map(lambda b: b.intersection(bounding_geom.geometry()))

    # 3. Convert to GeoDataFrame
    basins_gdf = ee.data.computeFeatures(
        {"expression": basins_clipped, "fileFormat": "GEOPANDAS_GEODATAFRAME"}
    )

    basins_gdf.crs = "EPSG:4326"
    return basins_gdf


def scale_modis_LST_to_Celsius(image):
    """
    Converts MODIS Land Surface Temperature (LST) data from Kelvin to Celsius.

    This function processes a MODIS image by selecting the daytime LST band, applying the
    appropriate scale factor to convert the values from scaled integers to Kelvin, and
    subtracting 273.15 to convert to Celsius. The resulting image retains the original
    properties from the input image (e.g., timestamp).

    Args:
        image (ee.Image): The input MODIS image containing the "LST_Day_1km" band.

    Returns:
        ee.Image: A new image with the "LST_Celsius" band, representing the LST in degrees Celsius,
                  while retaining the properties of the original image.
    """
    # Select the daytime LST band
    lst_day = image.select("LST_Day_1km")
    # Apply scale factor 0.02 to get Kelvin, then subtract 273.15 for Celsius
    lst_celsius = lst_day.multiply(0.02).subtract(273.15)
    # Rename the band (optional) and copy original properties (time, etc.)
    return lst_celsius.rename("LST_Celsius").copyProperties(
        image, image.propertyNames()
    )


def get_annual_precip_data(gee_id, basin_level_id, start_date, end_date, bounding_geom):
    """
    Computes annual precipitation data aggregated by basins and converts the results 
    into a pandas DataFrame.

    This function processes precipitation data from a specified GEE ImageCollection 
    and aggregates it annually for basins defined by a basin dataset. The annual 
    precipitation is computed in millimeters and converted to inches, and the results 
    are returned as a pandas DataFrame.

    Args:
        gee_id (str): The Earth Engine ImageCollection ID for the precipitation data 
                      (e.g., CHIRPS dataset ID).
        basin_level_id (str): The Earth Engine FeatureCollection ID for the basin dataset 
                              (e.g., HydroBASINS dataset ID).
        start_date (str): The start date for the analysis period in the format "YYYY-MM-DD".
        end_date (str): The end date for the analysis period in the format "YYYY-MM-DD".
        bounding_geom (ee.Geometry): The bounding geometry to filter the precipitation and 
                                      basin datasets.

    Returns:
        pandas.DataFrame: A DataFrame with annual precipitation data aggregated by basins. 
                          The DataFrame includes columns for basin ID (`HYBAS_ID`), year, 
                          annual precipitation in millimeters (`annual_precip_mm`), and 
                          annual precipitation in inches (`annual_precip_in`).
    """
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    precip = (
        ee.ImageCollection(gee_id)
        .select("precipitation")
        .filterDate(start_date, end_date)
        .filterBounds(bounding_geom)
    )

    basins = ee.FeatureCollection(basin_level_id).filterBounds(bounding_geom)

    # build annual precipitation sums image collection
    annual_chrips_precip_ic = annual_agg_ic(
        image_collection=precip,
        agg_type="sum",
        start_year=start_year,
        end_year=end_year,
    )

    # reduce images to basin level image collection (of feature collections)
    basin_reduced_col_precip = annual_chrips_precip_ic.map(
        lambda image: reduce_to_basin_means_annual(image, basins=basins)
    )

    # flatten into a single FeatureCollection
    all_features = basin_reduced_col_precip.flatten()

    precip_df = get_annual_pdf(
        feature_collection=all_features,
        agg_type="mean",
        output_col_name="annual_precip_mm",
    )

    mm_to_inches = 0.0393701
    precip_df["annual_precip_in"] = precip_df["annual_precip_mm"] * mm_to_inches

    return precip_df


def get_annual_data(
    gee_id,
    basin_level_id,
    start_date,
    end_date,
    bounding_geom,
    annual_agg_type,
    output_col_name,
    band_name=None,
) -> pd.DataFrame:
    """
    Computes annual aggregated data from a GEE ImageCollection and returns the results 
    as a pandas DataFrame, aggregated by basins.

    This function processes an ImageCollection over a specified date range, applies an 
    annual aggregation method, optionally selects and processes a specific band, and 
    aggregates the data at the basin level. The results are returned as a pandas DataFrame.

    Args:
        gee_id (str): The Earth Engine ImageCollection ID for the input data.
        basin_level_id (str): The Earth Engine FeatureCollection ID for the basin dataset 
                              (e.g., HydroBASINS dataset ID).
        start_date (str): The start date for the analysis period in the format "YYYY-MM-DD".
        end_date (str): The end date for the analysis period in the format "YYYY-MM-DD".
        bounding_geom (ee.Geometry): The bounding geometry to filter the input datasets.
        annual_agg_type (str): The type of annual aggregation to apply 
                               (e.g., "sum", "mean", "median").
        output_col_name (str): The column name for the aggregated values in the resulting DataFrame.
        band_name (str, optional): The name of the specific band to select from the 
                                   ImageCollection. If None, the entire ImageCollection is used.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the annual aggregated data. The DataFrame 
                      includes columns for basin ID (`HYBAS_ID`), year, and the specified 
                      output column name.
    """

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    if band_name:
        gee_ic = (
            ee.ImageCollection(gee_id)
            .select(band_name)
            .filterDate(start_date, end_date)
            .filterBounds(bounding_geom)
        )

        # add function to scale MODIS land surface temperature
        if gee_id == "GEE_ID" and band_name == "LST_Day_1km":
            gee_ic = gee_ic.map(scale_modis_LST_to_Celsius)
    else:
        gee_ic = (
            ee.ImageCollection(gee_id)
            .filterDate(start_date, end_date)
            .filterBounds(bounding_geom)
        )

    basins = ee.FeatureCollection(basin_level_id).filterBounds(bounding_geom)

    # build annual sums image collection
    annual_ic = annual_agg_ic(
        image_collection=gee_ic,
        agg_type=annual_agg_type,
        start_year=start_year,
        end_year=end_year,
    )

    # reduce images to basin level image collection (of feature collections)
    basin_reduced_col = annual_ic.map(
        lambda image: reduce_to_basin_means_annual(image, basins=basins)
    )

    # flatten into a single FeatureCollection
    all_features = basin_reduced_col.flatten()

    data_df = get_annual_pdf(
        feature_collection=all_features,
        agg_type="mean",
        output_col_name=output_col_name,
    )

    return data_df


def reproject_to_wgs84(in_tif, out_tif):
    """
    Reprojects a GeoTIFF to the WGS84 coordinate system (EPSG:4326).

    This function takes an input GeoTIFF file, reprojects it to the WGS84 geographic 
    coordinate system (EPSG:4326), and saves the reprojected raster to a new output 
    GeoTIFF file. The function uses Rasterio and performs reprojection with the 
    `Resampling.nearest` method.

    Args:
        in_tif (str): The file path to the input GeoTIFF to be reprojected.
        out_tif (str): The file path to save the reprojected GeoTIFF.

    Returns:
        None
    """
    with rasterio.open(in_tif) as src:
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        with rasterio.open(out_tif, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )


def ensure_wgs84(tif_path):
    """
    Ensures a GeoTIFF file is in the WGS84 coordinate system (EPSG:4326).

    This function checks the CRS (Coordinate Reference System) of a GeoTIFF file. 
    If the file is already in WGS84 (EPSG:4326), it returns the original file path. 
    If not, the file is reprojected to WGS84 and saved with a new file name 
    (appending "_4326" to the original name). If the source file has no CRS, 
    an exception is raised.

    Args:
        tif_path (str): The file path to the input GeoTIFF.

    Returns:
        str: The file path to the GeoTIFF in EPSG:4326 (either the original file 
             or the newly reprojected file).

    Raises:
        ValueError: If the source file does not have a CRS defined.
    """
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"Source file {tif_path} has no CRS!")
        if src.crs.to_string() == "EPSG:4326":
            print(f"{tif_path} is already in EPSG:4326")
            return tif_path

    base, ext = os.path.splitext(tif_path)
    out_tif = base + "_4326.tif"
    reproject_to_wgs84(tif_path, out_tif)
    return out_tif


def tif_to_png(in_tif, out_png):
    """
    Converts a single-band or multi-band GeoTIFF file to an 8-bit PNG image.

    This function reads a GeoTIFF file (assumed to be in EPSG:4326), scales the pixel 
    values to the 8-bit range [0, 255], and saves the result as a PNG image. 
    For single-band TIFFs, a simple min-max stretch is applied. For multi-band TIFFs, 
    the scaling is applied to each band individually. Geo-referencing metadata 
    (e.g., CRS, transform) is removed in the PNG output.

    Args:
        in_tif (str): The file path to the input GeoTIFF.
        out_png (str): The file path to save the output PNG.

    Returns:
        None
    """
    with rasterio.open(in_tif) as src:
        data = src.read()  # shape: (bands, height, width)

        # For single-band, do a simple min-max stretch to [0, 255]
        # If multi-band, you might want band-wise scaling
        min_val, max_val = data.min(), data.max()
        scaled = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # Update profile for 8-bit PNG
        profile = src.profile.copy()
        profile.update(
            {
                "driver": "PNG",
                "dtype": "uint8",
                "count": scaled.shape[0],
            }
        )

        # For PNG, typically remove geo-related keys
        profile.pop("transform", None)
        profile.pop("crs", None)
        profile.pop("nodata", None)

        with rasterio.open(out_png, "w", **profile) as dst:
            dst.write(scaled)


def tif_to_png_with_palette(in_tif, out_png, vis_params):
    """
    Converts a single-band GeoTIFF to a PNG image with a color palette applied.

    This function reads a single-band GeoTIFF, normalizes the pixel values based on 
    specified visualization parameters (`min`, `max`, and `palette`), applies a color 
    palette to create an RGBA image, and saves the result as a PNG. The palette can 
    be defined as a list of colors (e.g., `["black", "white"]`).

    Args:
        in_tif (str): The file path to the input single-band GeoTIFF.
        out_png (str): The file path to save the output PNG.
        vis_params (dict): Visualization parameters for scaling and coloring. 
                           Must include:
                           - `min` (float, optional): Minimum value for normalization. 
                             Defaults to the minimum value of the data.
                           - `max` (float, optional): Maximum value for normalization. 
                             Defaults to the maximum value of the data.
                           - `palette` (list of str, optional): List of colors for 
                             creating a custom colormap (e.g., `["black", "white"]`).

    Returns:
        None
    """
    with rasterio.open(in_tif) as src:
        # Read the data from the first band (assumes single-band TIF)
        data = src.read(1)  # shape: (height, width)

        # Get visualization parameters
        min_val = vis_params.get("min", data.min())
        max_val = vis_params.get("max", data.max())
        palette = vis_params.get("palette", ["black", "white"])

        # Normalize data to the range [0, 255]
        scaled = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        scaled = np.clip(scaled, 0, 255)  # Ensure values are within [0, 255]

        # Create a colormap based on the palette
        cmap = plt.get_cmap("viridis", len(palette))
        cmap = LinearSegmentedColormap.from_list("custom_palette", palette)

        # Apply the colormap
        rgba_image = cmap(scaled / 255.0)  # Normalize to [0, 1] for colormap
        rgba_image = (rgba_image * 255).astype(np.uint8)  # Convert to 8-bit RGBA

        # Save as PNG
        with rasterio.open(
            out_png,
            "w",
            driver="PNG",
            width=src.width,
            height=src.height,
            count=4,
            dtype="uint8",
        ) as dst:
            # Write RGBA bands to the output PNG
            dst.write(rgba_image[:, :, 0], 1)  # Red
            dst.write(rgba_image[:, :, 1], 2)  # Green
            dst.write(rgba_image[:, :, 2], 3)  # Blue
            dst.write(rgba_image[:, :, 3], 4)  # Alpha (optional)


def export_gee_image_to_local_png_with_palette(
    ee_image,
    roi,
    tif_file_path,
    png_file_path,
    vis_params,
    epsg_coordinate_system="EPSG:4326",
):
    """
    Exports an Earth Engine image to a GeoTIFF and converts it to a PNG file with a color palette applied.

    This function exports an Earth Engine image to a GeoTIFF file, ensures the GeoTIFF is in the 
    specified coordinate system (default is EPSG:4326), and then converts the GeoTIFF to a PNG 
    file with visualization parameters applied (e.g., color scaling and palette). The function 
    also prints the image bounds in latitude and longitude for use in mapping applications.

    Args:
        ee_image (ee.Image): The Earth Engine image to be exported.
        roi (ee.Geometry): The region of interest (ROI) to export.
        tif_file_path (str): The file path to save the exported GeoTIFF.
        png_file_path (str): The file path to save the converted PNG.
        vis_params (dict): Visualization parameters for scaling and coloring. Must include:
                           - `min` (float): Minimum value for normalization.
                           - `max` (float): Maximum value for normalization.
                           - `palette` (list of str): List of colors for the colormap.
        epsg_coordinate_system (str, optional): The EPSG code for the output coordinate 
                                                system (default is "EPSG:4326").

    Returns:
        None

    Notes:
        - The `geemap.ee_export_image` function is used for exporting the Earth Engine image to GeoTIFF.
        - The PNG is created using the `tif_to_png_with_palette` function, which applies the visualization parameters.
        - Image bounds are printed in latitude and longitude for mapping purposes.
    """
    # Export the Earth Engine image as a GeoTIFF
    geemap.ee_export_image(
        ee_image,
        filename=tif_file_path,  # or .png if the region is not too large
        scale=1000,
        region=roi,
        file_per_band=False,
        crs=epsg_coordinate_system,
    )

    region_bounds = ee.Geometry(roi).bounds().getInfo()["coordinates"][0]
    # Extract southwest (minLon, minLat) and northeast (maxLon, maxLat) corners
    southwest = region_bounds[0]
    northeast = region_bounds[2]
    bounds = [[southwest[1], southwest[0]], [northeast[1], northeast[0]]]

    print("If using this in leaflet, image bounds are: ", bounds)

    # Ensure the GeoTIFF is in EPSG:4326
    if epsg_coordinate_system == "EPSG:4326":
        tif_wgs84 = ensure_wgs84(tif_file_path)

    # Convert the GeoTIFF to PNG with the visualization parameters
    tif_to_png_with_palette(tif_wgs84, png_file_path, vis_params)
