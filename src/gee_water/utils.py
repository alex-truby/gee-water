import ee
import pandas as pd


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
    red = image.select('B4')
    nir = image.select('B5')
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
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
                reducer=ee.Reducer.mean(),
                geometry=roi,
                scale=10000,
                maxPixels=1e13
            )

            # Extract the mean value; could be None if dummy/empty
            mean_val = mean_dict.get(band_name).getInfo()
        except Exception as e:
            # If the band doesn't exist or another error occurs, set None
            print(f"Error for {year}-{month:02d}: {e}")
            mean_val = None

        # Add to records
        records.append({
            'year': year,
            'month': month,
            band_name: mean_val
        })

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(records)

    # Optional: create a 'date' column (use day=1 for monthly data)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1), errors='coerce')

    # Sort by date
    df = df.sort_values(by='date').reset_index(drop=True)

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
    end_date = start_date.advance(1, 'month')

    return (collection
            .filterDate(start_date, end_date)
            .select(band)
            .mean()
            .rename(f'{band}_{year}_{month}'))



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
