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
        end = ee.Date.fromYMD(year + 1, 1, 1)  # Up to (but not including) Jan 1 of next year
        annual_coll = image_collection.filterDate(start, end)
        if agg_type == 'sum':
            annual_img = annual_coll.sum().set({'year': year})
        elif agg_type == 'mean':
            annual_img = annual_coll.mean().set({'year': year})
        else:
            raise ValueError("Invalid agg_type. Expected 'sum' or 'mean'.")
        return annual_img

    images = []
    for y in range(start_year, end_year + 1):
        images.append(agg_for_year(y))
    return ee.ImageCollection(images)
