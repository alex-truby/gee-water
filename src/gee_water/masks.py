def mask_hls_l30_cloud_shadow(image):
    """
    Masks clouds, cloud shadows, snow/ice, and fill pixels from an HLS L30 image.

    This function uses the `Fmask` band from the Harmonized Landsat and Sentinel-2 
    (HLS) L30 dataset to exclude specific pixel types based on predefined values:
      - 0 = clear land (kept)
      - 1 = clear water (commented out, optional for exclusion)
      - 2 = cloud (masked)
      - 3 = cloud shadow (masked)
      - 4 = snow/ice (masked)
      - 255 = fill/no data (masked)

    Args:
        image (ee.Image): A Google Earth Engine Image object from the HLS L30 dataset, 
                          containing an `Fmask` band.

    Returns:
        ee.Image: The input image with a mask applied to exclude specified pixel types.
    """
    fmask = image.select("Fmask")
    # Keep only clear land (0) or clear water (optional)
    mask = (fmask.neq(2)   # Exclude cloud
            # .And(fmask.neq(1))  # Exclude water (optional, currently commented)
            .And(fmask.neq(3))  # Exclude cloud shadow
            .And(fmask.neq(4))  # Exclude snow/ice
            .And(fmask.neq(255))  # Exclude fill/no data
           )
    return image.updateMask(mask)
