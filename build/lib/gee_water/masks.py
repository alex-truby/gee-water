def mask_landsat_c2_l2_sr(image):
    """
    Masks clouds and cloud shadows from a Landsat Collection 2 Level-2 Surface Reflectance (C2 L2 SR) image.

    This function uses the 'QA_PIXEL' band from the input image to identify and mask out pixels flagged as 
    cloud or cloud shadow. The cloud and cloud shadow information is stored in bits 3 and 4 of the 'QA_PIXEL' band.

    Args:
        image (ee.Image): A Google Earth Engine Image object representing Landsat C2 L2 SR data. 
                          The image must contain a 'QA_PIXEL' band.

    Returns:
        ee.Image: The input image with a mask applied, masking out cloud and cloud shadow pixels.
    """
    # 'QA_PIXEL' band
    qa = image.select('QA_PIXEL')
    # Bits 3 and 4 are cloud and cloud shadow flags
    cloud_shadow = (1 << 3)
    clouds = (1 << 4)
    mask = qa.bitwiseAnd(cloud_shadow).eq(0).And(
           qa.bitwiseAnd(clouds).eq(0))
    return image.updateMask(mask)
