# Visualization parameters for LandSat RGB Imagery
landsat_vis_params = {
    'bands': ['SR_B4','SR_B3', 'SR_B2'],  # True color bands
    'min': 0,
    'max': 50000,
    'gamma': 1.4
}

# Visualization parameters for VIIRS Nighttime Lights
viirs_vis_params = {
    'bands': ['avg_rad'],
    'min': 0,
    'max': 60,  # Adjust based on the data range; 60 is a common upper limit
    # 'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red'] # optional colorful palette
    'palette': ['black', 'white']
}