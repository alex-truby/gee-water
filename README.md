# gee-water
Google Earth Engine Code Related to Water Resources

## Description
The purpose of this repo is to create a reusable Python package that may be utilized for Earth
observation workflows. The focus of this repo is Earth observation and analysis for water 
resources problems.

## Contents
This repo contains setup and code for Earth Observation using Google Earth Engine (GEE) 
for water resources purposes in a Python editor environment.

    <gee-water>/
    ├─ models/                          # serialized models (e.g., machine learning)
    ├─ notebooks/                       # notebooks for prototyping and research
    │  ├─ basin_analysis.ipynb          # workflow for basin-level water resources metrics (precipitation, groundwater anomalies, land surface temperature, etc.)
    │  ├─ development.ipynb             # workflow for urbanization evaluation with nighttime radiance
    │  ├─ ndvi_savvi.ipynb              # workflow for evaluating vegetation health
    ├─ src/                             # directory containing analysis code
    │  ├─ gee_water                     # Python package containing helpful functions for water resources analysis with GEE
    │     ├─ analysis.py                # analysis functions for GEE workflows
    │     ├─ masks.py                   # mask functions for GEE workflows
    │     ├─ utils.py                   # utility functions for GEE workflows
    │     ├─ vis_params.py              # parameters for visualizing common image types in GEE
    │     ├─ visualization_utils.py     # plotting functions
    │  ├─ gee_water.egg-info            # package configuration info
    ├─ static/                          # Contains static files (images, HTML) output from the evaluation
    ├─ .gitignore                       # standard Python gitignore
    ├─ pyproject.toml                   # pip install boilerplate
    ├─ setup.py                         # pip install boilerplate
    ├─ setup.cfg                        # pip install instructions

## Installation for Use
For direct use, this package can be installed using the configuration in `setup.cfg`. In the root directory:

    pip install .
