# gee-water
Google Earth Code Related to Water Resources

## Description
The purpose of this repo is to create a reusble python package that may be utilized for earth
observation workflows. The focus of this repo is water earth observation and anaysis for water 
resources problems.

## Contents
This repo contains set up and code allowing for Earth Observation using Google Earth Engine (GEE)
for water resources purposes in a python editor enviornment.

    <gee-water>/
    ├─ models/                          # serialised models (e.g. machine learning)
    ├─ notebooks/                       # notebooks for prototyping and research
    │  ├─ basin_analysis.ipynb          # workflow for basin level water resrouces metrics (precip, gw anamolies, land surface temperature, etc.)
    │  ├─ development.ipynb             # workflow for urbanization evaluation with nighttime radiance
    │  ├─ ndv_savii.ipynb               # workflow for evaluating vegetation health
    ├─ src/                             # directory containing analysis code
    │  ├─ gee_water                     # python package containing helfup functions for water resources analysis with GEE
    │     ├─ analysis.py                # analysis functions for GEE workflows
    │     ├─ masks.py                   # mask functions for GEE workflows
    │     ├─ utils.py                   # utility functions for GEE workflows
    │     ├─ vis_params.py              # parameters for visualizing commong image types in GEE
    │     ├─ visualization_utils.py     # plotting functions
    │  ├─ gee_water.egg-info            # package configuration ingo
    ├─ static/                             # Contains static files (images, html) output from the evaluation
    ├─ .gitignore                       # standard python gitignore
    ├─ pyproject.toml                   # pip install boilerplate
    ├─ setup.py                         # pip install boilerplate
    ├─ setup.cfg                        # pip install intructions

## Installation for Use
For direct use, this package can be installed using the configuration in `setup.cfg`. In the root directory:

    pip install .