import numpy as np
import plotly.graph_objects as go
import geemap
from matplotlib.colors import to_rgb



def plot_multiple_timeseries_with_trendlines_plotly(
    df_list,
    date_col='date',
    value_col='value',
    labels=None,
    x_interval_months=6,
    title='Time Series Comparison',
    ylabel='Value (units)'
):
    """
    Plots multiple time series (date vs. value) with linear trend lines in a single Plotly figure.
    Uses "days since earliest global date" as the x-values for linear regression, 
    which avoids extremely large timestamp numbers in nanoseconds.

    Args:
        df_list (list of pd.DataFrame):
            A list of DataFrames, each with a date column and a value column.
        date_col (str):
            Name of the column containing dates in each DataFrame.
        value_col (str):
            Name of the column containing the value to be plotted.
        labels (list of str or None):
            Legend labels for each DataFrame. If None, uses generic labels.
        x_interval_months (int):
            Interval (in months) for major x-axis ticks. Default=6 => tick every 6 months.
        title (str):
            Title of the plot.
        ylabel (str):
            Y-axis label.

    Returns:
        plotly.graph_objects.Figure:
            A Plotly figure object that can be displayed using fig.show().
    """

    # If no labels are given, assign default ones
    if labels is None:
        labels = [f"Dataset {i}" for i in range(len(df_list))]
    elif len(labels) != len(df_list):
        raise ValueError("Length of `labels` must match length of `df_list`.")

    # 1. Find a global earliest date among all DataFrames
    global_earliest_date = min(df[date_col].min() for df in df_list)

    # Create a Plotly figure
    fig = go.Figure()

    # 2. Generate each time series + trend lines
    for i, df in enumerate(df_list):
        # Sort by date just in case
        df_sorted = df.sort_values(by=date_col).reset_index(drop=True)

        # Calculate "days since global earliest date" for slope-fitting
        df_sorted['days_since_start'] = (
            df_sorted[date_col] - global_earliest_date
        ).dt.days

        # x-values for plotting (actual dates) vs. x-values for slope-fitting (days_since_start)
        x_dates = df_sorted[date_col]
        x_days = df_sorted['days_since_start']
        y_vals = df_sorted[value_col]

        # Plot the main time series
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=y_vals,
                mode='lines',
                name=labels[i]
            )
        )

        # Compute trend (slope) per day via np.polyfit
        coeffs = np.polyfit(x_days, y_vals, 1)  # linear fit
        slope_per_day, intercept = coeffs[0], coeffs[1]

        # Evaluate the linear model at each x_days
        trend_line_y = intercept + slope_per_day * x_days

        # Convert slope to "units per year"
        slope_per_year = slope_per_day * 365.25

        # Plot the trend line
        fig.add_trace(
            go.Scatter(
                x=x_dates,  # still plot trend line against real dates
                y=trend_line_y,
                mode='lines',
                line=dict(dash='dash'),
                name=f"{labels[i]} Trend (slope={slope_per_year:.5f}/yr)"
            )
        )

    # 3. Format the x-axis ticks and other layout details
    # Tick every x_interval_months months
    dtick_str = f"M{x_interval_months}"

    fig.update_layout(
        title=title,
        xaxis=dict(
            title='Date',
            dtick=dtick_str,
            tickformat='%Y-%m'
        ),
        yaxis=dict(
            title=ylabel
        ),
        legend=dict(
            x=0.01,
            y=0.99
        ),
        margin=dict(l=60, r=40, t=60, b=60)
    )

    return fig



def create_split_viirs_map(
        left_image, 
        right_image,
        left_image_label,
        right_image_label,
        vis_params,
        mep_center_coords,
        zoom,
        legend_title,
        legend_labels,
        html_export_location=None
):
    """
    Creates a split map to visually compare two VIIRS images side by side.

    This function generates an interactive map with a split screen feature, allowing users 
    to compare two Earth Engine images using geemap. The left and right images are labeled 
    and styled using the provided visualization parameters. A legend is added to the map, 
    and the map can optionally be exported to an HTML file.

    Args:
        left_image (ee.Image): The Earth Engine image to display on the left side of the map.
        right_image (ee.Image): The Earth Engine image to display on the right side of the map.
        left_image_label (str): Label to display on the left side of the map.
        right_image_label (str): Label to display on the right side of the map.
        vis_params (dict): Visualization parameters (e.g., palette, min, max) for styling the images.
        mep_center_coords (tuple): The map's center coordinates as (latitude, longitude).
        zoom (int): The initial zoom level of the map.
        legend_title (str): Title of the legend displayed on the map.
        legend_labels (list): Labels corresponding to the colors in the visualization palette.
        html_export_location (str, optional): File path to save the map as an HTML file. 
                                              If not provided, the map is not exported.

    Returns:
        geemap.Map: An interactive geemap Map object with the split map feature.
    """
    Map = geemap.Map(center=mep_center_coords, zoom=zoom)

    # Create Tile Layers for the images
    left_layer = geemap.ee_tile_layer(left_image, vis_params, left_image_label)
    right_layer = geemap.ee_tile_layer(right_image, vis_params, right_image_label)

    # Add a split map feature
    Map.split_map(left_layer=left_layer, right_layer=right_layer)

    # Add labels with inline styles
    left_label_position = "topleft"
    right_label_position = "topright"

    Map.add_text(left_image_label, position=left_label_position, fontcolor='gray', fontsize='28')
    Map.add_text(right_image_label, position=right_label_position, fontcolor='gray', fontsize='28')

    # Add a legend to the map
    Map.add_legend(
        legend_title=legend_title, 
        colors=[tuple(int(c * 255) for c in to_rgb(color)) for color in vis_params['palette']], 
        labels=legend_labels
    )

    if html_export_location:
        Map.to_html(html_export_location)

    # Return the interactive map
    return Map