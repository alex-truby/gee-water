import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

def plot_multiple_timeseries_with_trendlines_plotly(
    df_list,
    date_col='date',
    value_col='value',
    labels=None,
    restrict_pair=None,
    restrict_label=None,
    x_interval_months=6,
    title='Time Series Comparison',
    ylabel='Value (units)'
):
    """
    Plots multiple time series (date vs. value) with linear trend lines in a single Plotly figure.

    Args:
        df_list (list of pd.DataFrame):
            A list of DataFrames, each with a date column and a value column.
        date_col (str):
            Name of the column containing dates in each DataFrame.
        value_col (str):
            Name of the column containing the value to be plotted (e.g., soil moisture).
        labels (list of str or None):
            Legend labels for each DataFrame. If None, uses generic labels.
        restrict_pair (tuple or None):
            If you want to restrict one dataset to the date range of another,
            set this as (ref_index, target_index).
            Example: (0, 1) means restrict dataset #1 to the date range of dataset #0.
        restrict_label (str or None):
            Label for the restricted portion of the target dataset (e.g., "Dataset B (restricted)").
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
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # ---- Main Time Series and Trendlines ----
    for i, df in enumerate(df_list):
        # Sort by date just in case
        df_sorted = df.sort_values(by=date_col).reset_index(drop=True)

        # Extract x (dates) and y (values)
        x = df_sorted[date_col]
        y = df_sorted[value_col]

        # Convert dates to numeric for slope fitting
        x_num = pd.to_numeric(df_sorted[date_col])  # or use your own approach
        
        # Plot the main time series
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=labels[i]
        ))
        
        # Compute Trend (slope) via np.polyfit
        coeffs = np.polyfit(x_num, y, 1)  # 1 => linear
        trend_func = np.poly1d(coeffs)
        slope_per_day = coeffs[0]
        # Approx days per year
        slope_per_year = slope_per_day * 365.25
        
        # Add the trend line
        fig.add_trace(go.Scatter(
            x=x,
            y=trend_func(x_num),
            mode='lines',
            line=dict(dash='dash'),
            name=f"{labels[i]} Trend (slope={slope_per_year:.5f}/yr)"
        ))
    
    # ---- Restrict one dataset to the date range of another ----
    if restrict_pair is not None:
        ref_idx, target_idx = restrict_pair

        # Get date min/max from reference dataset
        ref_df = df_list[ref_idx].copy()
        ref_min_date = ref_df[date_col].min()
        ref_max_date = ref_df[date_col].max()

        # Target dataset to be restricted
        target_df = df_list[target_idx].copy()
        restricted_df = target_df[
            (target_df[date_col] >= ref_min_date) &
            (target_df[date_col] <= ref_max_date)
        ].copy()

        if len(restricted_df) > 0:
            restricted_df.sort_values(by=date_col, inplace=True)

            x_restrict = restricted_df[date_col]
            y_restrict = restricted_df[value_col]

            # Fit a new trend line for the restricted portion
            x_restrict_num = pd.to_numeric(x_restrict)
            restrict_coeffs = np.polyfit(x_restrict_num, y_restrict, 1)
            restrict_trend = np.poly1d(restrict_coeffs)
            slope_day = restrict_coeffs[0]
            slope_year = slope_day * 365.25

            # Plot restricted portion
            if restrict_label is None:
                restrict_label = f"{labels[target_idx]} (restricted)"

            fig.add_trace(go.Scatter(
                x=x_restrict,
                y=y_restrict,
                mode='lines',
                line=dict(dash='dot'),
                name=restrict_label
            ))
            fig.add_trace(go.Scatter(
                x=x_restrict,
                y=restrict_trend(x_restrict_num),
                mode='lines',
                line=dict(dash='dot'),
                name=f"{restrict_label} Trend (slope={slope_year:.5f}/yr)"
            ))
    
    # ---- Format layout and axes ----
    # Tick every x_interval_months months
    # Plotly supports "M1", "M2", etc. for monthly intervals
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
            x=0.01,  # position the legend
            y=0.99
        ),
        margin=dict(l=60, r=40, t=60, b=60)
    )

    return fig
