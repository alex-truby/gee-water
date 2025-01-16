from scipy.stats import linregress

def check_trend_significance(
    df,
    date_col='date',
    value_col='value',
    alpha=0.05
):
    """
    Checks whether the slope of a time-series is significantly different from zero.

    1. Sorts data by date.
    2. Converts each date to "days since earliest date" to get manageable x-values.
    3. Runs a linear regression using scipy.stats.linregress.
    4. Returns the slope (units per day), slope in units per year, p-value, and 
       a boolean for whether it is significant at the given alpha level.

    Args:
        df (pd.DataFrame):
            DataFrame with at least two columns: one for dates, one for the values.
        date_col (str, optional):
            Column name for the date. Defaults to 'date'.
        value_col (str, optional):
            Column name for the values to be analyzed. Defaults to 'value'.
        alpha (float, optional):
            Significance level for the slope test. Defaults to 0.05.

    Returns:
        slope_per_day (float): 
            Estimated slope in units per day.
        slope_per_year (float): 
            Estimated slope in units per year (slope_per_day * 365.25).
        p_value (float):
            P-value for the null hypothesis that slope == 0.
        is_significant (bool):
            True if p_value < alpha, else False.
    """
    # Sort by date just in case
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Convert each date to days since earliest date in this DataFrame
    earliest_date = df_sorted[date_col].min()
    df_sorted['days_since_start'] = (df_sorted[date_col] - earliest_date).dt.days
    
    x_days = df_sorted['days_since_start'].to_numpy()
    y_vals = df_sorted[value_col].to_numpy()
    
    # Run linear regression
    linreg_res = linregress(x_days, y_vals)
    slope_per_day = linreg_res.slope
    p_value = linreg_res.pvalue
    
    # Convert slope to units per year
    slope_per_year = slope_per_day * 365.25
    
    # Check if p_value < alpha => slope is significant
    is_significant = p_value < alpha
    
    return slope_per_day, slope_per_year, p_value, is_significant
