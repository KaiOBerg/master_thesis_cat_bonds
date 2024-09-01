def calc_rp(df, return_period):
    """
    Compute impacts/payouts for specific return periods using a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing impacts/payouts and their associated return periods.
        Should have columns 'Damage' and 'RP'.
    return_periods : Object
        The return period where we want to compute the exceedance impact.

    Returns
    -------
    pandas.DataFrame
        A List with the specified return period and their corresponding damage.
    """

    # Extract sorted return periods and impacts
    sorted_rp = df['RP'].values
    sorted_impact = df['Damage'].values

    # Interpolate impacts for the given return periods
    interp_impact = np.interp(return_period, sorted_rp, sorted_impact)

    return interp_impact
