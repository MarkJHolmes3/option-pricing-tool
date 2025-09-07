from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

import plotly.graph_objects as go
from scipy.interpolate import SmoothBivariateSpline
from scipy.optimize import brentq
from scipy.stats import norm


def get_latest_spot(ticker):
    """
    Get the latest stock price for a given stock's ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    
    Returns
    -------
    S
        Latest stock price rounded to 2 decimal places.
    """
    S = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
    S = np.round(S, 2)
    return S


def black_scholes(S, K, T, r, sigma, option_type, q=0):
    """
    Calculate the price of a call or put option for a given stock and inputs.

    Parameters
    ----------
    S : float
        The stock price at time t.
    K : float
        The exercise price of the option.
    T : float
        The time to option expiry (years).
    r : float
        The risk-free interest rate per annum.
    sigma : float
        The annual standard deviation of the stock price.
    option_type : str
        Call or put.
    q : float
        The dividend yield per annum.

    Returns
    -------
    float
        The theoretical option value as per the Black-Scholes model.
    """
    option_type_lower = option_type.lower()
    if option_type_lower not in ['call', 'put']:
        return False
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type_lower == 'call':
        option_value = (S * np.exp(-q * T)) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return option_value

    else:
        option_value = K * np.exp(-r * T) * norm.cdf(-d2) - (S * np.exp(-q * T)) * norm.cdf(-d1)
        return option_value


def implied_sigma(market_value, S, K, T, r, option_type, q=0):
    """
    Calculate the implied volatility an option.
    
    Paramters
    ---------
    market_value : float
        The market value of the option.
    S : float
        The stock price at time t.
    K : float
        The exercise price of the option.
    T : int
        The time to option expiry (years).
    r : float
        The risk-free rate per annum.
    option_type : str
        Call or put.
    q : float
        The dividend yield per annum.

    Returns
    -------
    float
        the implied volatility of the option for a given market price.
    """
    f = lambda sigma: black_scholes(S, K, T, r, sigma, option_type, q) - market_value
    return brentq(f, 1e-6, 5)


def get_options(ticker, option_type=None):
    """
    Retrieves and modifies an option chain for a given ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    option_type : str
        Call or put.

    Returns
    -------
    Pandas DataFrame
        modified options DataFrame.
    """
    if option_type is None:
        return False

    today = pd.Timestamp.today().normalize()
    ticker_object = yf.Ticker(ticker)
    all_options = []

    option_type = option_type.lower()
    
    if option_type not in ['call', 'put']:
        return False

    for expiry in ticker_object.options:
        opt_chain = ticker_object.option_chain(expiry)
        option_df = getattr(opt_chain, option_type + 's').copy()

        option_df['expiry_date'] = pd.to_datetime(expiry).normalize()
        option_df['daysToExpiry'] = (option_df['expiry_date'] - today).dt.days
        option_df['T'] = option_df['daysToExpiry'] / 365
        
        option_df['option_type'] = option_type
        
        S = ticker_object.history(period='1d')['Close'].iloc[-1]
        option_df['moneyness'] = option_df['strike'] / S
        
        all_options.append(option_df)

    options_df = pd.concat(all_options, ignore_index=True)

    return options_df


def get_options_liquid(ticker, option_type):
    """
    Retrieves and modifies an option chain for a given ticker from Yahoo Finance and removes iliquid options.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    option_type : str
        Call or put.
    
    Returns
    -------
    Pandas DataFrame
        Modified options chain that excludes iliquid options.
    """
    options_df = get_options(ticker, option_type)

    options_df = options_df.dropna(subset=['volume', 'openInterest', 'impliedVolatility', 'T', 'moneyness'])

    options_df = options_df[
        (options_df['volume'] > 0) &
        (options_df['openInterest'] > 0) &
        (options_df['impliedVolatility'] > 0.01) &
        (options_df['impliedVolatility'] < 3.00) &
        (options_df['T'] > 0) &
        (options_df['T'] <= 1) &
        (options_df['moneyness'] > 0.5) &
        (options_df['moneyness'] < 1.5)
    ]

    return options_df


def option_value_heatmap(S, K, T, r, sigma, option_type, q=0, S_min=None, S_max=None, sigma_min=None, sigma_max=None, steps=None):
    """
    Generate a heatmap of option values across varying stock prices and volatilities.

    Parameters
    ----------
    S : float
        Current stock price.
    K : float
        Option exercise (strike) price.
    T : float
        Time to expiration in years.
    r : float
        Annual risk-free interest rate (decimal).
    sigma : float
        Annual volatility of the stock price (decimal).
    option_type : str
        'call' or 'put'.
    q : float, optional
        Annual dividend yield (decimal), default is 0.
    S_min : float, optional
        Minimum stock price to plot; defaults to 80% of S.
    S_max : float, optional
        Maximum stock price to plot; defaults to 120% of S.
    sigma_min : float, optional
        Minimum volatility to plot; default is 0.10.
    sigma_max : float, optional
        Maximum volatility to plot; default is 0.50.
    steps : int, optional
        Number of points for stock price and volatility axes; default is 15.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A Plotly heatmap figure showing option values across stock price and volatility ranges.
    """
    if S_min is None:
        S_min = 0.8 * S
    
    if S_max is None:
        S_max = 1.2 * S
    
    if sigma_min is None:
        sigma_min = 0.10
    
    if sigma_max is None:
        sigma_max = 0.50
    
    if steps is None:
        steps = 15
    
    S_array = np.linspace(S_min, S_max, steps)
    sigma_array = np.linspace(sigma_min, sigma_max, steps)
    option_values = np.array([
        [black_scholes(S, K, T, r, sigma, option_type, q) for sigma in sigma_array]
        for S in S_array])

    fig = go.Figure(data = [
        go.Heatmap(
            x = S_array,
            y = sigma_array,
            z = option_values,
            colorscale = "Viridis"
        )
    ])

    fig.update_layout(
        title=f"{option_type} Value Heatmap",
        xaxis_title='Spot Price (S))',
        yaxis_title='Volatility',
        autosize=True,
        width=800,
        height=700
    )
    return fig


def sigma_smile(ticker, option_type, moneyness=False):
    """
    Plot the implied volatility smile for a given stock option.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    option_type : str
        'call' or 'put'.
    moneyness : bool, optional
        If True, plot implied volatility against moneyness (K/S);
        otherwise against strike price. Default is False.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A Plotly figure showing the implied volatility smile.
    """
    if option_type.lower() not in ['call', 'put']:
        return False
    
    options_df = get_options(ticker, option_type)

    if moneyness is True:
        fig = go.Figure(data = [
            go.Scatter(
                x = options_df['moneyness'],
                y = options_df['impliedVolatility'],
                mode='markers'
            )
        ])
        fig.update_layout(
        title=f"Volatility Smile for {ticker.upper()} {option_type}s",
        xaxis_title='Moneyness (K/S)',
        yaxis_title='Implied Volatility',
        autosize=True,
        width=800,
        height=700
        )
        
    else:
        fig = go.Figure(data = [
            go.Scatter(
                x = options_df['strike'],
                y = options_df['impliedVolatility'],
                mode='markers'
            )
        ])
        fig.update_layout(
        title=f"Volatility Smile: {ticker.upper()} {option_type}s",
        xaxis_title='Exercise Price (K))',
        yaxis_title='Implied Volatility',
        autosize=True,
        width=800,
        height=700
        )

    return fig


def create_sigma_surface(ticker, option_type, moneyness=False):
    """
    Build a smoothed implied volatility surface from option data.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    option_type : str
        'call' or 'put'.
    moneyness : bool, optional
        If True, use moneyness (K/S) as the horizontal axis; 
        otherwise use strike price. Default is False.

    Returns
    -------
    X_new : np.ndarray
        2D grid of expiry values for the interpolated surface.
    Y_new : np.ndarray
        2D grid of strike prices or moneyness values for the surface.
    Z_smooth : np.ndarray
        2D array of smoothed implied volatilities over the grid.
    """
    options_df = get_options(ticker, option_type)

    if moneyness is True:
        surface = (
            options_df[['daysToExpiry', 'moneyness', 'impliedVolatility']]
            .pivot_table(
                values='impliedVolatility', index='moneyness', columns='daysToExpiry'
            )
            .dropna()
        )
    else:
        surface = (
            options_df[['daysToExpiry', 'strike', 'impliedVolatility']]
            .pivot_table(
                values='impliedVolatility', index='strike', columns='daysToExpiry'
            )
            .dropna()
        )

    x = surface.columns.values
    y = surface.index.values
    X, Y = np.meshgrid(x, y)
    Z = surface.values

    x_new = np.linspace(x.min(), x.max(), 100)
    y_new = np.linspace(y.min(), y.max(), 100)
    X_new, Y_new = np.meshgrid(x_new, y_new)

    spline = SmoothBivariateSpline(
        X.flatten(), Y.flatten(), Z.flatten(), s=5 # A higher value used to account for sparse yfinance data
    )
    Z_smooth = spline(x_new, y_new)

    return X_new, Y_new, Z_smooth


def plot_sigma_surface(X, Y, Z, ticker, option_type, moneyness=False):
    """
    Plot a 3D implied volatility surface using Plotly.

    Parameters
    ----------
    X : np.ndarray
        2D grid of expiry values.
    Y : np.ndarray
        2D grid of strike prices or moneyness values.
    Z : np.ndarray
        2D array of implied volatilities over the grid.
    ticker : str
        Stock ticker symbol.
    option_type : str
        'call' or 'put'.
    moneyness : bool, optional
        If True, label y-axis as moneyness (K/S); otherwise as strike price. Default is False.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The Plotly figure of the volatility surface.
    """
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
        )
    ])

    if moneyness is True:
        fig.update_layout(
            title=f"Volatility Surface for {ticker.upper()} {option_type}s",
            scene=dict(
                xaxis_title='Days to Expiry',
                yaxis_title='Moneyness (K/S)',
                zaxis_title='Implied Volatility'
            ),
            autosize=True,
            width=800,
            height=700
        )

    else:
        fig.update_layout(
            title=f"Volatility Surface for {ticker.upper() } {option_type}s",
            scene=dict(
                xaxis_title='Days to Expiry',
                yaxis_title='Exercise Price (K)',
                zaxis_title='Implied Volatility'
            ),
            autosize=True,
            width=800,
            height=700
        )

    fig.show()

    return fig


def sigma_surface(ticker, option_type, moneyness=False):
    """
    Generate and plot the implied volatility surface for a given option type.

    This function builds a smoothed volatility surface from option data 
    and displays it as a 3D Plotly figure.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    option_type : str
        'call' or 'put'.
    moneyness : bool, optional
        If True, use moneyness (K/S) as the y-axis; otherwise use strike price. Default is False.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The Plotly figure displaying the volatility surface.
    """
    if moneyness is True:
        X_new, Y_new, Z_smooth = create_sigma_surface(ticker, option_type, moneyness=True)
        fig = plot_sigma_surface(X_new, Y_new, Z_smooth, ticker, option_type, moneyness=True)
    else:
        X_new, Y_new, Z_smooth = create_sigma_surface(ticker, option_type, moneyness=False)
        fig = plot_sigma_surface(X_new, Y_new, Z_smooth, ticker, option_type, moneyness=False)
    
    return fig