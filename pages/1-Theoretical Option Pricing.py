import sys
from pathlib import Path

import streamlit as st
import yfinance as yf

sys.path.append(str(Path(__file__).resolve().parent.parent))

from option_pricing_functions import (
    get_latest_spot,
    black_scholes,
    implied_sigma,
    option_value_heatmap
)


st.set_page_config(
    layout='centered'
)

st.title('Option Pricing Tool - Theoretical Option Pricing')
st.markdown(
    """
    Developed by Mark Holmes.
    <br> <br>
    Enter option parameters in the sidebar to compute theoretical option prices and 
    generate a value heatmap. 
    Use the buttons and selectors to adjust visualisations interactively.
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.subheader('Inputs for Theoretical Option Pricer:')
    S = st.number_input('Stock Price (S)', min_value=0.01, value=100.0)
    K = st.number_input('Strike Price (K)', min_value=0.01, value=100.0)
    T = st.number_input('Time to Maturity (T) in years', min_value=0.01, value=1.0)
    r = st.number_input('Risk-free Rate (r) in decimal', min_value=0.001, value=0.05)
    q = st.number_input('Annual Dividend Yield (q) in decimal', min_value=0.00, value=0.00)
    sigma = st.number_input('Volatility (sigma) in decimal', min_value=0.01, value=0.2)
    option_type = st.selectbox('Option Type', ('Call', 'Put'))

    with st.expander('Heatmap Specific Inputs'):
        S_min = st.number_input('Minimum Spot Price', min_value=0.01, value=0.5*S)
        S_max = st.number_input('Maximum Spot Price', min_value=0.01, value=1.5*S)
        sigma_min = st.number_input('Minimum Volatility for Heatmap', min_value=0.01, max_value=2.50, value=0.1)
        sigma_max = st.number_input('Maximum Volatility for Heatmap', min_value=0.01, max_value=2.50, value=1.0)
        steps = st.selectbox('Increments', [5, 10, 20, 50])

value = black_scholes(S, K, T, r, sigma, option_type, q)

st.markdown(
    f"<h1 style='text-align: center'>Theoretical {option_type} value: {value:.2f}</h1>", 
    unsafe_allow_html=True
)

fig_heatmap = option_value_heatmap(S, K, T, r, sigma, option_type, q, S_min, S_max, sigma_min, sigma_max, steps)
st.plotly_chart(fig_heatmap)

st.markdown(
    """
    <small>Special thanks to <a href="https://www.youtube.com/@CodingJesus" target="_blank">Coding Jesus</a> for the project inspiration.</small>
    <br> <br>
    <small>This application is intended for illustrative and educational purposes only. The developer accepts no responsibility for any losses or decisions arising from use of this application.</small>
    """,
    unsafe_allow_html=True
)