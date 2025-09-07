import sys
from pathlib import Path

import streamlit as st
import yfinance as yf

sys.path.append(str(Path(__file__).resolve().parent.parent))

from option_pricing_functions import (
    get_latest_spot,
    black_scholes,
    get_options,
    get_options_liquid,
    sigma_smile,
    sigma_surface,
)


st.set_page_config(
    layout='centered'
)

st.title('Option Pricing Tool - Market Data Visualisation')
st.markdown(
    """
    Developed by Mark Holmes.
    <br> <br>
    Enter ticker and option type parameters in the sidebar to visualise options data. 
    Use the buttons and selectors to adjust visualisations interactively.
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.subheader('Inputs for Market Data Visualisation')
    ticker = st.text_input('Ticker', value='SPY', key='ticker_input')
    fig_option_type = st.selectbox('Option Type', ('Call', 'Put'), key='option_type_select')
    S = get_latest_spot(ticker)
    st.markdown(f"# Current {ticker.upper()} spot price (*S*): {S}")

if st.button('Exercise Price (K) or Moneyness (K/S)'):
    fig_sigma_surface = sigma_surface(ticker, fig_option_type, moneyness=True)
    fig_sigma_smile = sigma_smile(ticker, fig_option_type, moneyness=True)
else:
    fig_sigma_surface = sigma_surface(ticker, fig_option_type, moneyness=False)
    fig_sigma_smile = sigma_smile(ticker, fig_option_type, moneyness=False)
st.plotly_chart(fig_sigma_surface)
st.plotly_chart(fig_sigma_smile)

st.markdown(
    """
    <small>Special thanks to <a href="https://www.youtube.com/@CodingJesus" target="_blank">Coding Jesus</a> for the project inspiration.</small>
    <br> <br>
    <small>This application is intended for illustrative and educational purposes only. The developer accepts no responsibility for any losses or decisions arising from use of this application.</small>
    """,
    unsafe_allow_html=True
)