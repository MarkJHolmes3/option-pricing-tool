import streamlit as st

st.title('Option Pricing Tool - Home Page')
st.markdown(
    """
    ## Welcome to *Option Pricing Tool* by Mark Holmes!  

    ### This app allows you to:  
    - Fetch the latest spot prices of assets.  
    - Compute option prices using the Black-Scholes model.  
    - Explore implied volatilities and volatility smiles.  
    - Visualise option value heatmaps and implied volatility surfaces.  

    ### Use the sidebar to navigate between pages and start exploring.
    """
)

my_linkedin = "https://www.linkedin.com/in/mark-j-holmes/"
st.link_button('My LinkedIn', my_linkedin)

my_git = "https://github.com/MarkJHolmes3"
st.link_button('My GitHub', my_git)