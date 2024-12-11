# Black-Scholes Calculator (for European-style options)


# Import libraries
import math
import numpy as np
from datetime import datetime
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  
import seaborn as sns
import streamlit as st


# Model setup
# Determine if current year is a leap year
current_year = datetime.now().year
days_in_year = 366 if (current_year % 4 == 0 and (current_year % 100 != 0 or current_year % 400 == 0)) else 365


# Define Yahoo! Finance functions
def get_risk_free_rate():
     # Fetch the 10-year Treasury yield from Yahoo! Finance
     tnx = yf.Ticker("^TNX")
     # Current yield, divided by 100 for decimal format (e.g., 4.5% becomes 0.045)
     risk_free_rate = tnx.history(period="1d")['Close'].iloc[-1] / 100
     return risk_free_rate


def get_top_10_apple_options():  
    # Define the ticker symbol for Apple  
    ticker_symbol = 'AAPL'  
      
    # Download the options data for the Apple ticker  
    apple = yf.Ticker(ticker_symbol)  
      
    # Get the list of expiration dates for the options  
    expiration_dates = apple.options  
      
    # Initialize an empty DataFrame to store all options  
    all_options = pd.DataFrame()  
      
    # Iterate over each expiration date  
    for expiration in expiration_dates:  
        # Get the options chain for the current expiration date  
        options_chain = apple.option_chain(expiration)  
          
        # Concatenate calls and puts data  
        options_data = pd.concat([options_chain.calls, options_chain.puts])  
          
        # Append to the all_options DataFrame  
        all_options = pd.concat([all_options, options_data], ignore_index=True)  
          
    # Sort options by volume in descending order and select the top 10  
    top_10_options = all_options.sort_values(by='volume', ascending=False).head(10)  
      
    return top_10_options  
  
# Call the function and display the top 10 options  
top_10_options = get_top_10_apple_options()  











# Define Black-Scholes functions
def calc_d1_d2(S, K, r, T, sigma):  
    d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))  
    d2 = d1 - sigma * math.sqrt(T)  
    return d1, d2  


def blackScholes(S, K, r, T, sigma, type ="c"):
    # Calculate Black-Scholes option price for a call or put options
    d1, d2 = calc_d1_d2(S, K, r, T, sigma)
    if type == "c":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif type == "p":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta(S, K, r, T, sigma, type ="c"):
    # Delta is the theoretical change of the option value in relation to the underlying S (first derivative)
    d1, d2 = calc_d1_d2(S, K, r, T, sigma)
    try:
        if type == "c":
            return norm.cdf(d1)
        elif type == "p":
            return norm.cdf(d1) - 1
    except:
        st.sidebar.error("Please confirm all input parameters!")


def gamma(S, K, r, T, sigma):
    # Gamma measures the rate of change in the delta, w.r.t. changes in the underlying price (second derivative)
    d1, d2 = calc_d1_d2(S, K, r, T, sigma)
    try: 
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))
    except:
        st.sidebar.error("Please confirm all input parameters!")


def vega(S, K, r, T, sigma):
    # Vega measures sensitivity to volativity. Vega is the derivative of the option value w.r.t.
    d1, d2 = calc_d1_d2(S, K, r, T, sigma)
    try: 
        return S * norm.pdf(d1) * math.sqrt(T) *0.01 # in basis points
    except:
        st.sidebar.error("Please confirm all input parameters!")


def theta(S, K, r, T, sigma, type ="c"):
    # Theta measures the sensitivity of the deriviative to the passage of time
    d1, d2 = calc_d1_d2(S, K, r, T, sigma)
    try:
        if type == "c":
            return (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))) - r * K * math.exp(-r * T) * norm.cdf(d2)  
        elif type == "p":
            return (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))) + r * K * math.exp(-r * T) * norm.cdf(-d2)
    except:
        st.sidebar.error("Please confirm all input parameters!")


def rho(S, K, r, T, sigma, type ="c"):
    # Rho measures the sensitivity of the interest rate
    d1, d2 = calc_d1_d2(S, K, r, T, sigma)
    try:
        if type == "c":
            return K * T * math.exp(-r * T) * norm.cdf(d2) * 0.01  # in basis points
        elif type == "p":
            return -K * T * math.exp(-r * T) * norm.cdf(-d2) * 0.01  # in basis points 
    except:
        st.sidebar.error("Please confirm all input parameters!")


#Streamlit setup
def setup_sidebar():
    st.sidebar.header("Black-Scholes Parameters")
    
    # Fetch and display current risk-free rate
    try:
        current_r = get_risk_free_rate()
        st.sidebar.write(f"Current 10-Y Treasury Yield (^TNX): [{current_r:.2%}](https://finance.yahoo.com/quote/%5ETNX/)")
     #https://finance.yahoo.com/quote/%5ETNX/
    
    except:
        st.sidebar.error("Unable to retrieve the current risk-free rate.")
        current_r = 0.04  # Set default if fetch fails
    
    r = st.sidebar.number_input("Risk-Free Rate", min_value=0.000, max_value=1.000, step=0.001, value=current_r)

    S = st.sidebar.number_input("Underlying Asset Price", min_value=1.00, step=0.10, value=30.00)

    K = st.sidebar.number_input("Strike Price", min_value=1.00, step=0.10, value=40.00)
    
    days_to_expiry = st.sidebar.number_input("Time to Maturity (in days)", min_value=1, step=1, value=240)
    # Time to maturity and the risk-free rate should be expressed on an annualized basis (p.a.)
    T = days_to_expiry / 365
    
    sigma = st.sidebar.number_input("Volatility (Sigma)", min_value=0.000, max_value=1.000, step=0.01, value=0.30)
    # Sigma is the implied (not historical) volatility of the underlying asset's returns
    
    type_input = st.sidebar.selectbox("Option Type", ["Call", "Put"])
    
    type = "c" if type_input == "Call" else "p"
    
    return S, K, r, T, sigma, type


# Calculation and display for Streamlit
def calculate_and_display(S, K, r, T, sigma, type):  
    with st.spinner("Please wait..."):  
        
        # Define a range of spot prices from 1 to S + 50 for sensitivity analysis
        spot_prices = np.linspace(1, S + 50, 100)  

        # Calculate option price and Greeks
        # We need to calculate and store the Greek values for each spot_price before plotting
        prices = [blackScholes(i, K, r, T, sigma, type) for i in spot_prices]
        deltas = [delta(i, K, r, T, sigma, type) for i in spot_prices]
        gammas = [gamma(i, K, r, T, sigma) for i in spot_prices]
        vegas = [vega(i, K, r, T, sigma) for i in spot_prices]
        thetas = [theta(i, K, r, T, sigma, type) for i in spot_prices]
        rhos = [rho(i, K, r, T, sigma, type) for i in spot_prices]

        # Plot all Greeks in a loop
        greeks = {
            "Option Price": prices,
            "Delta": deltas,
            "Gamma": gammas,
            "Vega": vegas,
            "Theta": thetas,
            "Rho": rhos,
        }

        # Web-app layout
        st.markdown("<h2 align='center'>Black-Scholes Option Price Calculator</h2>", unsafe_allow_html=True)
        st.markdown("<h5 align='center'>Made by the <a href='https://github.com/profit-prophet00'>profit-prophet</a></h5>", unsafe_allow_html=True)
        st.markdown("<h6 align='center'></h6>", unsafe_allow_html=True)
        
        st.markdown("<h6>See project's description: <a href='https://github.com/profit-prophet00/Options-price-calculator/blob/main/README.md'>here</a></h6>", unsafe_allow_html=True)
        st.markdown("<h6>See all my other projects here: <a href='https://github.com/profit-prophet00?tab=repositories'>here</a></h6>", unsafe_allow_html=True)
        st.markdown("<h6 align='center'></h6>", unsafe_allow_html=True)

        #st.markdown("<h3 align='center'>Top 10 APPLE options by volume)</h3>", unsafe_allow_html=True)
        #top_10_options
        st.markdown("<h7 align='center'></h7>", unsafe_allow_html=True)

        st.markdown("<h3 align='center'>Option Prices and Greeks</h3>", unsafe_allow_html=True)
        st.markdown("<h7 align='center'></h7>", unsafe_allow_html=True)

        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        col2.metric("Call Price", str(round(blackScholes(S, K, r, T, sigma,type="c"), 3)))
        col4.metric("Put Price", str(round(blackScholes(S, K, r, T, sigma,type="p"), 3)))

        bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
        bcol1.metric("Delta", str(round(delta(S, K, r, T, sigma,type = type), 3)))
        bcol2.metric("Gamma", str(round(gamma(S, K, r, T, sigma), 3)))
        bcol3.metric("Theta", str(round(theta(S, K, r, T, sigma,type = type), 3)))
        bcol4.metric("Vega", str(round(vega(S, K, r, T, sigma), 3)))
        bcol5.metric("Rho", str(round(rho(S, K, r, T, sigma,type = type), 3)))
        
        # Plotting prices and greeks, with custom layout and theme
        plots_per_row = 2

        # Create columns based on the number of plots per row
        cols = st.columns(plots_per_row)

        # Set a theme for the plots
        sns.set_theme(style="darkgrid")

        # Iterate over the greeks and plot each in a seperate column
        for i, (greek_name, values) in enumerate (greeks.items()):
            
            # Determine which column to use for the current plot  
            col = cols[i % plots_per_row] 

            # Create a figure and axis
            fig, ax = plt.subplots()
            
            # Plot data
            sns.lineplot(x=spot_prices, y=values, ax=ax, color='orange', linewidth = 3.5)

            # Set the figure background color  
            fig.patch.set_facecolor('#3d4350')

            # Change the color of the x-axis label  
            ax.xaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
  
            # Change the color of the y-axis label  
            ax.yaxis.label.set_color('white')  
            ax.tick_params(axis='y', colors='white')
            
            # Set the y-axis formatter to use commas as thousands separators  
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.4f}'))   

            # Set plot label and title
            ax.set_ylabel(None)
            ax.set_xlabel("Underlying Asset Price")
            ax.set_title(greek_name + " (" + type + ")", color = "white")
            
            # Display the plot in the corresponding column
            col.pyplot(fig)


# Call functions
# Get model parameters from side bar  
S, K, r, T, sigma, type = setup_sidebar()  
  
# Call the calculation and display function when parameters are changed  
calculate_and_display(S, K, r, T, sigma, type)
