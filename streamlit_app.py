import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Center all text in the app */
h1, h2, h3, h4, h5, h6, p, div {
    text-align: center;
}

/* Style the CALL and PUT value boxes */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px; /* Increased padding for better visibility */
    width: 100%;
    margin: 10px 0; /* Added margin for spacing */
}

.metric-call {
    background-color: #4CAF50; /* Green background for CALL */
    color: white; /* White font color */
    margin-right: 20px; /* Spacing between CALL and PUT */
    border-radius: 15px; /* More rounded corners */
    padding: 20px; /* Padding inside the box */
    width: 45%; /* Width of the box */
}

.metric-put {
    background-color: #F44336; /* Red background for PUT */
    color: white; /* White font color */
    border-radius: 15px; /* More rounded corners */
    padding: 20px; /* Padding inside the box */
    width: 45%; /* Width of the box */
}

/* Adjust the sidebar to the right using CSS */
.sidebar .sidebar-content {
    order: 2; /* Move sidebar to the right */
}

.main .block-container {
    order: 1; /* Ensure main content stays on the left */
}
</style>
""", unsafe_allow_html=True)

# BlackScholes class definition
class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
        purchase_price: float,
        option_type: str
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.purchase_price = purchase_price
        self.option_type = option_type

    def calculate_prices(self):
        d1 = (
            log(self.current_price / self.strike) +
            (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity
        ) / (
            self.volatility * sqrt(self.time_to_maturity)
        )
        d2 = d1 - self.volatility * sqrt(self.time_to_maturity)

        call_price = self.current_price * norm.cdf(d1) - (
            self.strike * exp(-(self.interest_rate * self.time_to_maturity)) * norm.cdf(d2)
        )
        put_price = (
            self.strike * exp(-(self.interest_rate * self.time_to_maturity)) * norm.cdf(-d2)
        ) - self.current_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        # Calculate PnL
        if self.option_type == 'call':
            self.pnl = call_price - self.purchase_price
        else:
            self.pnl = put_price - self.purchase_price

        # GREEKS
        # Delta
        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)

        # Gamma
        self.call_gamma = norm.pdf(d1) / (
            self.strike * self.volatility * sqrt(self.time_to_maturity)
        )
        self.put_gamma = self.call_gamma  # Corrected typo

        return call_price, put_price

# Function to generate heatmaps
def plot_heatmap(bs_model, spot_range, vol_range, strike, purchase_price_call, purchase_price_put):
    call_pnl = np.zeros((len(vol_range), len(spot_range)))
    put_pnl = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            # Calculate CALL PnL
            bs_temp_call = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate,
                purchase_price=purchase_price_call,
                option_type='call'
            )
            call_price, _ = bs_temp_call.calculate_prices()
            call_pnl[i, j] = call_price - purchase_price_call

            # Calculate PUT PnL
            bs_temp_put = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate,
                purchase_price=purchase_price_put,
                option_type='put'
            )
            _, put_price = bs_temp_put.calculate_prices()
            put_pnl[i, j] = put_price - purchase_price_put
    
    # Update heatmap coloring to use RdYlGn for PnL (red for negative, green for positive)
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_pnl, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), 
                annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax_call)
    ax_call.set_title('CALL Option PnL')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_pnl, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), 
                annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax_put)
    ax_put.set_title('PUT Option PnL')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put

# Sidebar for User Inputs
with st.sidebar:
    st.title("Options Pricing Visualizer")
    
    # Model Selection
    model_option = st.selectbox(
        "Select Pricing Model",
        ("Black-Scholes", "Monte Carlo", "Binomial")
    )
    
    if model_option == "Black-Scholes":
        current_price = st.number_input("Current Asset Price", value=100.0)
        strike = st.number_input("Strike Price", value=100.0)
        time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
        volatility = st.number_input("Volatility (σ)", value=0.2)
        interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)
        
        # New inputs for purchase price
        purchase_price_call = st.number_input("Call Option Purchase Price", value=10.0)
        purchase_price_put = st.number_input("Put Option Purchase Price", value=8.0)

        st.markdown("---")
        
        # No Generate Heatmaps button; proceed immediately
        spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
        spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
        vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
        vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)
        
        spot_range = np.linspace(spot_min, spot_max, 10)
        vol_range = np.linspace(vol_min, vol_max, 10)
    else:
        st.markdown("---")
        st.write("### Model not implemented yet.")
        st.stop()

if model_option == "Black-Scholes":
    # Main Page for Output Display
    st.title("Black-Scholes Pricing Model")

    # Instantiate BlackScholes class
    bs_model = BlackScholes(
        time_to_maturity=time_to_maturity,
        strike=strike,
        current_price=current_price,
        volatility=volatility,
        interest_rate=interest_rate,
        purchase_price=purchase_price_call,  # Initialize with call purchase price
        option_type='call'  # Initial type doesn't affect heatmap
    )
    call_price, put_price = bs_model.calculate_prices()

    # Display Call and Put Values in styled boxes
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-call">
                <div>
                    <div class="metric-label">CALL Value</div>
                    <div class="metric-value">${call_price:.2f}</div>
                </div>
            </div>
            <div class="metric-put">
                <div>
                    <div class="metric-label">PUT Value</div>
                    <div class="metric-value">${put_price:.2f}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.header("Options P&L - Interactive Heatmap")
    st.info("Explore how option PnL fluctuates with varying 'Spot Prices' and 'Volatility' levels based on your input parameters.")

    # Generate Heatmaps
    heatmap_fig_call, heatmap_fig_put = plot_heatmap(
        bs_model,
        spot_range,
        vol_range,
        strike,
        purchase_price_call,
        purchase_price_put
    )

    # Display Heatmaps Side by Side
    col1, col2 = st.columns([1,1], gap="small")

    with col1:
        st.subheader("Call Option PnL Heatmap")
        st.pyplot(heatmap_fig_call)

    with col2:
        st.subheader("Put Option PnL Heatmap")
        st.pyplot(heatmap_fig_put)
