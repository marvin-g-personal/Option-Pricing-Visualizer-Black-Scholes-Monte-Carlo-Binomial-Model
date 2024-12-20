import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns
from models.Binomial import BinomialOptionPricing
from models.MonteCarlo import monte_carlo_sim, calc_opt_price, visualize, stats

#######################
# Page configuration
st.set_page_config(
    page_title="Option Pricing Model",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Move sidebar to the right */
section[data-testid="stSidebar"] {
    left: unset !important;
    right: 0 !important;
    padding-top: 20px; /* Added padding to the top of the sidebar */
}

/* Center all text in the app */
h1, h2, h3, h4, h5, h6, p, div {
    text-align: center;
}

/* Center text in the sidebar, except input boxes */
section[data-testid="stSidebar"] .stSelectbox, 
section[data-testid="stSidebar"] .stNumberInput, 
section[data-testid="stSidebar"] .stSlider {
    text-align: left; /* Keep input boxes left-aligned */
}

section[data-testid="stSidebar"] .stMarkdown {
    text-align: center; /* Center text in markdown */
}

/* Adjust spacing for the separator */
section[data-testid="stSidebar"] hr {
    margin-top: 20px; /* Add space above the separator */
    margin-bottom: 20px; /* Add space below the separator */
}

.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
    width: 100%;
    margin: 10px 0;
}

.metric-call {
    background-color: #4CAF50;
    color: white;
    margin-right: 20px;
    border-radius: 15px;
    padding: 20px;
    width: 45%;
}

.metric-put {
    background-color: #F44336;
    color: white;
    border-radius: 15px;
    padding: 20px;
    width: 45%;
}

/* Custom cursor for the entire page */
html, body {
    cursor: url('https://cdn-icons-png.flaticon.com/512/25/25231.png'), crosshair; /* Custom cursor */
}
</style>
""", unsafe_allow_html=True)

#######################
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

#######################
# Function to generate heatmaps
def plot_heatmap(bs_model, spot_range, vol_range, strike, purchase_price_call, purchase_price_put):
    # Calculate PnL matrices
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

    plt.style.use('dark_background')
    fig_call, ax_call = plt.subplots(figsize=(15, 12)) 
    fig_put, ax_put = plt.subplots(figsize=(15, 12))   

    # Custom colormap
    custom_cmap = sns.diverging_palette(10, 133, as_cmap=True)

    # Calculate vmin and vmax for consistent color scaling
    vmin = min(call_pnl.min(), put_pnl.min())
    vmax = max(call_pnl.max(), put_pnl.max())
    
    abs_max = max(abs(vmin), abs(vmax))
    scale_factor = 0.5
    vmin, vmax = -abs_max * scale_factor, abs_max * scale_factor

    sns.heatmap(
        call_pnl,
        ax=ax_call,
        cmap=custom_cmap,
        center=0,
        annot=True,
        fmt='.1f',
        cbar_kws={
            'label': 'PnL ($)',
            'orientation': 'horizontal',
            'pad': 0.1,        # Reduced padding
            'aspect': 50,      # Made colorbar thinner
            'shrink': 0.8      # Made colorbar shorter
        },
        xticklabels=[f'{x:.1f}' for x in spot_range],
        yticklabels=[f'{x:.2f}' for x in vol_range],
        annot_kws={'size': 12},  # Increased annotation size
        square=True,
        vmin=vmin,
        vmax=vmax
    )

    # Plot PUT heatmap with enhanced styling and reduced margins
    sns.heatmap(
        put_pnl,
        ax=ax_put,
        cmap=custom_cmap,
        center=0,
        annot=True,
        fmt='.1f',
        cbar_kws={
            'label': 'PnL ($)',
            'orientation': 'horizontal',
            'pad': 0.1,        # Reduced padding
            'aspect': 50,      # Made colorbar thinner
            'shrink': 0.8      # Made colorbar shorter
        },
        xticklabels=[f'{x:.1f}' for x in spot_range],
        yticklabels=[f'{x:.2f}' for x in vol_range],
        annot_kws={'size': 12},  # Increased annotation size
        square=True,
        vmin=vmin,
        vmax=vmax
    )

    # Customize plots with reduced padding
    for ax, title in [(ax_call, 'CALL Option PnL'), (ax_put, 'PUT Option PnL')]:
        ax.set_title(title, fontsize=20, pad=10, color='white', fontweight='bold')
        ax.set_xlabel('Spot Price ($)', fontsize=14, color='white', labelpad=5)
        ax.set_ylabel('Volatility', fontsize=14, color='white', labelpad=5)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color='white', fontsize=12)
        plt.setp(ax.get_yticklabels(), rotation=0, color='white', fontsize=12)

    # Adjust layouts with minimal padding
    for fig in [fig_call, fig_put]:
        fig.patch.set_facecolor('#1E1E1E')
        fig.tight_layout(pad=1.0)  # Reduced padding

    return fig_call, fig_put

#######################
# Sidebar for User Inputs
with st.sidebar:
    
    # Model Selection
    model_option = st.selectbox(
        "Select Pricing Model",
        ("Black-Scholes", "Monte Carlo", "Binomial")
    )

    st.markdown("---")
    
    if model_option == "Black-Scholes":
        current_price = st.number_input("Current Asset Price", value=100.0)
        strike = st.number_input("Strike Price", value=100.0)
        time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
        volatility = st.number_input("Volatility (σ)", value=0.2)
        interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)
        
        purchase_price_call = st.number_input("Call Option Purchase Price", value=10.0)
        purchase_price_put = st.number_input("Put Option Purchase Price", value=8.0)

        st.markdown("---")
        
        spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
        spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
        vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
        vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)
        
        spot_range = np.linspace(spot_min, spot_max, 10)
        vol_range = np.linspace(vol_min, vol_max, 10)
    elif model_option == "Monte Carlo":
        current_price = st.number_input("Initial Stock Price ($)", value=100.0)
        strike = st.number_input("Strike Price ($)", value=100.0)
        interest_rate = st.number_input("Risk-Free Rate", value=0.03)
        volatility = st.number_input("Volatility (σ)", value=0.25)
        time_to_maturity = st.number_input("Time to Maturity (Years)", value=0.5)
        steps = st.number_input("Number of Time Steps", value=100)
        num_sims = st.number_input("Number of Simulations", value=1000)
    elif model_option == "Binomial":
        stock_price = st.number_input("Stock Price ($)", value=80.0)
        strike_price = st.number_input("Strike Price ($)", value=100.0)
        expiration_time = st.number_input("Expiration Time (Years)", value=2.0)
        no_risk_int = st.number_input("Risk-Free Rate", value=0.05)
        sigma = st.number_input("Volatility (σ)", value=0.3)
        steps = st.number_input("Number of Steps", value=10)
    else:
        st.markdown("---")
        st.write("### Model not implemented yet.")
        st.stop()

#######################
# Main Content
if model_option == "Black-Scholes":
    # Main Page for Output Display
    st.title("🔮 Black-Scholes Model")

    # Instantiate BlackScholes class
    bs_model = BlackScholes(
        time_to_maturity=time_to_maturity,
        strike=strike,
        current_price=current_price,
        volatility=volatility,
        interest_rate=interest_rate,
        purchase_price=purchase_price_call, 
        option_type='call'
    )
    call_price, put_price = bs_model.calculate_prices()

    # Display Call and Put Values
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
    st.header("P&L Heatmaps | Call & Put Options")
    st.info("Explore how option P&L fluctuates with varying 'Spot Prices' and 'Volatility' levels based on given input parameters.")

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
        st.subheader("Call P&Ls:")
        st.pyplot(heatmap_fig_call)

    with col2:
        st.subheader("Put P&Ls:")
        st.pyplot(heatmap_fig_put)

elif model_option == "Monte Carlo":
    st.title("🧮 Monte Carlo Model")
    
    # Run simulation
    sims = monte_carlo_sim(current_price, interest_rate, volatility, time_to_maturity, steps, num_sims)
    
    # Calculate option prices
    call_price, call_se = calc_opt_price(current_price, interest_rate, volatility, time_to_maturity, 
                                     steps, num_sims, strike, 'call')
    put_price, put_se = calc_opt_price(current_price, interest_rate, volatility, time_to_maturity, 
                                   steps, num_sims, strike, 'put')
    
    # Display prices in styled boxes
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-call">
                <div>
                    <div class="metric-label">CALL Value (± {call_se:.3f})</div>
                    <div class="metric-value">${call_price:.2f}</div>
                </div>
            </div>
            <div class="metric-put">
                <div>
                    <div class="metric-label">PUT Value (± {put_se:.3f})</div>
                    <div class="metric-value">${put_price:.2f}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monte Carlo Simulation Paths")
        fig_sim = visualize(sims)
        st.pyplot(fig_sim)
    
    with col2:
        st.subheader("Price Convergence Distribution")
        # Modify visualize_convergence to return figure
        fig_conv = plt.figure(figsize=(10, 8))
        x1 = np.linspace(call_price-3*call_se, call_price-call_se, 100)
        x2 = np.linspace(call_price-call_se, call_price+call_se, 100)
        x3 = np.linspace(call_price+call_se, call_price+3*call_se, 100)
        
        s1 = stats.norm.pdf(x1, call_price, call_se)
        s2 = stats.norm.pdf(x2, call_price, call_se)
        s3 = stats.norm.pdf(x3, call_price, call_se)
        
        plt.fill_between(x1, s1, color='tab:blue', label='> 1 StDev')
        plt.fill_between(x2, s2, color='cornflowerblue', label='±1 StDev')
        plt.fill_between(x3, s3, color='tab:blue')
        
        plt.plot([call_price, call_price], [0, max(s2)*1.1], 'k',
                label='Theoretical Value')
        
        plt.ylabel("Probability Density Function")
        plt.xlabel("Option Price ($)")
        plt.title("Option Price Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_conv)

elif model_option == "Binomial":
    st.title("🌳 Binomial Model")
    
    # Create Binomial model instances for both Call and Put
    binomial_call = BinomialOptionPricing(
        stock_price=stock_price,
        strike_price=strike_price,
        expiration_time=expiration_time,
        no_risk_int=no_risk_int,
        sigma=sigma,
        steps=steps,
        option_type="call"
    )
    
    binomial_put = BinomialOptionPricing(
        stock_price=stock_price,
        strike_price=strike_price,
        expiration_time=expiration_time,
        no_risk_int=no_risk_int,
        sigma=sigma,
        steps=steps,
        option_type="put"
    )
    
    # Calculate both option prices
    call_price = binomial_call.calculate_option()
    put_price = binomial_put.calculate_option()
    
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
    
    st.subheader("Binomial Tree Visualization")
    fig_tree = binomial_call.visualize_tree()
    plt.title("Binomial Tree Model for Call & Put Options", fontsize=14, pad=20, color='white')
    st.pyplot(fig_tree)

else:
    st.markdown("---")
    st.write("### Model not implemented yet.")
    st.stop()