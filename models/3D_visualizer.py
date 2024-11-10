# pages/black_scholes_3d.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import pandas as pd
import plotly.io as pio
from datetime import datetime
import io
import base64

def black_scholes_3d(S, K, r, sigma, T, q=0, option_type='call'):
    """Calculate Black-Scholes option price for multiple spots and times"""
    S, T = np.meshgrid(S, T)
    
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:  # put
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
    
    return price

def create_3d_surface(strike, rate, volatility, dividend_yield, option_type, template="plotly"):
    """Create 3D surface plot of option prices"""
    # Generate range of spot prices and times
    spot_prices = np.linspace(0.5*strike, 1.5*strike, 50)
    times = np.linspace(0.1, 2, 50)
    
    # Calculate option prices
    prices = black_scholes_3d(spot_prices, strike, rate, volatility, times, dividend_yield, option_type)
    
    # Create 3D surface
    fig = go.Figure(data=[go.Surface(
        x=np.array([spot_prices]*len(times)),
        y=np.array([times]*len(spot_prices)).T,
        z=prices,
        colorscale='Viridis'
    )])
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Black-Scholes {option_type.title()} Option Prices',
            x=0.5,
            y=0.95
        ),
        scene=dict(
            xaxis_title='Spot Price',
            yaxis_title='Time to Maturity (years)',
            zaxis_title='Option Price',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=800,
        template=template
    )
    
    return fig, prices, spot_prices, times

def convert_surface_to_csv(prices, spot_prices, times):
    """Convert surface data to CSV format"""
    df = pd.DataFrame(prices, columns=spot_prices, index=times)
    df.index.name = 'Time to Maturity'
    df.columns.name = 'Spot Price'
    return df

def get_download_link(df, filename):
    """Generate a download link for the DataFrame"""
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

def export_plot(fig, filename):
    """Export plot as HTML file"""
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_bytes = buffer.getvalue().encode()
    
    return html_bytes

def main():
    st.set_page_config(page_title="Black-Scholes 3D Visualizer", layout="wide")
    
    # Title and description
    st.title("Black-Scholes 3D Option Price Visualizer")
    st.markdown("""
    This tool visualizes how option prices change with respect to spot price and time to maturity.
    Adjust the parameters using the sidebar controls to see their impact on option prices.
    """)
    
    # Sidebar controls with tabs for organization
    st.sidebar.header("Model Parameters")
    
    tabs = st.sidebar.tabs(["Basic Parameters", "Advanced Parameters", "Visualization Settings"])
    
    with tabs[0]:
        strike = st.slider(
            "Strike Price",
            min_value=50.0,
            max_value=150.0,
            value=100.0,
            step=5.0,
            help="The strike price of the option"
        )
        
        option_type = st.selectbox(
            "Option Type",
            options=['call', 'put'],
            help="Type of option to visualize"
        )

    with tabs[1]:
        rate = st.slider(
            "Risk-free Rate",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="Annual risk-free interest rate"
        )
        
        volatility = st.slider(
            "Volatility",
            min_value=0.10,
            max_value=0.50,
            value=0.30,
            step=0.05,
            format="%.2f",
            help="Annual volatility of the underlying asset"
        )
        
        dividend_yield = st.slider(
            "Dividend Yield",
            min_value=0.0,
            max_value=0.05,
            value=0.0,
            step=0.01,
            format="%.2f",
            help="Annual dividend yield"
        )

    with tabs[2]:
        plot_template = st.selectbox(
            "Plot Theme",
            options=['plotly', 'plotly_dark', 'plotly_white', 'seaborn'],
            help="Visual theme for the plot"
        )
    
    # Create and display the 3D surface plot
    fig, prices, spot_prices, times = create_3d_surface(
        strike, rate, volatility, dividend_yield, option_type, plot_template
    )
    
    plot_container = st.container()
    with plot_container:
        st.plotly_chart(fig, use_container_width=True)
    
    # Export options in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Data")
        df = convert_surface_to_csv(prices, spot_prices, times)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"black_scholes_surface_{timestamp}.csv"
        
        csv_data = df.to_csv().encode()
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv"
        )
    
    with col2:
        st.subheader("Export Plot")
        html_bytes = export_plot(fig, f"black_scholes_plot_{timestamp}.html")
        st.download_button(
            label="Download Interactive Plot",
            data=html_bytes,
            file_name=f"black_scholes_plot_{timestamp}.html",
            mime="text/html"
        )
    
    # Additional information in expandable sections
    with st.expander("About the Black-Scholes Model"):
        st.markdown("""
        The Black-Scholes model is a mathematical model used to determine the fair price of a European-style option.
        
        ### Key Assumptions:
        - The underlying asset follows a geometric Brownian motion
        - No arbitrage opportunities
        - Risk-free interest rate is constant
        - No transaction costs or taxes
        - European-style options (can only be exercised at maturity)
        
        ### Model Parameters:
        - **Strike Price (K)**: The price at which the option can be exercised
        - **Risk-free Rate (r)**: The interest rate of a risk-free investment
        - **Volatility (σ)**: A measure of the underlying asset's price variability
        - **Time to Maturity (T)**: Time until the option expires
        - **Dividend Yield (q)**: Annual dividend yield of the underlying asset
        """)
    
    with st.expander("Surface Plot Guide"):
        st.markdown("""
        ### How to Use the 3D Plot:
        1. **Rotate**: Click and drag to rotate the surface
        2. **Zoom**: Use the scroll wheel or pinch gesture
        3. **Pan**: Right-click and drag
        4. **Reset View**: Double-click on the plot
        
        ### Understanding the Visualization:
        - **X-axis**: Spot price of the underlying asset
        - **Y-axis**: Time to maturity in years
        - **Z-axis**: Option price
        - **Colors**: Indicate option price levels (darker = lower, brighter = higher)
        """)
    
    # Display current parameter values
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current Parameters")
    st.sidebar.markdown(f"""
    - Strike Price: ${strike:.2f}
    - Risk-free Rate: {rate:.2%}
    - Volatility: {volatility:.2%}
    - Dividend Yield: {dividend_yield:.2%}
    - Option Type: {option_type.title()}
    """)
    
    # Add a footer with timestamp
    st.markdown("---")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    main()


