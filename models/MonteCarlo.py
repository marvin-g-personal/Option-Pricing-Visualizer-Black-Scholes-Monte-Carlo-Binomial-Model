# import dependencies
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# create a monte carlo simulation
def monte_carlo_sim(S: float, r: float, vol: float, T: float, steps: int, num: int) -> np.ndarray:
    """
    Generates Monte Carlo simulation paths for stock prices.

    Parameters:
        S (float): Initial stock price.
        r (float): Risk-free rate.
        vol (float): Volatility of the stock.
        T (float): Time to maturity in years.
        steps (int): Number of time steps.
        num (int): Number of simulations.

    Returns:
        np.ndarray: Simulated stock price paths.
    """
    dt = T / steps
    nudt = (r - 0.5 * vol ** 2) * dt
    sidt = vol * np.sqrt(dt)

    random_shocks = np.random.normal(0, 1, (num, steps))
    sims = np.zeros((num, steps + 1))
    sims[:, 0] = S
    sims[:, 1:] = S * np.exp(np.cumsum(nudt + sidt * random_shocks, axis=1))
    
    return sims


def visualize(sims: np.ndarray) -> plt.Figure:
    """
    Plots the Monte Carlo simulation paths with diverse colors.

    Parameters:
        sims (np.ndarray): Simulated stock price paths.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    
    colors = [
        '#FF0000',  
        '#00FF00',  
        '#0000FF', 
        '#FFD700', 
        '#FF1493',  
        '#00FFFF',  
        '#FF4500',  
        '#9400D3',  
        '#32CD32', 
        '#FF69B4', 
        '#4169E1',  
        '#FF8C00',  
        '#8A2BE2',  
        '#20B2AA',  
        '#FF6347' 
    ]
    
    num_paths = sims.shape[0]
    colors = colors * (num_paths // len(colors) + 1)
    
    for i, path in enumerate(sims):
        ax.plot(path, color=colors[i], alpha=0.6, linewidth=1)
    
    mean_path = np.mean(sims, axis=0)
    ax.plot(mean_path, color='white', linewidth=2, label='Mean Path', zorder=num_paths+1)
    
    ax.set_xlabel("Time Steps", fontsize=12, color='white')
    ax.set_ylabel("Stock Price ($)", fontsize=12, color='white')
    ax.set_title("Monte Carlo Simulation Paths", fontsize=14, color='white', pad=20)
    ax.grid(True, alpha=0.2)
    
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    ax.legend(facecolor='#1E1E1E', edgecolor='white', labelcolor='white')
    
    fig.tight_layout()
    
    return fig


def calc_opt_price(S, r, vol, T, steps, num, K, option_type='call'):
    """
    Calculate option price and standard error.

    Parameters:
        S (float): Initial stock price.
        r (float): Risk-free rate.
        vol (float): Volatility.
        T (float): Time to maturity.
        steps (int): Number of steps.
        num (int): Number of simulations.
        K (float): Strike price.
        option_type (str): 'call' or 'put'.

    Returns:
        tuple: (Option Price, Standard Error)
    """
    sims = monte_carlo_sim(S, r, vol, T, steps, num)
    if option_type.lower() == 'call':
        payoffs = np.maximum(sims[:, -1] - K, 0)
    elif option_type.lower() == 'put':
        payoffs = np.maximum(K - sims[:, -1], 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    discounted_payoffs = np.exp(-r * T) * payoffs
    C0 = np.mean(discounted_payoffs)
    SE = np.std(discounted_payoffs) / np.sqrt(num)
    
    return C0, SE


def visualize_convergence(C0: float, SE: float, prem: float) -> plt.Figure:
    """
    Visualizes the convergence of option prices with standard deviations.

    Parameters:
        C0 (float): Theoretical option price.
        SE (float): Standard error.
        prem (float): Market option price.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    
    x1 = np.linspace(C0 - 3 * SE, C0 - SE, 100)
    x2 = np.linspace(C0 - SE, C0 + SE, 100)
    x3 = np.linspace(C0 + SE, C0 + 3 * SE, 100)
    
    s1 = stats.norm.pdf(x1, C0, SE)
    s2 = stats.norm.pdf(x2, C0, SE)
    s3 = stats.norm.pdf(x3, C0, SE)

    ax.fill_between(x1, s1, color='tab:blue', alpha=0.3, label='> 1 StDev')
    ax.fill_between(x2, s2, color='cornflowerblue', alpha=0.6, label='±1 StDev')
    ax.fill_between(x3, s3, color='tab:blue', alpha=0.3)
    
    ax.plot([C0, C0], [0, stats.norm.pdf(C0, C0, SE)], 'white', linewidth=2, label='Theoretical Value')
    ax.plot([prem, prem], [0, stats.norm.pdf(C0, C0, SE)], 'r', linewidth=2, label='Market Value')
    
    ax.set_ylabel("Probability Density", fontsize=12, color='white')
    ax.set_xlabel("Option Price ($)", fontsize=12, color='white')
    ax.set_title("Option Price Distribution", fontsize=14, color='white', pad=20)
    
    max_pdf = max(np.max(s1), np.max(s2), np.max(s3))
    ax.set_ylim(0, max_pdf * 1.1)
    
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2)
    for spine in ax.spines.values():
        spine.set_color('white')
    
    ax.legend(facecolor='#1E1E1E', edgecolor='white', labelcolor='white')
    
    fig.tight_layout()
    return fig


def run_monte_carlo_example():
    """
    Runs a Monte Carlo simulation example.
    """
    # Initial parameters
    S = 100  # Starting stock price
    K = 110  # Strike price
    r = 0.03  # Risk-free rate
    vol = 0.25  # Volatility
    T = 0.5  # Time to maturity
    steps = 1000  # Number of time steps
    num = 1000  # Number of simulations
    
    # Run simulation
    sims = monte_carlo_sim(S, r, vol, T, steps, num)
    fig_sim = visualize(sims)
    plt.show(fig_sim)
    
    # Calculate option price
    C0, SE = calc_opt_price(S, r, vol, T, steps, num, K, 'call')
    
    # Visualize convergence
    fig_conv = visualize_convergence(C0, SE, C0)  # Using C0 as premium for example
    plt.show(fig_conv)

if __name__ == "__main__":
    run_monte_carlo_example()
