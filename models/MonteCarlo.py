# import dependencies
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# create a monte carlo simulation
def monte_carlo_sim(S, r, vol, T, steps, num):
    """ paramters:
    S = asset's initial price
    r = asset's historical return
    vol = asset's volatility
    T = time to maturity
    steps = number of time steps
    num = number of simulations
    K = strike price
    rate = risk-free rate
    prem = option premium """
    sims = np.zeros((num, 1 + steps))
    sims[:, 0] = S
    dt = T / steps
    nudt = (r - 0.5 * vol **2) *dt
    sidt = vol * np.sqrt(dt)

    for i in range(num):
        for j in range(steps):
            sims[i, j+1] = sims[i, j] * np.exp(nudt + sidt * np.random.normal())
    return sims


# visualize the monte carlo simulation
def visualize(sims):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(len(sims)):
        ax.plot(sims[i], alpha=0.3)
    ax.set_xlabel("Number of Time Steps", fontsize=12)
    ax.set_ylabel("Stock Price ($)", fontsize=12)
    ax.set_title("Monte Carlo Simulation for Stock Price", fontsize=14)
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    return fig


def calc_opt_price(S, r, vol, T, steps, num, K, option_type='call'):
    """
    Parameters:
        S (float): Initial stock price
        r (float): Risk-free rate
        vol (float): Volatility
        T (float): Time to maturity
        steps (int): Number of time steps
        num (int): Number of simulations
        K (float): Strike price
        option_type (str): 'call' or 'put'
    Returns:
        C0 (float): Option price
        SE (float): Standard error
    """
    dt = T / steps
    nudt = (r - 0.5 * vol **2) * dt
    sidt = vol * np.sqrt(dt)
    lnS = np.log(S)

    # Initialize sums
    sum_CT = 0
    sum_CT2 = 0

    for i in range(num):
        lnSt = lnS
        for j in range(steps):
            lnSt += nudt + sidt * np.random.normal()
        ST = np.exp(lnSt)
        if option_type == 'call':
            CT = max(0, ST - K)
        else:
            CT = max(0, K - ST)
        sum_CT += CT
        sum_CT2 += CT**2

    # Compute expectation and standard error
    C0 = np.exp(-r*T) * sum_CT / num
    sigma = np.sqrt((sum_CT2 - (sum_CT**2)/num) * np.exp(-2*r*T) / (num -1))
    SE = sigma / np.sqrt(num)

    return C0, SE


def visualize_convergence(C0, SE, prem):
    x1 = np.linspace(C0-3*SE, C0-1*SE, 100)
    x2 = np.linspace(C0-1*SE, C0+1*SE, 100)
    x3 = np.linspace(C0+1*SE, C0+3*SE, 100)
    
    s1 = stats.norm.pdf(x1, C0, SE)
    s2 = stats.norm.pdf(x2, C0, SE)
    s3 = stats.norm.pdf(x3, C0, SE)
    
    plt.fill_between(x1, s1, color='tab:blue',label='> StDev')
    plt.fill_between(x2, s2, color='cornflowerblue',label='1 StDev')
    plt.fill_between(x3, s3, color='tab:blue')
    
    plt.plot([C0,C0],[0, max(s2)*1.1], 'k',
            label='Theoretical Value')
    plt.plot([prem,prem],[0, max(s2)*1.1], 'r',
            label='Market Value')
    
    plt.ylabel("Probability")
    plt.xlabel("Option Price")
    plt.legend()
    plt.show()


def run_monte_carlo_example():
    # Initial parameters
    S = 100  # Starting stock price
    K = 110  # Strike price
    r = 0.03  # Risk-free rate
    vol = 0.25  # Volatility
    T = 0.5  # Time to maturity
    steps = 1000  # Reduced from 10000 for faster execution
    num = 100  # Reduced from 1000 for faster execution
    
    # Run simulation
    sims = monte_carlo_sim(S, r, vol, T, steps, num)
    visualize(sims)
    
    # Calculate option price
    C0, SE = calc_opt_price(S, r, vol, T, steps, num, K, r)
    
    # Visualize convergence
    visualize_convergence(C0, SE, C0)  # Using C0 as premium for example
