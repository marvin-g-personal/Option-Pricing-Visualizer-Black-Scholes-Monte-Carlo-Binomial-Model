# import dependencies
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

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

# create a monte carlo simulation
def monte_carlo_sim(S, r, vol, T, steps, num):
    sims = np.zeros((num, 1 + steps))
    sims[:, 0] = S
    dt = T / steps
    nudt = (r - 0.5 * vol **2) *dt
    sidt = vol * np.sqrt(dt)

    for i in range(0, Nrep):
        for j in range(0, Nsteps):
            sims[i, j+1] = sims[i, j] * np.exp(nudt + sidt * np.random.normal())
    return sims

# setup example testing parameters
S = 100
K = 110
r = 0.03
vol = 0.25
T = 0.5
steps = 10000
num = 1000
sims = monte_carlo_sim(S0, r, vol, T, steps, num)

# visualize the monte carlo simulation
def visualize(sims):
    plt.figure(figsize = (10,8))
    for i in range(len(sims)):
        plt.plot(sims[i])
    plt.xlabel("number of time steps")
    plt.ylabel("stock price")
    plt.title("monte carlo simulation for stock price")
    plt.show

# visualize(sims)

# calculate call value
def calc_opt_price(S, r, vol, T, steps, num, K, rate):
    dt = T / steps
    nudt = (rate - 0.5 * vol **2) *dt
    sidt = vol * np.sqrt(dt)
    lnS = np.log(S)

    # standard error placeholders
    sum_CT = 0
    sum_CT2 = 0

    for i in range(num):
        lnSt = lnS
        for j in range(steps):
            lnSt = lnSt + nudt + sidt *np.random.normal()

        ST = np.exp(lnSt)
        CT = max(0, ST - K)
        sum_CT = sum_CT + CT
        sum_CT2 = sum_CT2 + CT*CT

    # compute expectation and SE
    C0 = np.exp(-rate*T)*sum_CT/num
    sigma = np.sqrt( (sum_CT2 - sum_CT*sum_CT/num)*np.exp(-2*rate*T) / (num -1) )
    SE = sigma/np.sqrt(num)

    return C0, SE

# setup example testing parameters for call option
S = 101.15
K = 98.01
r = 0.01
vol = 0.0991
T = 0.5
steps = 1000
num = 10
rate = 0.01
prem = 3.86
# calc_opt_price(S, r, vol, T, steps, num, K, rate)

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

# setup example testing parameters for call option
C0 = 3.955686199025182
SE = 0.15199116854795436
prem = 3.86

# visualize_convergence(C0, SE, prem)
