import scipy.stats as scipy
from numpy import exp, sqrt, log
import matplotlib.pyplot as plt

class black_scholes_model:
    def __init__(self,
        strike_price: float,
        interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        spot_price: float,
        purchase_price: float,
        option_type: str):

        self.strike_price = strike_price
        self.interest_rate = interest_rate
        self.volatility = volatility
        self.time_to_maturity = time_to_maturity
        self.spot_price = spot_price
        self.purchase_price = purchase_price
        self.option_type = option_type

    def calculate(self):

        d1 = (((log(spot_price / strike_price)) +
               ((interest_rate + ((volatility ** 2) * .5)) * time_to_maturity))
               / (volatility * sqrt(time_to_maturity)))

        d2 = d1 - (volatility * sqrt(time_to_maturity))

        put = ((strike_price * exp(-(interest_rate * time_to_maturity)) * scipy.norm.cdf(-d2))
             - (spot_price * scipy.norm.cdf(-d1)))

        call = (spot_price * scipy.norm.cdf(d1)) - (scipy.norm.cdf(d2) * strike_price *
            exp(-interest_rate * time_to_maturity))

        self.call = call
        self.put = put
        self.call_delta = scipy.norm.cdf(d1)
        self.put_delta = 1 - scipy.norm.cdf(d1)
        self.call_gamma = scipy.norm.pdf(d1) / (spot_price * volatility * sqrt(time_to_maturity))
        self.put_gamma = self.call_gamma
        self.call_rho = (strike_price * .01 *
            time_to_maturity * exp(-interest_rate * time_to_maturity) * scipy.norm.cdf(d2))
        self.put_rho = -(strike_price * .01 *
            time_to_maturity * exp(-interest_rate * time_to_maturity) * scipy.norm.cdf(-d2))
        self.vega = (spot_price * .01 * sqrt(time_to_maturity) * scipy.norm.pdf(d1))

    def calculate_price(self):
        if self.option_type == 'call':
            return self.call
        elif self.option_type == 'put':
            return self.put

    def calculate_pnl(self):
        if option_type.title() == 'Call':
            current_price = self.call
        elif option_type.title() == 'Put':
            current_price = self.put

        pnl = (current_price - self.purchase_price)
        return pnl

# test
strike_price = 100
interest_rate = 0.05
volatility = 0.3
time_to_maturity = 2
spot_price = 80
purchase_price = 6
option_type = "put"

MODEL = black_scholes_model(
strike_price = strike_price
, interest_rate = interest_rate
, volatility = volatility
, time_to_maturity = time_to_maturity
, spot_price = spot_price
, purchase_price = purchase_price
,option_type = option_type)

MODEL.calculate()

print(f"option price ({MODEL.option_type}): {MODEL.calculate_price()}")

print(f"pnl: {MODEL.calculate_pnl()}")