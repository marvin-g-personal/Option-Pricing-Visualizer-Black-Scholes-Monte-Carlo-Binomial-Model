import numpy as np
import matplotlib.pyplot as plt

class BinomialOptionPricing:
    def __init__(self, stock_price, strike_price, expiration_time, no_risk_int, sigma, steps, option_type = "call"):
        self.stock_price = stock_price
        self.strike = strike_price
        self.time = expiration_time
        self.interest = no_risk_int
        self.sigma = sigma
        self.steps = steps
        self.option_type = option_type

        self.interval = expiration_time / steps
        self.up = np.exp(sigma * np.sqrt(self.interval))
        self.down = np.exp(-sigma * np.sqrt(self.interval))
        self.rnp = (np.exp(no_risk_int * self.interval) - self.down) / (self.up - self.down)  # risk neutral probability

    def calculate_option(self):
        asset_p = np.zeros(steps + 1)
        option_p = np.zeros(steps + 1)

        for i in range(self.steps + 1):
            asset_p[i] = self.stock_price * (self.up ** (self.steps - i)) * (self.down ** i)

        for i in range(self.steps + 1):
            if self.option_type == "call":
                option_p[i] = max(0, asset_p[i] - self.strike)
            elif self.option_type == "put":
                option_p[i] = max(0, self.strike - asset_p[i])

            # backward induction
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                # price at node
                spot_price = self.stock_price * (self.up ** (i - j)) * (self.down ** j)
                hold_value = (self.rnp * option_p[j] + (1 - self.rnp) * option_p[j + 1]) * np.exp(-self.interest * self.interval)

                if self.option_type == "call":
                    exercise_value = max(0, spot_price - self.strike)
                else:  # put option
                    exercise_value = max(0, self.strike - spot_price)

                # should you hold??????
                option_p[j] = max(hold_value, exercise_value)

        return option_p[0]

    def visualize_tree(self):
        asset_prices = np.zeros((self.steps + 1, self.steps + 1))

        for i in range(self.steps + 1):
            for j in range(i + 1):
                asset_prices[j, i] = self.stock_price * (self.up ** (i - j)) * (self.down ** j)

        plt.figure(figsize=(10, 7))
        for i in range(self.steps + 1):
            plt.plot([i] * (i + 1), asset_prices[:i + 1, i], "o")
            if i < self.steps:
                for j in range(i + 1):
                    plt.plot([i, i + 1], [asset_prices[j, i], asset_prices[j, i + 1]], "b-", alpha = 0.5)
                    plt.plot([i, i + 1], [asset_prices[j, i], asset_prices[j + 1, i + 1]], "b-", alpha = 0.5)

        plt.title(f"Binomial Tree for {self.option_type.title()} Option")
        plt.xlabel("Steps")
        plt.ylabel("Price")
        plt.grid(True, alpha = 0.7)
        plt.tight_layout()
        plt.show()


stock_price = 80
strike_price = 100
expiration_time = 2
no_risk_int = 0.05
sigma = 0.3
steps = 10



call_price = BinomialOptionPricing(stock_price = stock_price,
    strike_price = strike_price,
    expiration_time = expiration_time,
    no_risk_int = no_risk_int,
    sigma = sigma,
    steps = steps,
    option_type='call')

put_price = BinomialOptionPricing(stock_price = stock_price,
    strike_price = strike_price,
    expiration_time = expiration_time,
    no_risk_int = no_risk_int,
    sigma = sigma,
    steps = steps,
    option_type='put')

answer_call = call_price.calculate_option()
answer_put = put_price.calculate_option()

print(f"Call Option Price: {answer_call}")
print(f"Put Option Price: {answer_put}")
call_price.visualize_tree()