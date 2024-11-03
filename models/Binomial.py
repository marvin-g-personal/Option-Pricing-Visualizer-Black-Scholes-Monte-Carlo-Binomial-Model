import numpy as np
import matplotlib.pyplot as plt

class BinomialOptionPricing:
    def __init__(self, stock_price, strike_price, expiration_time, no_risk_int, sigma, steps, option_type="call"):
        self.stock_price = stock_price
        self.strike = strike_price
        self.expiration_time = expiration_time
        self.interest = no_risk_int
        self.sigma = sigma
        self.steps = steps
        self.option_type = option_type

        self.interval = expiration_time / steps
        self.up = np.exp(sigma * np.sqrt(self.interval))
        self.down = np.exp(-sigma * np.sqrt(self.interval))
        self.rnp = (np.exp(no_risk_int * self.interval) - self.down) / (self.up - self.down)  # risk-neutral probability

    def calculate_option(self):
        asset_p = np.zeros(self.steps + 1)
        option_p = np.zeros(self.steps + 1)

        for i in range(self.steps + 1):
            asset_p[i] = self.stock_price * (self.up ** (self.steps - i)) * (self.down ** i)

        for i in range(self.steps + 1):
            if self.option_type == "call":
                option_p[i] = max(0, asset_p[i] - self.strike)
            elif self.option_type == "put":
                option_p[i] = max(0, self.strike - asset_p[i])

        # Backward induction
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                # Price at node
                spot_price = self.stock_price * (self.up ** (i - j)) * (self.down ** j)
                hold_value = (self.rnp * option_p[j] + (1 - self.rnp) * option_p[j + 1]) * np.exp(-self.interest * self.interval)

                if self.option_type == "call":
                    exercise_value = max(0, spot_price - self.strike)
                else:  # put option
                    exercise_value = max(0, self.strike - spot_price)

                option_p[j] = max(hold_value, exercise_value)

        return option_p[0]

    def visualize_tree(self):
        # Set dark style
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Set dark background
        ax.set_facecolor('#1E1E1E')
        fig.patch.set_facecolor('#1E1E1E')
        
        asset_prices = np.zeros((self.steps + 1, self.steps + 1))

        for i in range(self.steps + 1):
            for j in range(i + 1):
                asset_prices[j, i] = self.stock_price * (self.up ** (i - j)) * (self.down ** j)

        for i in range(self.steps + 1):
            plt.plot([i] * (i + 1), asset_prices[:i + 1, i], "o", 
                    color='cornflowerblue', markersize=8)
            if i < self.steps:
                for j in range(i + 1):
                    plt.plot([i, i + 1], [asset_prices[j, i], asset_prices[j, i + 1]], 
                            color='skyblue', linestyle='-', alpha=0.7, linewidth=2)
                    plt.plot([i, i + 1], [asset_prices[j, i], asset_prices[j + 1, i + 1]], 
                            color='skyblue', linestyle='-', alpha=0.7, linewidth=2)

        plt.title(f"Binomial Tree for {self.option_type.title()} Option", 
                 fontsize=14, pad=20, color='white')
        plt.xlabel("Steps", fontsize=12, color='white')
        plt.ylabel("Price ($)", fontsize=12, color='white')
        plt.grid(True, alpha=0.2)  # Reduced grid opacity for dark theme
        
        # Make tick labels white
        plt.tick_params(colors='white')
        
        plt.tight_layout()
        return fig