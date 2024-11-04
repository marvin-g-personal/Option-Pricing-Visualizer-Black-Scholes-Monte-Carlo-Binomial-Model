import numpy as np
import matplotlib.pyplot as plt

class BinomialOptionPricing:
    """
    Binomial Option Pricing Model.
    """
    def __init__(self, stock_price: float, strike_price: float, expiration_time: float,
                 no_risk_int: float, sigma: float, steps: int, option_type: str="call"):
        """
        Initializes the Binomial Option Pricing model with given parameters.

        Parameters:
            stock_price (float): Current stock price.
            strike_price (float): Strike price of the option.
            expiration_time (float): Time to expiration in years.
            no_risk_int (float): Risk-free interest rate.
            sigma (float): Volatility of the stock.
            steps (int): Number of steps in the binomial tree.
            option_type (str): Type of option ('call' or 'put').
        """
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
        self.rnp = (np.exp(no_risk_int * self.interval) - self.down) / (self.up - self.down)  # Risk-neutral probability

    def calculate_option(self) -> float:
        """
        Calculates the option price using the binomial tree.

        Returns:
            float: The calculated option price.
        """
        asset_p = np.zeros(self.steps + 1)
        option_p = np.zeros(self.steps + 1)

        # Calculate asset prices at maturity
        for i in range(self.steps + 1):
            asset_p[i] = self.stock_price * (self.up ** (self.steps - i)) * (self.down ** i)

        # Calculate option values at maturity
        for i in range(self.steps + 1):
            if self.option_type.lower() == "call":
                option_p[i] = max(0, asset_p[i] - self.strike)
            elif self.option_type.lower() == "put":
                option_p[i] = max(0, self.strike - asset_p[i])
            else:
                raise ValueError("Option type must be 'call' or 'put'.")

        # Backward induction for option pricing
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                # Price at node
                spot_price = self.stock_price * (self.up ** (i - j)) * (self.down ** j)
                hold_value = (self.rnp * option_p[j] + (1 - self.rnp) * option_p[j + 1]) * np.exp(-self.interest * self.interval)

                if self.option_type.lower() == "call":
                    exercise_value = max(0, spot_price - self.strike)
                else:  # put option
                    exercise_value = max(0, self.strike - spot_price)

                option_p[j] = max(hold_value, exercise_value)

        return option_p[0]

    def visualize_tree(self) -> plt.Figure:
        """
        Visualizes the binomial tree for the option pricing.

        Returns:
            plt.Figure: The matplotlib figure object.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Set dark background
        ax.set_facecolor('#1E1E1E')
        fig.patch.set_facecolor('#1E1E1E')
        
        asset_prices = np.zeros((self.steps + 1, self.steps + 1))

        # Calculate asset prices for each node
        for i in range(self.steps + 1):
            for j in range(i + 1):
                asset_prices[j, i] = self.stock_price * (self.up ** (i - j)) * (self.down ** j)

        # Plot asset prices and connections
        for i in range(self.steps + 1):
            plt.plot([i] * (i + 1), asset_prices[:i + 1, i], "o", 
                     color='cornflowerblue', markersize=8)
            if i < self.steps:
                for j in range(i + 1):
                    # Up move
                    plt.plot([i, i + 1], [asset_prices[j, i], asset_prices[j, i + 1]], 
                             color='skyblue', linestyle='-', alpha=0.7, linewidth=2)
                    # Down move
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

# Example Usage
if __name__ == "__main__":
    stock_price = 80
    strike_price = 100
    expiration_time = 2.0
    no_risk_int = 0.05
    sigma = 0.3
    steps = 10
    option_type = "put"

    binomial_model = BinomialOptionPricing(
        stock_price=stock_price,
        strike_price=strike_price,
        expiration_time=expiration_time,
        no_risk_int=no_risk_int,
        sigma=sigma,
        steps=steps,
        option_type=option_type
    )

    option_price = binomial_model.calculate_option()
    print(f"Option Price ({option_type}): {option_price:.2f}")

    fig = binomial_model.visualize_tree()
    plt.show(fig)