# 💰 Options Pricing Dashboard

Welcome to the **Options Pricing Dashboard**, an interactive platform designed to help you visualize and analyze option prices using various financial models. Whether you're a trader, student, or finance enthusiast, this dashboard offers a comprehensive toolset to understand how different factors influence option valuations.

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue)]

## 📚 Table of Contents

- [Overview](#overview)
- [🚀 Features](#-features)
- [🔍 Models Explained](#-models-explained)
  - [1. Black-Scholes Model](#1-black-scholes-model)
  - [2. Monte Carlo Simulation](#2-monte-carlo-simulation)
  - [3. Binomial Model](#3-binomial-model)
- [📸 Screenshots](#-screenshots)
- [🛠️ Getting Started](#️-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## Overview

The **Options Pricing Dashboard** provides an intuitive interface to explore and analyze option pricing using three prominent financial models:

- **Black-Scholes Model**
- **Monte Carlo Simulation**
- **Binomial Model**

Each model offers unique insights into how various parameters such as Spot Price, Volatility, Time to Maturity, and Risk-Free Interest Rate impact the pricing of Call and Put options.

## 🚀 Features

1. **Options Pricing Visualization**:
   - Display both Call and Put option prices using interactive heatmaps.
   - Real-time updates as you adjust parameters like Spot Price, Volatility, and Time to Maturity.

2. **Interactive Dashboard**:
   - Input different values for Spot Price, Volatility, Strike Price, Time to Maturity, and Risk-Free Interest Rate.
   - Immediate calculation and display of both Call and Put option prices for easy comparison.

3. **Model-Specific Insights**:
   - **Black-Scholes**: Understand the theoretical pricing mechanism.
   - **Monte Carlo**: Simulate a wide range of possible price paths.
   - **Binomial Model**: Explore discrete-time option pricing trees.

4. **Customizable Parameters**:
   - Set custom ranges for Spot Price and Volatility to generate comprehensive views under diverse market conditions.

5. **Visualization Tools**:
   - Heatmaps for P&L analysis.
   - Simulation paths and price convergence distributions.

## 🔍 Models Explained

### 1. Black-Scholes Model

The **Black-Scholes Model** is a mathematical model for pricing an options contract. It estimates the variation over time of financial instruments, specifically European-style options.

- **Key Features**:
  - Calculates theoretical option prices.
  - Provides insights into the Greeks (Delta, Gamma, etc.) for risk management.
  - Assumes constant volatility and interest rates.

### 2. Monte Carlo Simulation

**Monte Carlo Simulation** is a computational algorithm that relies on repeated random sampling to obtain numerical results. In option pricing, it simulates a large number of possible price paths for the underlying asset.

- **Key Features**:
  - Handles complex and path-dependent options.
  - Provides probabilistic distribution of option prices.
  - Useful for pricing options with multiple sources of uncertainty.

### 3. Binomial Model

The **Binomial Model** is a discrete-time model for the varying price over time of financial instruments, primarily used for pricing options.

- **Key Features**:
  - Builds a price tree to evaluate option prices at different nodes.
  - Flexible and can model American options.
  - Easier to implement for options with early exercise features.

## 📸 Screenshots

<!-- Add your screenshots below by replacing the placeholders with your image links -->

### 1. Black-Scholes Model

![Black-Scholes Model](path/to/black-scholes-screenshot.png)

### 2. Monte Carlo Simulation

![Monte Carlo Simulation](path/to/monte-carlo-screenshot.png)

### 3. Binomial Model

![Binomial Model](path/to/binomial-model-screenshot.png)

## 🛠️ Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have the following installed:

- **Python 3.7 or higher**
- **Git**

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/options-pricing-dashboard.git
   cd options-pricing-dashboard
   ```

2. **Create a Virtual Environment**

   It's good practice to use a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**

   - **Windows**:

     ```bash
     venv\Scripts\activate
     ```

   - **macOS/Linux**:

     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided, you can install the necessary packages manually:*

   ```bash
   pip install streamlit numpy pandas scipy matplotlib seaborn plotly
   ```

### Running the Project

1. **Navigate to the Project Directory**

   Ensure you're in the project's root directory.

2. **Run the Streamlit App**

   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the Dashboard**

   After running the above command, Streamlit will provide a local URL (e.g., `http://localhost:8501`). Open this URL in your web browser to interact with the dashboard.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add some feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open a Pull Request**

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

*Feel free to reach out with any questions or feedback!*


