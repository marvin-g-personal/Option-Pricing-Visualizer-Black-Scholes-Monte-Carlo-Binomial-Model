# 💰 Options Pricing Dashboard

An interactive platform designed to help you visualize and analyze option prices using various financial models. Whether you're a trader, student, or finance enthusiast, this dashboard offers tools to understand how different factors influence option valuations. 

This project was presented to over a 100 students the DATA Club's semester-long project contest, 
getting up to 200 users it's first week. It has been deployed and useable at the following webpage:

[[Streamlit App](https://options-pricing-visualizer.streamlit.app/)]

## 🚀 Features
1. **Interactive Visualization**:
   - Real-time option price calculations
   - Interactive heatmaps for Call and Put options
   - Adjustable parameters (Spot Price, Volatility, Strike Price, etc.)

2. **Three Pricing Models**:
   - **Black-Scholes**: Theoretical pricing for European options w/ PnL Heatmaps
   - **Monte Carlo**: Price simulation through multiple random paths
   - **Binomial Model**: Discrete-time price tree analysis

# 🔍 Running Locally

1. Clone the repository
```bash
git clone https://github.com/yourusername/options-pricing-dashboard.git
cd options-pricing-dashboard
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run Home.py
```

## 🛠️ Prerequisites
- Python 3.7 or higher
- Required packages: streamlit, numpy, pandas, scipy, matplotlib, seaborn, plotly

## 🤝 Contributing
Contributions welcome! Fork the repository and submit a pull request.

---
Created by Marvin Gandhi, Bennett Franciosi, Elaine Zou, Vaibhav Singh & Tafari Darosa-Levy

*Feel free to reach out with any questions or feedback!*

## 📄 License
MIT License

Copyright (c) 2024 Marvin Gandhi, Bennett Franciosi, Elaine Zou, Vaibhav Singh & Tafari Darosa-Levy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


