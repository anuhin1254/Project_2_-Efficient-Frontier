#!/usr/bin/env python
# coding: utf-8

# Project #2
# Katerina Uruci
# Abdullah Nuhin

# In[91]:


import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class Asset:
    def __init__(self, ticker, start_date=None):
        self.ticker = ticker
        self.start_date = start_date if start_date else (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
        self.data = None

    def download_data(self):
        """
        Downloads historical open and close data for the asset using yfinance.
        """
        self.data = yf.download(self.ticker, start=self.start_date)

    def get_open_close_data(self):
        """
        Returns the open and close price data for the asset.
        """
        return self.data[['Open', 'Close']]

def get_multiple_assets_data(tickers, start_date=None):
    """
    It fetches the opening and closing prices for selected stocks over the past two years.
    """
    assets = [Asset(ticker, start_date) for ticker in tickers]
    for asset in assets:
        asset.download_data()
    return {asset.ticker: asset.get_open_close_data() for asset in assets}

# Example usage - automatically sets start date to 2 years ago if not specified
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B']
asset_data = get_multiple_assets_data(tickers)

# Convert the dictionary to a multi-level DataFrame for easier manipulation
# It creates a DataFrame with a MultiIndex (ticker, date) and columns for Open and Close prices
prices_df = pd.concat(asset_data, axis=1)


# In[63]:


import numpy as np
import matplotlib.pyplot as plt

# Function to Calculate Portfolio Statistics
def calculate_portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Function to Generate Multiple Portfolios
def generate_portfolios(mean_returns, covariance_matrix, num_portfolios=10000):
    num_assets = len(mean_returns)
    results_matrix = np.zeros((4, num_portfolios))  # 3 metrics + index for weights
    all_weights = np.zeros((num_portfolios, num_assets))  # Store all portfolio weights
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Normalize weights
        all_weights[i, :] = weights  # Save weights
        portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_statistics(weights, mean_returns, covariance_matrix)
        results_matrix[0, i] = portfolio_return
        results_matrix[1, i] = portfolio_volatility
        results_matrix[2, i] = sharpe_ratio
        results_matrix[3, i] = i  # Index placeholder for identification
    
    return results_matrix, all_weights

# Example Visualization Function
def plot_portfolios(results):
    plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o', alpha=0.5, s=20)
    plt.title('Efficient Frontier')
    plt.xlabel('Portfolio Volatility (Standard Deviation)')
    plt.ylabel('Portfolio Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

# Placeholder for Mean Returns and Covariance Matrix
# Assume mean_returns and covariance_matrix are already calculated from your assets' daily returns
mean_returns = np.array([0.1, 0.12, 0.14])  # Example data
covariance_matrix = np.array([[0.01, 0.0012, 0.0014], [0.0012, 0.02, 0.0024], [0.0014, 0.0024, 0.03]])  # Example data

# Generate Portfolios
results, all_weights = generate_portfolios(mean_returns, covariance_matrix)

# Plot the generated portfolios to visualize the Efficient Frontier
plot_portfolios(results)


# In[71]:


def efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray, num_portfolios: int = 10000, risk_free_rate: float = 0.03) -> np.ndarray:
    """
    It creates a bunch of smart investment mixes using average profits, how these profits move together,
    and how many mixes you want to look at.
    Args:
        mean_returns (numpy.ndarray): Mean daily returns of the assets.
        cov_matrix (numpy.ndarray): Covariance matrix of the assets' daily returns.
        num_portfolios (int): Number of portfolios to generate. Default: 10000
        risk_free_rate (float): Risk-free rate. Default: 0.03

    Returns:
        numpy.ndarray: A 2D array with columns representing portfolio return, volatility, Sharpe ratio,
        and the index of the asset with the highest weight.
    """
    num_assets = len(mean_returns)
    results = np.zeros((4, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio
        results[3, i] = weights.argmax()

    return results


# In[68]:


def efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.03):
    num_assets = len(mean_returns)
    results = np.zeros((4, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = sharpe_ratio
        results[3,i] = weights.argmax()
    
    return results


# In[65]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# Assuming mean_returns and covariance_matrix are already calculated
# Example placeholders for mean_returns and covariance_matrix
mean_returns = np.array([0.12, 0.18, 0.14])  # Example mean returns
covariance_matrix = np.array([[0.1, 0.01, 0.02], [0.01, 0.1, 0.03], [0.02, 0.03, 0.15]])  # Example covariance matrix

def objective_function(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio  # Minimize the negative Sharpe Ratio

def optimize_portfolio(mean_returns, cov_matrix, num_assets, risk_free_rate=0.03):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)  # Weights must sum to 1.
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]  # Start with equal weights
    
    opt_result = minimize(objective_function, initial_guess, args=(mean_returns, cov_matrix, risk_free_rate),
                          method='SLSQP', bounds=bounds, constraints=constraints)
    
    return opt_result

num_assets = len(mean_returns)
opt_result = optimize_portfolio(mean_returns, covariance_matrix, num_assets)

optimal_weights = opt_result.x
expected_return = np.dot(optimal_weights, mean_returns)
expected_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
sharpe_ratio = (expected_return - 0.03) / expected_volatility

print(f"Optimal weights: {optimal_weights}")
print(f"Expected return: {expected_return}")
print(f"Expected volatility: {expected_volatility}")
print(f"Sharpe Ratio: {sharpe_ratio}")

# Visualizing the Efficient Frontier would typically involve plotting multiple optimized portfolios
# For simplicity, here we just plot the single optimized point

plt.scatter(expected_volatility, expected_return, c='red', s=50, marker='*', label='Optimized Portfolio')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.legend()
plt.show()

We chose to use JupyterHub for coding this project because it made the process easier. JupyterHub provides an accessible, web-based interface for writing and running code, which is especially helpful for interactive data analysisand visualization tasks.We crafted our code with help from several platforms, using GitHub
for version control, GeeksforGeeks, and Reddit for troubleshooting and advice. Sites like
Stack Overflow, Investopedia, and Kaggle, alongside official documentation for NumPy and 
pandas, as well as the Matplotlib Gallery and Medium's Towards Data Science, provided 
essential coding support, financial insights, and data visualization inspiration. 
Together, these resources equipped us with the comprehensive knowledge necessary for
our project.

# In[ ]:





# In[ ]:




