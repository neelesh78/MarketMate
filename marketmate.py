#!/usr/bin/env python3
"""
MarketMate - Personal Portfolio Tracker
A comprehensive tool for analyzing stock portfolios with advanced metrics and visualization.

Author: MarketMate Team
Version: 2.0
Date: 2025
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class MarketMate:
    """
    Advanced Portfolio Analysis Tool
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize MarketMate with configuration
        """
        # Default configuration
        self.config = {
            "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "weights": [0.25, 0.20, 0.20, 0.20, 0.15],
            "period": "1y",
            "interval": "1d",
            "risk_free_rate": 0.02,  # 2% annual risk-free rate
            "confidence_level": 0.95,
            "moving_average_windows": [20, 50, 200],
            "volatility_window": 30
        }
        
        # Load custom config if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
            
        # Ensure data directories exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        # Initialize data containers
        self.raw_data = None
        self.daily_returns = None
        self.portfolio_returns = None
        self.metrics = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_config(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
            logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    def validate_inputs(self) -> bool:
        """Validate portfolio configuration"""
        stocks = self.config["stocks"]
        weights = self.config["weights"]
        
        if len(stocks) != len(weights):
            logger.error("Number of stocks must match number of weights")
            return False
            
        if abs(sum(weights) - 1.0) > 1e-6:
            logger.error(f"Portfolio weights must sum to 1.0, got {sum(weights)}")
            return False
            
        if any(w < 0 for w in weights):
            logger.error("Portfolio weights must be non-negative")
            return False
            
        return True
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch stock data with error handling and retry logic"""
        logger.info("Fetching market data...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Fetch raw data
                raw_data = yf.download(
                    self.config["stocks"], 
                    period=self.config["period"], 
                    interval=self.config["interval"],
                    progress=False
                )
                
                if raw_data.empty:
                    raise ValueError("No data retrieved")
                
                # Handle different data structures
                if isinstance(raw_data.columns, pd.MultiIndex):
                    # Multi-stock data with MultiIndex columns
                    if "Adj Close" in raw_data.columns.levels[0]:
                        data = raw_data["Adj Close"]
                    elif "Close" in raw_data.columns.levels[0]:
                        data = raw_data["Close"]
                    else:
                        raise ValueError("No price data found")
                else:
                    # Single stock data or already processed data
                    if "Adj Close" in raw_data.columns:
                        data = raw_data["Adj Close"].to_frame()
                        data.columns = self.config["stocks"]
                    elif "Close" in raw_data.columns:
                        data = raw_data["Close"].to_frame()
                        data.columns = self.config["stocks"]
                    else:
                        # Assume it's already price data
                        data = raw_data
                
                # Handle single stock case
                if len(self.config["stocks"]) == 1 and isinstance(data, pd.Series):
                    data = data.to_frame()
                    data.columns = self.config["stocks"]
                
                # Check for missing data
                missing_data = data.isnull().sum()
                if missing_data.any():
                    logger.warning(f"Missing data detected: {missing_data[missing_data > 0].to_dict()}")
                
                # Forward fill missing values
                data = data.fillna(method='ffill')
                
                # Drop any remaining NaN rows
                data = data.dropna()
                
                if data.empty:
                    raise ValueError("No valid data after cleaning")
                
                self.raw_data = data
                logger.info(f"Successfully fetched data for {len(self.config['stocks'])} stocks")
                logger.info(f"Data range: {data.index[0]} to {data.index[-1]}")
                logger.info(f"Data shape: {data.shape}")
                
                return data
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("Max retries exceeded. Data fetch failed.")
                    raise
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive portfolio and stock metrics"""
        logger.info("Calculating financial metrics...")
        
        if self.raw_data is None:
            raise ValueError("No data available. Please fetch data first.")
        
        data = self.raw_data
        weights = np.array(self.config["weights"])
        
        # Basic calculations
        self.daily_returns = data.pct_change().dropna()
        self.portfolio_returns = (self.daily_returns * weights).sum(axis=1)
        
        # Moving averages
        moving_averages = {}
        for window in self.config["moving_average_windows"]:
            moving_averages[f"MA_{window}"] = data.rolling(window=window).mean()
        
        # Volatility (rolling standard deviation)
        volatility = self.daily_returns.rolling(
            window=self.config["volatility_window"]
        ).std() * np.sqrt(252)  # Annualized
        
        # Portfolio metrics
        portfolio_volatility = self.portfolio_returns.std() * np.sqrt(252)
        portfolio_return = self.portfolio_returns.mean() * 252
        
        # Sharpe Ratio
        excess_return = portfolio_return - self.config["risk_free_rate"]
        sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Sortino Ratio
        downside_returns = self.portfolio_returns[self.portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (VaR)
        confidence_level = self.config["confidence_level"]
        var_95 = np.percentile(self.portfolio_returns, (1 - confidence_level) * 100)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = self.portfolio_returns[self.portfolio_returns <= var_95].mean()
        
        # Beta calculation (relative to market - using SPY as proxy)
        try:
            market_raw = yf.download("SPY", period=self.config["period"], progress=False)
            
            # Handle different data structures for market data
            if isinstance(market_raw.columns, pd.MultiIndex):
                if "Adj Close" in market_raw.columns.levels[0]:
                    market_data = market_raw["Adj Close"]["SPY"]
                elif "Close" in market_raw.columns.levels[0]:
                    market_data = market_raw["Close"]["SPY"]
                else:
                    raise ValueError("No market price data found")
            else:
                if "Adj Close" in market_raw.columns:
                    market_data = market_raw["Adj Close"]
                elif "Close" in market_raw.columns:
                    market_data = market_raw["Close"]
                else:
                    market_data = market_raw.iloc[:, 0]  # Take first column
            
            market_returns = market_data.pct_change().dropna()
            
            # Align dates
            common_dates = self.portfolio_returns.index.intersection(market_returns.index)
            if len(common_dates) > 30:  # Need at least 30 data points
                portfolio_aligned = self.portfolio_returns.loc[common_dates]
                market_aligned = market_returns.loc[common_dates]
                
                beta = np.cov(portfolio_aligned, market_aligned)[0][1] / np.var(market_aligned)
            else:
                logger.warning("Insufficient overlapping data for beta calculation")
                beta = np.nan
        except Exception as e:
            logger.warning(f"Could not calculate beta: {e}")
            beta = np.nan
        
        # Information Ratio (if benchmark available)
        try:
            active_return = portfolio_return - (market_returns.mean() * 252)
            tracking_error = (self.portfolio_returns - market_returns.reindex(self.portfolio_returns.index)).std() * np.sqrt(252)
            information_ratio = active_return / tracking_error if tracking_error > 0 else 0
        except:
            information_ratio = np.nan
        
        # Store all metrics
        self.metrics = {
            "portfolio": {
                "annual_return": portfolio_return,
                "annual_volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "beta": beta,
                "information_ratio": information_ratio,
                "total_return": (cumulative_returns.iloc[-1] - 1),
                "num_positive_days": (self.portfolio_returns > 0).sum(),
                "num_negative_days": (self.portfolio_returns < 0).sum(),
                "best_day": self.portfolio_returns.max(),
                "worst_day": self.portfolio_returns.min()
            },
            "individual_stocks": {
                "annual_returns": (self.daily_returns.mean() * 252).to_dict(),
                "annual_volatility": (self.daily_returns.std() * np.sqrt(252)).to_dict(),
                "sharpe_ratios": ((self.daily_returns.mean() * 252 - self.config["risk_free_rate"]) / 
                                 (self.daily_returns.std() * np.sqrt(252))).to_dict()
            },
            "moving_averages": moving_averages,
            "volatility": volatility,
            "correlation_matrix": self.daily_returns.corr().to_dict()
        }
        
        logger.info("Metrics calculation completed")
        return self.metrics
    
    def save_data(self):
        """Save all data to CSV files"""
        logger.info("Saving data to files...")
        
        # Raw data
        self.raw_data.to_csv("data/raw_prices.csv")
        
        # Returns
        self.daily_returns.to_csv("data/daily_returns.csv")
        
        # Portfolio returns
        pd.Series(self.portfolio_returns, name="Portfolio_Returns").to_csv("data/portfolio_returns.csv")
        
        # Moving averages
        for ma_name, ma_data in self.metrics["moving_averages"].items():
            ma_data.to_csv(f"data/{ma_name.lower()}.csv")
        
        # Volatility
        self.metrics["volatility"].to_csv("data/volatility.csv")
        
        # Portfolio summary
        portfolio_summary = pd.DataFrame([self.metrics["portfolio"]])
        portfolio_summary.to_csv("results/portfolio_summary.csv", index=False)
        
        # Individual stock metrics
        stock_summary = pd.DataFrame(self.metrics["individual_stocks"])
        stock_summary.to_csv("results/individual_stocks_summary.csv")
        
        # Correlation matrix
        correlation_df = pd.DataFrame(self.metrics["correlation_matrix"])
        correlation_df.to_csv("results/correlation_matrix.csv")
        
        logger.info("Data saved successfully")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Stock Price History with Moving Averages
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Price chart
        ax1 = axes[0, 0]
        for stock in self.config["stocks"]:
            ax1.plot(self.raw_data.index, self.raw_data[stock], label=stock, linewidth=2)
        ax1.set_title("Stock Price History", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Portfolio cumulative returns
        ax2 = axes[0, 1]
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        ax2.plot(cumulative_returns.index, cumulative_returns.values, 
                label='Portfolio', linewidth=3, color='darkblue')
        ax2.set_title("Portfolio Cumulative Returns", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Cumulative Return")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Rolling volatility
        ax3 = axes[1, 0]
        portfolio_vol_rolling = self.portfolio_returns.rolling(window=30).std() * np.sqrt(252)
        ax3.plot(portfolio_vol_rolling.index, portfolio_vol_rolling.values, 
                color='red', linewidth=2)
        ax3.set_title("Portfolio Rolling Volatility (30-day)", fontsize=14, fontweight='bold')
        ax3.set_ylabel("Annualized Volatility")
        ax3.grid(True, alpha=0.3)
        
        # Drawdown chart
        ax4 = axes[1, 1]
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        ax4.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax4.plot(drawdown.index, drawdown.values, color='darkred', linewidth=2)
        ax4.set_title("Portfolio Drawdown", fontsize=14, fontweight='bold')
        ax4.set_ylabel("Drawdown")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("results/portfolio_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation Heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.daily_returns.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title("Stock Correlation Matrix", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig("results/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Risk-Return Scatter Plot
        plt.figure(figsize=(12, 8))
        annual_returns = self.daily_returns.mean() * 252
        annual_volatility = self.daily_returns.std() * np.sqrt(252)
        
        # Individual stocks
        for i, stock in enumerate(self.config["stocks"]):
            plt.scatter(annual_volatility[stock], annual_returns[stock], 
                       s=self.config["weights"][i]*1000, alpha=0.7, label=stock)
        
        # Portfolio
        plt.scatter(self.metrics["portfolio"]["annual_volatility"], 
                   self.metrics["portfolio"]["annual_return"], 
                   s=200, color='red', marker='*', label='Portfolio', edgecolor='black')
        
        plt.xlabel("Annual Volatility")
        plt.ylabel("Annual Return")
        plt.title("Risk-Return Profile", fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/risk_return_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Returns Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(self.portfolio_returns, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        ax1.axvline(self.portfolio_returns.mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.portfolio_returns.mean():.4f}')
        ax1.axvline(self.metrics["portfolio"]["var_95"], color='orange', linestyle='--', 
                   label=f'VaR 95%: {self.metrics["portfolio"]["var_95"]:.4f}')
        ax1.set_title("Portfolio Daily Returns Distribution")
        ax1.set_xlabel("Daily Return")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(self.portfolio_returns, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normal Distribution)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("results/returns_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations created successfully")
    
    def generate_report(self) -> str:
        """Generate a comprehensive text report"""
        report = []
        report.append("="*60)
        report.append("MARKETMATE PORTFOLIO ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Period: {self.config['period']}")
        report.append("")
        
        # Portfolio Overview
        report.append("PORTFOLIO OVERVIEW")
        report.append("-" * 30)
        report.append(f"Stocks: {', '.join(self.config['stocks'])}")
        report.append(f"Weights: {[f'{w:.1%}' for w in self.config['weights']]}")
        report.append("")
        
        # Performance Metrics
        portfolio = self.metrics["portfolio"]
        report.append("PERFORMANCE METRICS")
        report.append("-" * 30)
        report.append(f"Annual Return: {portfolio['annual_return']:.2%}")
        report.append(f"Annual Volatility: {portfolio['annual_volatility']:.2%}")
        report.append(f"Total Return: {portfolio['total_return']:.2%}")
        report.append(f"Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")
        report.append(f"Sortino Ratio: {portfolio['sortino_ratio']:.3f}")
        report.append(f"Maximum Drawdown: {portfolio['max_drawdown']:.2%}")
        if not np.isnan(portfolio['beta']):
            report.append(f"Beta (vs SPY): {portfolio['beta']:.3f}")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-" * 30)
        report.append(f"Value at Risk (95%): {portfolio['var_95']:.2%}")
        report.append(f"Conditional VaR (95%): {portfolio['cvar_95']:.2%}")
        report.append(f"Best Day: {portfolio['best_day']:.2%}")
        report.append(f"Worst Day: {portfolio['worst_day']:.2%}")
        report.append(f"Positive Days: {portfolio['num_positive_days']}")
        report.append(f"Negative Days: {portfolio['num_negative_days']}")
        report.append("")
        
        # Individual Stock Performance
        report.append("INDIVIDUAL STOCK PERFORMANCE")
        report.append("-" * 40)
        stocks = self.metrics["individual_stocks"]
        for stock in self.config["stocks"]:
            report.append(f"{stock}:")
            report.append(f"  Annual Return: {stocks['annual_returns'][stock]:.2%}")
            report.append(f"  Annual Volatility: {stocks['annual_volatility'][stock]:.2%}")
            report.append(f"  Sharpe Ratio: {stocks['sharpe_ratios'][stock]:.3f}")
        
        report_text = "\n".join(report)
        
        # Save report
        with open("results/portfolio_report.txt", "w") as f:
            f.write(report_text)
        
        return report_text
    
    def run_analysis(self) -> Dict:
        """Run complete portfolio analysis"""
        logger.info("Starting MarketMate Analysis...")
        
        try:
            # Validate inputs
            if not self.validate_inputs():
                raise ValueError("Invalid portfolio configuration")
            
            # Fetch data
            self.fetch_data()
            
            # Calculate metrics
            self.calculate_metrics()
            
            # Save data
            self.save_data()
            
            # Create visualizations
            self.create_visualizations()
            
            # Generate report
            report = self.generate_report()
            
            logger.info("Analysis completed successfully!")
            logger.info("Results saved to 'data/' and 'results/' directories")
            
            return {
                "success": True,
                "metrics": self.metrics,
                "report": report
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    # Initialize MarketMate
    analyzer = MarketMate()
    
    print("🚀 Welcome to MarketMate - Personal Portfolio Tracker")
    print("=" * 55)
    
    # Run analysis
    results = analyzer.run_analysis()
    
    if results["success"]:
        print("\n✅ Analysis completed successfully!")
        print("\n📊 Key Portfolio Metrics:")
        portfolio = results["metrics"]["portfolio"]
        print(f"   • Annual Return: {portfolio['annual_return']:.2%}")
        print(f"   • Annual Volatility: {portfolio['annual_volatility']:.2%}")
        print(f"   • Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")
        print(f"   • Max Drawdown: {portfolio['max_drawdown']:.2%}")
        
        print("\n📁 Files Generated:")
        print("   • Data files saved to 'data/' directory")
        print("   • Charts saved to 'results/' directory")
        print("   • Report saved as 'results/portfolio_report.txt'")
        
        print("\n🎯 Next Steps:")
        print("   • Review the generated report and charts")
        print("   • Run 'streamlit run marketmate_ui.py' for interactive dashboard")
        print("   • Customize config in 'config.json' for different portfolios")
        
    else:
        print(f"\n❌ Analysis failed: {results['error']}")
        print("   Please check your configuration and try again.")
