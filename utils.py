#!/usr/bin/env python3
"""
MarketMate Utilities
Helper functions and utilities for portfolio analysis

Author: MarketMate Team
Version: 2.0
Date: 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import logging
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_portfolio_weights(weights: List[float], tolerance: float = 1e-6) -> Tuple[bool, str]:
        """Validate portfolio weights sum to 1.0"""
        # Check for negative weights first
        if any(w < 0 for w in weights):
            return False, "All weights must be non-negative"
        
        # Then check if they sum to 1.0
        if abs(sum(weights) - 1.0) > tolerance:
            return False, f"Weights sum to {sum(weights):.3f}, must sum to 1.0"
        
        return True, "Valid weights"
    
    @staticmethod
    def validate_stock_symbols(symbols: List[str]) -> Tuple[List[str], List[str]]:
        """Validate stock symbols exist and return valid/invalid lists"""
        valid_symbols = []
        invalid_symbols = []
        
        for symbol in symbols:
            try:
                # Quick check by fetching 1 day of data
                test_data = yf.download(symbol, period="1d", progress=False)
                if not test_data.empty:
                    valid_symbols.append(symbol.upper())
                else:
                    invalid_symbols.append(symbol)
            except:
                invalid_symbols.append(symbol)
        
        return valid_symbols, invalid_symbols

class PerformanceCalculator:
    """Performance metrics calculation utilities"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility != 0 else 0
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        return excess_returns / downside_deviation if downside_deviation != 0 else 0
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> Tuple[float, datetime, datetime]:
        """Calculate maximum drawdown and its start/end dates"""
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_drawdown = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find the peak before max drawdown
        peak_date = running_max[running_max.index <= max_dd_date].idxmax()
        
        return max_drawdown, peak_date, max_dd_date
    
    @staticmethod
    def calculate_var_cvar(returns: pd.Series, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional VaR"""
        var = np.percentile(returns, (1 - confidence_level) * 100)
        cvar = returns[returns <= var].mean()
        return var, cvar
    
    @staticmethod
    def calculate_beta(portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate portfolio beta relative to market"""
        # Align the series by date
        aligned_data = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
        if aligned_data.shape[1] != 2 or len(aligned_data) < 30:
            return np.nan
        
        portfolio_aligned = aligned_data.iloc[:, 0]
        market_aligned = aligned_data.iloc[:, 1]
        
        covariance = np.cov(portfolio_aligned, market_aligned)[0, 1]
        market_variance = np.var(market_aligned)
        
        return covariance / market_variance if market_variance != 0 else np.nan

class DataFetcher:
    """Data fetching utilities with caching and error handling"""
    
    @staticmethod
    def fetch_stock_data(symbols: List[str], period: str = "1y", interval: str = "1d", 
                        max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch stock data with retry logic"""
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    symbols, 
                    period=period, 
                    interval=interval, 
                    progress=False
                )["Adj Close"]
                
                if data.empty:
                    continue
                
                # Handle single stock case
                if len(symbols) == 1 and isinstance(data, pd.Series):
                    data = data.to_frame()
                    data.columns = symbols
                
                # Forward fill missing values
                data = data.fillna(method='ffill')
                
                # Drop rows with all NaN values
                data = data.dropna(how='all')
                
                return data
                
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logging.error(f"Failed to fetch data after {max_retries} attempts")
                    return None
        
        return None
    
    @staticmethod
    def fetch_market_data(symbol: str = "SPY", period: str = "1y") -> Optional[pd.Series]:
        """Fetch market benchmark data"""
        try:
            data = yf.download(symbol, period=period, progress=False)["Adj Close"]
            return data.fillna(method='ffill') if not data.empty else None
        except:
            return None

class ReportGenerator:
    """Report generation utilities"""
    
    @staticmethod
    def generate_summary_stats(returns: pd.Series, portfolio_name: str = "Portfolio") -> Dict:
        """Generate comprehensive summary statistics"""
        return {
            "portfolio_name": portfolio_name,
            "total_days": len(returns),
            "start_date": returns.index[0].strftime('%Y-%m-%d'),
            "end_date": returns.index[-1].strftime('%Y-%m-%d'),
            "mean_daily_return": returns.mean(),
            "std_daily_return": returns.std(),
            "annual_return": returns.mean() * 252,
            "annual_volatility": returns.std() * np.sqrt(252),
            "positive_days": (returns > 0).sum(),
            "negative_days": (returns < 0).sum(),
            "zero_days": (returns == 0).sum(),
            "best_day": returns.max(),
            "worst_day": returns.min(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis()
        }
    
    @staticmethod
    def generate_correlation_analysis(returns: pd.DataFrame) -> Dict:
        """Generate correlation analysis"""
        correlation_matrix = returns.corr()
        
        # Find pairs with highest/lowest correlations
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        corr_pairs = correlation_matrix.where(mask).stack().sort_values(ascending=False)
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "highest_correlations": dict(corr_pairs.head(5)),
            "lowest_correlations": dict(corr_pairs.tail(5)),
            "average_correlation": correlation_matrix.mean().mean()
        }

class ConfigManager:
    """Configuration management utilities"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {config_path} not found, using defaults")
            return {}
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in config file {config_path}")
            return {}
    
    @staticmethod
    def save_config(config: Dict, config_path: str) -> bool:
        """Save configuration to JSON file"""
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
            return False
    
    @staticmethod
    def get_default_config() -> Dict:
        """Get default configuration"""
        return {
            "stocks": ["AAPL", "MSFT", "GOOGL"],
            "weights": [0.4, 0.3, 0.3],
            "period": "1y",
            "interval": "1d",
            "risk_free_rate": 0.02,
            "confidence_level": 0.95,
            "moving_average_windows": [20, 50, 200],
            "volatility_window": 30
        }

class ExportManager:
    """Data export utilities"""
    
    @staticmethod
    def export_to_excel(data_dict: Dict[str, pd.DataFrame], filename: str) -> bool:
        """Export multiple DataFrames to Excel sheets"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                for sheet_name, df in data_dict.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=True)
            return True
        except Exception as e:
            logging.error(f"Failed to export to Excel: {e}")
            return False
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame, filename: str) -> bool:
        """Export DataFrame to CSV"""
        try:
            df.to_csv(filename, index=True)
            return True
        except Exception as e:
            logging.error(f"Failed to export to CSV: {e}")
            return False

# Portfolio presets
PORTFOLIO_PRESETS = {
    "Conservative": {
        "stocks": ["JNJ", "PG", "KO", "PFE", "VZ"],
        "weights": [0.25, 0.20, 0.20, 0.20, 0.15],
        "description": "Large-cap dividend stocks with low volatility"
    },
    "Growth": {
        "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        "weights": [0.25, 0.20, 0.20, 0.20, 0.15],
        "description": "High-growth technology stocks"
    },
    "Value": {
        "stocks": ["BRK-B", "JPM", "V", "UNH", "HD"],
        "weights": [0.25, 0.20, 0.20, 0.20, 0.15],
        "description": "Undervalued large-cap stocks"
    },
    "Diversified": {
        "stocks": ["SPY", "QQQ", "GLD", "TLT", "VTI"],
        "weights": [0.30, 0.25, 0.15, 0.15, 0.15],
        "description": "Mixed asset allocation including ETFs"
    },
    "International": {
        "stocks": ["VEA", "VWO", "EFA", "IEMG", "VXUS"],
        "weights": [0.25, 0.20, 0.20, 0.20, 0.15],
        "description": "International market exposure"
    }
}

# Risk level mappings
RISK_LEVELS = {
    "Conservative": {"max_volatility": 0.15, "min_sharpe": 0.5},
    "Moderate": {"max_volatility": 0.20, "min_sharpe": 0.8},
    "Aggressive": {"max_volatility": 0.30, "min_sharpe": 1.0}
}
