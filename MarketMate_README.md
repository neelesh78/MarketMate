
# 🚀 MarketMate - Personal Portfolio Tracker

**MarketMate** is a comprehensive Python-based portfolio analysis tool that provides professional-grade financial metrics, interactive visualizations, and automated reporting for your investment portfolio.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)

---

## 🎯 Project Overview

MarketMate empowers investors to make data-driven decisions by providing:
- **Advanced portfolio analytics** with 20+ financial metrics
- **Interactive web dashboard** built with Streamlit
- **Professional visualizations** with correlation analysis and risk profiling
- **Automated data fetching** from Yahoo Finance
- **Comprehensive reporting** with export capabilities

---

## ✨ Key Features

### 📊 Advanced Analytics
- **Portfolio Performance**: Annual returns, volatility, total return
- **Risk Metrics**: Sharpe ratio, Sortino ratio, VaR, CVaR, maximum drawdown
- **Market Analysis**: Beta calculation, correlation matrix, rolling volatility
- **Individual Stock Analysis**: Per-stock metrics and contribution analysis

### 📈 Professional Visualizations
- Stock price charts with moving averages
- Portfolio cumulative returns vs benchmarks
- Risk-return scatter plots
- Correlation heatmaps
- Drawdown analysis charts
- Returns distribution analysis

### 🖥️ Interactive Dashboard
- Real-time portfolio analysis
- Customizable stock selection and weights
- Multiple portfolio presets (Growth, Conservative, etc.)
- Interactive charts with Plotly
- Data export capabilities (Excel, CSV)

### 🔧 Advanced Features
- Multiple portfolio presets
- Configurable analysis parameters
- Automated report generation
- Comprehensive test suite
- Error handling and data validation
- Benchmarking against market indices

---

## 🛠 Technology Stack

- **Python 3.8+** - Core language
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **yfinance** - Market data fetching
- **matplotlib & seaborn** - Static visualizations
- **plotly** - Interactive charts
- **streamlit** - Web dashboard framework
- **scipy** - Statistical analysis

---

## 📁 Project Structure

```
MarketMate/
├── 📄 marketmate.py           # Main analysis engine
├── 🌐 marketmate_ui.py        # Streamlit web dashboard  
├── 🔧 utils.py                # Utility functions and helpers
├── ⚙️ config.json             # Configuration file
├── 🧪 test_marketmate.py      # Comprehensive test suite
├── 🚀 setup.py                # Installation and setup script
├── 📋 requirements.txt        # Python dependencies
├── 📖 MarketMate_README.md     # This documentation
├── 📊 data/                   # Generated data files
│   ├── raw_prices.csv
│   ├── daily_returns.csv
│   ├── portfolio_returns.csv
│   ├── ma_20.csv, ma_50.csv, ma_200.csv
│   └── volatility.csv
├── 📈 results/                # Generated analysis results
│   ├── portfolio_overview.png
│   ├── correlation_heatmap.png
│   ├── risk_return_scatter.png
│   ├── returns_distribution.png
│   ├── portfolio_summary.csv
│   ├── individual_stocks_summary.csv
│   ├── correlation_matrix.csv
│   └── portfolio_report.txt
└── 📁 exports/                # Additional export files
```

---

## ⚡ Quick Start Guide

### 1. 🔽 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MarketMate.git
cd MarketMate

# Run the automated setup (recommended)
python setup.py
```

Or install manually:
```bash
# Create virtual environment
python -m venv marketmate_env

# Activate virtual environment
# Windows:
marketmate_env\Scripts\activate
# macOS/Linux:
source marketmate_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. 🚀 Basic Usage

#### Command Line Analysis
```bash
# Run with default portfolio (AAPL, MSFT, GOOGL, AMZN, TSLA)
python marketmate.py

# Run with custom configuration
python marketmate.py --config custom_config.json
```

#### Interactive Web Dashboard
```bash
# Start the web dashboard
streamlit run marketmate_ui.py

# Open browser to: http://localhost:8501
```

### 3. ⚙️ Configuration

Edit `config.json` to customize your portfolio:

```json
{
  "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
  "weights": [0.25, 0.20, 0.20, 0.20, 0.15],
  "period": "1y",
  "risk_free_rate": 0.02,
  "moving_average_windows": [20, 50, 200]
}
```

---

## 📊 Usage Examples

### Basic Portfolio Analysis

```python
from marketmate import MarketMate

# Initialize with default configuration
analyzer = MarketMate()

# Run complete analysis
results = analyzer.run_analysis()

if results["success"]:
    metrics = results["metrics"]
    print(f"Annual Return: {metrics['portfolio']['annual_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['portfolio']['sharpe_ratio']:.3f}")
```

### Custom Portfolio

```python
# Create custom configuration
custom_config = {
    "stocks": ["AAPL", "GOOGL", "MSFT"],
    "weights": [0.5, 0.3, 0.2],
    "period": "2y",
    "risk_free_rate": 0.025
}

# Save to file
import json
with open('my_portfolio.json', 'w') as f:
    json.dump(custom_config, f)

# Analyze custom portfolio
analyzer = MarketMate('my_portfolio.json')
results = analyzer.run_analysis()
```

---

## 📈 Key Metrics Explained

### Portfolio Performance
- **Annual Return**: Annualized portfolio return
- **Total Return**: Cumulative return over the analysis period
- **Annual Volatility**: Annualized standard deviation of returns

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **VaR (95%)**: Value at Risk at 95% confidence level
- **CVaR (95%)**: Conditional Value at Risk (Expected Shortfall)

### Market Metrics
- **Beta**: Portfolio sensitivity to market movements
- **Correlation Matrix**: Inter-stock correlation analysis
- **Information Ratio**: Active return vs tracking error

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_marketmate.py

# Run specific test categories
python -m unittest test_marketmate.TestDataValidator
python -m unittest test_marketmate.TestPerformanceCalculator
```

---

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM
- **Storage**: 100MB for installation + data storage

### Python Dependencies
- `yfinance>=0.2.18` - Market data fetching
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Plotting
- `streamlit>=1.28.0` - Web dashboard
- `seaborn>=0.11.0` - Statistical plotting
- `plotly>=5.0.0` - Interactive charts
- `scipy>=1.7.0` - Statistical functions

---

## 🎨 Portfolio Presets

MarketMate includes several pre-configured portfolios:

| Preset | Stocks | Strategy |
|--------|--------|----------|
| **Growth** | AAPL, MSFT, GOOGL, AMZN, TSLA | High-growth technology stocks |
| **Conservative** | JNJ, PG, KO, PFE, VZ | Stable dividend-paying stocks |
| **Diversified** | SPY, QQQ, GLD, TLT, VTI | Mixed asset ETFs |
| **Value** | BRK-B, JPM, V, UNH, HD | Undervalued large-cap stocks |
| **International** | VEA, VWO, EFA, IEMG, VXUS | Global market exposure |

---

## 🔧 Advanced Configuration

### Custom Risk Parameters
```json
{
  "risk_free_rate": 0.025,
  "confidence_level": 0.95,
  "volatility_window": 30,
  "benchmark": "SPY"
}
```

### Analysis Periods
- `"1mo"` - 1 month
- `"3mo"` - 3 months  
- `"6mo"` - 6 months
- `"1y"` - 1 year (default)
- `"2y"` - 2 years
- `"5y"` - 5 years
- `"max"` - Maximum available data

---

## 📊 Output Files

### Data Files (`data/` directory)
- **raw_prices.csv**: Historical stock prices
- **daily_returns.csv**: Daily return calculations
- **portfolio_returns.csv**: Portfolio-level returns
- **ma_*.csv**: Moving average calculations
- **volatility.csv**: Rolling volatility metrics

### Results (`results/` directory)
- **portfolio_overview.png**: 4-panel overview chart
- **correlation_heatmap.png**: Stock correlation matrix
- **risk_return_scatter.png**: Risk vs return analysis
- **returns_distribution.png**: Return distribution analysis
- **portfolio_summary.csv**: Key metrics summary
- **portfolio_report.txt**: Comprehensive text report

---

## 🚨 Troubleshooting

### Common Issues

**"No data retrieved" Error**
- Check internet connection
- Verify stock symbols are valid
- Try a different time period

**"Weights must sum to 1.0" Error**
- Ensure portfolio weights add up to exactly 1.0
- Use decimals (0.25, 0.30, 0.45)

**Import Errors**
- Run `python setup.py` to install dependencies
- Ensure Python version is 3.8 or higher

**Streamlit Issues**
- Update Streamlit: `pip install streamlit --upgrade`
- Clear cache: Delete `.streamlit` folder

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Add tests** for new functionality
4. **Run tests**: `python test_marketmate.py`
5. **Submit** a pull request

### Development Setup
```bash
# Clone for development
git clone https://github.com/yourusername/MarketMate.git
cd MarketMate

# Install in development mode
pip install -e .

# Run tests
python test_marketmate.py
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**MarketMate is for educational and informational purposes only. It is not financial advice. Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.**

---

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free market data via yfinance
- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- **The Python community** for excellent financial libraries

---

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/MarketMate/issues)
- **Documentation**: This README and inline code comments
- **Email**: support@marketmate.com

---

**Happy analyzing! 📈🚀**

*Built with ❤️ by Neelesh Sharma*
