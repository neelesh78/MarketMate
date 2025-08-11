#!/usr/bin/env python3
"""
MarketMate Interactive Dashboard
Streamlit web application for portfolio analysis

Author: MarketMate Team
Version: 2.0
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from scipy import stats
import io

warnings.filterwarnings('ignore')

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="MarketMate - Portfolio Tracker", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .success-metric {
        color: #28a745;
        font-weight: bold;
    }
    .warning-metric {
        color: #ffc107;
        font-weight: bold;
    }
    .danger-metric {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HELPER FUNCTIONS
# =========================
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(stocks, period, interval="1d"):
    """Fetch stock data with caching"""
    try:
        data = yf.download(stocks, period=period, interval=interval, progress=False)["Adj Close"]
        if len(stocks) == 1:
            data = data.to_frame()
            data.columns = stocks
        return data.fillna(method='ffill')
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_portfolio_metrics(returns, weights, risk_free_rate=0.02):
    """Calculate comprehensive portfolio metrics"""
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Basic metrics
    annual_return = portfolio_returns.mean() * 252
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    total_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
    
    # Risk-adjusted metrics
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
    
    # Sortino ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # VaR and CVaR
    var_95 = np.percentile(portfolio_returns, 5)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    
    return {
        'portfolio_returns': portfolio_returns,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'best_day': portfolio_returns.max(),
        'worst_day': portfolio_returns.min(),
        'positive_days': (portfolio_returns > 0).sum(),
        'negative_days': (portfolio_returns < 0).sum(),
        'cumulative': cumulative,
        'drawdown': drawdown
    }

# =========================
# SIDEBAR CONFIGURATION
# =========================
st.sidebar.title("🎛️ Portfolio Configuration")

# Portfolio input method
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Manual Entry", "Upload CSV"]
)

if input_method == "Manual Entry":
    # Manual stock entry
    st.sidebar.subheader("📈 Stocks & Weights")
    
    # Popular portfolio presets
    preset = st.sidebar.selectbox(
        "Quick Presets:",
        ["Custom", "FAANG", "Blue Chips", "Tech Giants", "Diversified"]
    )
    
    if preset == "FAANG":
        default_stocks = "META,AAPL,AMZN,NFLX,GOOGL"
        default_weights = "0.2,0.2,0.2,0.2,0.2"
    elif preset == "Blue Chips":
        default_stocks = "AAPL,MSFT,JNJ,PG,KO"
        default_weights = "0.25,0.25,0.2,0.15,0.15"
    elif preset == "Tech Giants":
        default_stocks = "AAPL,MSFT,GOOGL,AMZN,TSLA"
        default_weights = "0.25,0.25,0.2,0.2,0.1"
    elif preset == "Diversified":
        default_stocks = "SPY,QQQ,GLD,TLT,VTI"
        default_weights = "0.3,0.25,0.15,0.15,0.15"
    else:
        default_stocks = "AAPL,MSFT,GOOGL"
        default_weights = "0.4,0.3,0.3"
    
    stocks_input = st.sidebar.text_area(
        "Stock Tickers (one per line or comma-separated):",
        value=default_stocks,
        height=100
    )
    
    weights_input = st.sidebar.text_area(
        "Portfolio Weights:",
        value=default_weights,
        height=60
    )
    
else:
    # CSV upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV with 'Symbol' and 'Weight' columns",
        type=['csv']
    )
    
    if uploaded_file:
        portfolio_df = pd.read_csv(uploaded_file)
        stocks_input = ','.join(portfolio_df['Symbol'].tolist())
        weights_input = ','.join(portfolio_df['Weight'].astype(str).tolist())
    else:
        stocks_input = "AAPL,MSFT,GOOGL"
        weights_input = "0.4,0.3,0.3"

# Parse inputs
try:
    if '\n' in stocks_input:
        STOCKS = [s.strip().upper() for s in stocks_input.split('\n') if s.strip()]
    else:
        STOCKS = [s.strip().upper() for s in stocks_input.split(',') if s.strip()]
    
    if '\n' in weights_input:
        WEIGHTS = [float(w.strip()) for w in weights_input.split('\n') if w.strip()]
    else:
        WEIGHTS = [float(w.strip()) for w in weights_input.split(',') if w.strip()]
except ValueError:
    st.sidebar.error("❌ Invalid input format. Please check your stocks and weights.")
    st.stop()

# Validation
if len(STOCKS) != len(WEIGHTS):
    st.sidebar.error(f"❌ Number of stocks ({len(STOCKS)}) must match number of weights ({len(WEIGHTS)})")
    st.stop()

if abs(sum(WEIGHTS) - 1.0) > 1e-6:
    st.sidebar.error(f"❌ Portfolio weights must sum to 1.0 (currently {sum(WEIGHTS):.3f})")
    st.stop()

if any(w < 0 for w in WEIGHTS):
    st.sidebar.error("❌ Portfolio weights must be non-negative")
    st.stop()

# Analysis parameters
st.sidebar.subheader("📊 Analysis Parameters")
period = st.sidebar.selectbox(
    "Data Period:",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

risk_free_rate = st.sidebar.slider(
    "Risk-free Rate (annual):",
    min_value=0.0,
    max_value=0.10,
    value=0.02,
    step=0.005,
    format="%.3f"
)

benchmark = st.sidebar.selectbox(
    "Benchmark:",
    ["SPY", "QQQ", "VTI", "IWM", "None"]
)

# =========================
# MAIN APPLICATION
# =========================
st.title("📊 MarketMate - Personal Portfolio Tracker")
st.markdown("### Professional Portfolio Analysis Dashboard")

# Display portfolio composition
st.subheader("🎯 Portfolio Composition")
col1, col2 = st.columns(2)

with col1:
    portfolio_df = pd.DataFrame({
        'Stock': STOCKS,
        'Weight': WEIGHTS,
        'Weight (%)': [f"{w:.1%}" for w in WEIGHTS]
    })
    st.dataframe(portfolio_df, use_container_width=True)

with col2:
    # Portfolio composition pie chart
    fig_pie = px.pie(
        values=WEIGHTS,
        names=STOCKS,
        title="Portfolio Allocation",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

# =========================
# DATA FETCHING
# =========================
with st.spinner("🔄 Fetching market data..."):
    data = fetch_stock_data(STOCKS, period)
    
    if data is None or data.empty:
        st.error("❌ Failed to fetch stock data. Please check your stock symbols.")
        st.stop()
    
    # Fetch benchmark data if selected
    benchmark_data = None
    if benchmark != "None":
        benchmark_data = fetch_stock_data([benchmark], period)

st.success(f"✅ Successfully fetched data for {len(STOCKS)} stocks ({data.shape[0]} trading days)")

# =========================
# CALCULATIONS
# =========================
with st.spinner("📈 Calculating metrics..."):
    # Basic calculations
    daily_returns = data.pct_change().dropna()
    
    # Portfolio metrics
    portfolio_metrics = calculate_portfolio_metrics(daily_returns, WEIGHTS, risk_free_rate)
    
    # Individual stock metrics
    stock_metrics = {
        'Annual Return': (daily_returns.mean() * 252).round(4),
        'Annual Volatility': (daily_returns.std() * np.sqrt(252)).round(4),
        'Sharpe Ratio': ((daily_returns.mean() * 252 - risk_free_rate) / (daily_returns.std() * np.sqrt(252))).round(3)
    }

# =========================
# KEY METRICS DISPLAY
# =========================
st.subheader("📊 Key Portfolio Metrics")

# Create metrics columns
metrics_cols = st.columns(6)

with metrics_cols[0]:
    st.metric(
        "Annual Return",
        f"{portfolio_metrics['annual_return']:.2%}",
        delta=None
    )

with metrics_cols[1]:
    st.metric(
        "Volatility",
        f"{portfolio_metrics['annual_volatility']:.2%}",
        delta=None
    )

with metrics_cols[2]:
    sharpe_color = "normal" if portfolio_metrics['sharpe_ratio'] > 1 else "inverse"
    st.metric(
        "Sharpe Ratio",
        f"{portfolio_metrics['sharpe_ratio']:.3f}",
        delta=None
    )

with metrics_cols[3]:
    st.metric(
        "Max Drawdown",
        f"{portfolio_metrics['max_drawdown']:.2%}",
        delta=None
    )

with metrics_cols[4]:
    st.metric(
        "Total Return",
        f"{portfolio_metrics['total_return']:.2%}",
        delta=None
    )

with metrics_cols[5]:
    st.metric(
        "VaR (95%)",
        f"{portfolio_metrics['var_95']:.2%}",
        delta=None
    )

# Additional metrics in expandable section
with st.expander("📈 Advanced Metrics"):
    adv_cols = st.columns(4)
    
    with adv_cols[0]:
        st.metric("Sortino Ratio", f"{portfolio_metrics['sortino_ratio']:.3f}")
        st.metric("CVaR (95%)", f"{portfolio_metrics['cvar_95']:.2%}")
    
    with adv_cols[1]:
        st.metric("Best Day", f"{portfolio_metrics['best_day']:.2%}")
        st.metric("Worst Day", f"{portfolio_metrics['worst_day']:.2%}")
    
    with adv_cols[2]:
        st.metric("Positive Days", f"{portfolio_metrics['positive_days']}")
        st.metric("Negative Days", f"{portfolio_metrics['negative_days']}")
    
    with adv_cols[3]:
        win_rate = portfolio_metrics['positive_days'] / (portfolio_metrics['positive_days'] + portfolio_metrics['negative_days'])
        st.metric("Win Rate", f"{win_rate:.1%}")
        st.metric("Total Days", f"{len(portfolio_metrics['portfolio_returns'])}")

# =========================
# INTERACTIVE CHARTS
# =========================
st.subheader("📈 Interactive Charts")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price History", "Returns", "Risk Analysis", "Correlation", "Performance"])

with tab1:
    st.markdown("#### Stock Price Evolution")
    
    # Price chart with moving averages
    fig_price = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Stock Prices with Moving Averages', 'Trading Volume (if available)'),
        row_heights=[0.7, 0.3]
    )
    
    # Add price lines
    for i, stock in enumerate(STOCKS):
        fig_price.add_trace(
            go.Scatter(
                x=data.index,
                y=data[stock],
                name=stock,
                line=dict(width=2),
                hovertemplate=f"{stock}: $%{{y:.2f}}<br>Date: %{{x}}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Add 20-day moving average
        ma_20 = data[stock].rolling(20).mean()
        fig_price.add_trace(
            go.Scatter(
                x=data.index,
                y=ma_20,
                name=f"{stock} MA20",
                line=dict(dash='dash', width=1),
                opacity=0.7,
                showlegend=False,
                hovertemplate=f"{stock} MA20: $%{{y:.2f}}<br>Date: %{{x}}<extra></extra>"
            ),
            row=1, col=1
        )
    
    fig_price.update_layout(
        height=600,
        title="Stock Price History with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_price, use_container_width=True)

with tab2:
    st.markdown("#### Portfolio Returns Analysis")
    
    returns_cols = st.columns(2)
    
    with returns_cols[0]:
        # Cumulative returns
        fig_cum = go.Figure()
        
        # Portfolio cumulative returns
        fig_cum.add_trace(
            go.Scatter(
                x=portfolio_metrics['cumulative'].index,
                y=(portfolio_metrics['cumulative'] - 1) * 100,
                name='Portfolio',
                line=dict(color='blue', width=3),
                hovertemplate="Portfolio: %{y:.2f}%<br>Date: %{x}<extra></extra>"
            )
        )
        
        # Add benchmark if available
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.pct_change().dropna()
            benchmark_cumulative = (1 + benchmark_returns.iloc[:, 0]).cumprod()
            fig_cum.add_trace(
                go.Scatter(
                    x=benchmark_cumulative.index,
                    y=(benchmark_cumulative - 1) * 100,
                    name=benchmark,
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate=f"{benchmark}: %{{y:.2f}}%<br>Date: %{{x}}<extra></extra>"
                )
            )
        
        fig_cum.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=400
        )
        
        st.plotly_chart(fig_cum, use_container_width=True)
    
    with returns_cols[1]:
        # Returns distribution
        fig_hist = go.Figure()
        
        fig_hist.add_trace(
            go.Histogram(
                x=portfolio_metrics['portfolio_returns'] * 100,
                nbinsx=50,
                name='Daily Returns',
                opacity=0.7,
                hovertemplate="Return: %{x:.2f}%<br>Frequency: %{y}<extra></extra>"
            )
        )
        
        # Add VaR line
        fig_hist.add_vline(
            x=portfolio_metrics['var_95'] * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR 95%: {portfolio_metrics['var_95']:.2%}"
        )
        
        fig_hist.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.markdown("#### Risk Analysis")
    
    risk_cols = st.columns(2)
    
    with risk_cols[0]:
        # Drawdown chart
        fig_dd = go.Figure()
        
        fig_dd.add_trace(
            go.Scatter(
                x=portfolio_metrics['drawdown'].index,
                y=portfolio_metrics['drawdown'] * 100,
                fill='tonexty',
                mode='lines',
                name='Drawdown',
                line=dict(color='red'),
                fillcolor='rgba(255,0,0,0.3)',
                hovertemplate="Drawdown: %{y:.2f}%<br>Date: %{x}<extra></extra>"
            )
        )
        
        fig_dd.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400
        )
        
        st.plotly_chart(fig_dd, use_container_width=True)
    
    with risk_cols[1]:
        # Rolling volatility
        rolling_vol = portfolio_metrics['portfolio_returns'].rolling(30).std() * np.sqrt(252) * 100
        
        fig_vol = go.Figure()
        
        fig_vol.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                name='30-Day Rolling Volatility',
                line=dict(color='orange', width=2),
                hovertemplate="Volatility: %{y:.2f}%<br>Date: %{x}<extra></extra>"
            )
        )
        
        fig_vol.update_layout(
            title="Rolling Volatility (30-day)",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility (%)",
            height=400
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)

with tab4:
    st.markdown("#### Correlation Analysis")
    
    corr_cols = st.columns([2, 1])
    
    with corr_cols[0]:
        # Correlation heatmap
        correlation_matrix = daily_returns.corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Stock Correlation Matrix",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with corr_cols[1]:
        st.markdown("**Correlation Insights:**")
        
        # Find highest and lowest correlations
        corr_flat = correlation_matrix.where(np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1))
        corr_pairs = corr_flat.stack().sort_values(ascending=False)
        
        st.markdown("**Highest Correlations:**")
        for i, (pair, corr) in enumerate(corr_pairs.head(3).items()):
            st.write(f"{pair[0]} - {pair[1]}: {corr:.3f}")
        
        st.markdown("**Lowest Correlations:**")
        for i, (pair, corr) in enumerate(corr_pairs.tail(3).items()):
            st.write(f"{pair[0]} - {pair[1]}: {corr:.3f}")

with tab5:
    st.markdown("#### Individual Stock Performance")
    
    # Stock performance comparison
    stock_perf_df = pd.DataFrame(stock_metrics).round(4)
    stock_perf_df['Weight'] = WEIGHTS
    stock_perf_df['Contribution'] = stock_perf_df['Weight'] * stock_perf_df['Annual Return']
    
    st.dataframe(stock_perf_df, use_container_width=True)
    
    # Risk-Return scatter plot
    fig_scatter = px.scatter(
        x=stock_perf_df['Annual Volatility'],
        y=stock_perf_df['Annual Return'],
        size=stock_perf_df['Weight'],
        hover_name=stock_perf_df.index,
        title="Risk-Return Profile",
        labels={
            'x': 'Annual Volatility',
            'y': 'Annual Return'
        },
        size_max=30
    )
    
    # Add portfolio point
    fig_scatter.add_trace(
        go.Scatter(
            x=[portfolio_metrics['annual_volatility']],
            y=[portfolio_metrics['annual_return']],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star'),
            name='Portfolio',
            hovertemplate="Portfolio<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>"
        )
    )
    
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

# =========================
# DATA EXPORT
# =========================
st.subheader("💾 Export Data")

export_cols = st.columns(3)

with export_cols[0]:
    if st.button("📊 Download Portfolio Report", type="primary"):
        # Generate comprehensive report
        report_data = {
            'Portfolio Summary': [{
                'Metric': 'Annual Return',
                'Value': f"{portfolio_metrics['annual_return']:.2%}"
            }, {
                'Metric': 'Annual Volatility',
                'Value': f"{portfolio_metrics['annual_volatility']:.2%}"
            }, {
                'Metric': 'Sharpe Ratio',
                'Value': f"{portfolio_metrics['sharpe_ratio']:.3f}"
            }, {
                'Metric': 'Max Drawdown',
                'Value': f"{portfolio_metrics['max_drawdown']:.2%}"
            }]
        }
        
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Portfolio summary
            pd.DataFrame(report_data['Portfolio Summary']).to_excel(
                writer, sheet_name='Portfolio Summary', index=False
            )
            
            # Stock performance
            stock_perf_df.to_excel(writer, sheet_name='Stock Performance')
            
            # Price data
            data.to_excel(writer, sheet_name='Price Data')
            
            # Returns data
            daily_returns.to_excel(writer, sheet_name='Daily Returns')
        
        st.download_button(
            label="📥 Download Excel Report",
            data=output.getvalue(),
            file_name=f"MarketMate_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

with export_cols[1]:
    if st.button("📈 Save Charts"):
        st.info("Charts can be saved individually using the download button on each chart")

with export_cols[2]:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    <p>MarketMate v2.0 | Built with ❤️ using Streamlit</p>
    <p>⚠️ This tool is for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
