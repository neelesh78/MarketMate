#!/usr/bin/env python3
"""
MarketMate Desktop GUI
Modern tkinter-based graphical user interface for portfolio analysis

Author: MarketMate Team
Version: 2.0
Date: 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import json
import os
from datetime import datetime
import webbrowser

# Import our MarketMate modules
from marketmate import MarketMate
from utils import PORTFOLIO_PRESETS, DataValidator

class MarketMateGUI:
    """Modern GUI for MarketMate Portfolio Analysis"""
    
    def __init__(self, root):
        self.root = root
        self.analyzer = None
        self.results = None
        self.setup_main_window()
        self.create_widgets()
        self.center_window()
        
    def setup_main_window(self):
        """Setup main window properties"""
        self.root.title("🚀 MarketMate - Personal Portfolio Tracker")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#26A69A',
            'warning': '#FF9800',
            'danger': '#F44336',
            'dark': '#263238',
            'light': '#FAFAFA'
        }
        
        # Configure custom styles
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground=self.colors['primary'])
        self.style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('Success.TLabel', foreground=self.colors['success'])
        self.style.configure('Warning.TLabel', foreground=self.colors['warning'])
        self.style.configure('Danger.TLabel', foreground=self.colors['danger'])
        
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Create main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        
        # Create header
        self.create_header()
        
        # Create input panel
        self.create_input_panel()
        
        # Create main content area (tabbed)
        self.create_content_area()
        
        # Create status bar
        self.create_status_bar()
        
    def create_header(self):
        """Create header section"""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Title
        title_label = ttk.Label(header_frame, text="🚀 MarketMate", style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Subtitle
        subtitle_label = ttk.Label(header_frame, text="Personal Portfolio Tracker & Analyzer")
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Menu buttons
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="💾 Save Config", command=self.save_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="📁 Load Config", command=self.load_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="🌐 Web UI", command=self.open_web_ui).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="❓ Help", command=self.show_help).pack(side=tk.LEFT, padx=2)
        
    def create_input_panel(self):
        """Create input panel for portfolio configuration"""
        input_frame = ttk.LabelFrame(self.main_frame, text="Portfolio Configuration", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 5))
        input_frame.columnconfigure(1, weight=1)
        
        # Portfolio preset selection
        ttk.Label(input_frame, text="Preset:", style='Heading.TLabel').grid(row=0, column=0, sticky=tk.W, pady=2)
        self.preset_var = tk.StringVar(value="Custom")
        preset_combo = ttk.Combobox(input_frame, textvariable=self.preset_var, 
                                   values=["Custom"] + list(PORTFOLIO_PRESETS.keys()),
                                   state="readonly", width=15)
        preset_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        preset_combo.bind('<<ComboboxSelected>>', self.on_preset_change)
        
        # Stocks input
        ttk.Label(input_frame, text="Stocks:", style='Heading.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        self.stocks_var = tk.StringVar(value="AAPL,MSFT,GOOGL,AMZN,TSLA")
        stocks_entry = ttk.Entry(input_frame, textvariable=self.stocks_var)
        stocks_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        
        # Weights input
        ttk.Label(input_frame, text="Weights:", style='Heading.TLabel').grid(row=2, column=0, sticky=tk.W, pady=2)
        self.weights_var = tk.StringVar(value="0.25,0.20,0.20,0.20,0.15")
        weights_entry = ttk.Entry(input_frame, textvariable=self.weights_var)
        weights_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        
        # Period selection
        ttk.Label(input_frame, text="Period:", style='Heading.TLabel').grid(row=3, column=0, sticky=tk.W, pady=2)
        self.period_var = tk.StringVar(value="1y")
        period_combo = ttk.Combobox(input_frame, textvariable=self.period_var,
                                   values=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                                   state="readonly", width=15)
        period_combo.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        
        # Risk-free rate
        ttk.Label(input_frame, text="Risk-free Rate:", style='Heading.TLabel').grid(row=4, column=0, sticky=tk.W, pady=2)
        self.risk_free_var = tk.StringVar(value="0.02")
        risk_free_entry = ttk.Entry(input_frame, textvariable=self.risk_free_var, width=15)
        risk_free_entry.grid(row=4, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        
        # Validate and Analyze buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=(10, 0))
        
        self.validate_btn = ttk.Button(button_frame, text="✓ Validate", command=self.validate_portfolio)
        self.validate_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.analyze_btn = ttk.Button(button_frame, text="📊 Analyze Portfolio", 
                                     command=self.start_analysis, style='Accent.TButton')
        self.analyze_btn.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(input_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Validation status
        self.validation_label = ttk.Label(input_frame, text="")
        self.validation_label.grid(row=7, column=0, columnspan=2, pady=(5, 0))
        
    def create_content_area(self):
        """Create main content area with tabs"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Summary Tab
        self.create_summary_tab()
        
        # Charts Tab
        self.create_charts_tab()
        
        # Data Tab
        self.create_data_tab()
        
        # Reports Tab
        self.create_reports_tab()
        
    def create_summary_tab(self):
        """Create portfolio summary tab"""
        summary_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(summary_frame, text="📊 Summary")
        
        # Configure grid
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.columnconfigure(1, weight=1)
        summary_frame.rowconfigure(1, weight=1)
        
        # Metrics frames
        performance_frame = ttk.LabelFrame(summary_frame, text="Performance Metrics", padding="10")
        performance_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 5), pady=(0, 5))
        
        risk_frame = ttk.LabelFrame(summary_frame, text="Risk Metrics", padding="10")
        risk_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N), padx=(5, 0), pady=(0, 5))
        
        # Performance metrics labels
        self.perf_labels = {}
        perf_metrics = ["Annual Return", "Total Return", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio"]
        for i, metric in enumerate(perf_metrics):
            ttk.Label(performance_frame, text=f"{metric}:", font=('Arial', 10, 'bold')).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.perf_labels[metric] = ttk.Label(performance_frame, text="-")
            self.perf_labels[metric].grid(row=i, column=1, sticky=tk.E, pady=2)
        
        # Risk metrics labels
        self.risk_labels = {}
        risk_metrics = ["Max Drawdown", "VaR (95%)", "CVaR (95%)", "Beta", "Best Day", "Worst Day"]
        for i, metric in enumerate(risk_metrics):
            ttk.Label(risk_frame, text=f"{metric}:", font=('Arial', 10, 'bold')).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.risk_labels[metric] = ttk.Label(risk_frame, text="-")
            self.risk_labels[metric].grid(row=i, column=1, sticky=tk.E, pady=2)
        
        # Portfolio composition
        composition_frame = ttk.LabelFrame(summary_frame, text="Portfolio Composition", padding="10")
        composition_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        
        # Treeview for portfolio composition
        self.composition_tree = ttk.Treeview(composition_frame, columns=('Weight', 'Annual Return', 'Volatility', 'Sharpe'), height=6)
        self.composition_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure treeview columns
        self.composition_tree.heading('#0', text='Stock')
        self.composition_tree.heading('Weight', text='Weight')
        self.composition_tree.heading('Annual Return', text='Annual Return')
        self.composition_tree.heading('Volatility', text='Volatility')
        self.composition_tree.heading('Sharpe', text='Sharpe Ratio')
        
        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(composition_frame, orient=tk.VERTICAL, command=self.composition_tree.yview)
        tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.composition_tree.configure(yscrollcommand=tree_scroll.set)
        
        composition_frame.columnconfigure(0, weight=1)
        composition_frame.rowconfigure(0, weight=1)
        
    def create_charts_tab(self):
        """Create charts tab with matplotlib plots"""
        charts_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(charts_frame, text="📈 Charts")
        
        # Configure grid
        charts_frame.columnconfigure(0, weight=1)
        charts_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Chart selection
        chart_controls = ttk.Frame(charts_frame)
        chart_controls.grid(row=1, column=0, pady=(10, 0))
        
        ttk.Label(chart_controls, text="Chart Type:").pack(side=tk.LEFT, padx=(0, 5))
        self.chart_type = tk.StringVar(value="Portfolio Overview")
        chart_combo = ttk.Combobox(chart_controls, textvariable=self.chart_type,
                                  values=["Portfolio Overview", "Price History", "Returns Distribution", 
                                         "Correlation Matrix", "Risk-Return Scatter"],
                                  state="readonly")
        chart_combo.pack(side=tk.LEFT, padx=(0, 10))
        chart_combo.bind('<<ComboboxSelected>>', self.update_chart)
        
        ttk.Button(chart_controls, text="💾 Save Chart", command=self.save_chart).pack(side=tk.LEFT, padx=5)
        
    def create_data_tab(self):
        """Create data tab for viewing raw data"""
        data_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(data_frame, text="📋 Data")
        
        # Configure grid
        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(1, weight=1)
        
        # Data type selection
        data_controls = ttk.Frame(data_frame)
        data_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(data_controls, text="Data Type:").pack(side=tk.LEFT, padx=(0, 5))
        self.data_type = tk.StringVar(value="Raw Prices")
        data_combo = ttk.Combobox(data_controls, textvariable=self.data_type,
                                 values=["Raw Prices", "Daily Returns", "Portfolio Returns", 
                                        "Moving Averages", "Volatility"],
                                 state="readonly")
        data_combo.pack(side=tk.LEFT, padx=(0, 10))
        data_combo.bind('<<ComboboxSelected>>', self.update_data_view)
        
        ttk.Button(data_controls, text="📤 Export", command=self.export_data).pack(side=tk.LEFT, padx=5)
        
        # Data display area
        data_display = ttk.Frame(data_frame)
        data_display.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        data_display.columnconfigure(0, weight=1)
        data_display.rowconfigure(0, weight=1)
        
        # Text widget for data display
        self.data_text = scrolledtext.ScrolledText(data_display, wrap=tk.NONE, state='disabled')
        self.data_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def create_reports_tab(self):
        """Create reports tab"""
        reports_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(reports_frame, text="📄 Reports")
        
        # Configure grid
        reports_frame.columnconfigure(0, weight=1)
        reports_frame.rowconfigure(1, weight=1)
        
        # Report controls
        report_controls = ttk.Frame(reports_frame)
        report_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(report_controls, text="📋 Generate Report", command=self.generate_report).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(report_controls, text="💾 Save Report", command=self.save_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(report_controls, text="📧 Email Report", command=self.email_report).pack(side=tk.LEFT, padx=5)
        
        # Report display
        self.report_text = scrolledtext.ScrolledText(reports_frame, wrap=tk.WORD, state='disabled')
        self.report_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def create_status_bar(self):
        """Create status bar"""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W, padding="5")
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def on_preset_change(self, event):
        """Handle preset selection change"""
        preset_name = self.preset_var.get()
        if preset_name != "Custom" and preset_name in PORTFOLIO_PRESETS:
            preset = PORTFOLIO_PRESETS[preset_name]
            self.stocks_var.set(",".join(preset["stocks"]))
            self.weights_var.set(",".join(map(str, preset["weights"])))
            self.status_var.set(f"Loaded {preset_name} preset: {preset['description']}")
        
    def validate_portfolio(self):
        """Validate current portfolio configuration"""
        try:
            # Parse inputs
            stocks = [s.strip().upper() for s in self.stocks_var.get().split(",") if s.strip()]
            weights = [float(w.strip()) for w in self.weights_var.get().split(",") if w.strip()]
            
            # Validate
            is_valid, message = DataValidator.validate_portfolio_weights(weights)
            
            if len(stocks) != len(weights):
                message = f"Number of stocks ({len(stocks)}) must match number of weights ({len(weights)})"
                is_valid = False
                
            # Update validation display
            if is_valid:
                self.validation_label.config(text=f"✓ {message}", style='Success.TLabel')
                self.analyze_btn.config(state='normal')
            else:
                self.validation_label.config(text=f"✗ {message}", style='Danger.TLabel')
                self.analyze_btn.config(state='disabled')
                
            self.status_var.set(message)
            
        except ValueError as e:
            self.validation_label.config(text=f"✗ Invalid input format", style='Danger.TLabel')
            self.analyze_btn.config(state='disabled')
            self.status_var.set("Invalid input format")
            
    def start_analysis(self):
        """Start portfolio analysis in background thread"""
        self.progress.start(10)
        self.analyze_btn.config(state='disabled')
        self.status_var.set("Running analysis...")
        
        # Start analysis in background thread
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
        
    def run_analysis(self):
        """Run the actual portfolio analysis"""
        try:
            # Parse inputs
            stocks = [s.strip().upper() for s in self.stocks_var.get().split(",") if s.strip()]
            weights = [float(w.strip()) for w in self.weights_var.get().split(",") if w.strip()]
            period = self.period_var.get()
            risk_free_rate = float(self.risk_free_var.get())
            
            # Create configuration
            config = {
                "stocks": stocks,
                "weights": weights,
                "period": period,
                "risk_free_rate": risk_free_rate
            }
            
            # Create analyzer and run analysis
            self.analyzer = MarketMate()
            self.analyzer.config.update(config)
            self.results = self.analyzer.run_analysis()
            
            # Update GUI in main thread
            self.root.after(0, self.analysis_complete)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.root.after(0, lambda: self.analysis_error(error_msg))
            
    def analysis_complete(self):
        """Handle completed analysis"""
        self.progress.stop()
        self.analyze_btn.config(state='normal')
        
        if self.results and self.results.get("success"):
            self.status_var.set("✅ Analysis completed successfully!")
            self.update_summary()
            self.update_chart()
            self.update_data_view()
            messagebox.showinfo("Analysis Complete", "Portfolio analysis completed successfully!")
        else:
            error_msg = self.results.get("error", "Unknown error") if self.results else "Analysis failed"
            self.status_var.set(f"❌ {error_msg}")
            messagebox.showerror("Analysis Failed", error_msg)
            
    def analysis_error(self, error_msg):
        """Handle analysis error"""
        self.progress.stop()
        self.analyze_btn.config(state='normal')
        self.status_var.set(f"❌ {error_msg}")
        messagebox.showerror("Analysis Error", error_msg)
        
    def update_summary(self):
        """Update summary tab with analysis results"""
        if not self.results or not self.results.get("success"):
            return
            
        metrics = self.results["metrics"]["portfolio"]
        individual = self.results["metrics"]["individual_stocks"]
        
        # Update performance metrics
        self.perf_labels["Annual Return"].config(text=f"{metrics['annual_return']:.2%}", style='Success.TLabel')
        self.perf_labels["Total Return"].config(text=f"{metrics['total_return']:.2%}", style='Success.TLabel')
        self.perf_labels["Annual Volatility"].config(text=f"{metrics['annual_volatility']:.2%}")
        self.perf_labels["Sharpe Ratio"].config(text=f"{metrics['sharpe_ratio']:.3f}")
        self.perf_labels["Sortino Ratio"].config(text=f"{metrics['sortino_ratio']:.3f}")
        
        # Update risk metrics
        self.risk_labels["Max Drawdown"].config(text=f"{metrics['max_drawdown']:.2%}", style='Danger.TLabel')
        self.risk_labels["VaR (95%)"].config(text=f"{metrics['var_95']:.2%}")
        self.risk_labels["CVaR (95%)"].config(text=f"{metrics['cvar_95']:.2%}")
        beta_text = f"{metrics['beta']:.3f}" if not np.isnan(metrics['beta']) else "N/A"
        self.risk_labels["Beta"].config(text=beta_text)
        self.risk_labels["Best Day"].config(text=f"{metrics['best_day']:.2%}", style='Success.TLabel')
        self.risk_labels["Worst Day"].config(text=f"{metrics['worst_day']:.2%}", style='Danger.TLabel')
        
        # Update composition treeview
        for item in self.composition_tree.get_children():
            self.composition_tree.delete(item)
            
        stocks = self.analyzer.config["stocks"]
        weights = self.analyzer.config["weights"]
        
        for i, stock in enumerate(stocks):
            annual_return = individual["annual_returns"].get(stock, 0)
            volatility = individual["annual_volatility"].get(stock, 0)
            sharpe = individual["sharpe_ratios"].get(stock, 0)
            
            self.composition_tree.insert('', 'end', text=stock, values=(
                f"{weights[i]:.1%}",
                f"{annual_return:.2%}",
                f"{volatility:.2%}",
                f"{sharpe:.3f}"
            ))
    
    def update_chart(self, event=None):
        """Update the selected chart"""
        if not self.results or not self.results.get("success"):
            return
            
        chart_type = self.chart_type.get()
        self.fig.clear()
        
        try:
            if chart_type == "Portfolio Overview":
                self.plot_portfolio_overview()
            elif chart_type == "Price History":
                self.plot_price_history()
            elif chart_type == "Returns Distribution":
                self.plot_returns_distribution()
            elif chart_type == "Correlation Matrix":
                self.plot_correlation_matrix()
            elif chart_type == "Risk-Return Scatter":
                self.plot_risk_return_scatter()
                
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating chart: {e}")
            
    def plot_portfolio_overview(self):
        """Plot portfolio overview (4-panel chart)"""
        axes = self.fig.subplots(2, 2, figsize=(12, 8))
        
        # Price history
        ax1 = axes[0, 0]
        for stock in self.analyzer.config["stocks"]:
            ax1.plot(self.analyzer.raw_data.index, self.analyzer.raw_data[stock], label=stock, linewidth=2)
        ax1.set_title("Stock Price History", fontweight='bold')
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Portfolio cumulative returns
        ax2 = axes[0, 1]
        cumulative_returns = (1 + self.analyzer.portfolio_returns).cumprod()
        ax2.plot(cumulative_returns.index, cumulative_returns.values, 
                linewidth=3, color='darkblue', label='Portfolio')
        ax2.set_title("Portfolio Cumulative Returns", fontweight='bold')
        ax2.set_ylabel("Cumulative Return")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Rolling volatility
        ax3 = axes[1, 0]
        rolling_vol = self.analyzer.portfolio_returns.rolling(30).std() * np.sqrt(252)
        ax3.plot(rolling_vol.index, rolling_vol.values, color='red', linewidth=2)
        ax3.set_title("Rolling Volatility (30-day)", fontweight='bold')
        ax3.set_ylabel("Annualized Volatility")
        ax3.grid(True, alpha=0.3)
        
        # Drawdown
        ax4 = axes[1, 1]
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        ax4.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax4.plot(drawdown.index, drawdown.values, color='darkred', linewidth=2)
        ax4.set_title("Portfolio Drawdown", fontweight='bold')
        ax4.set_ylabel("Drawdown")
        ax4.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        
    def plot_price_history(self):
        """Plot detailed price history"""
        ax = self.fig.add_subplot(111)
        for stock in self.analyzer.config["stocks"]:
            ax.plot(self.analyzer.raw_data.index, self.analyzer.raw_data[stock], 
                   label=stock, linewidth=2)
        ax.set_title("Stock Price History", fontsize=14, fontweight='bold')
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_returns_distribution(self):
        """Plot returns distribution"""
        ax = self.fig.add_subplot(111)
        ax.hist(self.analyzer.portfolio_returns, bins=50, alpha=0.7, density=True, 
               color='skyblue', edgecolor='black')
        ax.axvline(self.analyzer.portfolio_returns.mean(), color='red', linestyle='--', 
                  label=f'Mean: {self.analyzer.portfolio_returns.mean():.4f}')
        ax.set_title("Portfolio Daily Returns Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_correlation_matrix(self):
        """Plot correlation matrix heatmap"""
        ax = self.fig.add_subplot(111)
        correlation_matrix = self.analyzer.daily_returns.corr()
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = self.fig.colorbar(im, ax=ax, shrink=0.8)
        
        # Set ticks and labels
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45)
        ax.set_yticklabels(correlation_matrix.columns)
        
        # Add correlation values
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title("Stock Correlation Matrix", fontsize=14, fontweight='bold')
        
    def plot_risk_return_scatter(self):
        """Plot risk-return scatter plot"""
        ax = self.fig.add_subplot(111)
        
        annual_returns = self.analyzer.daily_returns.mean() * 252
        annual_volatility = self.analyzer.daily_returns.std() * np.sqrt(252)
        
        # Individual stocks
        for i, stock in enumerate(self.analyzer.config["stocks"]):
            ax.scatter(annual_volatility[stock], annual_returns[stock], 
                      s=self.analyzer.config["weights"][i]*1000, alpha=0.7, label=stock)
        
        # Portfolio
        portfolio_vol = self.results["metrics"]["portfolio"]["annual_volatility"]
        portfolio_ret = self.results["metrics"]["portfolio"]["annual_return"]
        ax.scatter(portfolio_vol, portfolio_ret, s=200, color='red', 
                  marker='*', label='Portfolio', edgecolor='black')
        
        ax.set_xlabel("Annual Volatility")
        ax.set_ylabel("Annual Return")
        ax.set_title("Risk-Return Profile", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def update_data_view(self, event=None):
        """Update data view based on selection"""
        if not self.results or not self.results.get("success"):
            return
            
        data_type = self.data_type.get()
        
        try:
            if data_type == "Raw Prices":
                data = self.analyzer.raw_data
            elif data_type == "Daily Returns":
                data = self.analyzer.daily_returns
            elif data_type == "Portfolio Returns":
                data = pd.Series(self.analyzer.portfolio_returns, name="Portfolio_Returns")
            elif data_type == "Moving Averages":
                data = pd.concat(self.results["metrics"]["moving_averages"], axis=1)
            elif data_type == "Volatility":
                data = self.results["metrics"]["volatility"]
            else:
                return
                
            # Display data in text widget
            self.data_text.config(state='normal')
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(1.0, data.to_string())
            self.data_text.config(state='disabled')
            
        except Exception as e:
            print(f"Error updating data view: {e}")
    
    def generate_report(self):
        """Generate comprehensive report"""
        if not self.results or not self.results.get("success"):
            messagebox.showwarning("No Data", "Please run analysis first.")
            return
            
        try:
            report_text = self.results.get("report", "No report available")
            
            self.report_text.config(state='normal')
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(1.0, report_text)
            self.report_text.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")
    
    def save_chart(self):
        """Save current chart to file"""
        if not self.results:
            messagebox.showwarning("No Data", "Please run analysis first.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Chart saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save chart: {e}")
    
    def export_data(self):
        """Export current data view to CSV"""
        if not self.results:
            messagebox.showwarning("No Data", "Please run analysis first.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                data_type = self.data_type.get()
                
                if data_type == "Raw Prices":
                    data = self.analyzer.raw_data
                elif data_type == "Daily Returns":
                    data = self.analyzer.daily_returns
                elif data_type == "Portfolio Returns":
                    data = pd.Series(self.analyzer.portfolio_returns, name="Portfolio_Returns")
                else:
                    data = self.analyzer.raw_data
                
                if filename.endswith('.xlsx'):
                    data.to_excel(filename)
                else:
                    data.to_csv(filename)
                    
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {e}")
    
    def save_report(self):
        """Save report to file"""
        if not self.results:
            messagebox.showwarning("No Data", "Please run analysis first.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                report_text = self.results.get("report", "No report available")
                with open(filename, 'w') as f:
                    f.write(report_text)
                messagebox.showinfo("Success", f"Report saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report: {e}")
    
    def email_report(self):
        """Email report (placeholder)"""
        messagebox.showinfo("Email Report", "Email functionality not implemented yet.\nYou can save the report and attach it to an email manually.")
    
    def save_config(self):
        """Save current configuration"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                config = {
                    "stocks": [s.strip().upper() for s in self.stocks_var.get().split(",") if s.strip()],
                    "weights": [float(w.strip()) for w in self.weights_var.get().split(",") if w.strip()],
                    "period": self.period_var.get(),
                    "risk_free_rate": float(self.risk_free_var.get())
                }
                
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                    
                messagebox.showinfo("Success", f"Configuration saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def load_config(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                # Update GUI with loaded config
                if "stocks" in config:
                    self.stocks_var.set(",".join(config["stocks"]))
                if "weights" in config:
                    self.weights_var.set(",".join(map(str, config["weights"])))
                if "period" in config:
                    self.period_var.set(config["period"])
                if "risk_free_rate" in config:
                    self.risk_free_var.set(str(config["risk_free_rate"]))
                
                self.preset_var.set("Custom")
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def open_web_ui(self):
        """Open web UI in browser"""
        try:
            webbrowser.open("http://localhost:8501")
            messagebox.showinfo("Web UI", "Opening web interface...\n\nIf it doesn't open automatically, run:\nstreamlit run marketmate_ui.py")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open web UI: {e}")
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
🚀 MarketMate - Help

QUICK START:
1. Enter stock symbols (comma-separated)
2. Enter portfolio weights (must sum to 1.0)
3. Select analysis period
4. Click "Validate" then "Analyze Portfolio"

FEATURES:
• Portfolio performance analysis
• Risk metrics (VaR, Sharpe, Sortino)
• Interactive charts and visualizations
• Data export capabilities
• Comprehensive reporting

PORTFOLIO PRESETS:
Use the preset dropdown to quickly load pre-configured portfolios:
• Growth: High-growth tech stocks
• Conservative: Stable dividend stocks
• Value: Undervalued large-cap stocks
• Diversified: Mixed asset allocation

CHARTS:
• Portfolio Overview: 4-panel summary
• Price History: Stock price trends
• Returns Distribution: Portfolio returns analysis
• Correlation Matrix: Inter-stock correlations
• Risk-Return Scatter: Risk vs return profile

DATA EXPORT:
Export any data view to CSV or Excel format for further analysis.

REPORTS:
Generate comprehensive text reports with all key metrics and analysis.

For more help, visit the MarketMate documentation.
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("MarketMate - Help")
        help_window.geometry("600x500")
        help_window.resizable(False, False)
        
        help_text_widget = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, padx=10, pady=10)
        help_text_widget.pack(fill=tk.BOTH, expand=True)
        help_text_widget.insert(1.0, help_text)
        help_text_widget.config(state='disabled')
        
        close_btn = ttk.Button(help_window, text="Close", command=help_window.destroy)
        close_btn.pack(pady=10)

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = MarketMateGUI(root)
    
    # Handle window closing
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit MarketMate?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.destroy()

if __name__ == "__main__":
    main()
