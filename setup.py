#!/usr/bin/env python3
"""
MarketMate Setup Script
Installs dependencies, configures environment, and validates setup

Author: MarketMate Team
Version: 2.0
Date: 2025
"""

import os
import sys
import subprocess
import importlib.util
import platform
from pathlib import Path

class MarketMateSetup:
    """Setup and configuration manager for MarketMate"""
    
    def __init__(self):
        self.required_packages = [
            'yfinance>=0.2.18',
            'pandas>=1.3.0',
            'numpy>=1.21.0',
            'matplotlib>=3.5.0',
            'streamlit>=1.28.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0',
            'scipy>=1.7.0',
            'requests>=2.25.0',
            'openpyxl>=3.0.0'
        ]
        
        self.optional_packages = [
            'ta-lib',
            'finnhub-python'
        ]
        
        self.directories = ['data', 'results', 'exports', 'logs']
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("Checking Python version...")
        version = sys.version_info
        
        if version < (3, 8):
            print(f"❌ Python {version.major}.{version.minor} detected.")
            print("❌ MarketMate requires Python 3.8 or higher.")
            return False
        
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected - Compatible!")
        return True
    
    def install_package(self, package):
        """Install a single package using pip"""
        try:
            print(f"Installing {package}...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ {package} installed successfully")
                return True
            else:
                print(f"❌ Failed to install {package}")
                print(f"Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")
            return False
    
    def check_package_installed(self, package_name):
        """Check if a package is installed"""
        # Remove version specifiers for checking
        clean_name = package_name.split('>=')[0].split('==')[0]
        
        try:
            importlib.import_module(clean_name)
            return True
        except ImportError:
            # Handle special cases
            if clean_name == 'ta-lib':
                try:
                    importlib.import_module('talib')
                    return True
                except ImportError:
                    pass
            return False
    
    def install_required_packages(self):
        """Install all required packages"""
        print("\n" + "="*50)
        print("INSTALLING REQUIRED PACKAGES")
        print("="*50)
        
        failed_packages = []
        
        for package in self.required_packages:
            package_name = package.split('>=')[0]
            
            if self.check_package_installed(package_name):
                print(f"✅ {package_name} already installed")
            else:
                if not self.install_package(package):
                    failed_packages.append(package)
        
        return failed_packages
    
    def install_optional_packages(self):
        """Install optional packages"""
        print("\n" + "="*50)
        print("INSTALLING OPTIONAL PACKAGES")
        print("="*50)
        
        for package in self.optional_packages:
            package_name = package.split('>=')[0] if '>=' in package else package
            
            if self.check_package_installed(package_name):
                print(f"✅ {package_name} already installed")
            else:
                print(f"⚠️ Installing optional package {package_name}...")
                if not self.install_package(package):
                    print(f"⚠️ Optional package {package_name} failed to install - continuing without it")
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n" + "="*50)
        print("CREATING DIRECTORIES")
        print("="*50)
        
        for directory in self.directories:
            try:
                Path(directory).mkdir(exist_ok=True)
                print(f"✅ Created/verified directory: {directory}")
            except Exception as e:
                print(f"❌ Failed to create directory {directory}: {e}")
    
    def validate_installation(self):
        """Validate that all components are working"""
        print("\n" + "="*50)
        print("VALIDATING INSTALLATION")
        print("="*50)
        
        validation_results = {}
        
        # Test core imports
        core_modules = [
            ('pandas', 'pd'),
            ('numpy', 'np'),
            ('matplotlib.pyplot', 'plt'),
            ('yfinance', 'yf'),
            ('streamlit', 'st'),
            ('seaborn', 'sns'),
            ('plotly.express', 'px')
        ]
        
        for module_name, alias in core_modules:
            try:
                module = importlib.import_module(module_name)
                print(f"✅ {module_name} - OK")
                validation_results[module_name] = True
            except ImportError as e:
                print(f"❌ {module_name} - FAILED: {e}")
                validation_results[module_name] = False
        
        # Test data fetching
        try:
            import yfinance as yf
            # Quick test with a reliable stock
            test_data = yf.download("AAPL", period="5d", progress=False)
            if not test_data.empty:
                print("✅ Data fetching - OK")
                validation_results['data_fetching'] = True
            else:
                print("❌ Data fetching - No data retrieved")
                validation_results['data_fetching'] = False
        except Exception as e:
            print(f"❌ Data fetching - FAILED: {e}")
            validation_results['data_fetching'] = False
        
        return validation_results
    
    def create_sample_files(self):
        """Create sample configuration and portfolio files"""
        print("\n" + "="*50)
        print("CREATING SAMPLE FILES")
        print("="*50)
        
        # Sample portfolio CSV
        sample_portfolio = """Symbol,Weight,Name
AAPL,0.25,Apple Inc
MSFT,0.20,Microsoft Corporation
GOOGL,0.20,Alphabet Inc
AMZN,0.20,Amazon.com Inc
TSLA,0.15,Tesla Inc"""
        
        try:
            with open('sample_portfolio.csv', 'w') as f:
                f.write(sample_portfolio)
            print("✅ Created sample_portfolio.csv")
        except Exception as e:
            print(f"❌ Failed to create sample portfolio: {e}")
        
        # Sample configuration
        sample_config = {
            "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "weights": [0.25, 0.20, 0.20, 0.20, 0.15],
            "period": "1y",
            "interval": "1d",
            "risk_free_rate": 0.02,
            "confidence_level": 0.95,
            "moving_average_windows": [20, 50, 200],
            "volatility_window": 30
        }
        
        try:
            import json
            with open('sample_config.json', 'w') as f:
                json.dump(sample_config, f, indent=2)
            print("✅ Created sample_config.json")
        except Exception as e:
            print(f"❌ Failed to create sample config: {e}")
    
    def print_usage_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*50)
        print("MARKETMATE USAGE INSTRUCTIONS")
        print("="*50)
        
        instructions = """
🚀 GETTING STARTED:

1. BASIC ANALYSIS:
   python marketmate.py

2. INTERACTIVE DASHBOARD:
   streamlit run marketmate_ui.py

3. RUN TESTS:
   python test_marketmate.py

4. CUSTOM CONFIGURATION:
   - Edit config.json for your portfolio
   - Or use sample_config.json as template

📁 FILE STRUCTURE:
   • marketmate.py       - Main analysis script
   • marketmate_ui.py    - Streamlit dashboard
   • utils.py           - Utility functions
   • config.json        - Main configuration
   • requirements.txt   - Package dependencies
   • test_marketmate.py - Test suite

📊 PORTFOLIO SETUP:
   • Edit 'stocks' and 'weights' in config.json
   • Weights must sum to 1.0
   • Use valid stock tickers (e.g., AAPL, MSFT, GOOGL)

⚡ QUICK START COMMANDS:
   # Run basic analysis
   python marketmate.py
   
   # Start web dashboard
   streamlit run marketmate_ui.py
   
   # Open dashboard in browser at: http://localhost:8501

🔧 TROUBLESHOOTING:
   • Check internet connection for data fetching
   • Verify stock symbols are valid
   • Ensure portfolio weights sum to 1.0
   • Check Python version (3.8+ required)

📈 FEATURES:
   ✓ Portfolio performance analysis
   ✓ Risk metrics (VaR, Sharpe, Sortino)
   ✓ Interactive visualizations
   ✓ Data export (CSV, Excel)
   ✓ Correlation analysis
   ✓ Drawdown analysis
   ✓ Moving averages
   ✓ Benchmarking

Happy analyzing! 🎯
"""
        print(instructions)
    
    def run_setup(self):
        """Run complete setup process"""
        print("🚀 Welcome to MarketMate Setup!")
        print("="*50)
        
        # Check Python version
        if not self.check_python_version():
            print("\n❌ Setup failed due to incompatible Python version.")
            return False
        
        # Install packages
        failed_packages = self.install_required_packages()
        
        if failed_packages:
            print(f"\n❌ Failed to install required packages: {failed_packages}")
            print("Please install them manually using: pip install <package_name>")
            return False
        
        # Install optional packages
        self.install_optional_packages()
        
        # Create directories
        self.create_directories()
        
        # Validate installation
        validation_results = self.validate_installation()
        
        # Check if core components are working
        core_success = all(validation_results.get(key, False) for key in 
                          ['pandas', 'numpy', 'matplotlib.pyplot', 'yfinance'])
        
        if not core_success:
            print("\n❌ Critical components failed validation.")
            print("Please check error messages above and resolve issues.")
            return False
        
        # Create sample files
        self.create_sample_files()
        
        # Print instructions
        self.print_usage_instructions()
        
        print("\n" + "="*50)
        print("✅ SETUP COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("MarketMate is ready to use!")
        print("\nNext steps:")
        print("1. Run: python marketmate.py")
        print("2. Or run: streamlit run marketmate_ui.py")
        print("="*50)
        
        return True

if __name__ == "__main__":
    setup = MarketMateSetup()
    success = setup.run_setup()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        
        # Ask user if they want to run a quick test
        try:
            response = input("\nRun a quick test? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                print("\nRunning quick validation test...")
                os.system(f"{sys.executable} -c \"from marketmate import MarketMate; print('✅ MarketMate import successful!')\"")
        except KeyboardInterrupt:
            print("\nSetup completed. You can run MarketMate anytime!")
    else:
        print("\n❌ Setup failed. Please resolve the issues above.")
        sys.exit(1)
