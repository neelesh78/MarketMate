#!/usr/bin/env python3
"""
MarketMate Test Suite
Comprehensive tests for portfolio analysis functionality

Author: MarketMate Team
Version: 2.0
Date: 2025
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marketmate import MarketMate
from utils import (
    DataValidator, PerformanceCalculator, DataFetcher, 
    ReportGenerator, ConfigManager, ExportManager
)

class TestDataValidator(unittest.TestCase):
    """Test data validation utilities"""
    
    def test_validate_portfolio_weights_valid(self):
        """Test valid portfolio weights"""
        weights = [0.4, 0.3, 0.3]
        is_valid, message = DataValidator.validate_portfolio_weights(weights)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Valid weights")
    
    def test_validate_portfolio_weights_invalid_sum(self):
        """Test invalid portfolio weights (sum != 1.0)"""
        weights = [0.5, 0.3, 0.3]
        is_valid, message = DataValidator.validate_portfolio_weights(weights)
        self.assertFalse(is_valid)
        self.assertIn("must sum to 1.0", message)
    
    def test_validate_portfolio_weights_negative(self):
        """Test negative portfolio weights"""
        weights = [0.6, 0.3, -0.1]
        is_valid, message = DataValidator.validate_portfolio_weights(weights)
        self.assertFalse(is_valid)
        self.assertIn("non-negative", message)

class TestPerformanceCalculator(unittest.TestCase):
    """Test performance calculation utilities"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample returns data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)  # For reproducible results
        self.returns = pd.Series(
            np.random.normal(0.001, 0.02, 252), 
            index=dates
        )
        
        self.cumulative = (1 + self.returns).cumprod()
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        sharpe = PerformanceCalculator.calculate_sharpe_ratio(self.returns)
        self.assertIsInstance(sharpe, float)
        # Should be reasonable value for normal returns
        self.assertGreater(sharpe, -2)
        self.assertLess(sharpe, 5)
    
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation"""
        sortino = PerformanceCalculator.calculate_sortino_ratio(self.returns)
        self.assertIsInstance(sortino, float)
        # Should be reasonable value
        self.assertGreater(sortino, -2)
        self.assertLess(sortino, 5)
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation"""
        max_dd, peak_date, trough_date = PerformanceCalculator.calculate_max_drawdown(self.cumulative)
        
        self.assertIsInstance(max_dd, float)
        self.assertLessEqual(max_dd, 0)  # Drawdown should be negative or zero
        self.assertIsInstance(peak_date, pd.Timestamp)
        self.assertIsInstance(trough_date, pd.Timestamp)
        self.assertLessEqual(peak_date, trough_date)  # Peak should come before or at trough
    
    def test_calculate_var_cvar(self):
        """Test VaR and CVaR calculation"""
        var, cvar = PerformanceCalculator.calculate_var_cvar(self.returns)
        
        self.assertIsInstance(var, float)
        self.assertIsInstance(cvar, float)
        self.assertLessEqual(cvar, var)  # CVaR should be less than or equal to VaR
    
    def test_calculate_beta(self):
        """Test beta calculation"""
        # Create market returns
        np.random.seed(43)
        market_returns = pd.Series(
            np.random.normal(0.0005, 0.015, 252),
            index=self.returns.index
        )
        
        beta = PerformanceCalculator.calculate_beta(self.returns, market_returns)
        self.assertIsInstance(beta, (float, np.floating))
        # Beta should be reasonable
        self.assertGreater(beta, -3)
        self.assertLess(beta, 3)

class TestMarketMate(unittest.TestCase):
    """Test main MarketMate functionality"""
    
    def setUp(self):
        """Set up test MarketMate instance"""
        # Create a test config
        test_config = {
            "stocks": ["AAPL", "MSFT"],
            "weights": [0.6, 0.4],
            "period": "3mo",
            "interval": "1d",
            "risk_free_rate": 0.02
        }
        
        # Save test config temporarily
        with open("test_config.json", "w") as f:
            import json
            json.dump(test_config, f)
        
        self.analyzer = MarketMate("test_config.json")
    
    def tearDown(self):
        """Clean up test files"""
        try:
            os.remove("test_config.json")
        except FileNotFoundError:
            pass
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs"""
        result = self.analyzer.validate_inputs()
        self.assertTrue(result)
    
    def test_validate_inputs_invalid_weights(self):
        """Test input validation with invalid weights"""
        self.analyzer.config["weights"] = [0.7, 0.4]  # Sum > 1
        result = self.analyzer.validate_inputs()
        self.assertFalse(result)
    
    def test_validate_inputs_mismatched_lengths(self):
        """Test input validation with mismatched stock/weight lengths"""
        self.analyzer.config["weights"] = [0.6, 0.3, 0.1]  # 3 weights, 2 stocks
        result = self.analyzer.validate_inputs()
        self.assertFalse(result)

class TestConfigManager(unittest.TestCase):
    """Test configuration management"""
    
    def test_get_default_config(self):
        """Test default configuration"""
        config = ConfigManager.get_default_config()
        self.assertIsInstance(config, dict)
        self.assertIn("stocks", config)
        self.assertIn("weights", config)
        
        # Validate weights sum to 1
        weights = config["weights"]
        self.assertAlmostEqual(sum(weights), 1.0, places=6)
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration"""
        test_config = {
            "stocks": ["TEST1", "TEST2"],
            "weights": [0.5, 0.5],
            "period": "1y"
        }
        
        # Save config
        filename = "test_save_config.json"
        success = ConfigManager.save_config(test_config, filename)
        self.assertTrue(success)
        
        # Load config
        loaded_config = ConfigManager.load_config(filename)
        self.assertEqual(loaded_config["stocks"], test_config["stocks"])
        self.assertEqual(loaded_config["weights"], test_config["weights"])
        
        # Clean up
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

class TestReportGenerator(unittest.TestCase):
    """Test report generation utilities"""
    
    def setUp(self):
        """Set up test data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        self.returns = pd.Series(
            np.random.normal(0.001, 0.02, 100),
            index=dates
        )
        
        # Create multi-stock returns
        self.multi_returns = pd.DataFrame({
            'STOCK1': np.random.normal(0.001, 0.02, 100),
            'STOCK2': np.random.normal(0.0005, 0.015, 100),
            'STOCK3': np.random.normal(0.0008, 0.018, 100)
        }, index=dates)
    
    def test_generate_summary_stats(self):
        """Test summary statistics generation"""
        stats = ReportGenerator.generate_summary_stats(self.returns)
        
        self.assertIsInstance(stats, dict)
        self.assertIn("portfolio_name", stats)
        self.assertIn("total_days", stats)
        self.assertIn("annual_return", stats)
        self.assertIn("annual_volatility", stats)
        self.assertEqual(stats["total_days"], 100)
    
    def test_generate_correlation_analysis(self):
        """Test correlation analysis"""
        corr_analysis = ReportGenerator.generate_correlation_analysis(self.multi_returns)
        
        self.assertIsInstance(corr_analysis, dict)
        self.assertIn("correlation_matrix", corr_analysis)
        self.assertIn("highest_correlations", corr_analysis)
        self.assertIn("lowest_correlations", corr_analysis)
        self.assertIn("average_correlation", corr_analysis)

class TestExportManager(unittest.TestCase):
    """Test data export utilities"""
    
    def setUp(self):
        """Set up test data"""
        self.test_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [100, 200, 300, 400, 500]
        })
    
    def test_export_to_csv(self):
        """Test CSV export"""
        filename = "test_export.csv"
        success = ExportManager.export_to_csv(self.test_df, filename)
        self.assertTrue(success)
        
        # Verify file exists and is readable
        self.assertTrue(os.path.exists(filename))
        loaded_df = pd.read_csv(filename, index_col=0)
        pd.testing.assert_frame_equal(self.test_df, loaded_df)
        
        # Clean up
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass
    
    def test_export_to_excel(self):
        """Test Excel export"""
        filename = "test_export.xlsx"
        data_dict = {
            "Sheet1": self.test_df,
            "Sheet2": self.test_df * 2
        }
        
        success = ExportManager.export_to_excel(data_dict, filename)
        self.assertTrue(success)
        
        # Verify file exists
        self.assertTrue(os.path.exists(filename))
        
        # Clean up
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def test_complete_analysis_workflow(self):
        """Test complete analysis workflow without actual data fetching"""
        # Create mock data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Mock the fetch_data method to return test data
        analyzer = MarketMate()
        analyzer.config = {
            "stocks": ["MOCK1", "MOCK2"],
            "weights": [0.6, 0.4],
            "period": "1y",
            "risk_free_rate": 0.02,
            "confidence_level": 0.95,
            "moving_average_windows": [20, 50],
            "volatility_window": 20
        }
        
        # Create mock data
        mock_data = pd.DataFrame({
            'MOCK1': np.cumsum(np.random.normal(0.001, 0.02, 252)) + 100,
            'MOCK2': np.cumsum(np.random.normal(0.0008, 0.018, 252)) + 50
        }, index=dates)
        
        analyzer.raw_data = mock_data
        
        # Test validation
        self.assertTrue(analyzer.validate_inputs())
        
        # Test metrics calculation
        metrics = analyzer.calculate_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("portfolio", metrics)
        self.assertIn("individual_stocks", metrics)
        
        # Test that all expected portfolio metrics are present
        portfolio_metrics = metrics["portfolio"]
        expected_keys = [
            "annual_return", "annual_volatility", "sharpe_ratio",
            "max_drawdown", "var_95", "total_return"
        ]
        for key in expected_keys:
            self.assertIn(key, portfolio_metrics)

if __name__ == '__main__':
    # Create a custom test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataValidator,
        TestPerformanceCalculator,
        TestMarketMate,
        TestConfigManager,
        TestReportGenerator,
        TestExportManager,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
