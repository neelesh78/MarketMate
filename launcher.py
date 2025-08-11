#!/usr/bin/env python3
"""
MarketMate Launcher
Launch MarketMate in different modes

Author: MarketMate Team
Version: 2.0
Date: 2025
"""

import sys
import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import webbrowser

class MarketMateLauncher:
    """Launcher GUI for MarketMate"""
    
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_widgets()
        
    def setup_window(self):
        """Setup launcher window"""
        self.root.title("🚀 MarketMate Launcher")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # Center window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
    def create_widgets(self):
        """Create launcher widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🚀 MarketMate", 
                               font=('Arial', 24, 'bold'))
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, text="Personal Portfolio Tracker",
                                  font=('Arial', 12))
        subtitle_label.pack(pady=(0, 30))
        
        # Launch options
        options_frame = ttk.LabelFrame(main_frame, text="Launch Options", padding="15")
        options_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Desktop GUI
        gui_frame = ttk.Frame(options_frame)
        gui_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(gui_frame, text="🖥️ Desktop GUI", 
                 font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
        ttk.Button(gui_frame, text="Launch", 
                  command=self.launch_gui).pack(side=tk.RIGHT)
        
        ttk.Label(gui_frame, text="Modern desktop interface with charts and analysis",
                 font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 0))
        
        # Web UI
        web_frame = ttk.Frame(options_frame)
        web_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(web_frame, text="🌐 Web Interface", 
                 font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
        ttk.Button(web_frame, text="Launch", 
                  command=self.launch_web).pack(side=tk.RIGHT)
        
        ttk.Label(web_frame, text="Interactive Streamlit dashboard in browser",
                 font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 0))
        
        # Command Line
        cli_frame = ttk.Frame(options_frame)
        cli_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(cli_frame, text="⚡ Command Line", 
                 font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
        ttk.Button(cli_frame, text="Run", 
                  command=self.launch_cli).pack(side=tk.RIGHT)
        
        ttk.Label(cli_frame, text="Quick analysis with default settings",
                 font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 0))
        
        # Utilities
        utils_frame = ttk.LabelFrame(main_frame, text="Utilities", padding="15")
        utils_frame.pack(fill=tk.X, pady=(0, 20))
        
        utils_buttons = ttk.Frame(utils_frame)
        utils_buttons.pack()
        
        ttk.Button(utils_buttons, text="🔧 Setup", 
                  command=self.run_setup).pack(side=tk.LEFT, padx=5)
        ttk.Button(utils_buttons, text="🧪 Run Tests", 
                  command=self.run_tests).pack(side=tk.LEFT, padx=5)
        ttk.Button(utils_buttons, text="📖 Help", 
                  command=self.show_help).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to launch MarketMate")
        status_label = ttk.Label(main_frame, textvariable=self.status_var,
                                font=('Arial', 9), foreground='gray')
        status_label.pack(side=tk.BOTTOM, pady=(20, 0))
        
        # Exit button
        ttk.Button(main_frame, text="Exit", 
                  command=self.root.quit).pack(side=tk.BOTTOM, pady=(10, 0))
        
    def launch_gui(self):
        """Launch desktop GUI"""
        self.status_var.set("Launching desktop GUI...")
        try:
            import marketmate_gui
            self.root.withdraw()  # Hide launcher
            
            gui_root = tk.Toplevel()
            gui_app = marketmate_gui.MarketMateGUI(gui_root)
            
            # Show launcher when GUI closes
            def on_gui_close():
                gui_root.destroy()
                self.root.deiconify()
                self.status_var.set("Desktop GUI closed")
                
            gui_root.protocol("WM_DELETE_WINDOW", on_gui_close)
            
        except ImportError:
            messagebox.showerror("Error", "GUI module not found. Please check installation.")
            self.status_var.set("Failed to launch GUI")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch GUI: {e}")
            self.status_var.set("Failed to launch GUI")
    
    def launch_web(self):
        """Launch web interface"""
        self.status_var.set("Starting web interface...")
        
        def start_streamlit():
            try:
                result = subprocess.run([
                    sys.executable, "-m", "streamlit", "run", 
                    "marketmate_ui.py", "--server.headless=true"
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", f"Failed to start web interface:\n{result.stderr}"))
                    
            except FileNotFoundError:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", "Streamlit not found. Please install with: pip install streamlit"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to start web interface: {e}"))
        
        # Start Streamlit in background thread
        thread = threading.Thread(target=start_streamlit)
        thread.daemon = True
        thread.start()
        
        # Open browser after short delay
        self.root.after(3000, lambda: webbrowser.open("http://localhost:8501"))
        self.status_var.set("Web interface starting... Opening browser...")
    
    def launch_cli(self):
        """Launch command line analysis"""
        self.status_var.set("Running command line analysis...")
        
        def run_cli():
            try:
                result = subprocess.run([
                    sys.executable, "marketmate.py"
                ], capture_output=True, text=True, cwd=os.getcwd())
                
                # Show results in popup
                self.root.after(0, lambda: self.show_cli_results(result))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to run CLI analysis: {e}"))
        
        thread = threading.Thread(target=run_cli)
        thread.daemon = True
        thread.start()
    
    def show_cli_results(self, result):
        """Show CLI results in popup"""
        results_window = tk.Toplevel(self.root)
        results_window.title("Analysis Results")
        results_window.geometry("600x400")
        
        text_widget = tk.Text(results_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        if result.returncode == 0:
            text_widget.insert(1.0, result.stdout)
            self.status_var.set("CLI analysis completed successfully")
        else:
            text_widget.insert(1.0, f"Error:\n{result.stderr}\n\nOutput:\n{result.stdout}")
            self.status_var.set("CLI analysis failed")
        
        text_widget.config(state='disabled')
        
        close_btn = ttk.Button(results_window, text="Close", 
                              command=results_window.destroy)
        close_btn.pack(pady=10)
    
    def run_setup(self):
        """Run setup script"""
        self.status_var.set("Running setup...")
        
        def setup():
            try:
                result = subprocess.run([
                    sys.executable, "setup.py"
                ], capture_output=True, text=True)
                
                self.root.after(0, lambda: self.show_setup_results(result))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to run setup: {e}"))
        
        thread = threading.Thread(target=setup)
        thread.daemon = True
        thread.start()
    
    def show_setup_results(self, result):
        """Show setup results"""
        if result.returncode == 0:
            messagebox.showinfo("Setup Complete", "MarketMate setup completed successfully!")
            self.status_var.set("Setup completed")
        else:
            messagebox.showerror("Setup Failed", f"Setup failed:\n{result.stderr}")
            self.status_var.set("Setup failed")
    
    def run_tests(self):
        """Run test suite"""
        self.status_var.set("Running tests...")
        
        def test():
            try:
                result = subprocess.run([
                    sys.executable, "test_marketmate.py"
                ], capture_output=True, text=True)
                
                self.root.after(0, lambda: self.show_test_results(result))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to run tests: {e}"))
        
        thread = threading.Thread(target=test)
        thread.daemon = True
        thread.start()
    
    def show_test_results(self, result):
        """Show test results"""
        results_window = tk.Toplevel(self.root)
        results_window.title("Test Results")
        results_window.geometry("700x500")
        
        text_widget = tk.Text(results_window, wrap=tk.WORD, font=('Courier', 9))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget.insert(1.0, result.stdout)
        text_widget.config(state='disabled')
        
        if result.returncode == 0:
            self.status_var.set("All tests passed")
        else:
            self.status_var.set("Some tests failed")
        
        close_btn = ttk.Button(results_window, text="Close", 
                              command=results_window.destroy)
        close_btn.pack(pady=10)
    
    def show_help(self):
        """Show help information"""
        help_text = """
🚀 MarketMate Launcher Help

LAUNCH OPTIONS:

🖥️ Desktop GUI
- Modern tkinter-based desktop application
- Interactive charts and visualizations
- Portfolio configuration and analysis
- Data export capabilities

🌐 Web Interface  
- Streamlit-based web dashboard
- Runs in your browser at localhost:8501
- Real-time interactive analysis
- Professional charts and metrics

⚡ Command Line
- Quick analysis with default portfolio
- Generates reports and charts automatically
- Results saved to data/ and results/ folders

UTILITIES:

🔧 Setup - Install dependencies and configure environment
🧪 Run Tests - Execute comprehensive test suite  
📖 Help - Show this help information

GETTING STARTED:
1. Run Setup if this is your first time
2. Choose your preferred interface
3. Configure your portfolio
4. Run analysis and review results

For detailed documentation, see MarketMate_README.md
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("MarketMate Help")
        help_window.geometry("600x500")
        
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=15, pady=15)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(1.0, help_text)
        text_widget.config(state='disabled')
        
        close_btn = ttk.Button(help_window, text="Close", 
                              command=help_window.destroy)
        close_btn.pack(pady=10)

def main():
    """Main launcher function"""
    root = tk.Tk()
    app = MarketMateLauncher(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.destroy()

if __name__ == "__main__":
    main()
