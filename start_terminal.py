#!/usr/bin/env python3
"""
AI Personal Agent System - Terminal Launcher
Simple launcher for the terminal interface.
"""

import os
import sys
import subprocess

def main():
    """Launch the terminal interface."""
    print("🚀 AI Personal Agent System - Terminal Mode")
    print("=" * 50)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected.")
        print("💡 Tip: Run 'source venv/bin/activate' first")
        print()
    
    # Check if required modules are available
    try:
        import torch
        import transformers
        print("✅ Required dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return
    
    # Launch terminal interface
    try:
        print("🎯 Starting terminal interface...")
        print("=" * 50)
        
        # Run the terminal app
        subprocess.run([sys.executable, "terminal_app.py"], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running terminal app: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
