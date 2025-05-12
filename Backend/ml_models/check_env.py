#!/usr/bin/env python
"""
Helper script to find and display Python environment information
"""
import sys
import os
import subprocess
import site

def get_environment_info():
    """Get environment information for debugging"""
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"sys.path:")
    for path in sys.path:
        print(f"  - {path}")
    
    print("\nChecking modules:")
    for module_name in ["pandas", "torch", "numpy"]:
        try:
            subprocess.run([sys.executable, "-c", f"import {module_name}; print(f'{module_name} version: ' + {module_name}.__version__)"], 
                          check=False)
        except subprocess.CalledProcessError:
            print(f"Error importing {module_name}")
    
    print("\nSite packages directories:")
    for path in site.getsitepackages():
        print(f"  - {path}")

if __name__ == "__main__":
    get_environment_info()
