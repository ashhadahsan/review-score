#!/usr/bin/env python3
"""
Build and serve ReviewScore documentation.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main function to build and serve documentation."""
    print("ReviewScore Documentation Builder")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("mkdocs.yml"):
        print("Error: mkdocs.yml not found. Please run from the project root.")
        sys.exit(1)
    
    # Install documentation dependencies
    print("\n1. Installing documentation dependencies...")
    if not run_command("pip install -r docs-requirements.txt", "Install docs requirements"):
        print("Failed to install documentation dependencies")
        sys.exit(1)
    
    # Build documentation
    print("\n2. Building documentation...")
    if not run_command("mkdocs build", "Build documentation"):
        print("Failed to build documentation")
        sys.exit(1)
    
    print("\n✓ Documentation built successfully!")
    print("\nTo serve the documentation locally, run:")
    print("  mkdocs serve")
    print("\nTo deploy to GitHub Pages, run:")
    print("  mkdocs gh-deploy")

if __name__ == "__main__":
    main()
