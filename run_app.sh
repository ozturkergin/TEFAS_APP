#!/bin/bash

# Install requirements
pip3 install -r requirements_new.txt 2>/dev/null || echo "Requirements installation failed, continuing..."

# Install TA-Lib dependencies (if not already installed)
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

brew install ta-lib 2>/dev/null || echo "TA-Lib C library installation failed, continuing..."
pip3 install ta-lib 2>/dev/null || echo "TA-Lib Python package installation failed. Visit https://ta-lib.org/install for manual installation, continuing..."

# Run Streamlit app
streamlit run app.py

# Optional: Wait for user input (equivalent to PAUSE)
read -p "Press Enter to continue..."