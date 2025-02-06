import json
import os
import streamlit as st

# Default configuration
default_config = {
    "weights": {
        "7d": 0.8,
        "1m": 0.9,
        "3m": 1.0,
        "6m": 1.1,
        "1y": 1.2,
        "3y": 1.4
    },
    "api_keys": {
        "example_api": "your_api_key_here"
    }
}

config_file_path = "config.json"

def load_config():
    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as file:
            config = json.load(file)
    else:
        config = default_config
        save_config(config)
    return config

def save_config(config):
    with open(config_file_path, "w") as file:
        json.dump(config, file, indent=4)

def get_weights():
    config = load_config()
    return config["weights"]

def get_api_keys():
    config = load_config()
    return config["api_keys"]

def update_config(new_config):
    save_config(new_config)

# Streamlit app to display and edit configurations
st.title("Ayarlar")

config = load_config()
new_config = config.copy()

col1, col2 = st.columns(2)
with col1:
    for period, weight in config["weights"].items():
        new_config["weights"][period] = st.number_input(f"Ağırlık {period}", value=weight, step=0.1, key=period)

if st.button("Sakla"):
    update_config(new_config)
    st.success("Configuration updated successfully!")