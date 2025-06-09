import json
import os
import streamlit as st
import pandas as pd
import requests

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
    "pa_username": "",  # Added for remember me
    "pa_api_token": ""  # Added for remember me
}

config_file_path = "config.json"

# --- PythonAnywhere upload config ---
PA_USERNAME = "your_username"  # Set your PythonAnywhere username
PA_API_TOKEN = "your_api_token"  # Set your PythonAnywhere API token

def load_config():
    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as file:
            config = json.load(file)
        # Ensure all keys exist (for backward compatibility)
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
    else:
        config = default_config
        save_config(config)
    return config

def save_config(config):
    with open(config_file_path, "w") as file:
        json.dump(config, file, indent=4)

def update_config(new_config):
    save_config(new_config)

def upload_to_pythonanywhere(file_path: str, pa_username: str, pa_api_token: str, pa_file_path: str):
    url = f"https://www.pythonanywhere.com/api/v0/user/{pa_username}/files/path{pa_file_path}"
    headers = {"Authorization": f"Token {pa_api_token}"}

    with open(file_path, "rb") as file:
        response = requests.post(url, headers=headers, files={"content": file})

    if response.status_code in (200, 201):
        st.success("File uploaded successfully to PythonAnywhere!")
    else:
        st.error(f"Failed to upload file: {response.status_code}, {response.text}")

def upload_myportfolio(pa_username, pa_api_token, pa_file_path):
    file_path = "data/myportfolio.csv"
    if not os.path.exists(file_path):
        st.error("myportfolio.csv file not found.")
        return
    upload_to_pythonanywhere(file_path, pa_username, pa_api_token, pa_file_path)

def download_from_pythonanywhere(pa_username: str, pa_api_token: str, pa_file_path: str, local_file_path: str):
    url = f"https://www.pythonanywhere.com/api/v0/user/{pa_username}/files/path{pa_file_path}"
    headers = {"Authorization": f"Token {pa_api_token}"}
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(local_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("File restored successfully from PythonAnywhere!")
    else:
        st.error(f"Failed to restore file: {response.status_code}, {response.text}")

# Streamlit app to display and edit configurations
st.title("Ayarlar")

config = load_config()
new_config = config.copy()

col1, col2, col3 = st.columns(3)
with col1:
    for period, weight in config["weights"].items():
        new_config["weights"][period] = st.number_input(f"Ağırlık {period}", value=weight, step=0.1, key=period)
with col2:
    st.subheader("")
with col3:
    st.subheader("Backup")
    remember_me = st.checkbox("Remember me", value=True, key="remember_me")
    pa_username = st.text_input("PythonAnywhere Username", value=config.get("pa_username", ""), key="pa_username")
    pa_api_token = st.text_input("PythonAnywhere API Token", value=config.get("pa_api_token", ""), key="pa_api_token", type="password")
    pa_file_path = f"/home/{pa_username}/myportfolio.csv"
    if st.button("Upload to PythonAnywhere"):
        upload_myportfolio(pa_username, pa_api_token, pa_file_path)
    if st.button("Restore from PythonAnywhere"):
        local_file_path = "data/myportfolio.csv"
        download_from_pythonanywhere(pa_username, pa_api_token, pa_file_path, local_file_path)
    # Save pa_username and pa_api_token if remember_me is checked
    if remember_me:
        if (config.get("pa_username", "") != pa_username) or (config.get("pa_api_token", "") != pa_api_token):
            # Only update and save if changed
            config["pa_username"] = pa_username
            config["pa_api_token"] = pa_api_token
            save_config(config)
        new_config["pa_username"] = pa_username
        new_config["pa_api_token"] = pa_api_token
    else:
        new_config["pa_username"] = ""
        new_config["pa_api_token"] = ""

if st.button("Sakla"):
    update_config(new_config)
    st.success("Configuration updated successfully.")