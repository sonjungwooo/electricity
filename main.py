import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset (you can replace this with a cleaned version)
@st.cache_data
def load_data():
    df = pd.read_csv("World Energy Consumption.csv")
    features = ['year', 'population', 'gdp', 'energy_per_capita', 'electricity_use_per_capita', 'electricity_demand']
    df = df[features].dropna()
    return df

# Train the model (you can save this and load as a .pkl later)
def train_model(df):
    X = df.drop(columns='electricity_demand')
    y = df['electricity_demand']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

# Main App
st.title("미래 전력 수요 예측기")
st.write("입력한 데이터를 바탕으로 미래 전력 수요 (TWh)를 예측합니다.")

df = load_data()
model, scaler = train_model(df)

# 사용자 입력
year = st.slider("연도", 2000, 2050, 2025)
population = st.number_input("인구 수 (명)", value=1_000_000)
gdp = st.number_input("GDP (USD)", value=1_000_000_000)
energy_per_capita = st.number_input("1인당 에너지 사용량 (kWh)", value=2000.0)
electricity_use_per_capita = st.number_input("1인당 전력 사용량 (kWh)", value=1000.0)

# 예측
input_data = np.array([[year, population, gdp, energy_per_capita, electricity_use_per_capita]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

st.subheader("예측 결과")
st.metric(label="예상 전력 수요 (TWh)", value=f"{prediction:.2f}")
