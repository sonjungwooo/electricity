import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.title("🔌 세계 전력 수요 분석 및 2030년까지 예측")

# 📁 CSV 불러오기 (같은 디렉토리에 있어야 함)
@st.cache_data
def load_data():
    df = pd.read_csv("World Energy Consumption.csv")
    features = ['year', 'population', 'gdp', 'energy_per_capita', 'electricity_demand']
    return df[features].dropna()

df = load_data()

# 🎯 예측 모델 학습
features = ['year', 'population', 'gdp', 'energy_per_capita']
target = 'electricity_demand'

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 📈 과거 데이터 그룹화
past_df = df.groupby("year")[target].mean().reset_index()

# 🔮 미래 예측: 2024~2030
future_years = np.arange(past_df["year"].max() + 1, 2031)
mean_vals = {
    "population": df["population"].mean(),
    "gdp": df["gdp"].mean(),
    "energy_per_capita": df["energy_per_capita"].mean()
}
future_data = pd.DataFrame({
    "year": future_years,
    "population": mean_vals["population"],
    "gdp": mean_vals["gdp"],
    "energy_per_capita": mean_vals["energy_per_capita"]
})
X_future_scaled = scaler.transform(future_data)
future_preds = model.predict(X_future_scaled)

future_df = pd.DataFrame({
    "year": future_years,
    "electricity_demand": future_preds
})

# 📊 전체 결합
combined_df = pd.concat([
    past_df.rename(columns={"electricity_demand": "전력 수요"}),
    future_df.rename(columns={"electricity_demand": "전력 수요"})
])

# 그래프 그리기
st.subheader("📊 전력 수요 추이 및 예측")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(past_df["year"], past_df["electricity_demand"], label="실제 수요", marker='o')
ax.plot(futu
