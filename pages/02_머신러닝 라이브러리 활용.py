import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title("🌍 전력 소비 기반 탄소 배출량 예측")
st.write("과거 데이터를 기반으로 전력 소비량이 늘어날 때 탄소 배출량(CO₂)이 얼마나 증가하는지 예측합니다.")

# 1. 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv("World Energy Consumption.csv")
    df = df[['year', 'electricity_demand', 'gdp', 'energy_per_capita']].dropna()

    # 탄소 배출량 계산: 1MWh당 0.4톤 CO₂ 배출 (예시 기준)
    df['co2_emissions'] = df['electricity_demand'] * 1_000_000 * 0.4 / 1_000_000_000  # → 단위: 억 톤 (billion tons)
    return df

df = load_data()

# 2. 모델 학습
features = ['electricity_demand', 'gdp', 'energy_per_capita']
target = 'co2_emissions'

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 3. 사용자 입력
st.subheader("📥 입력값을 기반으로 CO₂ 배출량 예측")
ed = st.number_input("전력 소비량 (TWh)", value=float(df['electricity_demand'].mean()))
gdp = st.number_input("GDP (USD)", value=float(df['gdp'].mean()))
epc = st.number_input("1인당 에너지 소비량 (kWh)", value=float(df['energy_per_capita'].mean()))

user_input = np.array([[ed, gdp, epc]])
user_scaled = scaler.transform(user_input)
prediction = model.predict(user_scaled)[0]

st.metric("예상 탄소 배출량 (억 톤)", f"{prediction:.3f}")

# 4. 시각화
st.subheader("📈 연도별 탄소 배출량 추이 (예상치)")
df_grouped = df.groupby("year")[target].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_grouped["year"], df_grouped[target], marker='o', color='green')
ax.set_xlabel("연도")
ax.set_ylabel("탄소 배출량 (억 톤)")
ax.set_title("과거 연도별 전력 소비 기반 CO₂ 배출량 추정")
ax.grid(True)
st.pyplot(fig)
