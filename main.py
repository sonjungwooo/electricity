import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.title("🔌 세계 전력 수요 예측기")
st.write("전 세계 데이터를 기반으로 연도와 경제 지표를 입력하면 전력 수요(TWh)를 예측합니다.")

# 1. 내장 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv("World Energy Consumption.csv")

df = load_data()

# 2. 필요한 열만 필터링
features = ['year', 'population', 'gdp', 'energy_per_capita']
target = 'electricity_demand'
df = df[features + [target]].dropna()

# 3. 학습 준비
X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 4. 사용자 입력
st.subheader("📥 입력 값")
input_data = []
for feature in features:
    default_val = float(df[feature].mean())
    val = st.number_input(f"{feature}", value=default_val)
    input_data.append(val)

# 5. 예측
prediction = model.predict(scaler.transform([input_data]))[0]

# 6. 결과 출력
st.subheader("🔮 예측 결과")
st.metric(label="예상 전력 수요 (TWh)", value=f"{prediction:.2f}")
