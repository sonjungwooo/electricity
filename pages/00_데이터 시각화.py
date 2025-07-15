import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 세계 전력 수요 추이 시각화")
st.write("이 그래프는 과거 연도별 전력 수요(TWh)를 나타냅니다.")

# 1. 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("World Energy Consumption.csv")
    return df[['year', 'electricity_demand']].dropna()

df = load_data()

# 2. 연도별 전력 수요 평균 계산
df_grouped = df.groupby("year")["electricity_demand"].mean().reset_index()

# 3. 그래프 출력
st.subheader("연도별 평균 전력 수요 (TWh)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_grouped["year"], df_grouped["electricity_demand"], marker='o', color='blue')
ax.set_xlabel("연도")
ax.set_ylabel("전력 수요 (TWh)")
ax.set_title("전 세계 연도별 전력 수요 변화")
ax.grid(True)
st.pyplot(fig)
