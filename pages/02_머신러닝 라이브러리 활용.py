import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.title("🌍 전력 소비 기반 탄소 배출량 및 배출권 비용 예측")
st.write("과거 데이터를 기반으로 전력 소비량이 늘어날 때 탄소 배출량(CO₂)과 탄소배출권 비용을 예측합니다.")

# 탄소배출권 가격 설정 (2025년 기준)
CARBON_PRICE_OPTIONS = {
    "EU ETS (유럽배출권거래제)": {"price_eur": 67.25, "description": "2025년 평균 예상 가격"},
    "K-ETS (한국배출권거래제)": {"price_eur": 15.0, "description": "약 15 EUR/tCO₂e (추정)"},
    "사용자 정의": {"price_eur": 50.0, "description": "직접 입력"}
}

# 환율 설정 (2025년 7월 기준)
EUR_TO_KRW = 1612  # 1 EUR = 1612 KRW (approximate)

# 1. 데이터 로드
@st.cache_data
def load_data():
    # 샘플 데이터 생성 (실제 사용 시 CSV 파일 사용)
    np.random.seed(42)
    years = list(range(2010, 2025))
    data = []
    
    for year in years:
        # 연도별 증가 추세 반영
        base_electricity = 1000 + (year - 2010) * 50 + np.random.normal(0, 100)
        base_gdp = 1e12 + (year - 2010) * 5e10 + np.random.normal(0, 1e10)
        base_energy_per_capita = 5000 + (year - 2010) * 100 + np.random.normal(0, 200)
        
        data.append({
            'year': year,
            'electricity_demand': max(base_electricity, 500),  # TWh
            'gdp': max(base_gdp, 5e11),  # USD
            'energy_per_capita': max(base_energy_per_capita, 3000)  # kWh
        })
    
    df = pd.DataFrame(data)
    # 탄소 배출량 계산: 1MWh당 0.4톤 CO₂ 배출
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

# 3. 사이드바 - 탄소배출권 설정
st.sidebar.header("🏷️ 탄소배출권 설정")
selected_market = st.sidebar.selectbox(
    "탄소배출권 시장 선택",
    options=list(CARBON_PRICE_OPTIONS.keys())
)

if selected_market == "사용자 정의":
    carbon_price_eur = st.sidebar.number_input(
        "탄소배출권 가격 (EUR/tCO₂e)",
        min_value=0.0,
        max_value=200.0,
        value=50.0,
        step=1.0
    )
else:
    carbon_price_eur = CARBON_PRICE_OPTIONS[selected_market]["price_eur"]
    st.sidebar.info(f"**{selected_market}**\n\n가격: {carbon_price_eur:.2f} EUR/tCO₂e\n\n{CARBON_PRICE_OPTIONS[selected_market]['description']}")

carbon_price_krw = carbon_price_eur * EUR_TO_KRW

# 4. 사용자 입력
st.subheader("📥 입력값을 기반으로 CO₂ 배출량 및 배출권 비용 예측")

col1, col2, col3 = st.columns(3)
with col1:
    ed = st.number_input("전력 소비량 (TWh)", value=float(df['electricity_demand'].mean()), min_value=0.0)
with col2:
    gdp = st.number_input("GDP (USD)", value=float(df['gdp'].mean()), min_value=0.0, format="%.0f")
with col3:
    epc = st.number_input("1인당 에너지 소비량 (kWh)", value=float(df['energy_per_capita'].mean()), min_value=0.0)

# 예측 수행
user_input = np.array([[ed, gdp, epc]])
user_scaled = scaler.transform(user_input)
prediction = model.predict(user_scaled)[0]

# 탄소배출권 비용 계산
co2_emissions_tons = prediction * 1_000_000_000  # 억 톤 → 톤 변환
carbon_cost_eur = co2_emissions_tons * carbon_price_eur
carbon_cost_krw = carbon_cost_eur * EUR_TO_KRW

# 5. 결과 표시
st.subheader("📊 예측 결과")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("예상 탄소 배출량", f"{prediction:.3f} 억 톤")
with col2:
    st.metric("탄소배출권 비용 (EUR)", f"€{carbon_cost_eur:,.0f}")
with col3:
    st.metric("탄소배출권 비용 (KRW)", f"₩{carbon_cost_krw:,.0f}")
with col4:
    st.metric("배출권 가격 (EUR/tCO₂e)", f"€{carbon_price_eur:.2f}")

# 6. 상세 분석
st.subheader("📈 상세 분석")

# 탄소배출권 시장별 비교
st.write("**탄소배출권 시장별 비용 비교**")
comparison_data = []
for market, info in CARBON_PRICE_OPTIONS.items():
    if market != "사용자 정의":
        cost_eur = co2_emissions_tons * info["price_eur"]
        cost_krw = cost_eur * EUR_TO_KRW
        comparison_data.append({
            "시장": market,
            "가격 (EUR/tCO₂e)": info["price_eur"],
            "총 비용 (EUR)": f"€{cost_eur:,.0f}",
            "총 비용 (KRW)": f"₩{cost_krw:,.0f}"
        })

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True)

# 7. 시각화
col1, col2 = st.columns(2)

with col1:
    st.write("**연도별 탄소 배출량 추이**")
    df_grouped = df.groupby("year")[target].mean().reset_index()
    
    fig = px.line(df_grouped, x="year", y=target, 
                  title="과거 연도별 전력 소비 기반 CO₂ 배출량 추정",
                  labels={target: "탄소 배출량 (억 톤)", "year": "연도"})
    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.write("**탄소배출권 비용 시뮬레이션**")
    
    # 전력 소비량 변화에 따른 비용 변화 시뮬레이션
    electricity_range = np.linspace(ed * 0.5, ed * 1.5, 10)
    costs = []
    
    for elec in electricity_range:
        sim_input = np.array([[elec, gdp, epc]])
        sim_scaled = scaler.transform(sim_input)
        sim_prediction = model.predict(sim_scaled)[0]
        sim_cost = sim_prediction * 1_000_000_000 * carbon_price_eur
        costs.append(sim_cost)
    
    sim_df = pd.DataFrame({
        'electricity_demand': electricity_range,
        'carbon_cost': costs
    })
    
    fig2 = px.line(sim_df, x='electricity_demand', y='carbon_cost',
                   title="전력 소비량 변화에 따른 탄소배출권 비용",
                   labels={'electricity_demand': '전력 소비량 (TWh)', 
                          'carbon_cost': '탄소배출권 비용 (EUR)'})
    st.plotly_chart(fig2, use_container_width=True)

# 8. 추가 정보
st.subheader("ℹ️ 추가 정보")

with st.expander("탄소배출권 시장 정보"):
    st.write("""
    **주요 탄소배출권 시장:**
    
    - **EU ETS (유럽배출권거래제)**: 세계 최대 탄소배출권 시장
    - **K-ETS (한국배출권거래제)**: 한국의 국가 배출권 거래제
    - **기타**: 캘리포니아, 중국, 캐나다 등 다양한 지역별 시장
    
    **가격 변동 요인:**
    - 경제 성장률 및 산업 활동
    - 재생에너지 확산 정도
    - 정부 정책 변화
    - 기후변화 관련 이벤트
    """)

with st.expander("계산 방법"):
    st.write(f"""
    **탄소 배출량 계산:**
    - 전력 소비량 1MWh당 0.4톤 CO₂ 배출 (가정)
    - 예측 배출량: {prediction:.3f} 억 톤 = {co2_emissions_tons:,.0f} 톤
    
    **탄소배출권 비용 계산:**
    - 총 배출량: {co2_emissions_tons:,.0f} 톤
    - 배출권 가격: €{carbon_price_eur:.2f}/tCO₂e
    - 총 비용: €{carbon_cost_eur:,.0f} (₩{carbon_cost_krw:,.0f})
    
    **환율:** 1 EUR = {EUR_TO_KRW} KRW
    """)

# 9. 결론 및 권장사항
st.subheader("💡 결론 및 권장사항")

if carbon_cost_krw > 1e10:  # 100억원 이상
    st.warning(f"⚠️ 예상 탄소배출권 비용이 **₩{carbon_cost_krw:,.0f}**로 상당히 높습니다. 탄소 감축 투자를 고려해보세요.")
elif carbon_cost_krw > 1e9:  # 10억원 이상
    st.info(f"ℹ️ 예상 탄소배출권 비용은 **₩{carbon_cost_krw:,.0f}**입니다. 중장기적 탄소 감축 계획을 수립하는 것이 좋습니다.")
else:
    st.success(f"✅ 예상 탄소배출권 비용은 **₩{carbon_cost_krw:,.0f}**로 관리 가능한 수준입니다.")

st.write("""
**탄소 비용 절감 방안:**
1. 재생에너지 전환 확대
2. 에너지 효율 개선
3. 탄소 포집 및 저장 기술 도입
4. 공정 최적화 및 디지털화
""")
