import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🌍 국가별 전력 소비 예측 & 경제·인구 분석 + 지도 시각화")

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    base_cols = ["country", "year", "electricity_demand"]
    additional_cols = ["population", "gdp", "energy_per_capita", "energy_per_gdp"]
    all_required = base_cols + additional_cols

    if not all(col in df.columns for col in all_required):
        st.error(f"필요한 컬럼이 누락되었습니다. 다음 컬럼이 모두 있어야 합니다:\n{all_required}")
        st.stop()

    df = df[all_required].dropna()
    df = df[df["year"].apply(lambda x: str(x).isnumeric())]
    df["year"] = pd.to_datetime(df["year"], format="%Y")

    countries = df["country"].unique().tolist()
    selected_countries = st.multiselect("국가(들)를 선택하세요 (최대 3개)", countries, default=countries[:1])

    if not selected_countries:
        st.warning("최소 한 개 이상의 국가를 선택하세요.")
        st.stop()

    if len(selected_countries) > 3:
        st.warning("최대 3개 국가까지만 선택 가능합니다.")
        st.stop()

    # 1) 다중 국가 전력 소비 및 ARIMA 예측 비교
    st.subheader(f"📈 선택한 국가들의 전력 소비 예측 비교 (ARIMA)")

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["blue", "green", "red"]  # 최대 3개 국가용 색상

    for i, country in enumerate(selected_countries):
        country_df = df[df["country"] == country].copy()
        country_df.set_index("year", inplace=True)
        ts = country_df["electricity_demand"]

        if len(ts) < 10:
            st.warning(f"⚠️ {country}의 데이터가 10년 미만으로 예측을 할 수 없습니다.")
            continue

        # 실제값
        ax.plot(ts.index, ts.values, label=f"{country} 실제", color=colors[i], linewidth=2)

        # ARIMA 예측
        try:
            model = ARIMA(ts, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=10)

            last_year = ts.index.max().year
            forecast_years = pd.date_range(start=f'{last_year+1}', periods=10, freq='Y')
            forecast_series = pd.Series(forecast, index=forecast_years)

            ax.plot(forecast_series.index, forecast_series.values,
                    label=f"{country} 예측 (10년)", color=colors[i], linestyle="--")
        except Exception as e:
            st.error(f"{country} 모델 훈련 오류: {e}")

    ax.set_ylabel("전력 소비량 (TWh)")
    ax.set_title("선택 국가별 전력 소비 실제값 및 10년 예측 비교")
    ax.legend()
    st.pyplot(fig)

    # 2) 다중 국가 경제/인구 지표 시각화 (별도 그래프, 겹쳐서 비교)
    st.subheader(f"📊 선택한 국가들의 전력 소비 vs 인구 · GDP 비교")

    fig2, ax2 = plt.subplots(figsize=(14, 7))
    for i, country in enumerate(selected_countries):
        country_df = df[df["country"] == country].copy()
        country_df.set_index("year", inplace=True)

        ax2.plot(country_df.index, country_df["electricity_demand"], label=f"{country} 전력 소비 (TWh)", color=colors[i], linewidth=2)
        ax2.plot(country_df.index, country_df["population"] / 1e6, label=f"{country} 인구 (백만명)", linestyle="--", color=colors[i])
        ax2.plot(country_df.index, country_df["gdp"] / 1e3, label=f"{country} GDP (천억)", linestyle="-.", color=colors[i])

    ax2.set_title("전력 소비 vs 인구 & GDP 비교")
    ax2.set_ylabel("값")
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig2)

    # 3) 에너지 효율성 지표 (여기도 다중국가 가능하게)
    st.subheader("🧠 선택 국가들의 에너지 효율성 지표 비교")

    fig3, ax3 = plt.subplots(figsize=(14, 6))
    for i, country in enumerate(selected_countries):
        country_df = df[df["country"] == country].copy()
        country_df.set_index("year", inplace=True)

        ax3.plot(country_df.index, country_df["energy_per_capita"], label=f"{country} 1인당 에너지 소비", color=colors[i])
        ax3.plot(country_df.index, country_df["energy_per_gdp"], label=f"{country} GDP당 에너지 소비", color=colors[i], linestyle="--")

    ax3.set_title("에너지 효율성 추이 비교")
    ax3.set_ylabel("에너지 단위")
    ax3.legend(loc="upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig3)

    # 4) 지도 시각화는 원본 데이터 전체에서 연도 선택 후 표시 (기존과 동일)
    st.subheader("🌍 세계 국가별 전력 소비량 지도")

    available_years = df["year"].dt.year.sort_values().unique()
    map_year = st.slider("지도로 볼 연도 선택", int(available_years.min()), int(available_years.max()), int(available_years.max()))

    map_df = df[df["year"].dt.year == map_year]
    map_fig = px.choropleth(
        map_df,
        locations="country",
        locationmode="country names",
        color="electricity_demand",
        color_continuous_scale="Blues",
        title=f"{map_year}년 국가별 전력 소비량 (TWh)",
        labels={"electricity_demand": "전력 소비량"}
    )
    st.plotly_chart(map_fig, use_container_width=True)

else:
    st.info("CSV 파일을 업로드하면 예측 및 분석 결과가 표시됩니다.")
