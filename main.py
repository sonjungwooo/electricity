import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("🔌 미래 전력 수요 예측기")
st.write("📊 인구, GDP 등의 정보를 바탕으로 미래 전력 수요(TWh)를 예측합니다.")
st.write("💾 먼저 CSV 파일을 업로드해주세요 (예: World Energy Consumption.csv)")

# 1. 파일 업로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    try:
        # 2. 데이터 로드
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV 파일이 성공적으로 로드되었습니다!")
        st.write("데이터프레임 미리보기:", df.head())

        # 3. 사용할 열 확인 및 필터링
        candidate_features = [
            'year', 'population', 'gdp',
            'energy_per_capita', 'electricity_use_per_capita',
            'electricity_demand'
        ]
        available_features = [col for col in candidate_features if col in df.columns]

        # 4. 타겟 열이 있는 경우에만 진행
        if 'electricity_demand' in available_features:
            target = 'electricity_demand'
            features = [f for f in available_features if f != target]
            df = df[features + [target]].dropna()

            # 5. 데이터 분할 및 모델 학습
            X = df[features]
            y = df[target]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)

            # 6. 사용자 입력
            st.subheader("📥 예측 입력값")
            input_data = []
            for feature in features:
                default_val = float(df[feature].mean())
                value = st.number_input(f"{feature}", value=default_val)
                input_data.append(value)

            # 7. 예측 및 출력
            prediction = model.predict(scaler.transform([input_data]))[0]
            st.subheader("🔮 예측 결과")
            st.metric(label="예상 전력 수요 (TWh)", value=f"{prediction:.2f}")

        else:
            st.error("❌ CSV 파일에 'electricity_demand' 열이 존재하지 않습니다. 해당 열이 포함된 파일을 업로드해주세요.")

    except Exception as e:
        st.error(f"🚨 파일 처리 중 오류 발생: {e}")

else:
    st.info("📂 왼쪽 사이드바에서 CSV 파일을 업로드하세요.")
