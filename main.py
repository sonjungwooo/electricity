import streamlit as st

def data_visualization_page():
    st.title("데이터 시각화 페이지")
    st.write("전력 소비 데이터를 시각화하는 기능입니다.")
    # 여기에 그래프 그리기 코드 등 추가

def ml_library_usage():
    st.title("머신러닝 라이브러리 활용")
    st.write("Scikit-learn, XGBoost 등의 라이브러리를 활용한 예측 모델입니다.")
    # 학습/예측 관련 코드 및 출력

def power_consumption_prediction():
    st.title("전력 소비량 기반 예측")
    st.write("머신러닝 모델로 미래 전력 소비량을 예측하고 시각화합니다.")
    # 예측 결과 시각화 코드

def main():
    st.sidebar.title("메뉴")
    page = st.sidebar.selectbox("페이지 선택", ["데이터 시각화", "머신러닝 활용", "전력 예측"])

    if page == "데이터 시각화":
        data_visualization_page()
    elif page == "머신러닝 활용":
        ml_library_usage()
    elif page == "전력 예측":
        power_consumption_prediction()

if __name__ == "__main__":
    main()
