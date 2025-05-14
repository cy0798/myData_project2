import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")
st.title("🎈나만의 시각화 ML 대시보드!")

# 1. 데이터 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state['data'] = df
    st.write('데이터 미리보기:')
    st.dataframe(df)

    # 2. Plotly 시각화
    st.subheader('📊 Plotly 인터랙티브 그래프')
    x = st.selectbox('X축 선택', df.columns)
    y = st.selectbox('Y축 선택', df.columns)
    color = st.selectbox('컬러 그룹 선택', df.columns)

    fig = px.scatter(df, x=x, y=y, color=color, hover_data=df.columns)
    st.plotly_chart(fig)

    # 3. AI 모델 학습
    st.subheader('🤖 모델 학습 및 저장')
    target = st.selectbox('타겟 컬럼 (예측 대상)', df.columns)
    features = st.multiselect('특징 컬럼들 (입력 변수)', [col for col in df.columns if col != target])

    if st.button('모델 학습하기') and features:
        df_clean = df.dropna(subset=[target] + features).copy()

        # 타겟 인코딩 (문자형이면)
        le_target = None
        if df_clean[target].dtype == 'object':
            le_target = LabelEncoder()
            df_clean[target] = le_target.fit_transform(df_clean[target])

        # 입력 변수 중 문자형 → One-Hot 인코딩
        X = pd.get_dummies(df_clean[features], drop_first=True)
        y = df_clean[target]

        model = LinearRegression()
        model.fit(X, y)

        joblib.dump((model, X.columns, le_target), 'model.pkl')
        st.success('모델 학습 및 저장 완료!')

    # 4. 사용자 입력 → 예측
    st.subheader('🧑‍💻 직접 입력해서 예측하기')
    if 'model.pkl' in st.session_state or st.button("예측 준비"):
        try:
            model, trained_cols, le_target = joblib.load('model.pkl')
            input_data = {}
            for col in features:
                if df[col].dtype == 'object':
                    input_data[col] = st.selectbox(f'{col} 선택', df[col].dropna().unique())
                else:
                    input_data[col] = st.number_input(f'{col} 입력', value=0.0)

            input_df = pd.DataFrame([input_data])
            input_df_encoded = pd.get_dummies(input_df)

            # 누락된 더미 변수 채우기
            for col in trained_cols:
                if col not in input_df_encoded.columns:
                    input_df_encoded[col] = 0
            input_df_encoded = input_df_encoded[trained_cols]

            prediction = model.predict(input_df_encoded)

            # 타겟이 라벨 인코딩된 경우 원래 값으로 복원
            if le_target:
                prediction = le_target.inverse_transform([round(prediction[0])])
                st.success(f'예측 결과: {prediction[0]}')
            else:
                st.success(f'예측 결과: {prediction[0]:.2f}')
        except:
            st.error("모델 또는 입력값 처리에 문제가 발생했습니다.")

    # 5. 데이터 다운로드
    st.subheader('⬇️ 데이터 다운로드')
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="현재 데이터 CSV 다운로드",
        data=csv,
        file_name='다운로드된_데이터.csv',
        mime='text/csv',
    )
