# streamlit을 포함한 프로그램에 필요한 라이브러리를 호출
import streamlit as st
import pandas as pd
import numpy as np

# 리스트 타입을 데이터 
x_real_list = [580, 700, 810, 840] # 공공장소 이용량 데이터
y_real_list = [374, 385, 375, 401] # 일일 확진자 수 데이터
y_pred_list = []                   # 예측한 결괏값을 담을 수 있는 빈 리스트 


# 일일 확진자 수를 예측하는 함수 제작(ml_model)
# 공공장소 이용량(x_real_list)을 전달받습니다. 
def ml_model(x_real_list):
    
    # 반복문을 활용하여 공공장소 이용량의 값을 하나씩 호출하여 1차 함수 수식에 적용합니다. 
    for x in x_real_list :

        # 1차 함수 수식 
        y_pred = x * a + b

        # 연산 후 예측값은 append() 함수를 활용하여 빈 리스트 y_pred_list 에 값을 저장합니다. 
        y_pred_list.append(y_pred)

    return y_pred_list


square_list = []                   # 제곱의 결과를 담을 수 있는 빈 리스트

# Sum of Square Error를 연산하는 함수 제작(sse)
# 실제 정답(y_real_list), 예측 결과(y_pred_list)를 전달받습니다. 
def sse(y_real_list, y_pred_list):
    
    for i in range(len(y_real_list)):

        # 실제 정답과 예측 결과의 편차를 연산합니다.
        diff = y_real_list[i] - y_pred_list[i]
        
        # 편차의 제곱을 수행합니다.
        square = np.square(diff)
        # 편차의 제곱을 모두 더하기 위해 하나의 리스트로 묶습니다.
        square_list.append(square)

    # 편차의 제곱을 모두 더하고, 이를 1/2 로 나눠주며 SSE 를 연산합니다.
    sse = 1/2 * (np.sum(square_list))

    return sse

# Streamlit 
st.title("Covid-19 일일 확진자 수 예측 ML Model")
st.caption("ML Model 구현에 필요한 값을 입력해주세요.")

# 임의의 기울기와 절편을 입력받습니다.
a = st.number_input(label = "임의의 기울기를 입력하여 주세요.(a)",
                    value = 2)
b = st.number_input(label = '임의의 절편을 입력하여 주세요.(b)', 
                    value= 1)

# 앞서 정의한 ml model() 과 sse() 를 실행시켜 결괏값을 변수로 정의합니다. 
y_pred_list = ml_model(x_real_list = x_real_list)
sse_list = sse(y_real_list = y_real_list, y_pred_list = y_pred_list)

# Pandas 를 활용하여 데이터 프레임을 생성합니다. 
df = pd.DataFrame({
    '공공장소 이용량 데이터(x_real_list)' : x_real_list,
    '일일 확진자 수 데이터(y_real_list)' : y_real_list,
    '머신러닝 모델의 예측(y_pred_list)' : y_pred_list,
})

# SSE 와 데이터 프레임을 화면에 출력하기 위한 함수를 사용합니다.
st.dataframe(data=df, use_container_width=True)
st.caption(f"ML Model 의 성능 : {sse_list}")