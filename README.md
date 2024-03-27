# 대전광역시 시내버스 승차 인원 예측

### 진행 배경

---

한남대학교에서 진행한 『지역사회 문제해결형 빅데이터 / AI 활용 공모전』에 참가하기 위해 진행하였습니다. 데이터는 위치 파라미터 데이터, 대전광역시 시내버스 승하차 데이터 (날짜, 위치, 시간, 승차인원, 하차인원)로 진행하였습니다. 
<br><br>
### 사용한 도구(Tool)

---

- python : 데이터 전처리 및 예측 모델제작할 때 사용하였습니다.
    - catboost를 이용하여 예측모델을 제작하였습니다.
<br><br>
### ML 제작

---

**최적의 하이퍼파라미터 모델의 검증 결과 시각화**

시내버스 이용객이 ‘요일’, 특히 ‘평일’과 ‘주말’에 차이가 나는지 확인하기 위해 일부 요일의 데이터로 검증과정을 진행하였으며, 그 결과 평일이 주말보다 이용객 수가 더 많다는 것을 확인하였습니다. 

따라서 ‘요일’ 칼럼을 추가하여 데이터를 재정비한 뒤, 요일, 시간, 하차인원을 주요 학습 데이터로 삼아서 승차 인원을 예측하도록 하였고, 그렇게 하여 임시 제작한 예측 모델의 실제 승차 인원과 예측 결과의 비교 그래프입니다.
![image03](https://github.com/hw20200500/Bus_traffic_forecasting_contest/assets/117514148/5fa71197-8002-4293-8b04-3da5172fa706)
![image02](https://github.com/hw20200500/Bus_traffic_forecasting_contest/assets/117514148/a7946840-855d-4175-b674-0843437d96c7)
![image01](https://github.com/hw20200500/Bus_traffic_forecasting_contest/assets/117514148/0e958e21-968c-4b57-8c0f-73ceaa5dea02)


<br><br>
최종 모델 제작

```jsx
# 독립 변수(X)와 종속 변수(y) 설정
warnings.filterwarnings("ignore")

x_train = data[['gid', 'TIME', 'DAY','ALIGHT_DEMAND']]
y_train = data['RIDE_DEMAND']

features = [0,1,2]

train_dataset = cb.Pool(x_train, y_train, cat_features = features) 
# test_dataset = cb.Pool(x_test, y_test, cat_features = features)

model = cb.CatBoostRegressor(iterations=800, depth=10, learning_rate=0.2, loss_function= 'RMSE', l2_leaf_reg = 1, task_type="CPU")

model.fit(train_dataset, verbose_eval=10)
```
![98_29%모델손실함수그래프](https://github.com/hw20200500/Bus_traffic_forecasting_contest/assets/117514148/f33ab327-e76e-4664-8cd1-64c1b4cc2e46)

