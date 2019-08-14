# Keras  
* Modularity: 라이브러리에 다양한 종류의 신경망 계층, 비용 함수, 옵티마이저, 초기화 스키마, 활성화 함수, 정규화 스키마들이 구현돼 있으므로 독립적인 모듈들을 레고 블록처럼 결합하여 신경망 구축.  
* Minimalism: 라이브러리는 파이썬으로 구현됨. 각각의모듈은 간단하고 바로 이해됨.
* 쉬운 확장성: 새로운 기능을 갖게 확장하기 쉽다.

## In Keras

### Tensor
- 다차원 배열 또는 행렬, tensorflow와 theano를 통해 기본 빌딩 블록인 텐서에 대해 효율적인 연산 가능

### Keras Model
- Sequential
```
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape(784,)))
model.add(Acivation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
mdoel.add(Dropout(DROPOUT))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()
```
- Functional
    - DAG(Directed Acyclic Graph), 공유 계층이 있는 모델, 다중 출력 모델 등

### 사전 정의 신경망
- Dense
> 완전 연결 신경망 계층.  

- Recurrent Neural Network
> 기본, LSTM, GRU  

- Convolution Layer, Pooling Layer
> 합성곱과 풀링 연산을 사용해 추상화 단계를 올림. 인간의 시각 모델과 유사.   

- Regularization
> overfitting을 방지하는 방법. regularizer parameter가 있으며 Dropout도 사용한다.

- Batch normalization
> 학습 속도를 더 빠르게 하고 일반적으로 더 나은 정확도를 달성하는 방법

- Activation Function
> sigmoid, ReLU, softmax, tanh, etc

- Loss function
> Accuracy는 분류 문제에 사용. 여러 종류의 Accuracy가 있으므로 목적에 맞게 사용
> 오차 손실은 예측값과 실제 관찰 값 차이 측정. mean squared error 류
> 힌지 손실은 일반적으로 분류기 학습하는데 사용.
> 범주 손실은 분류 문제에서 크로스엔트로피 계산.

- Metric
> 목적 함수와 유사하지만 모델 학습 시 메트릭 평가 결과를 사용하지 않음

- Optimizer
> SGD, RMSprop, Adam 등

- Save/Load Model
> Structure

```python
# JSON으로 저장
json_string = model.to_json()

# YAML로 저장
yaml_string = model.to_yaml()

# JSON으로 모델 복구
from keras.models import model_from_json
model = model_from_json(json_string)

# YAML로 모델 복구
model = model_from_yaml(yaml_string)
```
> Parameter
```python
from keras.models import load_model
#HDF5파일 'my_model.h5 생성'
model.save('my_model.h5')

#컴파일된 모델을 리턴, 이전 모델과 동일
model = load_model('my_model.h5')
```

- customizing
> callbacks을 통해 EarlyStopping, Tensorboard 등 사용.
