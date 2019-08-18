# MachineLearning    <br> 
  
## Perceptron
입력 벡터에 대한 출력 함수  
<pre>
if wx + b > 0:  
        return 1  
        *w = weight, b = bias, wx => inner product*  
    else:  
        return 0  
</pre>
  
## MLP(Multilayer Perceptron)  
여러 계층의 퍼셉트론    
처음에는 임의로 할당된 weight를 사용하여 활성화 한 이후 예측 값과 실제 값을 비교하여 오차를 계산하여 다시 앞 퍼셉트론으로 전달, 적절한 optimizer를 통해 오차를 줄이는 방향으로 weight 조정.  
출력에 대한 weight 및 bias 조정 시 활성화 함수 이용하는 것이 좋음.  

<br>

## CNN(Convolutional Neural Network)  
합성곱 신경망  
Layer 많으면 DCNN, 적은 Training data에 대해 좋은 효율을 보임  
  
<br>

## GAN(Generative Adversarial Networks)  //
적대적 생성 네트워크  
일반적으로 두 개의 네트워크로 구성할 때, 하나는 진짜처럼 보이는 합성 데이터를 위조하게 학습하고, 다른 하나는 위조한 데이터와 진짜 데이터를 구분하게 학습  
두 네트워크간 경쟁을 통한 향상  
- DCGAN - Deep Convolutional Generative Adversarial Networks  
CNN 활용하여 위조 생산 

<br>

## Word Embedding   //
One-Hot Encoding - 단어 간 유사성 표현 불가능  
NLP(Natural Language Processing)  
- word2vec
- GloVe  

<br>

## RNN(Recurrent Neural Network)   //
순환 신경망  
다중 퍼셉트론 신경망의 경우 모든 입력 변수가 독립이라 가정하지만, 시계열 데이터와 같은 순차적인 데이터의 경우 과거 데이터에 대해 의존성이 있다.  
순환 신경망의 셀에서는 특정 시점에 관한 함수로 은닉 상태 값과 입력 값을 표현  
RNN, LSTM, GRU  

<br>

## Autoencoder  
비지도학습  
역전파 이용 입력값 복원, 입력 차원과 출력 차원이 같다. 차원 압축 과정에서 중요한 정보들만 저장하는 것이 복원률이 좋으므로 이에 맞게 학습.    

<br>

## Supervised vs Unsupervised  
<pre>
Supervised - 정답지가 있음, 출력값과 label 비교  
    개는 1, 고양이는 0, 정답지 존재  
        x1 -> 1  
        x2 -> 0  
        x3 -> 1  
        ...  
Unsupervised - 정답 x, (k-means, density-based, etc)clustering / (in time-series)anomaly detector //  
    clustering  
        x1 -> group 1  
        x2 -> group 2  
        x3 -> group 1  
        ...  
    anomaly detector  
        x1 -> 0  
        x2 -> 0  
        x3 -> 0  
        x4 -> 0.1  
        **x5 -> 0.9**  
        x6 -> 0.3  
        ...  
Semisupervise - 개에 대해서 학습, 다른 data도 들어옴  
    x1 -> 1  
    x2 -> 1  
    x3 -> 1  
    x4 -> 1  
    a1 -> not 1  
</pre>

<br><br>

---------
## Use 

  <br>
  
### Data Preprocess
- 학습 시 필요한 data  
    - Training data, Validation data, Test data(+ label if classifier/supervised) 
    - data normalization 시 최댓값으로 나눠주는 것이 아니라, 0.9 또는 0.95값으로 나눠주는 것이 좋다(quantile)  
    
<br>

### Modeling
- PCA
    - 주성분 분석, 모든 벡터에 대해 분석하여 차원 축소  
    ```python
    pca = PCA(k)
    pca.fit(target_data)
    
    pca.transform(test_data)    
    ```
- MLP
    - bias, 활성화 함수가 없을 경우, 모든 벡터에 대해 가중치를 계산하므로 PCA와 결과값이 거의 비슷하게 나옴.
    - bias, 활성화 함수가 있을 경우, weight과 bias가 변하면서 영향력이 큰 인자에 대해 weight 및 bias가 조정됨
    ```python
    input_layer = Input(shape=(n, n)) or input_layer = Input(shape(m, ))
    flatten_layer = Flatten()(input_layer)
    ... hidden layers
    encoding_layer = Dense(k)(hidden_layer_x)
    
    encoder = Model(input_layer, encoding_layer)
    encoder.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy')  
    
    # check params
    encoder.summary() 
    
    encoder.fit(training_data, training_label, validation_data = (validation_data, validation_label), epochs = 1000, callbacks=[early_stopping, tensorboard]) # callbacks 내부 인자는 미리 선언해야함
    
    encoder.evaluate()
    ```
- CNN
    - Deep Learning, 이미지의 특성이 강한 경우 좋다.
    - 층이 깊어질수록 filter 수를 많이 하는 편
    - 압축이 목적이라면 stride 옵션을 따로 주거나 filter size를 크게 하여 다음 filter들의 크기를 작아지게 하거나, Pooling을 이용
    - 비교적 적은 parameter 사용, 많은 resource 소모
     ```python
    input_layer = Input(shape=(n, n, 1))
    conv1_layer = Conv2D(10, (3, 3), padding="same")(input_layer)
    mxp1_layer = MaxPooling2D((2, 2), padding="same")(conv1_layer) 
    ....
    encoding_layer = Conv2D(1, (3, 3), padding='same')(mxp3_deep)
    
    encoder = Model(input_layer, encoding_layer)
    encoder.compile(optimizer='rmsprop', loss='binary_crossentropy') 
    
    # check params
    encoder.summary() 
    
    encoder.fit(training_data, training_label, validation_data = (validation_data, validation_label), epochs = 1000, callbacks=[early_stopping, tensorboard]) # callbacks 내부 인자는 미리 선언해야함
    
    encoder.evaluate()
    ```
  
- In Model  
    - 모델 생성 시 parameter 수는 data 수에 비해 적은게 좋다(복잡한 모델 - 시간 소모 크고 overfitting 가능성 높음)  
    - Input layer 입력층  
    - Flatten layer (필요하다면) (n, n)행렬을 (n**2, 1) 행렬로  
    - Hidden layer 은닉층, Hidden layer 중첩을 통해 Deep Learning  
    - Output layer 출력층, 출력 값에 맞는 활성화 함수 사용(sigmoid, softmax, etc)  
    - Dropout 네트워크 내부로 전달되는 값 중 일부를 드롭아웃 확률로 무작위로 제거하여 성능 향상  
    - summary를 통해 parameter 수 확인 
    - 활성화함수
        - sigmoid: -inf to 0, inf to 1, (0,1)의 출력값을 갖는 출력층에 적절, [True, False]  
        - tanh: -inf to -1, inf to 1, (-1,1)  
        - ReLU: 은닉층에 적절, 활성화 함수 없이 사용할 경우 단순 PCA와 비슷한 결과  
        <pre>
        if x<0:  
            return 0  
        else:  
            return ax
        </pre>
        - softmax: 확률분포함수로 2개 이상의 분류 모델 출력층에 적절, [a, b, c, ...]  
  
- Compile(Model.compile)  
    - loss:  
        - mean squared error - 일반적으로 사용  
        - binary crossentropy - 출력 값과 멀어질수록 큰 penalty 받음, binary   
        - categorical crossentropy - 출력 값과 멀어질 수록 큰 penalty 받음, >2  
    - Metric:  
        - Accuracy(정확도) - 타겟을 정확히 예측한 비율  
        - Precision(정밀도) - 긍정이라고 예측한 것 중 실제로 참인 것의 비율  
        - Recall(재현율) - 올바르게 예측한 것 중 긍정으로 예측한 것이 실제로 참인 경우의 비율  
    - optimizer:  
        - SGD(Stochastic Gradient Descent) - 확률적 경사하강법  
        - RMSprop, Aam - SGD에 관성개념 포함, 많은 계산 비용 필요 시 빠른 수렴 가능  
  
- Learning(Model.fit)  
    - epochs - 모델이 학습 데이터셋 전체를 살펴본 횟수, 각 반복마다 optimizer를 통해 목적 함수가 최소화되도록 가중치 조정 // 모델의 복잡도   
    - batch_size - 옵티마이저가 weight 업데이트 하기 전까지 살펴본 학습 데이터의 수 // 수렴성  
    - Regularization  
        - kernel_regularizer weight matrix에 적용되는 일반화 함수  
        - bias_regularizer bias vector에 적용되는 일반화 함수  
        - activity_regularizer 계층의 출력에 적용되는 일반화 함수  
    - callback - Tensorboard, earlystopping  
  
  
- After Learning  
    - evaluate - 손실 값 계산  
    - predict_classes - 범주 출력 계산  
    - predict_proba - 범주 확률  
    
  <br>
  

## 분석 방법  

 <br> 
 
### tSNE
- 두 데이터의 분포 2차원 평면에 나타냄, 거리 정보 활용.  
  
### Confusion matrix  
  
|           | 식별 결과 양성 | 식별 결과 음성 | 합계 |    
| :-------: | :-------: | :-------: | :--------: |    
| 실제 양성 | TP(True Positive) | FN(False Negative) | TP + FN |    
| 실제 음성 | FP(False Positive) | TN(True Negative) | FP + TN |    
| 합계 | TP + FP | FN + TN | ALL |      
  
    - TP / ( TP + FN ) = 진양성률 = 민감도  
    - TP / ( TP + FP ) = 정밀도  
    - TP / ( TP + FN ) = 재현율   
    - FN / ( TP + FN ) = 거짓음성률  
    - FP / ( FP + TN ) = 거짓양성률  
    - TN / ( FP + TN ) = 진음성률 = 특이도   
    - 민감도 + 거짓음성률 = 1  
    - 특이도 + 거짓양성률 = 1  
    - Accuracy = ( TP + TN ) / ALL  
    - F measure = 2 * (정밀도 * 재현율) / (정밀도 + 재현율)  
  
### Receiver Operating Characteristic curve(ROC curve)  
    - 식별 결과가 (0, 1) 사이로 나온 경우 식별 결과와 정답 세트를 식별 결과의 점수 순서로 정렬하여 임계치를 설정하고, 임계치보다 위를 양성으로 하여 혼동행렬로 나타낸다면 진양성률과 거짓양성률을 계산할 수 있음.   
    - x축을 거짓양성률(1-특이도, False Positive Rate), y축을 진양성률(민감도, True Positive Rate)로 하여 임계치를 변화시키면 ROC curve 만들 수 있음.  
    - 기준 라인 NIR(No Information Rate) - 멍청한 모델로 만든 경우의 곡선  
    - AUC(Area Under Curve): ROC 곡선의 아랫부분 면적 값, 0.9 이상이면 정확도(accuracy)가 높음.  
    - Youden Index - AUC 값과 길이 0.5의 대각선 사이 거리 b가 가장 멀 때의 진양성률 + 거짓양성률, 클 수록 좋은 모델  
    
#### Confusion Matrix Example
  
Model 1 
  
|           | 식별 결과 개 | 식별 결과 고양이 | 합계 |    
| :-------: | :-------: | :-------: | :--------: |    
| 실제 개 | TP = 5000 | FN = 5000 | 10000 |    
| 실제 고양이 | FP = 5000 | TN = 5000 | 10000 |    
| 합계 | 10000 | 10000 | 20000 |    
  
실제 개 중 개라고 대답할 확률 -> 민감도 = 5000 / 10000 = 0.5  
개라고 대답할 때 개일 확률 -> 정밀도 = 5000 / 10000 = 0.5  
실제 고양이 중 고양이라고 대답할 확률 -> 특이도 = 5000 / 10000 = 0.5  
전체 정답 중 개의 비율 -> 재현율 = 5000 / 10000 = 0.5  
고양이라고 대답할 때 고양이가 있을 확률 - 5000 / 10000 = 0.5  
Accuracy = 10000 / 20000 = 0.5  
F = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5  
  
Model 2  
  
|           | 식별 결과 개 | 식별 결과 고양이 | 합계 |    
| :-------: | :-------: | :-------: | :--------: |    
| 실제 개 | TP = 9000 | FN = 1000 | 10000 |    
| 실제 고양이 | FP = 9000 | TN = 1000 | 10000 |    
| 합계 | 18000 | 2000 | 20000 |   
  
실제 개 중 개라고 대답할 확률 -> 민감도 = 9000 / 10000 = 0.7  
개라고 대답할 때 개일 확률 -> 정밀도 = 9000 / 18000 = 0.5  
실제 고양이 중 고양이라고 대답할 확률 -> 특이도 = 1000 / 10000 = 0.1  
전체 정답 중 개의 비율 -> 재현율 = 9000 / 10000 = 0.9  
고양이라고 대답할 때 고양이가 있을 확률 - 5000 / 10000 = 0.5  
Accuracy = 10000 / 20000 = 0.5  
F = 2 * ( 0.5 * 0.9 ) / ( 0.5 + 0.9 ) = 0.64285  
  
Model 3  
   
|           | 식별 결과 개 | 식별 결과 고양이 | 합계 |    
| :-------: | :-------: | :-------: | :--------: |    
| 실제 개 | TP = 7000 | FN = 3000 | 10000 |    
| 실제 고양이 | FP = 1000 | TN = 9000 | 10000 |    
| 합계 | 8000 | 12000 | 20000 |    
   
실제 개 중 개라고 대답할 확률 -> 민감도 = 7000 / 10000 = 0.7  
개라고 대답할 때 개일 확률 -> 정밀도 = 7000 / 8000 = 0.875  
실제 고양이 중 고양이라고 대답할 확률 -> 특이도 = 9000 / 10000 = 0.9  
전체 정답 중 개의 비율 -> 재현율 = 7000 / 10000 = 0.7  
고양이라고 대답할 때 고양이가 있을 확률 -> 9000 / 12000 = 0.75  
Accuracy = 16000 / 20000 = 0.8  
F = 2 * ( 0.875 * 0.7 ) / (0.875 + 0.7 )= 0.77778 
  
Model 4  
   
|           | 식별 결과 개 | 식별 결과 고양이 | 합계 |    
| :-------: | :-------: | :-------: | :--------: |    
| 실제 개 | TP = 9000 | FN = 1000 | 10000 |    
| 실제 고양이 | FP = 3000 | TN = 7000 | 10000 |    
| 합계 | 12000 | 8000 | 20000 |    
  
실제 개 중 개라고 대답할 확률 -> 민감도 = 9000 / 10000 = 0.9  
개라고 대답할 때 개일 확률 -> 정밀도 = 9000 / 12000 = 0.75  
실제 고양이 중 고양이라고 대답할 확률 -> 특이도 = 7000 / 10000 = 0.7  
전체 정답 중 개의 비율 -> 재현율 = 9000 / 10000 = 0.9  
고양이라고 대답할 때 고양이가 있을 확률 - 7000 / 8000 = 0.875  
Accuracy = 16000 / 20000 = 0.8  
F = 2 * ( 0.75 * 0.9) / (0.75 + 0.9) = 0.81818  
  
Model 5  
   
|           | 식별 결과 개 | 식별 결과 고양이 | 합계 |    
| :-------: | :-------: | :-------: | :--------: |    
| 실제 개 | TP = 0 | FN = 10000 | 10000 |    
| 실제 고양이 | FP = 0 | TN = 10000 | 10000 |    
| 합계 | 0 | 20000 | 20000 |    
  
실제 개 중 개라고 대답할 확률 -> 민감도 = 0 / 10000 = 0  
개라고 대답할 때 개일 확률 -> 정밀도 = 0 / 0 = 0  
실제 고양이 중 고양이라고 대답할 확률 -> 특이도 = 10000 / 10000 = 1.0  
전체 정답 중 개의 비율 -> 재현율 = 0/ 10000 = 0  
고양이라고 대답할 때 고양이가 있을 확률 - 10000 / 20000 = 0.5  
Accuracy = 10000 / 20000 = 0.5  
F = 2 * ( 0 * 0) / (0 + 0) = 0  
  
Model 6 
   
|           | 식별 결과 개 | 식별 결과 고양이 | 합계 |    
| :-------: | :-------: | :-------: | :--------: |    
| 실제 개 | TP = 10000 | FN = 0 | 10000 |    
| 실제 고양이 | FP = 10000 | TN = 0 | 10000 |    
| 합계 | 20000 | 0 | 20000 |    
  
실제 개 중 개라고 대답할 확률 -> 민감도 = 10000 / 10000 = 1.0  
개라고 대답할 때 개일 확률 -> 정밀도 = 10000 / 20000 = 0.5  
실제 고양이 중 고양이라고 대답할 확률 -> 특이도 = 0 / 10000 = 0  
전체 정답 중 개의 비율 -> 재현율 = 10000 / 10000 = 1.0  
고양이라고 대답할 때 고양이가 있을 확률 - 0 / 0 = 0  
Accuracy = 10000 / 20000 = 0.5  
F = 2 * ( 1 * 0.5) / (1 + 0.5) = 0.66667  
  
Model 7  
   
|           | 식별 결과 개 | 식별 결과 고양이 | 합계 |    
| :-------: | :-------: | :-------: | :--------: |    
| 실제 개 | TP = 10000 | FN = 0 | 10000 |    
| 실제 고양이 | FP = 0 | TN = 10000 | 10000 |    
| 합계 | 10000 | 10000 | 20000 |    
  
실제 개 중 개라고 대답할 확률 -> 민감도 = 10000 / 10000 = 1.0  
개라고 대답할 때 개일 확률 -> 정밀도 = 10000 / 10000 = 1.0  
실제 고양이 중 고양이라고 대답할 확률 -> 특이도 = 10000 / 10000 = 1.0  
전체 정답 중 개의 비율 -> 재현율 = 10000 / 10000 = 1.0  
고양이라고 대답할 때 고양이가 있을 확률 - 10000 / 10000 = 1.0  
Accuracy = 20000 / 20000 = 1.0  
F = 2 * ( 1.0 * 1.0) / (1.0 + 1.0) = 1.0  
  

