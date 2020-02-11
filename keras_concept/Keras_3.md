## Keras_3 신경망 시작하기

###  목차

#### 1. 딥러닝의 구성 단위

#### 2. 모델 정의 방법

#### 3. 손실함수 , 활성화함수, 옵티마이저

#### 4. 에포크와 배치 크기 이터레이션



#### 1. 딥러닝의 구성 단위

1-1) 텐서 포맷과 데이터 처리 방식

- (samples, features) 크기의 2D 텐서 : 완전 연결 층 (fully connected layer)이나 밀집층(Dense layer)이라고도 불리는 밀집 연결 층(densely connected layer)에 의해 처리되는 경우가 많다. 케라스에서 Dense 클래스

  ```
  from keras import layers
  layer = layers.Dense(32, input_shape= (784,)) #32개의 유닛으로 된 밀집 층
  ```

  

- (samples, timesteps, features) 크기의 3D 텐서(시퀀스 데이터) : LSTM같은 순환  층(recurrent layer)에 의해 처리

- 4D 텐서로 되어있는 이미지 데이터 : 일반적으로 2D 합성곱 층(convolution layer)에 의해 처리 (Conv2D 클래스)



#### 2. 모델 정의 방법

​	1) Sequential 클래스 - 순서대로 쌓아 올린 네트워크

```
from keras import models
from keras import layers

model = model.Sequential()
model.add(layers.Dense(32, activation = 'relu', input_shape=(784,)))
model.add(layers.Dense(10, activation = 'softmax'))
```



​	2) 함수형 API : 임의의 구조를 만들 수 있는 비순환 유향 그래프

 ```
from keras.models import Model
from keras.layers import Input

input_tensor = layers.Input(shape=(784,))
x = Dense(7,activation='relu')(input_tensor)
output_tensor = Dense(10, activation = 'softmax')(x)

model = Model(inputs = input_tensor, outputs= output_tensor) # 다중입력 출력은 리스트형식으로
 ```



#### 3. 손실함수 , 활성화함수, 옵티마이저,지표

```
from keras import optimizers

model.compile(optimizer = RMSprop, loss = 'mse', metrics= ['accuracy'])
```

##### 1). 손실함수  

* 2개의 클래스가 있는 분류 문제에는 : binary crossentropy= Logistic loss = Log loss
* 다중 클래스가 있는 분류 문제에는 : 범주형 크로스엔트로피 (categorical crossentropy) <- 원-핫사용            참고 : sparse_categorical_corssentropy - 범주형 교차엔트로피와 같지만 원-핫 인코딩이 된 상태일 필요없이 정수 인코딩 된 상태에서 수행가능
* 회귀 문제에는 : 평균 제곱 오차 (MSE)
* 시퀀스 학습 문제에는 : CTC (Connection Temporal Classification) - 음성 인식이나 필기 인식처럼 입력에 레이블 할당 위치를 정하기 어려운 연속적인 시퀀스를 다루는 문제에 사용하는 손실 함수

##### 2). 활성화 함수 - 신경망의 출력을 결정하는 식, 이 함수를 이용해 컴퓨터가 이해하기 쉽게 입력데이터를 가공

* 분류 (0/1) : 시그모이드 함수 sigmoid function  

<img src=".\picture\keras_3\sig.JPG" alt="sig" style="zoom:67%;" />



* tanh Function : sigmoid fuction을 보완하고자 나온 함수이다. 입력신호를 (−1,1−1,1) 사이의 값으로 normalization 해준다. 거의 모든 방면에서 sigmoid보다 성능이 좋다.

   <img src=".\picture\keras_3\tan.JPG" alt="tan" style="zoom:67%;" />

   

   

* 다중 분류 : 소프트맥스 함수 softmax function

* ReLU 함수 : 입력값이 0이상이면 입력 값을 그대로 출력하는 함수 , 회귀문제에 주로 사용 , relu 사진 삽입

  <img src=".\picture\keras_3\relu.JPG" alt="relu" style="zoom:67%;" />
  
  

##### 3). 옵티마이저 

<img src=".\picture\keras_3\opm.JPG" alt="opm" style="zoom:67%;" />



##### 4). 지표

​	1) 정확성(accuracy) : 분류모델 (이진분류,다중 클래스 분류) - 예측이 얼마나 정확한지를 의미



#### 4. 에포크와 배치 크기 이터레이션

<img src=".\picture\keras_3\epoch.JPG" alt="epoch" style="zoom:67%;" />



Iteration = batch 수     주의 * : 배치사이즈와 배치 수 헷갈 x

ex) 에포크(Epoch) :  에포크가 20이면 문제집을 20번 풀었다.

ex) 배치크기(Batch size) : 1000문제를 사람이 100문제 단위로 풀고 채점하면 이때 배치 크기는 100 

1 에포크에서 배치크기가 100이면 배치수(iteration) = 1000/100 = 10

