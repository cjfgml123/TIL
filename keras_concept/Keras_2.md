## Keras_2

###  목차

#### 1.  텐서의 핵심 속성

#### 2.  텐서의 실제 사례

#### 3. 텐서 크기 변환

#### 4. 옵티마이저



##### 1.  텐서의 핵심 속성

1-1) 축의 개수(랭크) : 3D 텐서에는 3개의 축 , 행렬에는 2개의 축

1-2) 크기(shape) 

- ex) : ([[1,2,3],[2,3,4]]) 행렬의 크기 (2,3)
- ex) : ([[[1,2,3],[2,3,4],[3,4,5],[4,5,6]]]) 3D텐서의 크기(3,4,3) : 축,행,열 순으로
- ex) : ([1,2,3]) 벡터의 크기 (3,)
- ex) : (12) 스칼리의 크기 없음

1-3) 데이터 타입 

- float32, uint8. float64 등 , 드물게 char 타입 사용



참고 : 슬라이싱(slicing) - 배열에 있는 특정 원소들을 선택하는 것

- ex) [10:100, : , :]   , [14, :]



##### 2. 텐서의 실제 사례

- 벡터 데이터 : (samples,features) 크기의 2D 텐서(행렬)  ex) 10만명의 나이,번호,소득 (100000,3) 크기의 텐서 

- 시계열 데이터(시퀀스) : (samples ,timesteps ,features) 3D 텐서

![](.\keras_concept\picture\keras_2\keras_2.1.JPG)



- 이미지 : 채널마지막방식(samples, height, width, channels) or 채널우선방식(samples, channels, height, width) 4D 텐서

  ex) 256 * 256 크기의 128개의 흑백 배치는 (128,256,256,1) 컬러는 (128,256,256,3) 

  주의 - 흑백이미지는 하나의 컬러 채널만을 가지고 있어 2D 텐서로 저장될 수 있지만 관례상 이미지 텐서는 항상 3D로 저장 

![image-20200203174129217](C:\Users\student\Desktop\철희\TIL\picture\keras_2\keras_2.2.JPG)

- 동영상 : (samples,frames, height, width, channels) or (samples,frames, channels, height, width) 5D 텐서

ex) 60초 짜리 144 * 256 유튜브 비디오 클립을 초당 4 프레임으로 샘플링하면 240 프레임이 된다. 이런 비디오 클립을 4개 가진 배치는 (4 ,240, 144, 256, 3) : (samples,frames, height, width, channels)

주의 - samples, frames 생략 가능 -> 3D , 4D 텐서로 저장될 수 있다. 



2-1) ReLU 함수

```
kears.layers.Dense(512,activation='relu')
```

- 입력이 0보다 크면 입력을 그대로 반환하고 0보다 작으면 0을 반환  relu(x) = max(x,0)



2-2) 브로드캐스팅(broadcasting) - 작은 텐서가 큰 텐서의 크기에 맞추는 것

- 단계 
- 1) 큰 텐서의 ndim에 맞도록 작은 텐서에 (브로드캐스팅 축이라고 부르는)축이 추가 된다.
- 2) 작은 텐서가 새 축을 따라서 큰 텐서의 크기에 맞도록 반복된다.



#### 3. 텐서 크기 변환 (중요)

```
x = np.array([[0,1],[2,3],[4,5]])
print(x.shape) # (3,2)

x= x.reshape((2,3))
x
# array([[0,1,2],[3,4,5]])

x = np.zeros((300,20)) # 모두 0으로 채워진 (300,20) 크기의 행렬
x = np.transpose(x)  # x[i, : ] 이 x[:, i]로 바뀐다.
print(x) # (20,300)  
```



#### 4. 옵티마이저



![](C:\Users\student\Desktop\철희\TIL\picture\keras_2\keras_2.3.JPG)

4-1) 확률적 경사하강법 - SGD(Stochastic Gradient Descent)- 배치크기가 1인 경사하강법 알고리즘

ex) 확률적 경사 하강법(Stochastic GD, SGD)은 한 번에 전체 데이터가 아닌 \(B\)개 씩을 계산하고 그 정보로 훈련을 진행합니다. 그리고 매번 어떤 데이터들을 쓸 지 전체 데이터 중 무작위로 뽑아 사용합니다. 그래서 확률적이라고 이야기합니다. 예를 들어, 10만 개의 데이터가 있고 100개씩 뽑아 훈련하는데 사용한다면 1000번 훈련하면 한 번 데이터 전체를 다 본 것이 됩니다. 이렇게 한 번 데이터 전체를 다 보는 것을 1 에포크(**epoch**)가 지났다고 하고, 보통 수십~수백 번의 에포크를 거쳐 훈련이 이루어집니다. 매 에포크마다 미니배치를 만드는 순서는 반드시 섞어(**shuffle**) 줘야 합니다!



데이터 전체는 배치(Batch, Full batch)라고 합니다. 데이터의 일부, 즉 \(B\)개는 미니배치(**mini-batch**)라고 하고, \(B\)의 크기를 배치 크기(**batch size**)라고 합니다.    batch size : 샘플 수 

4-2) 모멘텀(momentum) : 관성

- GD 알고리즘의 단점은 기울기 0인 점을 잘 탈출하지 못한다는 것 외에도 너무 훈련이 느리다는 점입니다. 이를 해결하기 위해서 보편적으로 사용되는 방법이 관성(**momentum**)을 적용하는 것

- 관성이란, 변수가 가던 방향으로 계속 가도록 하는 속도(velocity) 항을 추가하는 것으로 바른 방향으로 가고 있다면 점점 더 속도가 빨라지게 되어 더 빨리 훈련이 될 수도 있고 지역해에 빠져도 계속 빠르게 이동해 탈출 할 수 있다.

![](C:\Users\student\Desktop\철희\TIL\picture\keras_2\keras_2.5.JPG)

- 모멘텀을 사용한 옵티마이저 - 모멘텀을 사용한 SGD, Adagrad, RMSProp

  결론 : 모멘텀은 SGD에 있는 2개의 문제점인 수렴 속도와 지역 최솟값을 해결해준다. 

  ```
  keras.optimizers.SGD(lr=0.5, momentum=0.9,nesterov=True)
  # lr  : 학습속도 , 일반적으로 모멘텀은 0.9사용
  ```

  