
# 4. 신경망 학습
[손실함수] : 신경망 모델의 예측 값과 실제 값 간의 차이를 측정하는 함수
-> 오차의 합

## MSE(mean square error)
![alt text](image-7.png)

가장 널리 쓰이는 손실함수
회귀(regression)문제에 적합
## CEE(cross entropy error)(CEE)
교차 엔트로피
![alt text](image-8.png)
![alt text](image-9.png)

출력이 softmax를 거쳐나온 "분포"일 경우 활용 가능
분류(classification)문제에 적합


### 4.2.5. 왜 손실 함수를 설정하는가?
신경망을 학습할 때 정확도를 지표로 삼아서는 안 된다. 정확도를 지표로 하면 매개변수의 미분이 대부분의 장소에서 0이 되기 때문이다.

## 4.3. 수치 미분
경사법에서는 기울기 값을 기준으로 나아갈 방향을 정한다.

반올림 오차라는 문제를 일으키기 때문에 중심 차분 혹은 중앙차분을 해줘야함
```python
def numerical_diff(f,x):
    h= 1e-4 #0.0001
    return (f(x+h)-f(x-h))/(2*h)
```

### 4.4 편미분을 벡터로 정의한 것을 기울기(gradient)라고 함.
```python
def numerical_gradient(f,x):
    h=1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx]=f(x)
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad
```

**기울기가 가르키는 방향은 각 장소에서 함수의 출력 값을 가장 크게 줄이는 방향**

### 경사하강법
함수 모양이 복잡하고 변수의 개수가 많을 때 최소값을 구하는 방법  
초기값으로 출발해서 기울기의 음의 방향으로 조금씩 변경.

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
```
f는 최적화하려는 함수, init_x 는 초깃값, lr은 learning rate를 의미하는 학습률, step_nuum은 경사법에 따른 반복 횟수를 뜻합니다. 함수의 기울기는 numerical_gradient(f,x)로 구하고, 그 기울기에 학습률을 곱한 값으로 갱신하는 처리를 step_num번 반복합니다.

### 4.4.2 신경망에서의 기울기
가중치가 W, 손실함수가 L인 신경망
이 때 경사는 