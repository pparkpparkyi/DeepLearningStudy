
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



---
# 🧠 신경망 학습 정리

## 📘 1. 신경망 학습이란?

- **학습(Training)**:  
  훈련 데이터로부터 **가중치(Weight)와 편향(Bias)** 의 **최적값을 찾는 과정**  
- **목표**: 손실함수(Loss Function)를 최소화하는 방향으로 매개변수를 갱신하는 것  

---

## ⚖️ 2. 손실 함수 (Loss Function)

### 🎯 정의
- 신경망이 예측한 값 `y`와 실제 정답 `t` 간의 차이를 수치로 표현한 함수  
- **손실이 작을수록 모델이 잘 학습된 상태**

---

### (1) 오차 제곱합 (SSE, Sum of Squares for Error)

$
E = \frac{1}{2} \sum_k (y_k - t_k)^2
$

- 주로 **회귀(regression)** 문제에서 사용  
- 원-핫 인코딩된 출력에서 각 클래스 차이를 제곱해 더함  

```python
def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)
````

---

### (2) 교차 엔트로피 오차 (CEE, Cross Entropy Error)

$
E = - \sum_k t_k \log(y_k)
$

* **분류(classification)** 문제에서 주로 사용
* 정답 클래스의 확률이 높을수록 손실이 작아짐

```python
def cross_entropy_error(y, t):
    delta = 1e-7  # log(0) 방지
    return -np.sum(t * np.log(y + delta))
```

---

### (3) 미니배치 손실

$
E = -\frac{1}{N} \sum_n \sum_k t_{nk} \log(y_{nk})
$

* 전체 데이터를 한 번에 쓰지 않고, **일부(mini-batch)** 만 사용하여 계산
* 학습 효율 향상 + 일반화 성능 증가

---

### ❗ 손실 함수를 사용하는 이유

* **정확도(accuracy)** 는 미분 불가능한 구간이 많아 **경사 계산 불가**
* 손실함수는 **연속적이고 미분 가능**, → **경사하강법(Gradient Descent)** 으로 최적화 가능

---

## 🧮 3. 미분(Derivative)

### (1) 해석적 미분

$$\frac{df(x)}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

* 실제 수학적 정의이지만, 컴퓨터에서는 **반올림 오차 발생**

### (2) 수치 미분 (Numerical Differentiation)

* 근삿값으로 미분을 구하는 방법

$$\frac{df(x)}{dx} \approx \frac{f(x + h) - f(x - h)}{2h}$$

```python
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)
```

---

### (3) 편미분 (Partial Derivative)

$$f(x_0, x_1) = x_0^2 + x_1^2$$

→ ( x_0 )에 대한 편미분:  
$$\frac{\partial f}{\partial x_0} = 2x_0$$

* 여러 변수 중 하나만 변할 때의 변화율
* 신경망에서는 모든 가중치에 대해 편미분 수행

---

## 📉 4. 기울기 (Gradient)

* **기울기(Gradient)**: 함수의 출력 값을 가장 빠르게 감소시키는 방향의 벡터
* 손실함수를 줄이려면 → **기울기의 반대 방향으로 이동**

```python
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp = x[idx]
        x[idx] = tmp + h
        fxh1 = f(x)
        x[idx] = tmp - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp
    return grad
```

---

## 🧭 5. 경사하강법 (Gradient Descent)

$$x_{new} = x - \eta \frac{\partial f}{\partial x}$$

* $\eta$: **학습률 (Learning Rate)** — 갱신 폭을 조절하는 하이퍼파라미터

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
```

---

## 🧩 6. simpleNet 예시

```python
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규분포 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss
```
softmax와 cross_entropy_error를 이용해 손실 계산

W: 학습 대상 파라미터

loss: 손실함수

predict: 입력 → 출력 변환

* $dW = ∂L/∂W$ : 손실 함수 L을 W로 미분한 값 (기울기)
* $dW[i, j] > 0$ → 해당 가중치가 손실 증가 방향
* $dW[i, j] < 0$ → 손실 감소 방향 (이쪽으로 업데이트)

---

## 🔁 7. 신경망 학습 알고리즘 요약

| 단계            | 설명                             |
| :------------ | :----------------------------- |
| **① 미니배치 선택** | 훈련데이터 일부를 무작위로 선택 (Mini-Batch) |
| **② 기울기 계산**  | 각 가중치에 대해 손실함수의 기울기 계산         |
| **③ 매개변수 갱신** | 기울기 반대 방향으로 가중치 갱신             |
| **④ 반복**      | 위 과정을 여러 번 반복하여 손실 최소화         |

📌 미니배치를 사용하므로
이 알고리즘은 **확률적 경사하강법 (SGD, Stochastic Gradient Descent)** 이라고 부름.

---

## 💡 핵심 요약

| 개념           | 핵심 내용                  |
| :----------- | :--------------------- |
| **손실함수**     | 모델 예측값과 실제값의 차이 측정     |
| **미분 / 기울기** | 손실을 줄이는 방향을 찾기 위한 도구   |
| **경사하강법**    | 기울기의 반대 방향으로 매개변수 갱신   |
| **미니배치**     | 효율적이고 일반화에 강한 학습 방식    |
| **학습률 (lr)** | 너무 크면 발산, 너무 작으면 수렴 느림 |

---

## 🚀 한 줄 정리

> **신경망 학습은**
> “손실함수를 최소화하기 위해 기울기를 계산하고,
> 그 반대 방향으로 가중치를 반복 갱신하는 과정”이다.
