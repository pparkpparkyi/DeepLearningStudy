# 신경망 (Neural Network)
신경망은 매개변수의 적절한 값을 데이터로부터 자동으로 학습ㅂ하는 능력을갖추고 있음. 

신경망의 개요   
신경망이 입력데이터가 무엇인지, 식별하는 처리 과정.  
가중치와 매개변수 값을 학습하는 방법은 Neural Network Suttdy파트에서

---
## 정리 가능한 주제들

### 1. 기본 개념
- 신경망의 구조 (입력층, 은닉층, 출력층)
- 뉴런과 가중치
- 순전파 (Forward Propagation)

### 2. 활성화 함수
- 계단 함수 (Step Function)
- 시그모이드 (Sigmoid)
- ReLU (Rectified Linear Unit)
- tanh, Softmax 등

### 3. 학습 알고리즘
- 손실 함수 (Loss Function)
- 경사 하강법 (Gradient Descent)
- 역전파 (Backpropagation)

### 4. 신경망 구현
- NumPy를 이용한 구현
- 행렬 연산
- 배치 처리

### 5. 실전 응용
- MNIST 손글씨 인식
- 다중 분류 문제
- 과적합 방지 기법

---
# Neural Network (신경망) - 코드 중심 정리

## 개요

### 활성화 함수의 역할

활성화 함수는 **임계값을 경계로 출력값을 결정**한다.

**퍼셉트론의 계단 함수:**
$$h(x) = \begin{cases} 
0, & (x \leq 0) \\ 
1, & (x > 0) 
\end{cases}$$

**신경망의 일반적인 형태:**
$$a = b + w_1x_1 + w_2x_2$$
$$y = h(a)$$

### 🔑 비선형 함수를 사용해야 하는 이유

> **선형 함수를 사용하면 신경망의 층을 깊게 하는 의미가 없어진다!**

**증명:**
- 선형 함수 $h(x) = cx$를 사용한다면
- 3층 신경망: $y = h(h(h(x))) = c \cdot c \cdot c \cdot x = c^3x$
- 이는 결국 $y = ax$ 형태의 1층 신경망과 동일

따라서 **계단 함수**와 **시그모이드 함수** 모두 **비선형 함수**를 사용함.

---

## 1. Step Function (계단 함수)

### 기본 구현 (배열 처리 불가)

```python
def step_function(x):
    """실수만 받을 수 있는 버전"""
    if x > 0:
        return 1
    else:
        return 0

# 문제: 배열을 입력으로 받을 수 없음
```

### NumPy를 이용한 구현 (배열 처리 가능)

```python
import numpy as np

def step_function(x):
    """배열을 받을 수 있는 버전"""
    y = x > 0
    return y.astype(int)

# 또는
def step_function(x):
    return np.array(x > 0, dtype=int)

# 테스트
x = np.array([-1.0, 1.0, 2.0])
print(step_function(x))  # [0 1 1]
```

### 동작 원리

```python
import numpy as np

x = np.array([-1.0, 1.0, 2.0])
print(x > 0)           # [False  True  True]
y = x > 0
print(y.astype(int))   # [0 1 1]
```

---

## 2. Sigmoid Function (시그모이드 함수)

### 수식

$$h(x) = \frac{1}{1 + \exp(-x)}$$

### 구현

```python
import numpy as np

def sigmoid(x):
    """시그모이드 함수"""
    return 1 / (1 + np.exp(-x))

# 테스트
x = np.array([-1.0, 0.0, 1.0, 2.0])
print(sigmoid(x))
# [0.26894142 0.5 0.73105858 0.88079708]
```

### 시각화

```python
import matplotlib.pyplot as plt

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.title('Sigmoid Function')
plt.grid()
plt.show()
```

---

## 3. ReLU Function (Rectified Linear Unit)

### 개념

**입력이 0을 넘으면** 그 입력을 그대로 출력하고, **0 이하면 0을 출력**

$$\text{ReLU}(x) = \max(0, x) = \begin{cases}
x, & (x > 0) \\
0, & (x \leq 0)
\end{cases}$$

### 구현

```python
import numpy as np

def relu(x):
    """ReLU 함수"""
    return np.maximum(0, x)

# 테스트
x = np.array([-1.0, 0.0, 1.0, 2.0])
print(relu(x))  # [0. 0. 1. 2.]
```

---

## 4. 3층 신경망 구현

### 출력층 활성화 함수 선택

| 문제 유형 | 활성화 함수 |
|:---------|:-----------|
| 회귀 | 항등 함수 (Identity) |
| 2클래스 분류 | Sigmoid |
| 다중 클래스 분류 | Softmax |

---

### 4.1 입력층 → 1층

```python
import numpy as np

# 입력
X = np.array([1.0, 0.5])

# 가중치와 편향
W1 = np.array([[0.1, 0.3, 0.5],
               [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(f"W1 shape: {W1.shape}")  # (2, 3)
print(f"X shape: {X.shape}")    # (2,)
print(f"B1 shape: {B1.shape}")  # (3,)

# 1층 계산
A1 = np.dot(X, W1) + B1  # 가중합
Z1 = sigmoid(A1)          # 활성화 함수 적용

print(f"A1: {A1}")
print(f"Z1: {Z1}")
```

---

### 4.2 1층 → 2층

```python
# 가중치와 편향
W2 = np.array([[0.1, 0.4],
               [0.2, 0.5],
               [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(f"Z1 shape: {Z1.shape}")  # (3,) - 1층의 뉴런 수
print(f"W2 shape: {W2.shape}")  # (3, 2)
print(f"B2 shape: {B2.shape}")  # (2,)

# 2층 계산
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

print(f"A2: {A2}")
print(f"Z2: {Z2}")
```

---

### 4.3 2층 → 출력층

```python
def identity_function(x):
    """항등 함수 - 입력을 그대로 출력"""
    return x

# 가중치와 편향
W3 = np.array([[0.1, 0.3],
               [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

# 출력층 계산
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)  # 또는 Y = A3

print(f"Y: {Y}")
```

> **Note:** 출력층의 활성화 함수는 풀고자 하는 문제의 성질에 맞게 정의
> - **회귀**: 항등 함수
> - **2클래스 분류**: Sigmoid
> - **다중 클래스 분류**: Softmax

---

## 5. 신경망 구현 정리

### 전체 코드

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def init_network():
    """신경망 초기화"""
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5],
                              [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4],
                              [0.2, 0.5],
                              [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3],
                              [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward(network, x):
    """순전파 (Forward Propagation)"""
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    # 1층
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    
    # 2층
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    
    # 출력층
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y

# 실행
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(f"출력: {y}")
```

---

## 6. Softmax Function

### 특징

- 출력값이 **0에서 1.0 사이의 실수**
- 출력의 **총합이 1**
- 따라서 **확률로 해석 가능**

### 수식

$$y_k = \frac{\exp(a_k)}{\displaystyle\sum_{i=1}^{n}\exp(a_i)}$$

### 기본 구현

```python
def softmax(a):
    """소프트맥스 함수 (기본)"""
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

# 테스트
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)           # [0.01821127 0.24519181 0.73659691]
print(np.sum(y))   # 1.0
```

### 오버플로 문제

```python
# 오버플로 발생
a = np.array([1010, 1000, 990])
print(np.exp(a))  # [inf inf inf] - 오버플로!
```

### 개선된 구현 (오버플로 대책)

```python
def softmax(a):
    """오버플로 방지 소프트맥스"""
    c = np.max(a)
    exp_a = np.exp(a - c)  # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

# 큰 값에서도 안전
a = np.array([1010, 1000, 990])
print(softmax(a))  # [9.99954600e-01 4.53978686e-05 2.06106005e-09]
```

### 수학적 증명

$$y_k = \frac{\exp(a_k)}{\displaystyle\sum_{i=1}^{n}\exp(a_i)} = \frac{C\exp(a_k)}{C\displaystyle\sum_{i=1}^{n}\exp(a_i)}$$

$$= \frac{\exp(a_k + \log C)}{\displaystyle\sum_{i=1}^{n}\exp(a_i + \log C)}$$

$$= \frac{\exp(a_k + C')}{\displaystyle\sum_{i=1}^{n}\exp(a_i + C')}$$

보통 $C' = -\max(a)$로 설정합니다.

---

### Softmax의 성질

> **Softmax 함수를 적용해도 각 원소의 대소 관계는 변하지 않습니다.**

이유: $y = \exp(x)$가 **단조 증가 함수**이기 때문

```python
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)

print(f"입력: {a}")
print(f"출력: {y}")
print(f"최댓값 인덱스: {np.argmax(a)} == {np.argmax(y)}")
# 최댓값 인덱스: 2 == 2
```

### 사용 시점

- **추론 단계**: 사용 **X** (argmax만 사용)
- **학습 단계**: 출력층에서 사용 **O**

---

## 7. 추론 과정 구현

### 순전파 (Forward Propagation)

추론 과정은 **순전파(Forward Propagation)**라고 함.

**절차:**
1. **학습**: 훈련 데이터를 사용해 가중치 매개변수를 학습
2. **추론**: 학습한 매개변수를 사용하여 입력 데이터를 분류

### 기본 추론 코드

```python
# 데이터 로드 (가상 함수)
x, t = get_data()  # x: 입력 데이터, t: 정답 레이블

# 신경망 초기화
network = init_network()

# 추론
accuracy_cnt = 0
for i in range(len(x)):
    y = forward(network, x[i])
    p = np.argmax(y)  # 확률이 가장 높은 원소의 인덱스
    if p == t[i]:
        accuracy_cnt += 1

print(f"Accuracy: {float(accuracy_cnt) / len(x)}")
```

---

## 8. 배치 처리 (Batch Processing)

### 배치 처리의 장점

✅ **빠른 연산**: 행렬 연산 최적화  
✅ **효율적**: 한 번에 여러 데이터 처리  
✅ **GPU 활용**: 병렬 처리 가능

### 배치 처리 구현

```python
# 데이터 로드
x, t = get_data()
network = init_network()

batch_size = 100  # 배치 크기
accuracy_cnt = 0

# 배치 단위로 처리
for i in range(0, len(x), batch_size):  # start=0, end=len(x), step=batch_size
    x_batch = x[i:i+batch_size]  # 입력 데이터를 묶음
    y_batch = predict(network, x_batch)
    
    # 각 데이터의 확률이 가장 높은 원소의 인덱스
    p = np.argmax(y_batch, axis=1)
    # axis=1: 1번째 차원(행)을 따라 최댓값의 인덱스를 찾음
    
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print(f"Accuracy: {float(accuracy_cnt) / len(x)}")
```

### axis 매개변수 이해

```python
# 2D 배열 예시
x = np.array([[0.1, 0.8, 0.1],
              [0.3, 0.1, 0.6],
              [0.2, 0.5, 0.3],
              [0.8, 0.1, 0.1]])

# axis=0: 열 방향 (세로)
print(np.argmax(x, axis=0))  # [3 0 1]

# axis=1: 행 방향 (가로) - 각 샘플의 최댓값
print(np.argmax(x, axis=1))  # [1 2 1 0]
```

### 배치 처리 시각화

```
단일 처리:
x[0] → network → y[0]
x[1] → network → y[1]
x[2] → network → y[2]
...

배치 처리:
┌─────┐
│ x[0]│
│ x[1]│   →  network  →  ┌─────┐
│ x[2]│                   │ y[0]│
│ ... │                   │ y[1]│
└─────┘                   │ y[2]│
                          │ ... │
                          └─────┘
```

---

## 9. 완전한 MNIST 추론 예제

```python
import numpy as np
from dataset.mnist import load_mnist
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def get_data():
    """MNIST 데이터 로드"""
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )
    return x_test, t_test

def init_network():
    """학습된 가중치 로드"""
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    """순전파 예측"""
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

# 메인 실행
x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print(f"Accuracy: {float(accuracy_cnt) / len(x):.4f}")
```

---

## 10. 핵심 정리

### 활성화 함수 비교

| 함수 | 수식 | 특징 | 사용 위치 |
|:----:|:-----|:-----|:---------|
| **Step** | $h(x) = \begin{cases}0 & (x \leq 0) \\ 1 & (x > 0)\end{cases}$ | 불연속, 미분 불가 | 퍼셉트론 |
| **Sigmoid** | $h(x) = \frac{1}{1+e^{-x}}$ | 부드러운 곡선, 0~1 출력 | 은닉층, 이진 분류 |
| **ReLU** | $h(x) = \max(0, x)$ | 계산 간단, 기울기 소실 완화 | 은닉층 (현대적) |
| **Softmax** | $y_k = \frac{e^{a_k}}{\sum e^{a_i}}$ | 확률 분포 출력 | 다중 분류 출력층 |

### 신경망 설계 가이드

```python
# 전형적인 신경망 구조
"""
입력층 (784) → 은닉층1 (50) → 은닉층2 (100) → 출력층 (10)
   ↓              ↓               ↓              ↓
  입력          ReLU            ReLU         Softmax
"""
```

### 배치 처리 형상 변화

```python
# 단일 입력: (784,)
# 배치 입력: (100, 784)

# W1: (784, 50)
# 배치 출력: (100, 50)
```

**핵심:** 배치 처리는 첫 번째 차원(배치 크기)을 유지하며 전파된다!