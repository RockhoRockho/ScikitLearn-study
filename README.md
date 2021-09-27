# Today-study
사이킷런 마스터 프로젝트
==

http://suanlab.com/youtube/bd.html 참고
- 빅데이터 개념 영상 시청
- 머신러닝 개념 영상 시청
- 사이킷런  scikit-learn 제대로 시작하기 영상 1시간 시청

-----

## **Day 1 (2021-09-01)**

**API 란?**  

API를 본격적으로 알아보기 전에, 비유를 들어 쉽게 설명을 도와드리겠습니다. 여러분이 멋진 레스토랑에 있다고 가정해봅시다. 점원이 가져다준 메뉴판을 보면서 먹음직스러운 스테이크를 고르면, 점원이 주문을 받아 요리사에 요청을 할 텐데요. 그러면 요리사는 정성껏 스테이크를 만들어 점원에게 주고, 여러분은 점원이 가져다준 맛있는 음식을 먹을 수 있게 됩니다.  

여기서 점원의 역할을 한 번 살펴보겠습니다. 점원은 손님에게 메뉴를 알려주고, 주방에 주문받은 요리를 요청합니다. 그다음 주방에서 완성된 요리를 손님께 다시 전달하지요. API는 점원과 같은 역할을 합니다.  
API는 손님(프로그램)이 주문할 수 있게 메뉴(명령 목록)를 정리하고, 주문(명령)을 받으면 요리사(응용프로그램)와 상호작용하여 요청된 메뉴(명령에 대한 값)를 전달합니다.  
쉽게 말해, API는 프로그램들이 서로 상호작용하는 것을 도와주는 매개체로 볼 수 있습니다.  

 

**API의 역할은?**  
 

1. API는 서버와 데이터베이스에 대한 출입구 역할을 한다.  
: 데이터베이스에는 소중한 정보들이 저장되는데요. 모든 사람들이 이 데이터베이스에 접근할 수 있으면 안 되겠지요. API는 이를 방지하기 위해 여러분이 가진 서버와 데이터베이스에 대한 출입구 역할을 하며, 허용된 사람들에게만 접근성을 부여해줍니다.  

2. API는 애플리케이션과 기기가 원활하게 통신할 수 있도록 한다.  
: 여기서 애플리케이션이란 우리가 흔히 알고 있는 스마트폰 어플이나 프로그램을 말합니다. API는 애플리케이션과 기기가 데이터를 원활히 주고받을 수 있도록 돕는 역할을 합니다.  

3. API는 모든 접속을 표준화한다.  
API는 모든 접속을 표준화하기 때문에 기계/ 운영체제 등과 상관없이 누구나 동일한 액세스를 얻을 수 있습니다. 쉽게 말해, API는 범용 플러그처럼 작동한다고 볼 수 있습니다.  

 

**API유형은 어떤게 있을까?**  
 

1) private API  
: private API는 내부 API로, 회사 개발자가 자체 제품과 서비스를 개선하기 위해 내부적으로 발행합니다. 따라서 제 3자에게 노출되지 않습니다.  

2) public API  
: public API는 개방형 API로, 모두에게 공개됩니다. 누구나 제한 없이 API를 사용할 수 있는 게 특징입니다.  

3) partner API  
:partner API는 기업이 데이터 공유에 동의하는 특정인들만 사용할 수 있습니다. 비즈니스 관계에서 사용되는 편이며, 종종 파트너 회사 간에 소프트웨어를 통합하기 위해 사용됩니다.  

 

**API 사용하면 뭐가 좋을까?**  
API를 사용하면 많은 이점들이 있는데요. Private API를 이용할 경우, 개발자들이 애플리케이션 코드를 작성하는 방법을 표준화함으로써, 간소화되고 빠른 프로세스 처리를 가능하게 합니다. 또한, 소프트 웨어를 통합하고자 할 때는 개발자들 간의 협업을 용이하게 만들어줄 수 있죠.  
public API와 partner API 를 사용하면, 기업은 타사 데이터를 활용하여 브랜드 인지도를 높일 수 있습니다. 뿐만 아니라 고객 데이터베이스를 확장하여 전환율까지 높일 수 있지요.  


###**머신러닝 개념**

**머신러닝**  
- 명시적인 프로그래밍 없이 컴퓨터가 학습하는 능력을 갖추게 하는 연구분야  
- 머신 러닝은 데이터를 통해 다양한 패턴을 감지하고 스스로 학습할 수 있는 모델 개발에 초점  

####**지도학습**  
- 지도 학습은 주어진 입력으로부터 출력 값을 예측하고자 할 때 사용  
- 입력과 정답 데이터를 사용해 모델을 학습 시킨 후 새로운 입력 데이터에 대해 정확한 출력을 예측하도록 하는 것이 목표  
- 지도 학습 알고리즘의 학습 데이터를 만드는 것은 많은 사람들의 노력과 자원이 필요하지만 높은 성능을 기대할 수 있음  

분류와 회귀  
- 지도 학습 알고리즘은 크게 **분류(classification)와 회귀(regressino)로 구분**  
- 분류는 입력 데이터를 미리 정의된 여러개의 클래스 중 하나로 예측하는 것  
- 분류는 클래스의 개수가 2개인 이진 분류(binary classification)와 3개 이상인 다중 분류(multi-class classificatino)로 나눌 수 있음  
- 회귀는 연속적인 숫자를 예측하는 것으로 어떤 사람의 나이, 농작물의 수확량, 주식 가격 등 출력 값이 연속성을 갖는 다면 회귀 문제라고 할 수 있음  

지도학습 알고리즘  
- 선형회귀(Linear Regression)
- 로지스틱 회귀(Logistic Regression)
- 서포트 벡터 머신(Support Vector Machine)
- k-최근접 이웃(k-Nearest Neighbors)
- 결정 트리(Decision Tree)
- 앙상블(Ensemble)
- 신경망(Neural Networks)

#### **비지도 학습**  
- 비지도 학습 알고리즘은 크게 **클러스터링(Clustering), 차원 축소(Dimensionality Reductino), 연관 규칙(Association Rules)으로 구분**  
- 클러스터링은 공간상에서 서로 가깝고 유사한 데이터를 클러스터로 그룹화  
- 차원 축소는 고차원의 데이터에 대해서 너무 많은 정보를 잃지 않으면서 데이터를 축소시키는 방법  
- 연관 규칙은 데이터에서 특성 간의 연관성이 있는 흥미로운 규칙을 찾는 방법  

비지도 학습 알고리즘  
- 클러스터링(Clustering)  
  - k-Means  
  - DBSCAN  
  - 계층 군집 분석(Hierarchical Cluster Analysis)  
  - 이상치 탐지(Outlier Detection), 특이값 탐지(Novelty Detectino)  
- 차원축소(Dimensionality Reduction)  
  - 주성분 분석(Principal Component Analysis)  
  - 커널 PCA(Kernel PCA)  
  - t-SNE(t-Distributed Stochastic Neighbor Embedding)  
- 연관 규칙(Associatino Rule Learning)  
  - Apriori  
  - Eclat  
 
#### **준지도 학습(Semi-supervised Learning)**  
- 레이블이 있는 것과 없는 것이 혼합된 경우 사용  
- 일반적으로는 일부 데이터에만 레이블이 있음  
- 준지도 학습 알고리즘은 대부분 지도 학습 알고리즘과 비지도 학습 알고리즘 조합으로 구성  

#### **강화 학습(Reinforcement Learning)**  
- 동적 환경과 함께 상호 작용하는 피드백 기반 학습 방법  
- 에이전트(Agent)가 환경을 관찰하고, 행동을 실행하고, 보상(reward)또는 벌점(penality)를 받음  
- 에이전트는 이러한 피드백을 통해 자동으로 학습하고 성능을 향상시킴  
- 어떤 지도가 없이 일정한 목표를 수행  

### **온라인 vs. 배치**  
- 온라인 학습(Online Learning)  
  - 적은 데이터를 사용해 미니배치(mini-batch) 단위로 점진적으로 학습  
  - 실시간 시스템이나 메모리 부족의 경우 사용  
- 배치 학습(Batch Learning)  
  - 전체 데이터를 모두 사용해 오프라인에서 학습  
  - 컴퓨팅 자원이 풍부한 경우 사용  

### **사례 기반 vs. 모델 기반**  
- 사례 기반 학습(Instance-based Learning)  
  - 훈련 데이터를 학습을 통해 기억  
  - 예측을 위해 데이터 사이의 유사도를 측정  
  - 새로운 데이터와 학습된 데이터를 비교  
- 모델 기반 학습(Model-based Learning)  
  - 훈련 데이터를 사용해 모델을 훈련  
  - 훈련된 모델을 사용해 새로운 데이터를 예측  

### **일반화, 과대적합, 과소적함**  
일반화(generalinzation)  
- 일반적으로 지도 학습 모델은 학습 데이터로 훈련 시킨 뒤 평가 데이터에서 정확하게 예측하기를 기대함  
- 훈련된 모델이 처음보는 데이터에 대해 정확하게 예측한다면, 이러한 상태의 모델이 **일반화(generalization)** 되었다고 함  
- 모델이 항상 일반화 되는 것은 아님  

과대적합(overfitting)  
- 주어진 훈련 데이터에 비해 복잡한 모델을 사용한다면, 모델은 훈련 데이터에서만 정확한 데이터를 보이고, 평가 데이터에서는 낮은 성능을 보임  
- 즉, 모델이 주어진 훈련 데이터는 잘 예측하지만 일반적인 특징을 학습하지 못해 평가 데이터에서는 낮은 성능을 보이는 상태를 **과대적합(overfitting)**이라고 함  

과소적합(underfitting)  
- 과대적합과 반대로 주어진 훈련 데이터에 비해 너무 간단한 모델을 사용하면, 모델이 데이터에 존재하는 다양한 정보들을 제대로 학습하지 못함  
- 이러한 경우 모델은 훈련 데이터에서도 나쁜 성능을 보이고 평가 데이터에서도 낮은 성능을 보이는 **과소적합(underfitting)**되었다고 함  

### **모델 복잡도와 데이터셋 크기의 관계**  
- 데이터의 다양성이 클수록 더 복잡한 모델을 사용하면 좋은 성능을 얻을 수 있음  
- 일반적으로 더 큰 데이터셋(데이터 수, 특징 수)일수록 다양성이 높기 때문에 더 복잡한 모델을 사용할 수 있음  
- 하지만, 같은 데이터를 중복하거나 비슷한 데이터를 모으는 것은 다양성 증가에 도움이 되지 않음  
- 데이터를 더 많이 수집하고 적절한 모델을 만들어 사용하면 지도 학습을 사용해 놀라운 결과를 얻을 수 있음  

### **훈련 세트 vs. 테스트 세트 vs. 검증세트**  
- 머신러닝 모델의 일반화 성능을 측정하기 위해 훈련 세트, 테스트 세트로 구분  
- 훈련 세트로 모델을 학습하고 테스트 세트로 모델의 일반화 성능 측정  
- 하이퍼파라미터는 알고리즘을 조절하기 위해 사전에 정의하는 파라미터  
- 테스트 세트를 이용해 여러 모델을 평가하면 테스트 세트에 과대적합됨  
- 모델 선택을 위해 훈련 세트, 테스트 세트, 검증 세트로 구분  

---

**사이킷런**   

- scikit-learn 주요 모듈에 대해 알게됨  
- API 사용법에 대해 알게됨   
- LinearRegression으로 사용예제를 한번 해봄  
- 예제 분류/회귀용 데이터 세트에 대해 알게됨  
- 온라인 데이터 세트를 알게됨  
- 분류와 클러스터링을 위한 표본 데이터에 대해 알게됨  
- 예제 데이터 세트 구조를 파악함  
- model_selectino에 train_test_split, cross_val_score, gridSearchCV를 학습함  

------

## **2일차 study (2021-09-01)**  

- `preprocessing` 데이터 전처리 모듈(표준화, 정규화)방법에 대해 공부함
- `StandardScaler` 표준화 클래스를 학습
- `MinMaxScaler` 정규화 클래스를 학습
- 성능 평가지표(정확도, 오차행렬, 정밀도, 재현율, F1 score, ROC 곡선 & AUC)를 학습함
  - `accuracy_score`, `cofusion_matrix`, `precision_score`, `recall_score`, `f1_score`, `roc_curve`, `roc_auc_score`

------

## **3일차 study (2021-09-06)**

선형모델 정의 학습
- 선형 회귀(Linear Regression)학습

* 선형 모델은 과거 부터 지금 까지 널리 사용되고 연구 되고 있는 기계학습 방법
* 선형 모델은 입력 데이터에 대한 선형 함수를 만들어 예측 수행

* 회귀 분석을 위한 선형 모델은 다음과 같이 정의

\begin{equation}  
\hat{y}(w,x) = w_0 + w_1 x_1 + ... + w_p x_p  
\end{equation}  

  + $x$: 입력 데이터
  + $w$: 모델이 학습할 파라미터
  + $w_0$: 편향
  + $w_1$~$w_p$: 가중치

**선형 회귀(Linear Regression)** 또는 **최소제곱법(Ordinary Least Squares)** 은 가장 간단한 회귀 분석을 위한 선형 모델
* 선형 회귀는 모델의 예측과 정답 사이의 **평균제곱오차(Mean Squared Error)** 를 최소화 하는 학습 파라미터 $w$를 찾음
* 평균제곱오차는 아래와 같이 정의  

\begin{equation}  
MSE = \frac{1}{N} \sum_{i=1}^{N}(y_i - \hat{y}_i)^2  
\end{equation}  

  + $y$: 정답  
  + $\hat{y}$: 예측 값을 의미  

* 선형 회귀 모델에서 사용하는 다양한 오류 측정 방법  
  + MAE(Mean Absoulte Error)  
  + MAPE(Mean Absolute Percentage Error)  
  + MSE(Mean Squared Error)  
  + MPE(Mean Percentage Error)
  
보스턴 주택가격 데이터
- `pairplot()`실행
- 보스턴 주택 가격에 대한 선형 회귀 실행
- 교차검증`cross_val_score()`, 결정 계수`r2_score()` 실행

------

## **4일차 study (2021-09-07)**

캘리포니아 주택 가격에 대한 선형회기 실행
- `pairplot()`실행
- 주택 가격에 대한 선형 회귀 실행
- 교차검증`cross_val_score()`, 결정 계수`r2_score()` 실행

## 보스턴, 캘리포니아 주택 가격에 대한 선형모델학습


### 릿지회귀(Ridge)

  - from sklearn.linear_model import Ridge
  - from sklearn.model_selection import train_test_split
  - from sklearn.datasets import load_boston
  - from sklearn.datasets import fetch_california_housing

* 릿지 회귀는 선형 회귀를 개선한 선형 모델
* 릿지 회귀는 선형 회귀와 비슷하지만, 가중치의 절대값을 최대한 작게 만든다는 것이 다름
* 이러한 방법은 각각의 특성(feature)이 출력 값에 주는 영향을 최소한으로 만들도록 규제(regularization)를 거는 것
* 규제를 사용하면 다중공선성(multicollinearity) 문제를 방지하기 때문에 모델의 과대적합을 막을 수 있게 됨
* 다중공선성 문제는 두 특성이 일치에 가까울 정도로 관련성(상관관계)이 높을 경우 발생
* 릿지 회귀는 다음과 같은 함수를 최소화 하는 파라미터 $w$를 찾음

\begin{equation}
RidgeMSE = \frac{1}{N} \sum_{i=1}^{N}(y_i - \hat{y}_i)^2 + \alpha \sum_{i=1}^{p} w_i^2
\end{equation}

  + $\alpha$: 사용자가 지정하는 매개변수
  * $\alpha$가 크면 규제의 효과가 커지고, $\alpha$가 작으면 규제의 효과가 작아짐

* 릿지 회귀는 가중치에 제약을 두기 때문에 선형 회귀 모델보다 훈련 데이터 점수가 낮을 수 있음
* 일반화 성능은 릿지 회귀가 더 높기 때문에 평가 데이터 점수는 릿지 회귀가 더 좋음

* 일반화 성능에 영향을 주는 매개 변수인 $\alpha$ 값을 조정해 보면서 릿지 회귀 분석의 성능이 어떻게 변하는지 확인 필요


### 라쏘 회귀(Lasso Regression)

  - from sklearn.linear_model import Lasso
  - from sklearn.model_selection import train_test_split
  - from sklearn.datasets import load_boston
  - from sklearn.datasets import fetch_california_housing

* 선형 회귀에 규제를 적용한 또 다른 모델로 라쏘 회귀가 있음
* 라쏘 회귀는 릿지 회귀와 비슷하게 가중치를 0에 가깝게 만들지만, 조금 다른 방식을 사용

* 라쏘 회귀에서는 다음과 같은 함수를 최소화 하는 파라미터 $w$를 찾음

\begin{equation}
LassoMSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 + \alpha \sum_{i=1}^{p} |w_i|
\end{equation}

* 라쏘 회귀도 매개변수인 $\alpha$ 값을 통해 규제의 강도 조절 가능


### 신축망 (Elastic-Net)

  - from sklearn.linear_model import ElasticNet
  - from sklearn.model_selection import train_test_split
  - from sklearn.datasets import load_boston
  - from sklearn.datasets import fetch_california_housing

* 신축망은 릿지 회귀와 라쏘 회귀, 두 모델의 모든 규제를 사용하는 선형 모델
* 두 모델의 장점을 모두 갖고 있기 때문에 좋은 성능을 보임
* 데이터 특성이 많거나 서로 상관 관계가 높은 특성이 존재 할 때 위의 두 모델보다 좋은 성능을 보여 줌

* 신축망은 다음과 같은 함수를 최소화 하는 파라미터 $w$를 찾음

\begin{equation}
ElasticMSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i) + \alpha \rho \sum_{i=1}^{p} |w_i| + \alpha (1 - \rho) \sum_{i=1}^{p} w_i^2
\end{equation}

  + $\alpha$: 규제의 강도를 조절하는 매개변수
  + $\rho$: 라쏘 규제와 릿지 규제 사이의 가중치를 조절하는 매개변수


### 직교 정합 추구 (Orthogonal Matching Pursuit)

  - from sklearn.linear_model import OrthogonalMatchingPursuit
  - from sklearn.model_selection import train_test_split
  - from sklearn.datasets import load_boston
  - from sklearn.datasets import fetch_california_housing

* 직교 정합 추구 방법은 모델에 존재하는 가중치 벡터에 특별한 제약을 거는 방법

* 직교 정합 추구 방법은 다음을 만족하는 파라미터 $w$를 찾는것이 목표

\begin{equation}
\underset{w}{\arg \min} \; ||y - \hat{y}||^2_2 \; subject \; to \; ||w||_0 \leq k
\end{equation}

  + $||w||_0$: 가중치 벡터 $w$에서 0이 아닌 값의 개수

* 직교 정합 추구 방법은 가중치 벡터 $w$에서 0이 아닌 값이 $k$개 이하가 되도록 훈련됨
* 이러한 방법은 모델이 필요 없는 데이터 특성을 훈련 과정에서 자동으로 제거 하도록 만들 수 있음

* 직교 정합 추구 방법은 위에서 설명한 제약 조건 대신에 다음 조건을 만족하도록 변경 가능

\begin{equation}
\underset{w}{\arg \min} \; ||w||_0 \; subject \; to \; ||y - \hat{y}||^2_2 \leq tol
\end{equation}

  + $||y - \hat{y}||^2_2$는 $\sum_{i=1}^N (y - \hat{y})^2$와 같은 의미

* 위의 식을 통해서 직교 정합 추구 방법을 $y$와 $\hat{y}$ 사이의 오차 제곱 합을 $tol$ 이하로 하면서 $||w||_0$를 최소로 하는 모델로 대체 가능


### 다항 회귀 (Polynomial Regression)

  - from sklearn.preprocessing import PolynomialFeatures, StandardScaler
  - from sklearn.linear_model import LinearRegression
  - from sklearn.pipeline import make_pipeline
  - from sklearn.model_selection import train_test_split
  - from sklearn.datasets import load_boston
  - from sklearn.datasets import fetch_california_housing

* 입력 데이터를 비선형 변환 후 사용하는 방법
* 모델 자체는 선형 모델

\begin{equation}
\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_1^2 + w_5 x_2^2
\end{equation}

* 차수가 높아질수록 더 복잡한 데이터 학습 가능

![polynomial regression](https://scikit-learn.org/stable/_images/sphx_glr_plot_polynomial_interpolation_0011.png)

------

## **5일차 study (2021-09-09)**

### 로지스틱 회귀(Logistic Regression)

* 로지스틱 회귀는 이름에 회귀라는 단어가 들어가지만, 가능한 클래스가 2개인 이진 분류를 위한 모델  
* 로지스틱 회귀의 예측 함수 정의  

\begin{equation}  
\sigma(x) = \frac{1}{1 + e^{-x}} \\  
\hat{y} = \sigma(w_0 + w_1 x_1 + ... + w_p x_p)  
\end{equation}  

  + $\sigma$: 시그모이드 함수  
  
* 로지스틱 회귀 모델은 선형 회귀 모델에 시그모이드 함수를 적용  

* 로지스틱 회귀의 학습 목표는 다음과 같은 목적 함수를 최소화 하는 파라미터 $w$를 찾는 것  

\begin{equation}  
BinaryCrossEntropy = -\frac{1}{N}\sum_{i=1}^{N}y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)  
\end{equation}  

- 붓꽃 데이터, 유방암 데이터로 로지스틱 회귀 실행

### 확률적 경사 하강법(Stochastic Gradient Descent)

* 모델을 학습 시키기 위한 간단한 방법
* 학습 파라미터에 대한 손실 함수의 기울기를 구해 기울기가 최소화 되는 방향으로 학습

\begin{equation}  
\frac{\partial L}{\partial w} = \underset{h \rightarrow 0}{lim} \frac{L(w+h) - L(w)}{h} \\  
w^{'} = w - \alpha \frac{\partial L}{\partial w}  
\end{equation}  

* scikit-learn에서는 선형 SGD 회귀와 SGD 분류를 지원

- 붓꽃 데이터, 유방암 데이터로 SGD 분류 실행

------

## **5일차 study (2021-09-11)**

### 서포트 벡터 머신(Support Vector Machines)

* 회귀, 분류, 이상치 탐지 등에 사용되는 지도학습 방법
* 클래스 사이의 경계에 위치한 데이터 포인트를 서포트 벡터(support vector)라고 함
* 각 지지 벡터가 클래스 사이의 결정 경계를 구분하는데 얼마나 중요한지를 학습
* 각 지지 벡터 사이의 마진이 가장 큰 방향으로 학습
* 지지 벡터 까지의 거리와 지지 벡터의 중요도를 기반으로 예측을 수행

![support vector machine](https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Svm_separating_hyperplanes.png/220px-Svm_separating_hyperplanes.png)

* H3은 두 클래스의 점들을 제대로 분류하고 있지 않음
* H1과 H2는 두 클래스의 점들을 분류하는데, H2가 H1보다 더 큰 마진을 갖고 분류하는 것을 확인할 수 있음

- `SVM()`, `SVC()`

#### 커널기법
* 입력 데이터를 고차원 공간에 사상해서 비선형 특징을 학습할 수 있도록 확장하는 방법
* scikit-learn에서는 Linear, Polynomial, RBF(Radial Basis Function)등 다양한 커널 기법을 지원
- `kernel='rbf'`

![kernel trick](https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_svc_0011.png)

-----

## **6일차 study (2021-09-13)**

### 최근접 이웃(K-Nearest Neighbor)

* 특별한 예측 모델 없이 가장 가까운 데이터 포인트를 기반으로 예측을 수행하는 방법
* 분류와 회귀 모두 지원

![k nearest neighbor](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/KnnClassification.svg/220px-KnnClassification.svg.png)

#### K 최근접 이웃 분류

* 입력 데이터 포인트와 가장 가까운 k개의 훈련 데이터 포인트가 출력
* k개의 데이터 포인트 중 가장 많은 클래스가 예측 결과

- 붓꽃 데이터, 유방암 데이터, 와인 데이터로 실행

#### k 최근접 이웃 회귀

* k 최근접 이웃 분류와 마찬가지로 예측에 이웃 데이터 포인트 사용
* 이웃 데이터 포인트의 평균이 예측 결과

- 보스턴 데이터, 캘리포니아 데이터로 실행

-----

## **7일차 study (2021-09-14)**

### 나이브 베이스 분류기(Naive Bayes Classification)

* 베이즈 정리를 적용한 확률적 분류 알고리즘
* 모든 특성들이 독립임을 가정 (naive 가정)
* 입력 특성에 따라 3개의 분류기 존재
  * 가우시안 나이브 베이즈 분류기
  * 베르누이 나이브 베이즈 분류기
  * 다항 나이브 베이즈 분류기

### 나이브 베이즈 분류기의 확률 모델

* 나이브 베이즈는 조건부 확률 모델
* *N*개의 특성을 나타내는 벡터 **x**를 입력 받아 k개의 가능한 확률적 결과를 출력

\begin{equation}
p(C_k | x_1,...,x_n)
\end{equation}

* 위의 식에 베이즈 정리를 적용하면 다음과 같음

\begin{equation}
p(C_k | \textbf{x}) = \frac{p(C_k)p(\textbf{x}|C_k)}{p(\textbf{x})}
\end{equation}

* 위의 식에서 분자만이 출력 값에 영향을 받기 때문에 분모 부분을 상수로 취급할 수 있음

\begin{equation}
\begin{split}
p(C_k | \textbf{x}) & \propto p(C_k)p(\textbf{x}|C_k) \\
& \propto p(C_k, x_1, ..., x_n)
\end{split}
\end{equation}

* 위의 식을 연쇄 법칙을 사용해 다음과 같이 쓸 수 있음

\begin{equation}
\begin{split}
p(C_k, x_1, ..., x_n) & = p(C_k)p(x_1, ..., x_n | C_k) \\
& = p(C_k)p(x_1 | C_k)p(x_2, ..., x_n | C_k, x_1) \\
& = p(C_k)p(x_1 | C_k)p(x_2 | C_k, x_1)p(x_3, ..., x_n | C_k, x_1, x_2) \\
& = p(C_k)p(x_1 | C_k)p(x_2 | C_k, x_1)...p(x_n | C_k, x_1, x_2, ..., x_{n-1})
\end{split}
\end{equation}

* 나이브 베이즈 분류기는 모든 특성이 독립이라고 가정하기 때문에 위의 식을 다음과 같이 쓸 수 있음

\begin{equation}
\begin{split}
p(C_k, x_1, ..., x_n) & \propto p(C_k)p(x_1|C_k)p(x_2|C_k)...p(x_n|C_k) \\
& \propto p(C_k) \prod_{i=1}^{n} p(x_i|C_k)
\end{split}
\end{equation}

* 위의 식을 통해 나온 값들 중 가장 큰 값을 갖는 클래스가 예측 결과

\begin{equation}
\hat{y} = \underset{k}{\arg\max} \; p(C_k) \prod_{i=1}^{n} p(x_i|C_k)
\end{equation}

- 산림토양 데이터확인, 분류, 전처리
- 20 Newsgroup 데이터확인, 분류, 전처리
- 벡터화(`CountVectorizer()`, `HashingVectorizer()`, `tfidfVectorizer()`)

#### 가우시안 나이브 베이즈(`GaussianNB()`)

* 입력 특성이 가우시안(정규) 분포를 갖는다고 가정

#### 베르누이 나이브 베이즈(`BernoulliNB()`)

* 입력 특성이 베르누이 분포에 의해 생성된 이진 값을 갖는 다고 가정

#### 다항 나이브 베이즈(`MultinomialNB()`)

* 입력 특성이 다항분포에 의해 생성된 빈도수 값을 갖는 다고 가정

### **선형모델 복습**

선형모델 정의 복습 
- 선형 회귀(Linear Regression)학습

보스턴 주택가격 데이터  
- `pairplot()`실행
- 보스턴 주택 가격에 대한 선형 회귀 실행
- 교차검증`cross_val_score()`, 결정 계수`r2_score()` 실행

캘리포니아 주택 가격에 대한 선형회기 실행  
- `pairplot()`실행
- 주택 가격에 대한 선형 회귀 실행
- 교차검증`cross_val_score()`, 결정 계수`r2_score()` 실행

릿지회귀(Ridge)복습  
라쏘 회귀(Lasso Regression)복습  
신축망 (Elastic-Net)복습  
직교 정합 추구 (Orthogonal Matching Pursuit)복습  
다항 회귀 (Polynomial Regression)복습  

위 해당 모델을 따라치고 복습함

-----

## **8일차 study (2021-09-15)**

#### 결정 트리(Decision Tree)

* 분류와 회귀에 사용되는 지도 학습 방법
* 데이터 특성으로 부터 추론된 결정 규칙을 통해 값을 예측
* **if-then-else** 결정 규칙을 통해 데이터 학습
* 트리의 깊이가 깊을 수록 복잡한 모델
* 결정 트리 장점
  * 이해와 해석이 쉽다
  * 시각화가 용이하다
  * 많은 데이터 전처리가 필요하지 않다
  * 수치형과 범주형 데이터 모두를 다룰 수 있다
  * ...

#### 분류 - `DecisionTreeClassifier()`

* `DecisionTreeClassifier`는 분류를 위한 결정트리 모델
* 두개의 배열 X, y를 입력 받음
  * X는 [n_samples, n_features] 크기의 데이터 특성 배열
  * y는 [n_samples] 크기의 정답 배열

- 붓꽃데이터, 와인데이터, 유방암데이터로 결정트리 실행

#### 회귀 - `DecisionTreeRegressor()`

- 보스턴데이터, 당뇨병데이터로 결정트리회귀 실행

----

## **9일차 study(2021-09-16)**

### 앙상블(Ensemble)

* 일반화와 강건성(Robustness)을 향상시키기 위해 여러 모델의 예측 값을 결합하는 방법
* 앙상블에는 크게 두가지 종류가 존재
  * 평균 방법
    * 여러개의 추정값을 독립적으로 구한뒤 평균을 취함
    * 결합 추정값은 분산이 줄어들기 때문에 단일 추정값보다 좋은 성능을 보임
  * 부스팅 방법
    * 순차적으로 모델 생성
    * 결합된 모델의 편향을 감소 시키기 위해 노력
    * 부스팅 방법의 목표는 여러개의 약한 모델들을 결합해 하나의 강력한 앙상블 모델을 구축하는 것

#### Bagging meta-estimator

* bagging은 bootstrap aggregating의 줄임말
* 원래 훈련 데이터셋의 일부를 사용해 여러 모델을 훈련
* 각각의 결과를 결합해 최종 결과를 생성
* 분산을 줄이고 과적합을 막음
* 강력하고 복잡한 모델에서 잘 동작

- bagging을 사용한 분류
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `KNeighborsClassifier()`을 이용
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `SVC()`을 이용
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `DecisionTreeClassifier()`을 이용

- bagging을 사용한 회귀
 - 보스턴데이터, 당뇨병데이터에 `KNeighborsRegressor()`을 이용
 - 보스턴데이터, 당뇨병데이터에 `SVR()`을 이용
 - 보스턴데이터, 당뇨병암데이터에 `DecisionTreeRegressor()`을 이용

#### Forests of randomized trees

* `sklearn.ensemble` 모듈에는 무작위 결정 트리를 기반으로하는 두 개의 평균화 알고리즘이 존재
  * Random Forest
  * Extra-Trees
* 모델 구성에 임의성을 추가해 다양한 모델 집합이 생성
* 앙상블 모델의 예측은 각 모델의 평균

- Random Forests을 사용한 분류
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `KNeighborsClassifier()`을 이용
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `SVC()`을 이용
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `DecisionTreeClassifier()`을 이용

- Random Forests을 사용한 회귀
 - 보스턴데이터, 당뇨병데이터에 `KNeighborsRegressor()`을 이용
 - 보스턴데이터, 당뇨병데이터에 `SVR()`을 이용
 - 보스턴데이터, 당뇨병암데이터에 `DecisionTreeRegressor()`을 이용

- Extremely Randomized Tree 분류
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `KNeighborsClassifier()`을 이용
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `SVC()`을 이용
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `DecisionTreeClassifier()`을 이용

- Extremely Randomized Tree 회귀
 - 보스턴데이터, 당뇨병데이터에 `KNeighborsRegressor()`을 이용
 - 보스턴데이터, 당뇨병데이터에 `SVR()`을 이용
 - 보스턴데이터, 당뇨병암데이터에 `DecisionTreeRegressor()`을 이용

- Random Forest, Extra Tree 시각화
 - 결정 트리, Random Forest, Extra Tree의 결정 경계와 회귀식 시각화


## **10일차 study(2021-09-17)**

### AdaBoost

* 대표적인 부스팅 알고리즘
* 일련의 약한 모델들을 학습
* 수정된 버전의 데이터를 반복 학습 (가중치가 적용된)
* 가중치 투표(또는 합)을 통해 각 모델의 예측 값을 결합
* 첫 단계에서는 원본 데이터를 학습하고 연속적인 반복마다 개별 샘플에 대한 가중치가 수정되고 다시 모델이 학습
  * 잘못 예측된 샘플은 가중치 증가, 올바르게 예측된 샘플은 가중치 감소
  * 각각의 약한 모델들은 예측하기 어려운 샘플에 집중하게 됨

![AdaBoost](https://scikit-learn.org/stable/_images/sphx_glr_plot_adaboost_hastie_10_2_0011.png)

- Adaboost을 사용한 분류
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `KNeighborsClassifier()`을 이용
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `SVC()`을 이용
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `DecisionTreeClassifier()`을 이용

- Adaboost을 사용한 회귀
 - 보스턴데이터, 당뇨병데이터에 `KNeighborsRegressor()`을 이용
 - 보스턴데이터, 당뇨병데이터에 `SVR()`을 이용
 - 보스턴데이터, 당뇨병암데이터에 `DecisionTreeRegressor()`을 이용

### Gradient Tree Boosting

* 임의의 차별화 가능한 손실함수로 일반화한 부스팅 알고리즘
* 웹 검색, 분류 및 회귀 등 다양한 분야에서 모두 사용 가능

- Gradient Tree Boosting을 사용한 분류
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `KNeighborsClassifier()`을 이용
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `SVC()`을 이용
 - 붓꽃데이터, 와인데이터, 유방암데이터에 `DecisionTreeClassifier()`을 이용

- Gradient Tree Boosting을 사용한 회귀
 - 보스턴데이터, 당뇨병데이터에 `KNeighborsRegressor()`을 이용
 - 보스턴데이터, 당뇨병데이터에 `SVR()`을 이용
 - 보스턴데이터, 당뇨병암데이터에 `DecisionTreeRegressor()`을 이용

### 투표 기반 분류 (Voting Classifier)

* 서로 다른 모델들의 결과를 투표를 통해 결합
* 두가지 방법으로 투표 가능
  * 가장 많이 예측된 클래스를 정답으로 채택 (hard voting)
  * 예측된 확률의 가중치 평균 (soft voting)
* 결정경계 시각화 실행

### 투표 기반 회귀 (Voting Regressor)

* 서로 다른 모델의 예측 값의 평균을 사용
* 회귀식 시각화 실행

### 스택 일반화 (Stacked Generalization)

* 각 모델의 예측 값을 최종 모델의 입력으로 사용
* 모델의 편향을 줄이는데 효과적

- 스택 회귀, 분류 시각화 

## Sci study 로지스틱 회귀 복습

- 로지스틱 회귀 예제 따라 치기
- 붓꽃, 유방암 데이터 확인 및 로지스틱 회귀
- 확률적 하강 경사법
 - 붓꽃데이터 선형회귀 학습
 - 붓꽃, 유방암 데이터 선형분류 학습

------

## **11일차 study(2021-09-23)**

### 군집화(Clustering) 이론 학습

* 대표적인 비지도학습 알고리즘
* 레이블이 없는 데이터를 그룹화 하는 알고리즘

![clustering](https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_0011.png)

### K-평균 (K-Means) 이론 학습

* n개의 등분산 그룹으로 군집화
* 제곱합 함수를 최소화
* 군집화 개수를 지정해야 한다.
* 각 군집 $C$의 평균 $\mu_j$을 중심점 이라고 함
* 다음을 만족하는 중심점을 찾는것이 목표

\begin{equation}
\sum_{i=0}^{n} \underset{\mu_j \in C}{\min} (||x_i - \mu_j||^2)
\end{equation}

### 미니 배치 K-평균 (Mini Batch K-Means) 이론 학습

* 배치 처리를 통해 계산 시간을 줄인 K-평균
* K-평균과 다른 결과가 나올 수 있다.

### Affinity Propagation 이론 학습

* 샘플 쌍 끼리 메시지를 보내 군집을 생성
* 샘플을 대표하는 적절한 예를 찾을 때까지 반복
* 군집의 갯수를 자동으로 정함

![affinity propagation](https://scikit-learn.org/stable/_images/sphx_glr_plot_affinity_propagation_0011.png)

### Mean Shift 이론 학습

* 중심점 후보를 정해진 구역 내 평균으로 업데이트

### 스펙트럼 군집화 (Spectral Clustering) 이론 학습

### 계층 군집화 (Hierarchical Clustering) 이론 학습

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 이론 학습

### OPTICS (Ordering Points To Identify the Clustering Structure) 이론 학습

### Birch (Balanced iterative reducing and clustering using hierarchies) 이론 학습

### 손글씨 데이터 군집화 이론 학습

- K-Mean 손글씨 데이터 군집화
- Spectral Clustering 손글씨 데이터 군집화
- Hierarchical Clustering 손글씨 데이터 군집화
- Birch 손글씨 데이터 군집화

-----

### 서포트 벡터머신 이론학습

- SVM, SVR을 이용하여 회귀모델(boston), 분류모델(breast_cancer) 학습
- Linear, Polynomial, RBF를 이용하여 학습
- 매개변수 튜닝, 데이터 전처리 파악

------

## **12일차 study(2021-09-24)**

### 다양체 학습 (Manifold Learning)

* 높은 차원의 데이터를 저차원으로 축소하는 방법

![manifold](https://scikit-learn.org/stable/_images/sphx_glr_plot_compare_methods_0011.png)

* 고차원 데이터를 2차원 또는 3차원으로 축소해 시각화에 활용할 수 있음
* 차원 축소 과정에서 중요하지 않은 정보는 버려지고 중요한 정보만 남기 때문에 데이터 정제에 활용 가능

### Locally Linear Embedding (LLE) 이론 학습

* 국소 이웃 거리를 보존하는 저차원 임베딩을 찾음

### Local Tangent Space Alignment (LTSA) 이론 학습

* 탄젠트 공간을 통해 각 이웃의 국소 성질을 특성화
* 국소 탄젠트 공간을 정렬

### Hessian Eigenmapping 이론 학습

* LLE의 문제를 해결한 다른 방법
* 국소 선형 구조를 복원하기 위해 각 이웃에서 hessian 기반의 이차 형태를 중심으로 회전

### Modified Locally Linear Embedding 이론 학습

* 각 이웃에 여러 가중치 벡터를 사용
* n_neighbors > n_components를 만족해야 함

### Isomap 이론 학습

* 초기의 다양체 학습 알고리즘
* MDS와 커널 PCA의 확장으로 볼 수 있음
* 모든 점들 사이의 측지 거리를 유지하는 저차원 임베딩을 찾음

### Multi-Dimensional Scaling (MDS) 이론 학습

* 고차원 공간에서의 거리를 고려하는 저차원 공간을 찾음

### Spectral Embedding 이론 학습

* 스펙트럼 분해를 통해 데이터의 저차원 표현을 찾음
* 데이터의 점이 저차원 공간에서도 서로 가깝게 유지되도록 함

### t-distributed Stochastic Neighbor Embedding (t-SNE) 이론 학습

* 데이터 포인트의 유사성을 확률로 변환
* 국소 구조에 민감
* 국소 구조를 기반으로 샘플 그룹을 추출하는데 강함
* 항상 KL발산의 국소 최소값에서 끝남
* 계산 비용이 많이 듬
* 전역 구조를 보존하지 않음

### 정제된 표현을 이용한 학습

* 다양체 학습의 결과를 정제된 데이터로 생각할 수 있음
* 정제된 표현이기 때문에 분석에 비교적 용이함
* 기계학습 모델의 입력으로 사용했을때 성능향상을 기대할 수 있음

- 원본데이터 사용과 정제된 데이터 사용함으로 `KNeighborsClassifier`, `SVC`, `DecisionTreeClassifier`, `RandomForestClassifier` 

## 서포트 벡터머신 복습 2

- 유방암, 붓꽃, 와인데이터로 Linear SVC, kernel SVC 실행


------

## **13일차 study(2021-09-25)**

### 분해 (Decomposition)
* 큰 하나의 행렬을 여러개의 작은 행렬로 분해
* 분해 과정에서 중요한 정보만 남게됨

### Principal Component Analysis (PCA) 이론 학습
* PCA를 사용해 iris 데이터 변환
* $150 \times 4$ 크기의 데이터를 $150 \times 2$ 크기의 행렬로 압축
* PCA를 통해 학습된 각 컴포넌트 (6개)
* 각 컴포넌트는 얼굴의 주요 특징을 나타냄

### Incremental PCA 이론 학습

* PCA는 SVD 알고리즘 실행을 위해 전체 학습용 데이터 셋을 메모리에 올려야 함
* Incremental PCA는 학습 데이터를 미니 배치 단위로 나누어 사용
* 학습 데이터가 크거나 온라인으로 PCA 적용이 필요할 때 유용

### Kernel PCA 이론 학습

* 차원 축소를 위한 복잡한 비선형 투형

### Sparse PCA 이론 학습

* PCA의 주요 단점 중 하나는 주성분들이 보통 모든 입력 변수들의 선형결합으로 나타난다는 점
* 희소 주성분분석(Sparse PCA)는 몇 개 변수들만의 선형결합으로 주성분을 나타냄으로써 이러한 단점을 극복

### Truncated Singular Value Decomposition (Truncated SVD) 이론 학습

* PCA는 정방행렬에 대해서만 행렬 분해 가능
* SVD는 정방행렬 뿐만 아니라 행과 열이 다른 행렬도 분해 가능
* PCA는 밀집 행렬(Dense Matrix)에 대한 변환만 가능하지만, SVD는 희소 행렬(Sparse Matrix)에 대한 변환도 가능
* 전체 행렬 크기에 대해 Full SVD를 사용하는 경우는 적음
* 특이값이 0인 부분을 모두 제거하고 차원을 줄인 Truncated SVD를 주로 사용

### Dictionary Learning 이론 학습

* Sparse code를 사용하여 데이터를 가장 잘 나타내는 사전 찾기
* Sparse coding은 overcomplete 기저벡터(basis vector)를 기반으로 데이터를 효율적으로 표현하기 위해 개발
* 기저 벡터는 벡터 공간에 속하는 벡터의 집합이 선형 독립이고, 다른 모든 벡터 공간의 벡터들이 그 벡터 집합의 선형 조합으로 나타남

### Factor Analysis

* 요인 분석(Factor Analysis)은 변수들 간의 상관관계를 고려하여 저변에 내재된 개념인 요인들을 추출해내는 분석방법
* 요인 분석은 변수들 간의 상관관계를 고려하여 서로 유사한 변수들 끼리 묶어주는 방법
* PCA에서는 오차(error)를 고려하지 않고, 요인 분석에서는 오차(error)를 고려

### Independent Component Analysis(ICA) 이론 학습

* 독립 성분 분석(Independent Component Analysis, ICA)은 다변량의 신호를 통계적으로 독립적인 하부 성분으로 분리하는 계산 방법
* ICA는 주성분을 이용하는 점은 PCA와 유사하지만, 데이터를 가장 잘 설명하는 축을 찾는 PCA와 달리 가장 독립적인 축, 독립성이 최대가 되는 벡터를 찾음

### Non-negative Matrix Factorization 이론 학습

* 음수 미포함 행렬 분해(Non-negative matrix factorization, NMF)는 음수를 포함하지 않은 행렬 V를 음수를 포함하지 않은 행렬 W와 H의 곱으로 분해하는 알고리즘

### Latent Dirichlet Allocation (LDA) 이론 학습

* 잠재 디리클레 할당은 이산 자료들에 대한 확률적 생성 모형
* 디리클레 분포에 따라 잠재적인 의미 구조를 파악
* 
### Linear Discriminant Analysis (LDA) 이론 학습

* LDA는 PCA와 유사하게 입력 데이터 세트를 저차원 공간에 투영해 차원을 축소
* LDA는 지도학습 분류에서 사용하기 쉽도록 개별 클래스르 분별할 수 있는 기준을 최대한 유지하면서 차원 축소

### 압축된 표현을 사용한 학습
* 행렬 분해를 통해 압축된 데이터를 사용해 학습
- knn, svm, decision-tree, Random-forest

### 복원된 표현을 사용한 학습
* 분해 후 복원된 행렬을 사용해 학습
- knn, svm, decision-tree, Random-forest

### 이미지 복원 학습

------

## **14일차 study(2021-09-27)**

### XGBoost

* 트리 기반의 앙상블 기법
* 분류에 있어서 다른 알고리즘보다 좋은 예측 성능을 보여줌
* XGBoost는 GBM 기반이지만, GBM의 단점인 느린 수행 시간과 과적합 규제 부재 등의 문제를 해결
* 병렬 CPU 환경에서 빠르게 학습 가능

- XGBClassifier(붓꽃, 와인, 유방암데이터), XGBRegressor(보스턴, 당뇨병데이터) 실행

### LightGBM

* 빠른 학습과 예측 시간
* 더 적은 메모리 사용
* 범주형 특징의 자동 변환과 최적 분할

- LGBClassifier(붓꽃, 와인, 유방암데이터), LGBRegressor(보스턴, 당뇨병데이터) 실행
