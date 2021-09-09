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

-------

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

------

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
