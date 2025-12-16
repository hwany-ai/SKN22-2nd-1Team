# Online Shoppers Intention EDA 가이드 (이탈 원인 인사이트 중심)

이 데이터셋에서 **이탈(비구매)과 구매를 가르는 핵심 요인**은 대략 다음과 같이 정리할 수 있습니다.

- **PageValues(페이지 가치) > 0인지 여부가 전환의 가장 강력한 신호**이고, 값이 0인 세션은 대부분 이탈로 끝납니다.
- **ExitRates / BounceRates가 높을수록 이탈 확률이 급격히 증가**하며, 매우 낮을 때만 구매가 잘 일어납니다.
- **ProductRelated/Administrative/Informational 페이지 수와 체류시간이 많을수록 전환 가능성이 높아지지만**, 품질이 나쁜 트래픽(높은 이탈률)에서는 오히려 전환이 거의 일어나지 않습니다.
- **신규 방문자가 재방문자보다 전환율이 더 높고**, **11월·10월·9월 같은 기간적 요인(프로모션 시즌)도 전환에 큰 영향을 줍니다.**
- 데이터는 **결측치는 없고, 약 5.5:1의 클래스 불균형과 일부 극단값(이상치)이 존재**하므로, **이상치 로버스트 처리와 불균형 대응 전략을 전제로 EDA·모델링**을 설계하는 것이 좋습니다.

타깃: `Revenue`를 “구매 / 이탈”로 보고, 최종적으로는 “이탈(비구매) 예측” 모델로 이어지도록 설계

***

## 1. 기본 세팅 & 데이터 로딩

```python
# 1. 기본 세팅
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use("seaborn-v0_8")
sns.set(font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)

# 2. 데이터 로드
df = pd.read_csv("online_shoppers_intention.csv")

# 3. 기본 정보 확인
display(df.head())
print(df.shape)
print(df.dtypes)
```

***

## 2. 데이터 품질 점검 (결측치, 중복, 타깃 분포)

```python
# 결측치 확인
df.isnull().sum()

# 중복 확인
dup_cnt = df.duplicated().sum()
dup_ratio = dup_cnt / len(df) * 100
print(f"중복 행 개수: {dup_cnt} ({dup_ratio:.2f}%)")

# 필요하면 중복 제거 (모델링 전에 고려)
# df = df.drop_duplicates().reset_index(drop=True)

# 타깃 분포 확인
target_col = "Revenue"
print(df[target_col].value_counts())
print(df[target_col].value_counts(normalize=True))

import matplotlib.ticker as mtick

fig, ax = plt.subplots()
df[target_col].value_counts().plot(kind="bar", ax=ax)
ax.set_xticklabels(["No Purchase", "Purchase"], rotation=0)
ax.set_title("Revenue 분포 (구매 vs 비구매)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=len(df)))
plt.show()
```

**인사이트 방향**
- 중복은 약 1% 수준이므로, **모델링 시에는 제거하는 편이 안전**.
- `Revenue=True` 비율이 약 15% 수준 → **클래스 불균형(약 5.5:1)** → 이후 모델링 시  
  - `class_weight="balanced"`  
  - SMOTE/언더샘플링 등 고려 필요.

***

## 3. 수치형/범주형 변수 분리 & 기술통계

```python
# 수치형 / 범주형 분리
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print("수치형 변수:", num_cols)
print("범주형 변수:", cat_cols)

# 기술 통계
display(df[num_cols].describe().T)
```

```python
# 주요 범주형 변수 분포
for col in ["Month", "VisitorType", "Weekend"]:
    print(f"\n=== {col} 분포 ===")
    display(df[col].value_counts())
    df[col].value_counts().plot(kind="bar")
    plt.title(f"{col} 분포")
    plt.show()
```

**인사이트 방향**

- `OperatingSystems, Browser, Region, TrafficType`는 숫자형이지만 **실질적으로 범주형(코드)** →  
  모델링 시 **카테고리로 인코딩** (one-hot, target encoding 등) 고려.
- `Month`, `VisitorType`, `Weekend`는 초기 EDA에서 **전환율 차이 확인용 그룹핑 변수**로 활용.

***

## 4. 이상치 탐지 (특히 체류시간/페이지 뷰/페이지 가치)

```python
def iqr_outlier_ratio(s):
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR
    mask = (s < lb) | (s > ub)
    return mask.mean() * 100, lb, ub

outlier_summary = []
for col in num_cols:
    # 명백한 코드형 변수는 제외
    if col in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        continue
    ratio, lb, ub = iqr_outlier_ratio(df[col])
    outlier_summary.append({
        "col": col,
        "outlier_ratio(%)": round(ratio, 2),
        "lower_bound": round(lb, 2),
        "upper_bound": round(ub, 2)
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values("outlier_ratio(%)", ascending=False)
display(outlier_df)
```

```python
# 예시: PageValues, ProductRelated_Duration 분포 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x="Revenue", y="PageValues", data=df, showfliers=False, ax=axes[0])
axes[0].set_title("Revenue별 PageValues (이상치 제외 시각화)")
sns.boxplot(x="Revenue", y="ProductRelated_Duration", data=df, showfliers=False, ax=axes[1])
axes[1].set_title("Revenue별 ProductRelated_Duration (이상치 제외 시각화)")
plt.tight_layout()
plt.show()
```

**인사이트 방향**

- `PageValues`, `Informational`, `Informational_Duration` 등은 **IQR 기준 이상치 비율이 20% 수준**으로 높음.
- 이 값들은 **실제 비즈니스 행동(매우 긴 체류, 높은 가치 페이지)** 를 반영할 수 있으므로  
  단순 제거보다는
  - **로그 스케일 변환**  
  - **윈저라이징(상하위 1% 클리핑)**  
  방식으로 로버스트하게 다루는 것을 추천.

***

## 5. 타깃(Revenue) 기준 그룹별 비교

### 5.1 주요 수치형 지표 평균 비교

```python
key_cols = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues"
]

rev_group = df.groupby("Revenue")[key_cols].mean().T
rev_group.columns = ["No Purchase", "Purchase"]
display(rev_group)
```

```python
# 시각화: 스케일이 많이 다른 변수는 정규화해서 비교
rev_group_norm = (rev_group - rev_group.min()) / (rev_group.max() - rev_group.min())

rev_group_norm.plot(kind="bar", figsize=(12, 6))
plt.title("구매 여부에 따른 주요 지표 평균 (정규화)")
plt.ylabel("정규화된 값")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
```

**주요 인사이트**

- 구매 세션은 비구매 세션에 비해  
  - **더 많은 ProductRelated/Administrative/Informational 페이지를 보고**,  
  - **해당 페이지에서 더 오래 체류**.
- `BounceRates`, `ExitRates`는 구매 세션에서 **의미 있게 낮음**.
- `PageValues`는 구매 세션에서 **압도적으로 높음** → 전환과 거의 직결되는 지표.

### 5.2 PageValues의 임팩트

```python
cond_zero = df["PageValues"] == 0
pv_zero = df[cond_zero]["Revenue"].value_counts(normalize=True) * 100
pv_nonzero = df[~cond_zero]["Revenue"].value_counts(normalize=True) * 100

print("PageValues = 0:")
print(pv_zero)
print("\nPageValues > 0:")
print(pv_nonzero)

sns.barplot(x=["PageValues=0", "PageValues>0"],
            y=[pv_zero.get(True, 0), pv_nonzero.get(True, 0)])
plt.ylabel("Purchase Rate (%)")
plt.title("PageValues 여부에 따른 구매율")
plt.show()
```

**인사이트**

- `PageValues = 0`인 세션의 구매율은 **약 3~4%**에 불과.
- `PageValues > 0`인 세션의 구매율은 **50% 이상**으로, **약 10배 이상 차이**.
- → **유의미한 가치 페이지(결제 직전 페이지 등)에 도달하지 못하는 트래픽이 곧 이탈**로 볼 수 있음.

***

## 6. 이탈과 가장 직결된 지표: Bounce / Exit / PageValues

### 6.1 Revenue와의 상관관계 (Point-Biserial)

```python
df["Revenue_int"] = df["Revenue"].astype(int)

corr_list = []
for col in key_cols:
    r, p = stats.pointbiserialr(df["Revenue_int"], df[col])
    corr_list.append({"col": col, "r": r, "p": p})

corr_df = pd.DataFrame(corr_list).sort_values("r", ascending=False)
display(corr_df)
```

```python
# 시각화: 상관계수 막대 그래프
fig, ax = plt.subplots(figsize=(8, 5))
corr_sorted = corr_df.sort_values("r", ascending=True)
colors = ["red" if r < 0 else "green" for r in corr_sorted["r"]]
ax.barh(corr_sorted["col"], corr_sorted["r"], color=colors)
ax.axvline(0, color="black", linewidth=1)
for i, v in enumerate(corr_sorted["r"]):
    ax.text(v, i, f"{v:.2f}", va="center",
            ha="left" if v > 0 else "right")
ax.set_title("수치형 변수와 Revenue 간 상관계수 (Point-Biserial)")
plt.tight_layout()
plt.show()
```

**핵심 해석**

- **가장 강한 양의 상관**: `PageValues` (~0.49)
- 그 다음: `ProductRelated`, `ProductRelated_Duration`, `Administrative` 등.
- **음의 상관**: `ExitRates` (~-0.21), `BounceRates` (~-0.15)
- → **“많이 보고, 오래 머물고, 가치 있는 페이지까지 간 세션”에서 구매가 발생**하고,  
  **“초반에 튕기거나(높은 Bounce), 중간 단계에서 이탈(높은 Exit)” 하면 비구매로 끝남**.

### 6.2 구간별 전환율

```python
# BounceRate 구간화
df["Bounce_bin"] = pd.cut(
    df["BounceRates"],
    bins=[0, 0.01, 0.05, 0.1, 0.2],
    labels=["0-1%", "1-5%", "5-10%", "10-20%"]
)
bounce_tab = pd.crosstab(df["Bounce_bin"], df["Revenue"], normalize="index") * 100
display(bounce_tab)

bounce_tab[True].plot(kind="bar")
plt.ylabel("Purchase Rate (%)")
plt.title("Bounce Rate 구간별 구매율")
plt.show()

# ExitRate 구간화
df["Exit_bin"] = pd.cut(
    df["ExitRates"],
    bins=[0, 0.025, 0.05, 0.1, 0.2],
    labels=["0-2.5%", "2.5-5%", "5-10%", "10-20%"]
)
exit_tab = pd.crosstab(df["Exit_bin"], df["Revenue"], normalize="index") * 100
display(exit_tab)

exit_tab[True].plot(kind="bar", color="orange")
plt.ylabel("Purchase Rate (%)")
plt.title("Exit Rate 구간별 구매율")
plt.show()
```

**인사이트**

- `BounceRates` / `ExitRates`가 **매우 낮은 구간에서만 두 자릿수 이상의 전환율**을 보임.
- 10% 이상 구간에서는 전환율이 거의 0에 수렴 → **실제 이탈 funnel 구간**으로 추정 가능.
- 이 값들은 웹 분석 지표 관점에서 **초기 랜딩 페이지 품질, 중간/체크아웃 단계 UX 문제 등과 직결**된다고 해석할 수 있음(웹 분석 일반론과 일치).

***

## 7. 체류시간(총 Duration)과 이탈

```python
df["TotalDuration"] = (
    df["Administrative_Duration"] +
    df["Informational_Duration"] +
    df["ProductRelated_Duration"]
)

print(df.groupby("Revenue")["TotalDuration"].mean())

sns.kdeplot(data=df, x="TotalDuration", hue="Revenue", common_norm=False)
plt.xlim(0, df["TotalDuration"].quantile(0.99))  # 꼬리 잘라내기
plt.title("총 체류시간 분포 (상위 1% 잘라서 표시)")
plt.show()
```

**인사이트**

- 구매 고객의 평균 총 체류시간이 비구매 고객보다 **약 15분 가량 더 길다**는 결과.
- 단, **“오래 머무른다고 무조건 구매”는 아님** → **Bounce/Exit와 함께 복합적으로 해석**해야 함.

***

## 8. 범주형 변수와 전환율 (이탈 관점)

### 8.1 월별 전환율(시즌성 / 프로모션 영향)

```python
month_conv = pd.crosstab(df["Month"], df["Revenue"], normalize="index") * 100
month_conv = month_conv.rename(columns={False: "No_Purchase", True: "Purchase"})
display(month_conv.sort_index())

# 월 순서 정의
order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_conv_ord = month_conv.reindex(order)

month_conv_ord["Purchase"].plot(kind="bar", color="skyblue")
plt.ylabel("Purchase Rate (%)")
plt.title("월별 구매 전환율")
plt.xticks(rotation=45)
plt.axhline(df["Revenue"].mean() * 100, color="red", linestyle="--", label="전체 평균")
plt.legend()
plt.tight_layout()
plt.show()
```

**인사이트**

- 11월(아마 블랙 프라이데이 포함)이 **가장 높은 전환율**.
- 2월은 전환율이 매우 낮음.
- → **계절성과 프로모션(특별 행사) 효과를 명확히 반영**하는 데이터 →  
  모델링 시 **Month를 단순 카테고리로 두거나, high-season vs low-season으로 이분화** 가능.

### 8.2 방문자 유형별 전환율

```python
visitor_conv = pd.crosstab(df["VisitorType"], df["Revenue"], normalize="index") * 100
visitor_conv = visitor_conv.rename(columns={False: "No_Purchase", True: "Purchase"})
display(visitor_conv)

visitor_conv["Purchase"].sort_values(ascending=False).plot(kind="bar", color="limegreen")
plt.ylabel("Purchase Rate (%)")
plt.title("방문자 유형별 구매 전환율")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

**인사이트**

- **New_Visitor의 전환율이 Returning_Visitor보다 높음.**
  - 일반적인 “충성 고객 > 신규” 패턴과 다르게, **프로모션/광고 유입 신규 트래픽이 강하게 구매를 일으키는 구조**로 해석 가능.
- “Other”는 샘플 수가 적어 해석에 주의.

### 8.3 주말 vs 평일

```python
weekend_conv = pd.crosstab(df["Weekend"], df["Revenue"], normalize="index") * 100
weekend_conv = weekend_conv.rename(index={False: "Weekday", True: "Weekend"})
display(weekend_conv)

weekend_conv["Purchase"].plot(kind="bar", color=["steelblue", "orange"])
plt.ylabel("Purchase Rate (%)")
plt.title("주말 여부별 구매 전환율")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

**인사이트**

- 주말이 약간 더 높은 전환율을 보이지만, **월/프로모션 효과만큼 극적이지는 않음**.
- 이탈 예측에서는 **보조적인 feature로 활용**.

***

## 9. 이탈(비구매) 원인 요약 & 해결 방향

결과를 **“이탈 원인” + “해결책/실행 방향”**으로 정리하면 다음과 같습니다.

### 9.1 이탈(비구매)로 이어지는 패턴

1. **PageValues = 0인 세션이 대부분 이탈**
   - 결제/구매 가치가 있는 페이지에 도달하지 못한 세션.
   - 주로 **초기 탐색만 하다가 이탈**하는 패턴.

2. **BounceRates / ExitRates가 높은 세션**
   - 첫 페이지에서 바로 이탈(높은 Bounce)  
   - 상품/정보 페이지 중간 단계에서 이탈(높은 Exit)
   - 이는 **랜딩 페이지의 매칭 실패, 페이지 속도, 콘텐츠 품질, UI/UX 문제** 등을 시사.

3. **짧은 체류시간과 적은 페이지 뷰**
   - ProductRelated/Administrative/Informational 페이지 수가 적고, 체류시간도 짧은 세션은 전환율 낮음.
   - 즉, **상품 탐색 깊이가 얕은 세션**은 구매로 이어지지 않음.

4. **비효율적인 트래픽 소스**
   - Region, TrafficType, Browser, OperatingSystems 별 전환율 분석을 추가로 해보면  
     일부 트래픽 소스는 **거의 전환이 없는 “노이즈 트래픽”**일 가능성.
   - 이들은 모델에서 **이탈 클래스 쪽으로 강한 신호**가 될 수 있음.

### 9.2 해결책/실행 가능한 액션 아이디어

EDA 기반으로, 실제 비즈니스에서 취할 수 있는 방향을 데이터로 뒷받침할 수 있습니다.

- **PageValues 높이기(가치 페이지 유도)**
  - 연관 상품 추천, 장바구니/결제 버튼 노출 강화, 장바구니 유지 기능 등으로  
    사용자가 결제 플로우에 진입하기 쉽게 설계.
- **Bounce/Exit 집중 구간 최적화**
  - Bounce/Exit가 높은 페이지(특히 상품 리스트, 상세 페이지, 장바구니/결제 단계)를 찾아  
    - 로딩 속도, 정보량(가격, 리뷰, 재고, 배송 정보), 신뢰 요소(리뷰, 보증),  
    - 폼 간소화, 결제 수단 다양화 등을 개선.
- **고품질 트래픽 확보 및 저품질 트래픽 축소**
  - 전환율이 높은 TrafficType/Region/VisitorType 패턴을 기반으로  
    - 마케팅 캠페인 타겟팅 최적화  
    - 전환이 거의 없는 채널/캠페인의 예산 축소.
- **시즌성 전략**
  - 11월/10월 등 전환이 높은 시점에 프로모션 극대화,  
  - 전환이 낮은 기간(2월 등)에는 **관여도/브랜딩 캠페인 위주** 운영.

***

## 10. 이후 모델링을 위한 데이터 가공 방향

EDA를 바탕으로 **이탈 예측 모델**을 구축할 때 권장하는 전처리/피처 엔지니어링 방향은 다음과 같습니다.

1. **타깃 정의**
   - `y = (~Revenue).astype(int)`로 **이탈(비구매)=1**로 두면 churn 관점에 직관적.
   - 또는 `Revenue` 그대로 사용해도 무방하나, 해석을 “구매 예측”으로 두게 됨.

2. **수치형 변환**
   - `TotalDuration`, `PageValues`, `*_Duration` 등에 대해
     - 로그 변환 `np.log1p`  
     - 상하위 1~2% 클리핑으로 스케일 안정화.
   - `BounceRates`, `ExitRates`는 0~0.2 사이라 그대로 사용 가능.

3. **범주형 인코딩**
   - `Month`, `VisitorType`, `Weekend`, `Region`, `TrafficType`, `Browser`, `OperatingSystems` 등
     - **원-핫 인코딩** 또는  
     - 트리 계열 모델(LightGBM, CatBoost) 사용 시 **카테고리로 직접 사용**.

4. **불균형 대응**
   - 베이스라인:  
     - 트리 계열 모델 + `class_weight` 혹은 `scale_pos_weight`.
   - 고도화:  
     - Train set에서 SMOTE/언더샘플링 조합,  
     - 평가 지표는 **ROC-AUC + F1, Recall, Precision-Recall AUC** 중심.

5. **평가지표**
   - 단순 Accuracy 대신,  
     - Recall(이탈을 얼마나 잘 잡는지),  
     - Precision(잡은 이탈이 실제 이탈인지),  
     - F1-score, ROC-AUC를 함께 보고 **비즈니스 코스트 구조에 맞는 threshold 튜닝**.

***

## 11. 마크다운 결론

```markdown
## EDA 결론 및 이탈(비구매) 인사이트 요약

1. **데이터 품질**
   - 결측치는 존재하지 않으며, 중복 데이터는 약 1% 수준으로 모델링 전 제거 가능하다.
   - 타깃(`Revenue`)은 약 15%가 구매, 85%가 비구매로 클래스 불균형(약 5.5:1)이 존재한다.

2. **이탈과 가장 관련이 깊은 행동 특성**
   - `PageValues`는 구매 여부와 가장 강한 양의 상관을 보이며, 0인 세션은 대부분 비구매로 끝난다.
   - `BounceRates`와 `ExitRates`는 전환과 음의 상관을 보이며, 값이 높을수록 구매율이 급격히 감소한다.
   - `ProductRelated`/`Administrative`/`Informational` 페이지 수와 체류시간은 많을수록 구매 확률이 높다.

3. **사용자 여정 관점의 이탈 패턴**
   - 가치 페이지(`PageValues > 0`)에 도달하지 못한 세션에서 이탈이 집중적으로 발생한다.
   - 초기 랜딩에서 바로 나가거나(높은 Bounce), 상품/체크아웃 중간 단계에서 떠나는(높은 Exit) 세션이 많다.
   - 구매 세션은 비구매 세션보다 총 체류시간이 약 15분 정도 더 길다.

4. **세그먼트 관점 인사이트**
   - 신규 방문자의 전환율이 재방문자보다 높은 특이 패턴이 존재하며, 이는 프로모션/광고 유입의 영향으로 해석할 수 있다.
   - 11월 등 특정 월(프로모션 시즌)에 전환율이 크게 상승하는 계절성이 존재한다.
   - 주말은 평일보다 다소 높은 전환율을 보이나, 월별 효과만큼 크지는 않다.

5. **이탈 감소를 위한 실행 방향**
   - 가치 페이지 유도를 강화하여 `PageValues > 0` 세션 비율을 높인다(추천 상품, 장바구니 진입 유도 등).
   - Bounce/Exit가 높은 핵심 페이지(상품 리스트, 상세, 장바구니/결제)의 UX와 신뢰 요소를 개선한다.
   - 전환율이 낮은 트래픽 소스를 줄이고, 전환율이 높은 채널/세그먼트에 마케팅 자원을 집중한다.
   - 시즌성과 방문자 유형을 고려한 맞춤 프로모션 전략을 수립한다.

6. **모델링을 위한 제안**
   - 타깃은 이탈(비구매)을 1로 두어 churn 관점에서 예측 모델을 구축한다.
   - Duration 및 PageValues 관련 변수는 로그 변환과 클리핑을 통해 이상치 영향을 완화한다.
   - 범주형 변수는 적절한 인코딩 후, 불균형 대응 기법과 함께 트리 계열 모델(LightGBM, CatBoost 등)을 사용하는 것을 추천한다.
   - 평가 지표는 Accuracy 대신 ROC-AUC, F1, Recall, Precision-Recall AUC를 중심으로 설정한다.
```

***

## 12. 정리


1. **데이터 품질 점검 → 이상치/분포 확인 → 타깃 기준 그룹 비교 → 상관분석 → 세그먼트별 전환율 분석 → 인사이트 정리** 
2. 이 EDA는 바로 뒤에 **“이탈(비구매) 예측 모델”**을 붙이기 좋은 구조
   - 피처 엔지니어링 코드
   - Train/Validation 분할 및 불균형 처리
   - 베이스라인 모델 (예: LightGBM, CatBoost)
   로 자연스럽게 확장할 수 있습니다.
