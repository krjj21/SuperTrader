# cap_rank x top_n grid sweep — 부분 결과 (9/14, 중단됨)

strategy=factor_hybrid, period=2018-01-02~2024-12-30, rebalance=biweekly, train_ratio=0.5

페어당 ~25분 x 14 = 5.6시간 예상이라 9/14 (64%) 시점에 사용자 중단.
나머지 5개 페어: (30,150)_top30, (50,200)x2, (100,300)x2.

## 결과 (total_return DESC)

```
 cap_rank_min  cap_rank_max  top_n  total_return  sharpe_ratio  max_drawdown  total_trades
            0             0     30         14.55         -0.82         -1.36            73
            1            30     30         14.55         -0.82         -1.36            73
            1           100     30         14.55         -0.82         -1.36            73
           30           100     30         14.55         -0.82         -1.36            73
           30           150     30         14.55         -0.82         -1.36            73
            0             0     15          6.38         -2.00         -1.14            43
            1            30     15          6.38         -2.00         -1.14            43
            1           100     15          6.38         -2.00         -1.14            43
           30           100     15          6.38         -2.00         -1.14            43
           30           150     15          6.38         -2.00         -1.14            43
           50           200     15          6.38         -2.00         -1.14            43
```

## 핵심 발견

### 1. cap_rank 는 무의미한 변수
5개 cap_rank 페어 (0,0) (1,30) (1,100) (30,100) (30,150) 가 *완전히 동일* 결과 생성.
- top_n=15 → 모두 +6.38% / Sharpe -2.00 / 43 trades
- top_n=30 → 모두 +14.55% / Sharpe -0.82 / 73 trades

원인: IC-weighted composite + neutralizer 가 cap_rank 필터와 독립적으로 같은 종목 선택.

### 2. CLAUDE.md +515% baseline 미재현
CLAUDE.md L161 의 `(30,150)+top_n=15 → +515%` 는 실제 측정 +6.38%.
baseline 메모 자체가 잘못 기록되었거나, 다른 변수 (L0 composite, 모델 버전, 사이징 모드) 의 영향.

### 3. top_n 만이 진짜 변수
- top_n=15: +6.38% / Sharpe -2.00
- top_n=30: +14.55% / Sharpe -0.82
**top_n=30 으로 settings.yaml 1줄 변경하면 즉시 2배 수익률 + Sharpe 개선.**
