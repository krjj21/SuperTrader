# SELL cooldown(3d) + RL prob floor(0.45) A/B — 20260507_174056

strategy=factor_hybrid + scale, MockLLM, period=2018-01-01~2024-12-31, train_ratio=0.5

```
                         OFF          ON     delta
total_return        5.204110    5.587681  0.383570
cagr                0.739680    0.792952  0.053272
sharpe_ratio       -1.110315   -0.926142  0.184173
sortino_ratio      -1.976335   -1.545082  0.431253
max_drawdown       -5.067087   -8.632003 -3.564916
calmar_ratio        0.145977    0.091862 -0.054116
win_rate           40.875912   35.338346 -5.537567
profit_factor       1.130413    1.124394 -0.006019
total_trades      137.000000  133.000000 -4.000000
avg_holding_days   12.094891   19.135338  7.040448
cooldown_blocked    0.000000    8.000000  8.000000
```
