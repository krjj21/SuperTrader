# XGB threshold sweep — 20260508_062117

strategy=factor_hybrid + scale, MockLLM, period=2018-01-01~2024-12-31, train_ratio=0.5, OOS 92회

```
                  buy045_sell050  buy050_sell055  buy055_sell060  buy060_sell065
total_return          -11.069352        2.219968        5.454102        5.108226
sharpe_ratio           -1.002298       -0.863406       -1.105915       -1.659185
sortino_ratio          -1.455044       -1.364200       -1.977660       -3.172953
max_drawdown          -16.066646       -6.498563       -4.759303       -3.521886
calmar_ratio           -0.105169        0.049159        0.162717        0.206235
win_rate               39.102564       39.661017       40.740741       43.396226
profit_factor           0.852810        1.029947        1.144645        1.474835
total_trades          624.000000      295.000000      135.000000       53.000000
avg_holding_days       12.580128       13.254237       11.881481       12.528302
```
