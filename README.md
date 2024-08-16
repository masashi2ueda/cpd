# cpd
change point detection

## CUMSUM
### お気持ち
今ままでの値より大きく外れているものが続いたら、変化点。

### 擬似コード
```math
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
```
1. μとσを初期化