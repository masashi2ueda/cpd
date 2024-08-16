# cpd
change point detection

## CUMSUM
### お気持ち
今ままでの値より大きく外れているものが続いたら、変化点。

### 擬似コード

- ハイパーパラメータ
    - $k$:累積を減衰させる値 
    - $h$:変化点を検出する閾値

0. 平均$\mu$と分散$\sigma^2$、累積はずれ値$S$と$T$を初期化
    - $\mu_0=0$
    - $\sigma_0^2=1$
    - $S_0=0$
    - $T_0=0$
1. 新しい値$x_i$が入ってくる
1. $\mu$と$\sigma^2$を更新
    - $\mu_i \gets \frac{n-1}{n}\mu_{i-1} + \frac{1}{n}x_i$
    - $\sigma_i^2 \gets \frac{n-2}{n-1}\sigma_{i-1}^2 + \frac{(x_i - \mu_i)(x_{i-1} - \mu_{i-1})}{n-1}$
1. $x_i$を正規化
    - $\hat{x}_i \gets \frac{x_i - \mu_i}{\sigma_i} $
1. $S$と$T$を更新
    - $S_i \gets max(0, S_{i-1} -k + \hat{x}_i)$
    - $T_i \gets max(0, T_{i-1} -k - \hat{x}_i)$
1. 変化点を検出
    - $S_i > h$ or $T_i > h$なら$i$は変化点
    - $i$が変化点の場合は$S_i \gets 0, T_i \gets 0$
1. 1に戻る
### 解説
- 値が通常より大きいのを$S$で、小さいのを$T$でそれぞれ捉える
- 両方$k$により常に減衰する（$\hat{x}_i$は$k$より大きくならないとたまらない）
    