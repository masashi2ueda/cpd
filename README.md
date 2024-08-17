# cpd
change point detection

## CUMSUM
### お気持ち
今ままでの値より大きく外れているものが続いたら、変化点。

### 擬似コード
- ハイパーパラメータ
    - $k$:累積を減衰させる値 
    - $h$:変化点を検出する閾値

0. $平均\muと分散\sigma^2、累積はずれ値SとTを初期化$
    - $\mu_0=0$
    - $\sigma_0^2=1$
    - $S_0=0$
    - $T_0=0$
1. $新しい値x_iが入ってくる$
1. $\muと\sigma^2を更新$
    - $\mu_i \gets \frac{n-1}{n}\mu_{i-1} + \frac{1}{n}x_i$
    - $\sigma_i^2 \gets \frac{n-2}{n-1}\sigma_{i-1}^2 + \frac{(x_i - \mu_i)(x_{i-1} - \mu_{i-1})}{n-1}$
1. $x_iを正規化$
    - $\hat{x}_i \gets \frac{x_i - \mu_i}{\sigma_i} $
1. $SとTを更新$
    - $S_i \gets max(0, S_{i-1} -k + \hat{x}_i)$
    - $T_i \gets max(0, T_{i-1} -k - \hat{x}_i)$
1. 変化点を検出
    - $S_i > h もしくは T_i > hならiは変化点$
    - $iが変化点の場合はS_i \gets 0, T_i \gets 0$
1. 1に戻る
### 解説
- $値が通常より大きいのをSで、小さいのをTでそれぞれ捉える$
- $両方kにより常に減衰する（\hat{x}_iはkより大きくならないとたまらない）$

## EWMA: Exponentially Weighted Moving Average algorithm

### お気持ち
今ままでの値より大きく外れているものがあったら変化点

### 擬似コード
- ハイパーパラメータ
    - $r:値を更新する際の現在の値の重み$
    - $L:この値*偏差値より値がズレていたら変化点$

0. $平均\muと分散\sigma^2、Zを初期化$
    - $\mu_0=0$
    - $\sigma_0^2=1$
    - $Z_0=0$
1. $新しい値x_iが入ってくる$
1. $xを更新$
    - $Z_i \gets (1-r)Z_{i-1}+rx_i$
1. $\muと\sigma^2を更新$
    - $\mu_i \gets \frac{n-1}{n}\mu_{i-1} + \frac{1}{n}x_i$
    - $\sigma_i^2 \gets \frac{n-2}{n-1}\sigma_{i-1}^2 + \frac{(x_i - \mu_i)(x_{i-1} - \mu_{i-1})}{n-1}$
1. 変化点を検出
    - $Z_i > \mu_i+L\sigma_Z もしくは Z_i < \mu_i-L\sigma_Z ならiは変化点$
1. 1に戻る
### 解説
- $現在の値のiirを取りながら、平均、分散の統計値より外れていたら変化点$
- CUMSUMと比べて、内部で正規化するか、変化点検出時に正規化するかの違い？

## Two-sample test algorithm

### お気持ち
変化点前後のデータの分布が異なるか検定する

### 擬似コード
- ハイパーパラメータ
    - $s:検定手法(例:Wilcoxonの順位和検定) $
    - $h:検定の閾値$

0. $最後の変化点\tau、最後の変化点からの値群を初期化$
    - $\tau=0$
    - $xs=[]$
1. $新しい値x_iが入ってくる→xを追加$
    - $xs.append(x_i)$
1. $現在のxs(サイズ=n)の各変化点候補で検定$
    1. $k = 1, Ds = []$
    1. $xs1 = xs[1:k], xs2 = xs[k+1:n]$
    1. $Ds.appen(xs1とxs2の分布が異なる確率by s)$
    1. $if k == n: 終了, else: k \gets k+1でiへ$
    1. $もしmax(Ds)>hなら、Dsが最大の位置が変化点→\tau$
1. 1に戻る
### 解説
- 最後の変化点から現在の値までの各位置で検定する
- 変化点が見つからないとどんどん計算量が増えていく→xsを保存するサイズの上限を設定する？

