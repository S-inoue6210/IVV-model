# IVV-model
Gu, Qihang, et al. "Inferring venue visits from GPS trajectories." Proceedings of the 25th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems. 2017.

## 実装詳細

本リポジトリでは、GPS軌跡データから来訪施設を推定するIVV (Inferring Venue Visits) モデルを実装しています。

### 1. データ準備 (`src/preprocessing.py`)

入力データは、滞在点（stop）と候補施設（candidate venues）が紐付けられた状態を想定しています。

*   **ランク計算**: 各滞在点において、候補施設を距離順にランク付けします（最も近い施設がランク0）。
*   **特徴量の離散化**: 距離（Distance）とランク（Rank）をそれぞれ40個のバケットに離散化します。
    *   距離: 0〜500mを40等分
    *   ランク: 0〜39にクリッピング
*   **データ分割**: `stay_id` 単位で学習用（80%）と評価用（20%）に分割します。

### 2. モデル定義 (`src/model.py`)

離散選択モデル（Discrete Choice Model）に基づき、以下のスコアリング関数を学習します。

*   **スコア関数**: $s(v,m) = \Phi(D) \cdot \Phi(R)$
    *   $\Phi(D)$: 距離バケットに対する重み
    *   $\Phi(R)$: ランクバケットに対する重み
*   **目的関数**: 対数尤度（Log Likelihood）の最大化
    *   $LL = \sum_C \log s(v(m),m) - \sum_C \log \left( \sum_{v' \in V_m} s(v',m) \right)$
*   **最適化手法**: 勾配上昇法（Gradient Ascent）
    *   パラメータの更新には乗法的な更新則（Multiplicative Update）を使用しています。

### 3. 評価指標 (`src/evaluation.py`)

モデルの性能は以下の指標で評価します。

*   **NDCG@k** (Normalized Discounted Cumulative Gain): 上位k件のランキング精度 (k=1, 5, 10)
*   **MAP** (Mean Average Precision): 正解施設の順位に基づく平均適合率

## 実行方法

```bash
python main.py
```

実行すると、データの読み込み、前処理、モデルの学習、およびテストデータでの評価が行われます。
