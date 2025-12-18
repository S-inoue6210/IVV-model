# IVV-model Repository

このリポジトリは、IVV (In-Venue Visit) モデルの実装を含んでいます。
主に、滞在データ（stay）とPOI候補（candidate）の距離やランクなどの特徴量を用いて、真の来訪POIを推定するモデルです。

## 構成

- `src/`: ソースコード（前処理、モデル定義、評価関数など）
- `notebook01.ipynb`: 基本モデル（距離とランクを使用）の実行例
- `notebook02_general.ipynb`: 汎用モデル（任意の特徴量をバケット化して使用）の実行例
- `data/`: データセット用ディレクトリ
- `outputs/`: 出力結果用ディレクトリ

## ソースコード (`src/`) の解説

`src` ディレクトリには、モデルの実装やデータ処理のための Python モジュールが含まれています。

### 1. `preprocessing.py`
基本モデル用の前処理関数群です。
- **`load_data`**: CSVファイルからデータを読み込みます。
- **`calculate_rank`**: 各滞在（`stay_id`）内での候補POIの距離ランクを計算します。
- **`filter_valid_stays`**: 正解ラベル（`label=1`）が1つだけ存在する有効な滞在データのみを抽出します。
- **`discretize_features`**: 距離（`distance_from_poi`）とランク（`rank`）をバケット化（離散化）します。
- **`train_test_split_by_stay`**: `stay_id` 単位で学習データとテストデータに分割します。

### 2. `model.py`
基本モデル（IVVModel）の実装です。距離とランクの2つの特徴量に特化しています。
- **`IVVModel` クラス**:
    - `__init__`: バケット数を受け取り、パラメータ（`phi_dist`, `phi_rank`）を初期化します。
    - `score`: 各候補のスコアを計算します（各特徴量の重みの積）。
    - `predict_proba`: スコアを滞在ごとに正規化し、来訪確率を算出します。
    - `log_likelihood`: データの対数尤度を計算します。
    - `train`: 勾配上昇法（Gradient Ascent）を用いてモデルのパラメータを学習します。

### 3. `evaluation.py`
モデルの評価を行うための関数群です。
- **`calculate_ndcg`**: NDCG@k を計算し、ランキングの精度を評価します。
- **`plot_confidence_margin`**: Confidence Margin（1位と2位の確率差）の分布をヒストグラムで可視化します。正解/不正解ごとの分布を確認できます。
- **`calculate_entropy`**: 予測確率分布のエントロピーを計算し、モデルの予測の不確実性を評価します。
- **`extract_mismatch_predictions`**: 予測と正解が一致しなかった事例を抽出します。

### 4. `preprocessing_general.py`
汎用モデル用の前処理関数群です。
- **`discretize_features`**: ユーザーが指定した特徴量リスト（`discretize_feature`）に基づいて、任意の特徴量をバケット化します。
- その他、`load_data`, `filter_valid_stays`, `train_test_split_by_stay` は基本版と同様の機能を提供します。

### 5. `model_general.py`
汎用モデル（IVVModel）の実装です。任意個の特徴量に対応しています。
- **`IVVModel` クラス**:
    - 初期化時に特徴量の設定（名前とバケット数）のリストを受け取ります。
    - 各特徴量に対してパラメータ（`phi`）を保持し、それらの積でスコアを計算します。
    - 学習ロジックは基本モデルと同様に、勾配上昇法を用いて各特徴量のパラメータを更新します。

## ノートブックの解説

### `notebook01.ipynb`
基本モデルのワークフローを示すノートブックです。
1. **データの読み込み**: `data/data_01.csv` などを読み込みます。
2. **前処理**: ランク計算、有効データフィルタリング、特徴量（距離・ランク）の離散化、Train/Test分割を行います。
3. **モデル学習**: `src.model.IVVModel` を使用して学習を行い、対数尤度の推移を表示します。
4. **評価**: NDCGの計算、Confidence Marginのプロット、エントロピーの計算を行い、Train/Testセットでの性能を確認します。
5. **結果の保存**: 予測結果（確率付与済み）をCSVとして保存します。

### `notebook02_general.ipynb`
汎用モデルのワークフローを示すノートブックです。
1. **設定**: 使用する特徴量とバケット数を辞書リストで定義します（例：`distance_from_poi_bucket`）。
2. **前処理**: `src.preprocessing_general` を使用して、指定された特徴量の離散化などを行います。
3. **モデル学習**: `src.model_general.IVVModel` を初期化・学習します。特徴量定義を渡すことで、柔軟にモデルを構築できます。
4. **評価・保存**: 基本モデルと同様に評価指標の計算と結果の保存を行います。
