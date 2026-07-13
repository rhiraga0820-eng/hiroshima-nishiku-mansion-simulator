# 広島マンション価格・売却シミュレーター

## 主な機能

- 駅名から駅ランクへ自動変換
- 間取りから部屋数へ自動変換
- XGBoostによる現在価格予測
- 最大50年先までの価格予測
- 政策金利の期間別シナリオ
- 広島市CPIの期間別シナリオ
- 固定金利・元利均等ローン残債
- ボーナス追加返済
- 売却費用控除
- 売却後手残り
- 価格・残債・手残りグラフ
- 年次結果CSV出力
- 学習範囲外警告
- モデル精度・特徴量重要度表示

## フォルダ構成

```text
mansionv2/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ model/
│  └─ mansion_xgb_interaction_bundle_v5.joblib
└─ src/
   ├─ __init__.py
   ├─ prediction.py
   ├─ scenarios.py
   └─ loan.py
```

## モデル配置

Colabで作成した以下のファイルを `model` フォルダへ入れてください。

```text
mansion_xgb_interaction_bundle_v5.joblib
```

## Windows・VS Codeでの起動

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## GitHubへ登録

```powershell
git init
git add .
git commit -m "Complete mansion price simulator"
git branch -M main
git remote add origin <GitHubリポジトリURL>
git push -u origin main
```

## 注意事項

- 20年超、特に50年予測は参考シナリオです。
- XGBoostは学習範囲外を線形外挿しません。
- ローン計算は固定金利・元利均等の簡易試算です。
- 税金、管理費、修繕積立金、譲渡所得税等は含みません。
