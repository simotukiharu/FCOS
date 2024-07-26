物体検出をするためのリポジトリ
アノテーションのクラス数は背景含め4としているので、目的に応じて変更してください。

機械学習で使用するファイル
train.py
config.py(モジュールとして使用)
load_dataset_annot.py(モジュールとして使用)

検出結果であるバウンディングボックスを応用して日の丸構図と三分割構図に変換します。
日の丸構図で使用するソースコード
composition_center.py(モジュールとして使用)
conversion_1img_center.py

3分割構図で使用するソースコード
composition_Ro3.py(モジュールとして使用)
conversion_1img_Ro3.py
