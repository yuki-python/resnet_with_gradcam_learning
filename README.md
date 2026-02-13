# ResNet50 with Gradcam
ResNet50の転移学習時に、Gradcamによる注目領域の制御をしながらモデル学習をするプログラムです。


下記のような構成でデータセットを作成する必要があります（ここではポケットに手を入れているかどうかの2値分類を行うモデルを例としています）。
ただし、ここで、モデル学習時、trainデータから7:3の割合で学習するようにしており、valデータはモデル学習後の評価用データになります。
```
dataset/
├── train/
│   ├── pocket_positive_labelme/
│   │   ├── 00000.jpg
│   │   ├── 00000.json
│   │   ├── 00001.jpg
│   │   ├── 00001.json
│   │   └── ...
│   └── pocket_negative_labelme/
│       ├── 00000.jpg
│       ├── 00000.json
│       ├── 00001.jpg
│       ├── 00001.json
│       └── ...
├── val/
│   ├── pocket_positive_labelme/
│   │   ├── 00000.jpg
│   │   ├── 00000.json
│   │   └── ...
│   └── pocket_negative_labelme/
│       ├── 00000.jpg
│       ├── 00000.json
│       └── ...
```


jsonファイルは下記形式です。labelmeを使ってアノテーションすることを想定しています。
（positive（手がポケットに入っている）例）
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "hand_in_pocket",
      "points": [[120.5, 80.2], [130.1, 85.0], [128.3, 100.6], [115.0, 98.4]],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": "00012.jpg",
  "imageHeight": 224,
  "imageWidth": 224
}


（negative（手がポケットに入っていない、人がいない）例）
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [],
  "imagePath": "00045.jpg",
  "imageHeight": 224,
  "imageWidth": 224
}

