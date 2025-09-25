# 発話口形状の再現AR

## 注意事項
本プロジェクトは以下の理由により、**純粋な Ubuntu 環境での実行を推奨**します。
- カメラ入力を直接使用しているため、WSL や他の仮想環境では動作しません。
- ShellScript を用いた実行手順が含まれており、Ubuntu 固有のコマンドを使用しています。

そのため、Windows や macOS 上の WSL などでは正しく動作しない可能性があります。

## 1. 環境構築
### 1.1 プロジェクトのダウンロード
```bash
git clone https://github.com/kohama-yujin/b4-lip-sync-ar.git
```

### 1.2 仮想環境の作成
```bash
cd b4-lip-sync-ar
python3 -m venv b4lip
source b4lip/bin/activate
```
(b4lip) が付けば仮想環境有効化成功  
`(b4lip) ---@~~~:/b4-lip-sync-ar$`
> ※無効化したいときは`deactivate`を実行

### 1.3 ライブラリのインストール
```bash
pip install -r requirements.txt
```
> ※場合によっては以下のようにパッケージをインストールする必要があるかも  
> ```bash
> sudo apt update
> sudo apt-get install -y portaudio19-dev python3-dev build-essential
> sudo apt install -y libgtk-3-dev pkg-config
> ```

### 1.4 フレーム補間の学習済みモデルをダウンロード
1. [Google Drive](https://drive.google.com/drive/folders/1s9pbFx_bSbinhx5PChJwZqPsyRIlehmZ)を開き`variables.data-00000-of-00001`をダウンロード
1. ダウンロードしたファイルを`frame-interpolation/pretrained_models/film_net/Style/saved_model/variables`に配置
1. 以下のようなディレクトリ構造となっていることを確認

b4-lip-sync-ar/  
└── frame-interpolation/  
　　 └── pretrained_models/  
　　　　　└── film_net/  
　　　　　　　 └── Style/  
　　　　　　　　　　└── saved_model/  
　　　　　　　　　　　　 ├── keras_metadata.pb  
　　　　　　　　　　　　 ├── saved_model.pb  
　　　　　　　　　　　　 └── variables/  
　　　　　　　　　　　　　　　├── **variables.data-00000-of-00001** ←ここに配置  
　　　　　　　　　　　　　　　└── variables.index  


## 2. 実行
```bash
cd Scripts
```
### 2.1 3Dモデル作成 
```bash
./create_project.sh
```
- 名前入力  
`create_face_model/mqodata/model/`に3Dモデルを格納するフォルダが作成される。
- 撮影  
ウィンドウの指示に従う。撮影後、3Dモデルの作成が始まる。作成終了まで8分程かかる。

### 2.2 AR実行 
```bash
./run_project.sh
```
- 名前入力  
3Dモデル作成時に入力した名前を入力する。
> ※3Dモデル作成日が当日でない場合、3Dモデル作成から実行する必要がある。
- アルファ変更  
作成した3Dモデルの輪郭をなめらかにする。
- AR実行  
AR実行ウィンドウが起動する。実行キーは以下のとおりである。

| キー | 説明 |
| :--- | :--- |
| Qキー    | 終了 |
| Mキー    | 発話再現 |
| Tキー | テキスト入力 |
| Vキー | 音声入力 |
