# 発話口形状の再現AR

## 推奨環境
- OS: Ubuntu 22.04 などの標準的な Linux ディストリビューション
- 理由:
  - カメラ入力を直接利用するため、WSL や仮想環境では動作しない
  - ShellScript によるセットアップを行うため、Ubuntu 固有のコマンドを使用

## 1. 環境構築
### 1.1 プロジェクトのダウンロード
```bash
git clone https://github.com/kohama-yujin/b4-lip-sync-ar.git
```

### 1.2 仮想環境の作成
```bash
cd b4-lip-sync-ar
# 作成
python3 -m venv b4lip
# 有効化
source b4lip/bin/activate
```
> 以下のように(b4lip) が付けば仮想環境有効化成功
> (b4lip) ---@~~~:/b4-lip-sync-ar$
> ※ 無効化したいときは`deactivate`を実行

### 1.3 パッケージのインストール  
 ```bash
# パッケージ情報の更新
sudo apt update
# GTK3 開発用ライブラリ（GUIアプリ作成に必要）
sudo apt install -y libgtk-3-dev pkg-config
# 音声処理ライブラリや開発ツール
sudo apt-get install -y portaudio19-dev python3-dev build-essential
# Python GUI 用ライブラリ Tkinter
sudo apt-get install -y python3-tk
# 日本語形態素解析ライブラリ MeCab
sudo apt install -y mecab libmecab-dev mecab-ipadic-utf8
# 動画・音声処理ライブラリ
sudo apt install -y ffmpeg
# 開発用ライブラリ（圧縮・SDL2 など）
sudo apt install -y build-essential zlib1g-dev libsdl2-dev
# Perl インタプリタ（Julius スクリプト実行に必要）
sudo apt-get install -y perl
```

### 1.4 ライブラリのインストール
```bash
pip install -r requirements.txt
```

### 1.5 フレーム補間の学習済みモデルをダウンロード
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

### 1.6 音声認識エンジンをダウンロード
```bash
# 任意の作業フォルダに移動
cd ~/Downloads
git clone https://github.com/julius-speech/julius.git
cd julius
# 2. 設定
./configure
# 3. コンパイル
make
# 4. システムへのインストール
sudo make install
```

## 2. 実行
```bash
cd b4-lip-sync-ar
# 仮想環境有効化
source b4lip/bin/activate
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
