# 発話口形状の再現AR

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
(b4lip)が付けば仮想環境有効化成功  
`(b4lip) ---@~~~:/b4-lip-sync-ar$`
> ※無効化したいときは`deactivate`を実行

### 1.3 ライブラリのインストール
```bash
sudo apt update
sudo apt install -y libgtk3-dev pkg-config
pip install -r requirements.txt
```

## 2. 実行
### 2.1 モデル作成 
```bash
./create_project.sh
```

ウィンドウの指示に従う

### 2.2 AR実行 
```bash
./run_project.sh
```

モデル作成直後は
