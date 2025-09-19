# 発話再現

## 環境構築
### 1. プロジェクトのダウンロード
```bash
git clone https://github.com/kohama-yujin/B4-graduation-project
```

### 2. 仮想環境の作成
```bash
cd B4-graduation-project
python3 -m venv b4grad
source b4grad/bin/activate
```

### 3. ライブラリのインストール
```bash
sudo apt update
sudo apt install -y libgtk3-dev pkg-config
pip install -r requirements.txt
```

## 実行
### モデル作成 
```bash
./create_project.sh
```
