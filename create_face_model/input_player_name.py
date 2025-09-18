import TkEasyGUI as sg
import sys

# データ保存ディレクトリの作成
# 
# GUI設定
#sg.theme("DefaultNoMoreNagging")
sg.theme("clam")
layout = [[sg.Text("ローマ字で名前を入力してください")]
        ,[sg.Text("例）田中太郎→TanakaTaro")]
        ,[sg.InputText(key='-Input-')]
        ,[sg.Button('OK', key='-Btn-')]]
window = sg.Window('名前入力', layout)
while True:
    event, value = window.read()  # イベントの入力を待つ
    # イベント処理
    if event in (None, 'Cancel'):
        # 右上のボタンでウィンドウを閉じた場合、プログラムを終了
        sys.exit("強制終了")
    elif event == '-Btn-':
        player_name = value['-Input-']
        break
    elif event is None:
        break
window.close()

sys.stdout.write(player_name)