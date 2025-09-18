# main.py
# editor : tagawa kota, sugano yasuyuki
# last updated : 2023/6/9
# overview : 
# Input and set 3dpoint, 3dmodel and imagefile. 
# Run main loop of app.

import sys
import numpy as np
import cv2
import glfw
from mqoloader.loadmqo import LoadMQO
import Application
# 追加
import USBCamera as cam
from create_MQO import CreateMQO
from create_MQO import CreateMQO
from TextAnalysis import TextAnalysis
from ContourAlpha import ContourAlpha
from SpeechRecognition import SpeechRecognition
import os
#import PySimpleGUI as sg
import TkEasyGUI as sg
import argparse
import time
import datetime
from datetime import date, timedelta
from OpenGL.GL import *

import concurrent.futures

# カレントディレクトリを実行中のファイル(このファイル)のディレクトリに移動
#os.chdir(os.path.dirname(os.path.abspath(__file__)))
#
# メインクラス
#        
class Main:
    
    #
    # コンストラクタ
    # (@param kwargs : image = "image_filename"
    #                  texture = "texture_filename")
    #
    def __init__(self, texture, image, draw_landmark, draw_poseline, run, use_cut, player_name, interp_frames):

        if not texture == None:
            self.take_texture = False
            texture_filename = texture
        else:
            self.take_texture = True
        
        if not image == None:
            # 画像入力の場合
            self.use_camera = False
            # 入力画像をセット
            image_filename = image
            input_image = cv2.imread(image_filename)
            width = input_image.shape[1]
            height = input_image.shape[0]            
        else:
            # カメラ入力の場合
            self.use_camera = True
            # ディスプレイサイズ
            width  = 640
            height = 480
        #use_api = cv2.CAP_DSHOW # Windowsで使用する場合こちらを使う
        use_api = 0 # Linuxで使用する場合はこちらを使う

        if interp_frames == None:
            interp_frames = 0
        else:
            interp_frames = int(interp_frames)+1

        mouth_list = ['a', 'i', 'u', 'e', 'o', 'n']
        #self.today = str(date.today() - timedelta(days=1)).replace('-','')
        self.today = str(datetime.date.today()).replace('-','')
        self.data_folda = "mqodata/model/"+player_name
        
        # 音声認識クラス
        self.sp_recognizer = SpeechRecognition(self.data_folda, self.today)

        #
        # アプリケーション設定
        # Applicationクラスのインスタンス生成
        #
        title = 'Q:終了, M:発話再現, T:テキスト入力, V:音声入力'
        self.app = Application.Application(title, width, height, use_api, self.use_camera, use_cut, self.data_folda, interp_frames)

        #
        # 描画設定
        #
        #self.app.set_draw_landmark(draw_landmark)
        #self.app.set_draw_poseline(draw_poseline)
        
        #
        # データ保存ディレクトリの作成
        #
        if not os.path.isdir(self.data_folda):
            os.makedirs(self.data_folda)
            for m in mouth_list:
                os.makedirs('{}/{}'.format(self.data_folda,m))
            roop1 = 0
            roop2 = 0
            for m1 in mouth_list:
                roop2 = 0
                for m2 in mouth_list:
                    if roop1 < roop2:
                        os.makedirs('{}/{}-{}'.format(self.data_folda, m1, m2))
                    roop2 += 1
                roop1 += 1
            os.makedirs('{}/free'.format(self.data_folda))
            os.makedirs('{}/speech'.format(self.data_folda))

        if run: # 実行時
            mouth_shape = "free"
            texture_filename = "mouthShape_n.png"
            #
            # モデル読み込み
            # CreateMQOクラスのインスタンス生成
            #
            start = time.time()
            mqo = CreateMQO(texture_filename, use_cut, mouth_shape, self.data_folda)
            model_path = os.getcwd() +"/"+ mqo.model_filename
            end = time.time()
            print("Created. "+str(end-start)+'sec.')
        else: # データ作成時            
            #
            # テクスチャ撮影
            # 初期テクスチャを使用する場合、回転の補正は使用しない(モデルの向きがおかしくなるため)
            #
            if self.take_texture:
                # GUI設定
                #sg.theme("DefaultNoMoreNagging")
                sg.theme("clam")
                #layout = [[sg.Text("テクスチャ画像を撮影しますか？(口の形：{})".format(mouth_shape))]
                #        ,[sg.Button('Yes'), sg.Button('No')]]
                layout = [[sg.Text("あ(a)、い(i)、う(u)、え(e)、お(o)、ん(n)の発音形状を撮影します")]
                        ,[sg.Text("ウィンドウ上部の口の形を参照してください。")]
                        ,[sg.Text("撮影：Sキー")]
                        ,[sg.Text("※ なるべく顔の位置が変わらないようにしてください。")]
                        ,[sg.Text("※ 大げさにせず、日常会話時の口の開きをイメージしてください。")]
                        ,[sg.Button('OK', key='-Yes-')]]
                self.window = sg.Window('撮影', layout)
                # ウィンドウ読み込み
                event, values = self.window.read()
                # イベント処理
                if event in (None, 'Cancel'):
                    # 右上のボタンでウィンドウを閉じた場合、プログラムを終了
                    sys.exit("強制終了")
                elif event == "-Yes-":
                    for m in mouth_list:
                        mouth_shape = m
                        texture_filename = "mouthShape_{}.png".format(mouth_shape)
                        while True:
                            # カメラ画像読み込み
                            success, image = self.app.camera.CaptureImage()
                            if not success:
                                print("error : video open error")
                                return
                            # 画像を表示
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            cv2.imshow("{}".format(mouth_shape), image)
                            cv2.moveWindow('{}'.format(mouth_shape), 2500, 200)
                            # `s`キーを押すと画像を撮影しループ終了
                            k = cv2.waitKey(1)
                            if k == ord('s'):
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                self.app.camera.SaveImage(image, self.data_folda+'/'+texture_filename)
                                break
                            # `q`キーを押すと画像を撮影せずにループ終了
                            if k == ord('q'):
                                break
                        #
                        # モデル読み込み
                        # CreateMQOクラスのインスタンス生成
                        #
                        start = time.time()
                        mqo = CreateMQO(texture_filename, use_cut, mouth_shape, self.data_folda)
                        model_path = os.getcwd() +"/"+ mqo.model_filename
                        end = time.time()
                        print("Created. "+str(end-start)+'sec.')                        
                        # ウィンドウの終了
                        cv2.destroyAllWindows()
                elif event == "No":
                    # 画像を撮影せずに、デフォルトのテクスチャ画像を使用
                    pass
                # ウィンドウクローズ
                self.window.close()

        # カメラを閉じる
        self.app.camera.Close()

        # フレーム数の指定がない場合表示しない
        if interp_frames == 0:
            return
        
        # 輪郭のアルファ処理
        # GUI設定
        sg.theme("clam")
        #layout = [[sg.Text("テクスチャのアルファ値を変更しますか？")]
        #        ,[sg.Button('Yes'), sg.Button('No')]]
        layout = [[sg.Text("実行準備をします")]
                ,[sg.Button('OK', key='-Yes-')]]
        self.window = sg.Window('アルファ変更', layout, (200, 80))        
        # ウィンドウ読み込み
        event, values = self.window.read()     
        # イベント処理
        if event in (None, 'Cancel'):
            # 右上のボタンでウィンドウを閉じた場合、プログラムを終了
            sys.exit("強制終了")
        elif event == "-Yes-":
            # スプライン補間を行うかどうか
            use_spline = False
            # 親テクスチャのアルファ処理
            for m in mouth_list:
                self.contour_alpha = ContourAlpha(self.data_folda+'/mouthShape_'+m+'.png', use_cut, False, use_spline)
            # 補間テクスチャのアルファ処理
            dict_mouth = {"a":0, "i":1, "u":2, "e":3, "o":4, "n":5}
            for mouth1 in mouth_list:
                for mouth2 in mouth_list:
                    if dict_mouth[mouth1] < dict_mouth[mouth2]:
                        for now_frame in range(interp_frames+1):
                            if use_cut:
                                texture_name = os.getcwd() + "/{}/{}-{}/cut_{}_{}-{}_{}-{}.png".format(self.data_folda, mouth1, mouth2, self.today, mouth1, mouth2, interp_frames, now_frame)
                            else:
                                texture_name = os.getcwd() + "/{}/{}-{}/{}_{}-{}_{}-{}.png".format(self.data_folda, mouth1, mouth2, self.today, mouth1, mouth2, interp_frames, now_frame)
                            self.contour_alpha = ContourAlpha(texture_name, use_cut, False, use_spline)
                            print("Changed {}-{}_{}-{} alpha value... ".format(mouth1, mouth2, interp_frames, now_frame))
        elif event == "No":
            # テクスチャのアルファ値は設定済み
            pass
        # ウィンドウクローズ
        self.window.close()

        # 3次元データリストの読み込み
        self.app.set_3D_list(mqo.cnt, mqo.delete_datalist, mqo.datalist, mqo.datalist1, mqo.datalist2, mqo.right_list, mqo.left_list, mqo.mesh)
        #
        # モデルの読み込み
        #
        model_list = [0 for i in range(len(mouth_list)+1)]

        for i, m in enumerate(mouth_list):
            if use_cut:
                model_filename = os.getcwd() + "/{}/{}/cut_{}_{}.mqo".format(self.data_folda, m, self.today, m)
            else:
                model_filename = os.getcwd() + "/{}/{}/{}_{}.mqo".format(self.data_folda, m, self.today, m)
            
            model_list[i] = self.load_model(model_filename)
        model_list[-1] = self.load_model(model_path)

        # モデルリストのセット
        self.app.set_parent_list(model_list)

        # 現在のモデルを表示
        self.app.display(model_list[-1])
        
        if self.use_camera:
            if self.take_texture:
                # アプリケーションのカメラオープン
                self.app.camera.Open(width, height, None, use_api)
            else:
                pass
        else:
            # 入力画像をセット
            self.app.set_image(input_image)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        executor.submit(self.speech)
        executor.submit(self.mainroop(texture, image, draw_landmark, draw_poseline, run, use_cut, player_name, interp_frames))
    
    def mainroop(self, texture, image, draw_landmark, draw_poseline, run, use_cut, player_name, interp_frames):
        #
        # アプリケーションのメインループ
        #
        self.end = False
        while not self.app.glwindow.window_should_close():
            # カメラ映像の表示(メインの処理が記述される)
            self.app.display_func(self.app.glwindow.window)
            # イベントを待つ
            glfw.poll_events()
            
            self.app.mouth_motion() 

        # glfwの終了処理
        glfw.terminate()
        
        # Applicationクラスのインスタンス削除
        del self.app

        self.end = True
    
    def speech(self):
        while True:
            # wavファイル再生
            if self.app.on_playwav:
                self.sp_recognizer.play_wav(self.app.text)
                self.app.on_playwav = False
            
            # 音声認識・文字起こし
            if self.app.on_transcription:
                self.sp_recognizer.transcription()
                # text上書き
                self.app.text = self.sp_recognizer.text
                self.app.on_transcription = False

                # wavファイルを保存
                wav_name = '{}/speech/{}_{}.wav'.format(self.data_folda, self.today, self.app.text)
                with open(wav_name, 'wb') as f:
                    f.write(self.sp_recognizer.audio.get_wav_data())

                # wavファイルをffmpegで変換
                print(wav_name)
                os.system('ffmpeg -y -i {} -ac 1 -ar 16000 -acodec pcm_s16le {}/speech/tmp.wav'.format(wav_name, self.data_folda))
                os.remove(wav_name)
                os.rename('{}/speech/tmp.wav'.format(self.data_folda), wav_name)
                
                # txtファイルを保存
                self.text_analyzer = TextAnalysis(self.app.text, self.data_folda, self.today)
                self.text_analyzer.save_hiraText()

                # labファイルを作成
                os.chdir('../segmentation-kit')
                os.system('perl segment_julius.pl ../create_face_model/{}/speech'.format(self.data_folda))
                os.chdir('../create_face_model')

            time.sleep(0.1)
                    
            # 終了条件
            if self.end:
                break

    def load_model(self, model_filename):        
        #
        # 3次元モデルの読み込み
        #   (OpenGLのウィンドウを作成してからでないとテクスチャが反映されない)
        #
        msg = 'Loading %s ...' % model_filename
        print(msg)
        
        start = time.time()
        #
        # 第3引数をTrueにすると面の法線計算を行い、陰影がリアルに描画されます（実行できない）
        # その代わりに計算にかなり時間がかかります
        #
        self.app.use_normal = False
        model_scale = 10.0
        model = LoadMQO(model_filename, model_scale, self.app.use_normal)

        end = time.time()
        print('Loaded. '+str(end-start)+'sec.')
        
        return model

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(
        usage =  "python main.py <-t texture_filename> <-i image_filename> <-n player_name> <-f frames>",
        description = "description for commandline arguments",
        epilog = "end",
        add_help = True,
    )
    parser.add_argument("-t", "--texture", help = "texture_filename")
    parser.add_argument("-i", "--image", help = "image_filename")
    parser.add_argument('-l', action='store_true', help = "flag for draw_landmark")
    parser.add_argument('-p', action='store_true', help = "flag for draw_poseline")
    parser.add_argument('-r', action='store_true', help = "flag for run")
    parser.add_argument('-c', action='store_true', help = "flag for use_cut")
    parser.add_argument("-n", "--name", help = "player_name")
    parser.add_argument('-f', "--frames", help = "interp_frames")
    args = parser.parse_args()

    Main(args.texture, args.image, args.l, args.p, args.r, args.c, args.name, args.frames)
    end = time.time()
    print('All Run Time. '+str(end-start)+'sec.\n')