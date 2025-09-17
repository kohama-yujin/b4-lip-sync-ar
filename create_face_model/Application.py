# Application.py
# editor : tagawa kota, sugano yasuyuki
# last updated : 2023/6/9
# overview : 
# Display camera footage and 3D model and face landmark.
# Describe the processing of most of the app


import numpy as np
import datetime
import cv2
from OpenGL.GL import *
import glfw

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)  # For MacOS compatibility


import mediapipe as mp

import GLWindow
import PoseEstimation as ps
import USBCamera as cam
# 追加
from create_MQO import CreateMQO
from TextAnalysis import TextAnalysis
from SpeechRecognition import SpeechRecognition
#import PySimpleGUI as sg
import TkEasyGUI as sg
import os
from mqoloader.loadmqo import LoadMQO
from ultralytics import YOLO
import insightface
import time
from datetime import date, timedelta
import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

#
# MRアプリケーションクラス
#
class Application:

    #
    # コンストラクタ
    #
    # @param width    : 画像の横サイズ
    # @param height   : 画像の縦サイズ
    #
    def __init__(self, title, width, height, use_api, use_camera, use_cut, data_folda, interp_frames):        
        self.width   = width
        self.height  = height
        self.channel = 3
        self.use_camera = use_camera
        self.use_cut = use_cut
        self.data_folda = data_folda
        self.interp_frames = int(interp_frames)

        # 動的再現実行フラグ
        self.on_mouth_motion = False
        # 音声再生フラグ
        self.on_playwav = False
        # 音声文字起こしフラグ
        self.on_transcription = False

        # 日付の設定
        #self.today = str(date.today() - timedelta(days=1)).replace('-','')
        self.today = str(datetime.date.today()).replace('-','')
        # テキスト
        self.text = "こんにちは"
        # 箱ひげ図用のデータセット
        self.all_switch_list = []
        
        #
        # USBカメラの設定
        # USBCameraクラスのインスタンス生成
        #
        if use_camera:
            self.camera = cam.USBCamera(width, height, use_api)
        else:
            self.camera = cam.USBCamera(640, 480, use_api)
            #self.camera = cam.USBCamera(1280, 960, use_api)
            
        #
        # GLウィンドウの設定
        # GLウィンドウクラスのインスタンス生成
        #
        self.glwindow = GLWindow.GLWindow(title, width, height, self.display_func, self.keyboard_func)

        #
        # カメラの内部パラメータ(usbカメラ)
        #
        self.focus = 700.0
        self.u0    = width / 2.0
        self.v0    = height / 2.0

        #
        # OpenGLの表示パラメータ
        #
        scale = 0.01
        self.viewport_horizontal = self.u0 * scale
        self.viewport_vertical   = self.v0 * scale
        self.viewport_near       = self.focus * scale
        self.viewport_far        = self.viewport_near * 1.0e+6
        self.modelview           = (GLfloat * 16)()
        self.draw_axis           = False
        self.use_normal          = False
        
        #
        # カメラ姿勢を推定の設定
        # PoseEstimationクラスのインスタンス生成
        #
        self.estimator = ps.PoseEstimation(self.focus, self.u0, self.v0)
        self.point_3D = np.array([])
        self.point_3D_yolov8 = np.array([])
        self.point_list = np.array([])
        
        
        #
        # マスク着用有無の推論モデルYOLOv8(未使用)
        # train : Yolov8, datasets : face mask dataset(Yolo format)
        # initial_weight : yolov8n.pt , epoch : 200 , image_size : 640
        #
        self.use_mask = False
        if self.use_mask:
            self.mask_model = YOLO("./yolov8n/detect/train/weights/best.pt")
            self.mask = False # mask未着用
        else:
            self.mask = True # mask着用
        
        #
        # 高精度顔検出モデルinsightface(未使用)
        #
        self.use_faceanalysis = False
        if self.use_faceanalysis:
            # load detection model
            self.detector = insightface.model_zoo.get_model("models/det_10g.onnx")
            self.detector.prepare(ctx_id=-1, input_size=(640, 640))
        else:
            self.detect = False
        
        #
        # mediapipeを使った顔検出モデル
        # Mediapipe FaceMeshのインスタンス生成
        #
        self.use_facemesh = True
        self.face_mesh = None
        
        self.face_mesh_solution = mp.solutions.face_mesh.FaceMesh(
            static_image_mode = not self.use_camera,
            min_detection_confidence = 0.25,
            min_tracking_confidence = 0.25)

        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            thickness = 1, 
            circle_radius = 1)
        
        # 顔検出に用いる対応点に関する変数(顔全体の場合0)
        self.detect_stable = 0
        
        # 顔の方向ベクトルを記述するかどうか
        self.draw_headpose = False
        # 顔のランドマークを記述するかどうか
        self.draw_landmark = False
        # 顔向き
        self.angle = [0,0,0]
        
        # カウント用変数
        self.count_img = 0
        self.count_rec = 0
        self.count_func = 0
        self.exec_time = 0
        
        # 録画用変数
        self.use_record = False # 初期値はFalse
        self.video = None
        
        # その他の機能
        self.flag_save_matrix = 0
        
    
    #
    # カメラの内部パラメータの設定関数
    # 
    def SetCameraParam(self, focus, u0, v0):
        self.focus = focus
        self.u0    = u0
        self.v0    = v0

    #
    # マスクの着用判別(実行に時間がかかるため、リアルタイムでの使用が難しく未使用)
    #
    def Yolov8(self):
        if self.count_func % 100 == 0:
            # 画像に対して顔の占める割合が大きすぎると誤判別するため、リサイズ
            image = cv2.cvtColor (self.image, cv2.COLOR_BGRA2RGBA)
            img_resized = cv2.resize(image, dsize=(self.width // 2, self.height //2))
            back = np.zeros((self.height, self.width, 3))
            back[0:self.height // 2, 0:self.width // 2] = img_resized
            # save=Trueで結果を保存
            results = self.mask_model(back, max_det=1) 
            if(len(results[0]) == 1):
                names = results[0].names
                # 画像サイズを半分にしているため、座標を2倍してもとのスケールに戻す
                cls = results[0].boxes.cls
                # conf = results[0].boxes.conf
                name = names[abs(int(cls) - 1)]
                if name == "no-mask":
                    self.mask = False
                else:
                    self.mask = True
            else:
                # 検出できなかった場合、self.maskはそのまま
                pass
        else:
            pass
        
    #
    # 顔認識(マスクを着用している場合でも高成度で顔検出を行えるが、実行に時間がかかるため未使用)
    #
    def Retinaface(self):
        if self.use_faceanalysis:
            bboxes, kpss = self.detector.detect(self.image, max_num=1)
            if len(bboxes) == 1:
                self.bbox = bboxes[0]
                self.kps = kpss[0]
                return True
            else:
                return False
        
    #
    # カメラ映像を表示するための関数
    # ここに作成するアプリケーションの大部分の処理を書く
    #
    def display_func(self, window):
        # バッファを初期化
        #glClearColor(GLclampf red, GLclampf green , GLclampf blue, GLclampf alpha) # バッファを初期化するカラー情報
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # 画像の読み込み
        success = False
        if self.use_camera:
            success, self.image = self.camera.CaptureImage()
        else:
            success = True

        if not success:
            print("error : video error")
            return
    
        # 描画設定
        self.image.flags.writeable = False
       
        # マスク検出のメソッドを実行
        if self.use_mask:
            self.Yolov8()
        
        #
        # 顔特徴点検出(FaceMesh)を実行
        #
        if self.use_facemesh:
            self.face_mesh = self.face_mesh_solution.process(self.image)
        
        #
        # 画像の描画を実行
        #
        self.image.flags.writeable = True

        # ランドマークの描画
        if self.draw_landmark:
            # ランドマークを描画するメソッドを実行
            self.draw_landmarks(self.image)

        # 画像を描画するメソッドを実行
        self.glwindow.draw_image(self.image)
        
        
        # 
        # カメラ姿勢推定
        # 顔のランドマーク検出
        #
        if self.face_mesh.multi_face_landmarks:
            #
            # 座標の正規化用リスト
            #
            point_2D = []
            point_3D = []
            cnt = 0
            #
            # 対応点を指定(顔全体を用いる場合は0)
            #
            if self.detect_stable == 0:
                # print("all")
                point_list = self.point_list
                point_3D = self.point_3D
            elif self.detect_stable == 1:
                # print("upper")
                point_list = self.point_list1
                point_3D = self.point_3D1
            elif self.detect_stable == 2:
                # print("selected")
                point_list = self.point_list2
                point_3D = self.point_3D2
            elif self.detect_stable == 3:
                # print("left-right")
                if self.angle[0] >= 5:
                    point_list = self.left_list
                    point_3D = self.left
                elif self.angle[0] < -5:
                    point_list = self.right_list
                    point_3D = self.right
                else:
                    point_list = self.point_list1
                    point_3D = self.point_3D1
            else:
                point_list = self.point_list
                point_3D = self.point_3D
            
            #
            # 顔の特徴点を取得
            #
            for landmarks in self.face_mesh.multi_face_landmarks:
                for idx, p in enumerate(landmarks.landmark):
                    cnt += 1
                    if idx in point_list:
                        # 画像サイズに合わせて正規化  
                        point_2D.append([p.x * self.width, p.y * self.height])

            #
            # カメラ位置、姿勢計算
            #
            success, vector, angle = self.compute_camera_pose(point_2D, point_3D)
            self.angle = angle
            
            #
            # 顔向きの描画
            #
            if self.draw_headpose:
                # 顔向きを描画するメソッドを実行
                # 顔の中心(0,0)座標を基準にして、顔向きのベクトルを描画する
                scale = (int(point_2D[1][0]), int(point_2D[1][1]))
                self.draw_poseline(self.image, vector, scale)
            #
            # マスク着用時、モデルを描画
            #
            if success and self.mask:
                #pass
                self.draw_model()
                
    
        else:
            #
            # 検出が安定しない
            #
            print("not detection")    
        
        if self.count_func == 0:
            # 検出できなかった場合もとりあえず実行（一回顔を検出しないとカメラが起動しない）
            model_scale_X = 0.0
            model_scale_Y = 0.0
            model_scale_Z = 0.0
            glScalef(model_scale_X, model_scale_Y, model_scale_Z)
            self.model.draw()
            
        # 関数実行回数を更新
        self.count_func += 1
        
        # バッファを入れ替えて画面を更新
        glfw.swap_buffers(window)
        
        # 画像入力の場合、画像を保存しフラグをセットする
        """
        if not self.use_camera:
            self.save_image()
            self.count_img += 1
            # window_should_closeフラグをセットする。
            glfw.set_window_should_close(self.glwindow.window, GL_TRUE)
        """
        # 録画している場合画面を保存
        if self.use_record:
            frame = self.save_image()
            self.video.write(frame)

    #
    # モデル描画に関する処理を行う関数
    #
    def draw_model(self, scale_x = 1.0, scale_y = 1.0):
        #
        # モデル表示に関するOpenGLの値の設定
        #
        # 射影行列を選択
        glMatrixMode(GL_PROJECTION)
        # 単位行列
        glLoadIdentity()
        # 透視変換行列を作成            
        glFrustum(-self.viewport_horizontal, self.viewport_horizontal, -self.viewport_vertical, self.viewport_vertical, self.viewport_near, self.viewport_far)
        # モデルビュー行列を選択
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # モデルビュー行列を作成(よくわかってない)
        glLoadMatrixf(self.modelview)

        # 照明をオン
        if self.use_normal:
            # 光のパラメータの設定(光源0,照明位置,照明位置パラメータ)
            glLightfv(GL_LIGHT0, GL_POSITION, self.camera_pos)
            # GL_LIGHTNING(光源0)の機能を有効にする
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)

        model_shift_X = 0.0
        model_shift_Y = 0.0
        model_shift_Z = 0.0
        model_scale_X = 1.0 * scale_x
        model_scale_Y = 1.0 * scale_y
        model_scale_Z = 1.0 
    
        # 世界座標系の描画
        if self.draw_axis:
            mesh_size = 200.0
            mesh_grid = 10.0
            # カメラを平行移動
            glTranslatef(model_shift_X, model_shift_Y, model_shift_Z)
            # 回転(x方向に90度)
            glRotatef(90.0, 1.0, 0.0, 0.0)
            # 世界座標系の軸を描画する関数
            
            # xz平面のグリッドを記述するメソッド
            #self.glwindow.draw_XZ_plane(mesh_size, mesh_grid)
            # カメラをもとに戻す
            glRotatef(90.0, -1.0, 0.0, 0.0)
            glTranslatef(-model_shift_X, -model_shift_Y, -model_shift_Z)


        # 3次元モデルを描画
        glTranslatef(model_shift_X, model_shift_Y, model_shift_Z)
        # 3次元モデルのスケールに変更
        glScalef(model_scale_X, model_scale_Y, model_scale_Z)
        glRotatef(0.0, 1.0, 0.0, 0.0)
        # 3次元モデルを記述(mqoloderクラスのdrawメソッド)
        self.model.draw()

        # 照明をオフ
        if self.use_normal:
            # GL_LIGHTNING(光源0)の機能を無効にする            
            glDisable(GL_LIGHTING)
            glDisable(GL_LIGHT0)
    
        
    #
    # 検出した顔のランドマークを描画するかを設定する関数
    #
    def set_draw_landmark(self, draw_landmark):
        self.draw_landmark = draw_landmark
        
    #
    # 検出したランドマークを画像上に描画する関数
    #
    def draw_landmarks(self, image):
        if self.face_mesh.multi_face_landmarks:
            for face_landmarks in self.face_mesh.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    face_landmarks,
                    # 描画モード
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    self.drawing_spec,
                    self.drawing_spec)
    
    #
    # 顔向きを画像上に描画するかを設定する関数
    #
    def set_draw_poseline(self, draw_pose):
        self.draw_headpose = draw_pose
        
    #
    # 顔向きを画像上に描画する関数
    #
    def draw_poseline(self, image, vector, scale):
        vector = [ x-y for x, y in zip(vector, scale) ]
        self.glwindow.draw_line(vector)
        
    #
    # キー関数
    #
    def keyboard_func(self, window, key, scancode, action, mods):
        # Qで終了
        if key == glfw.KEY_Q:
            if self.use_record:
                print("録画を終了します")
                self.use_record = False
            # Notoフォントを指定（例: Noto Sans CJK JP）
            font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'  # 適切なフォントパスを指定
            font_prop = fm.FontProperties(fname=font_path)
            
            # 音素別の描画時間の折れ線グラフの作成
            if len(self.phoneme_list_x) != 0 and len(self.char_per_sec) != 0:
                # silB, silEを削除
                self.true_total_list_y = self.true_total_list_y[1:-1]
                self.phoneme_list_x = self.phoneme_list_x[1:-1]
                self.excess_time_list_y = self.excess_time_list_y[1:-1]
                self.total_time_list_y = self.total_time_list_y[1:-1]
                # ラベル
                x = range(len(self.phoneme_list_x))
                plt.xlabel("音素", fontproperties=font_prop)
                plt.ylabel("総合描画時間[秒]", fontproperties=font_prop)
                # 補間
                df = pd.DataFrame({"phoneme":self.phoneme_list_x, "total_time":self.total_time_list_y})
                df["total_time"] = df["total_time"].interpolate(method="linear")
                # プロット
                plt.plot(x, self.true_total_list_y, marker=".", label="理論値")
                plt.plot(x, df["total_time"].values, marker=".", label="実測値")
                plt.legend(prop=font_prop)
                # 目盛り
                plt.xticks(x, self.phoneme_list_x)
                plt.yticks(np.arange(0, round(self.total_time_list_y[-1]), step=0.5))
                # 保存
                plt.savefig("mqodata/output/graph/total_{}_{}.png".format(self.today, self.speech_text))
                print("音素別の総合描画時間に関する折れ線グラフを作成しました。")
                # リセット
                plt.figure()

                # ラベル
                plt.xlabel("音素", fontproperties=font_prop)
                plt.ylabel("超過時間[秒]", fontproperties=font_prop)
                # 補間
                df = pd.DataFrame({"phoneme":self.phoneme_list_x, "excess_time":self.excess_time_list_y})
                df["excess_time"] = df["excess_time"].interpolate(method="linear")
                # プロット
                plt.plot(x, df["excess_time"].values, marker=".", label="改善前")
                plt.legend(loc="upper left", prop=font_prop)
                # 目盛り
                plt.xticks(x, self.phoneme_list_x)
                plt.yticks(np.arange(-0.04, 0.05, step=0.01))
                # 保存
                plt.savefig("mqodata/output/graph/excess_{}_{}.png".format(self.today, self.speech_text), transparent=False)
                print("音素別の超過時間に関する折れ線グラフを作成しました。")
            
            # モデル変更時間の箱ひげ図の作成
            if len(self.all_switch_list) != 0:
                plt.figure()
                fig, ax = plt.subplots()
                bp = ax.boxplot(self.all_switch_list)
                plt.xlabel("実行回数[回]", fontproperties=font_prop)
                plt.ylabel("モデルの描画時間[秒]", fontproperties=font_prop)
                plt.ylim([0, round(max(self.all_switch_list[0]), 2)+0.02])
                plt.savefig("mqodata/output/graph/switch_interval_{}_{}.png".format(self.today, len(self.all_switch_list)))
                print("モデルの描画時間に関する箱ひげ図を作成しました。")
            # window_should_closeフラグをセットする。
            glfw.set_window_should_close(self.glwindow.window, GL_TRUE)

        # Sで画像の保存
        if action == glfw.PRESS and key == glfw.KEY_S:
            if self.use_record:
                print("録画実行中です...録画を終了してから画像の保存を実行できます")
            else:
                print("画像を保存します...")
                # 画像を保存する関数を実行
                self.save_image()
                # ランドマークを保存する関数を実行
                # self.save_landmarks()
                # 回転行列、並進ベクトルを保存するフラッグを立てる
                #self.flag_save_matrix = 0
                # 画像カウントを+1する
                self.count_img += 1
            
        # Wでウィンドウを表示(モデル変更用)
        if action == glfw.PRESS and key == glfw.KEY_W:
            #
            # GUI設定
            #
            #sg.theme("DefaultNoMoreNagging")
            sg.theme("clam")
            layout=[[sg.Text("ボタンを押してテクスチャファイルを選択してください")]
                    ,[sg.Button('man'), sg.Button('woman')]]
            self.window = sg.Window('select texture', layout)
            event, values = self.window.read()
            if event in (None, 'Cancel'):
                pass
            elif event == "man":
                mqo = CreateMQO('man.png')
            elif event == "woman":
                mqo = CreateMQO('woman.png')
            # 3次元モデルの対応点であるdatfileの場所を記述
            point_filename = "data/face_3D.dat"
            # 3次元データの読み込み
            point_3D = np.loadtxt(point_filename, delimiter=' ', dtype="double")
            self.estimator.set_3D_points(point_3D)
            model_filename = os.getcwd() +"/"+ mqo.model_filename
            self.display(model_filename)
            self.window.close()
        
        # Rで画面録画開始
        if action == glfw.PRESS and key == glfw.KEY_R:
            if self.use_record == False:
                # 録画用変数をTrueに
                self.use_record = True
                #　録画を保存する関数を実行
                self.video = self.save_record()
                self.count_rec += 1
            else:
                print("録画を終了します")
                self.use_record = False
        
        # Pで対応点を変更        
        if action == glfw.PRESS and key == glfw.KEY_P:
            if self.detect_stable == 0:
                self.detect_stable = 1
                print("対応点をモード1(顔上部)に変更")
            elif self.detect_stable == 1:
                self.detect_stable = 2
                print("対応点をモード2(ずれが小さいランドマーク選択)に変更")
            elif self.detect_stable == 2:
                self.detect_stable = 3
                print("対応点をモード3(顔上部左右)に変更")
            elif self.detect_stable == 3:
                self.detect_stable = 0
                print("対応点をモード0(顔全体)に変更")
            else:
                pass

        # Aでモデル変更(あ)
        if action == glfw.PRESS and key == glfw.KEY_A:
            print("「あ」のモデル表示")
            self.display(self.parent_model[0])
            self.next_mouth = 'a'
        # Iでモデル変更(い)
        if action == glfw.PRESS and key == glfw.KEY_I:
            print("「い」のモデル表示")
            self.display(self.parent_model[1])
            self.next_mouth = 'i'
        # Uでモデル変更(う)
        if action == glfw.PRESS and key == glfw.KEY_U:
            print("「う」のモデル表示")
            self.display(self.parent_model[2])
            self.next_mouth = 'u'
        # Eでモデル変更(え)
        if action == glfw.PRESS and key == glfw.KEY_E:
            print("「え」のモデル表示")
            self.display(self.parent_model[3])
            self.next_mouth = 'e'
        # Oでモデル変更(お)
        if action == glfw.PRESS and key == glfw.KEY_O:
            print("「お」のモデル表示")
            self.display(self.parent_model[4])
            self.next_mouth = 'o'
        # Nでモデル変更(ん)
        if action == glfw.PRESS and key == glfw.KEY_N:
            print("「ん」のモデル表示")
            self.display(self.parent_model[5])
            self.next_mouth = 'n'

        # Mで動的再現のON/OFF
        if action == glfw.PRESS and key == glfw.KEY_M:

            # 発話内容保存
            self.speech_text = self.text
            # フラグ処理
            if self.on_mouth_motion:
                self.on_mouth_motion = False
            else:
                self.on_mouth_motion = True

                # 口形状の正順序
                self.dict_vowel_n = {"a":0, "i":1, "u":2, "e":3, "o":4, "n":5}

                # テキスト
                self.text_analyzer = TextAnalysis(self.text, self.data_folda, self.today)
                
                # 動的再現パラメータ
                self.now_mouth = "n" 
                self.now_frame = self.interp_frames
                self.mouth_positive_order = False
                self.half = False
                
                # labファイルを読み込み
                lab_name = '{}/speech/{}_{}.lab'.format(self.data_folda, self.today, self.text)

                # 1行分のデータ
                self.lab_content = []
                # １音にかかる時間
                self.char_per_sec = []
                
                if os.path.isfile(lab_name):
                    # 折れ線用のデータセット
                    self.true_total_list_y = []
                    self.phoneme_list_x = []
                    self.excess_time_list_y = []
                    self.total_time_list_y = []
                    with open(lab_name, 'r') as f:
                        # 1行ずつ取り込み
                        self.lab_content = f.read().split("\n")
                    # 1列ずつ取り出し
                    for i, row in enumerate(self.lab_content):
                        # １音にかかる時間
                        self.char_per_sec.append(row.split(" "))
                    self.char_per_sec = self.char_per_sec[0:-1]
                    for i, row in enumerate(self.char_per_sec):
                        self.true_total_list_y.append(float(self.char_per_sec[i][0]))
                        self.phoneme_list_x.append(self.char_per_sec[i][2])
                        self.excess_time_list_y.append(None)
                        self.total_time_list_y.append(None)

                self.next_mouth = self.text_analyzer.extract_vowel_and_n(self.now_mouth)
                self.switch_interval = 0.04

                # モデル変更のタイミング
                self.last_switch_time = time.time()
                # 発話口形状変更のタイミング
                self.last_change_time = time.time()                                            
                # 音声再生
                self.on_playwav = True

                # labファイルがある時
                if len(self.char_per_sec) != 0:

                    # 現在の発話文字の発話にかかる時間
                    self.before_pro_time = self.char_per_sec[0][1]
                    self.now_pro_time = self.char_per_sec[self.text_analyzer.num_char_text][0]
                    self.switch_interval = (float(self.now_pro_time) - float(self.before_pro_time)) / 5
                    print("text:", self.text_analyzer.text)
                    print("pro_char:", self.char_per_sec[self.text_analyzer.num_char_text][2])
                    self.save_pro_num = self.text_analyzer.num_char_text
                    print(self.save_pro_num)

                    while True:
                        current_time = time.time()
                        if current_time - self.last_switch_time >= float(self.before_pro_time):
                            break

                self.total_time = 0.0
                # 箱ひげ図1回分用のデータセット
                self.switch_list = []

        # Tでテキスト入力
        if action == glfw.PRESS and key == glfw.KEY_T:
            # GUI設定
            #sg.theme("DefaultNoMoreNagging")
            sg.theme("clam")
            layout = [[sg.Text("現在のテキスト「{}」".format(self.text))]
                    ,[sg.Text("発音を再現したいテキストを入力してください")]
                    ,[sg.InputText(key='-Input-')]
                    ,[sg.Button('OK', key='-Btn-')]]
            window = sg.Window('テキスト入力', layout)
            while True:
                event, value = window.read()  # イベントの入力を待つ
                if event == '-Btn-':
                    self.text = value['-Input-']
                    break
                elif event is None:
                    break
            window.close()
            
        # Vでテキスト入力
        if action == glfw.PRESS and key == glfw.KEY_V:
            self.on_transcription = True

    #
    # モデルの動的再現
    #
    def mouth_motion(self):
        if not self.on_mouth_motion:
            return
        while True:
            current_time = time.time()
            if current_time - self.last_switch_time >= self.switch_interval:
                break
        self.switch_list.append(current_time-self.last_switch_time)
        print('Switched. '+str(current_time-self.last_switch_time)+'sec.'+'Interval:'+str(self.switch_interval))
        allset_start = time.time()

        read_next_mouth = False

        # 口形状の更新チェック
        if self.mouth_positive_order:
            threshold = self.interp_frames+1
        else:
            threshold = -1
        if self.now_frame == threshold:
                read_next_mouth = True

        # 口形状を更新
        if read_next_mouth:

            self.half = False
            self.now_mouth = self.next_mouth
            self.next_mouth = self.text_analyzer.extract_vowel_and_n(self.now_mouth)
            
            #if not self.text_analyzer.half:
            print('\nChanged. '+str(time.time()-self.last_change_time)+'sec.')
            self.total_time += time.time()-self.last_change_time
            if len(self.char_per_sec) != 0:
                self.total_time_list_y[self.save_pro_num] = self.total_time
            self.last_change_time = time.time()
            print('Total Time.', self.total_time)
            # labファイルがある時
            if len(self.char_per_sec) != 0:
                self.excess_time = float(self.total_time) - float(self.now_pro_time)
                print('Excess Time.', self.excess_time)
                self.excess_time_list_y[self.save_pro_num] = self.excess_time
            print("========================================================")

            # テキスト読み込み終了
            if self.next_mouth == None:
                self.on_mouth_motion = False
                self.next_mouth = self.text_analyzer.input
                self.text_analyzer = TextAnalysis(self.text, self.data_folda, self.today)
                print(self.text_analyzer.kunrei_text)
                print("「%s」と言いました。"% self.speech_text)

                self.all_switch_list.append(self.switch_list[1:])

                return
            print("%s-%s" % (self.now_mouth, self.next_mouth))
            print("text:", self.text_analyzer.text)
            # labファイルがある時
            if len(self.char_per_sec) != 0:
                print("pro_char:", self.char_per_sec[self.text_analyzer.num_char_text][2])
                self.save_pro_num = self.text_analyzer.num_char_text
                print(self.save_pro_num)

            # labファイルがある時
            if len(self.char_per_sec) != 0:
                # 現在の発話文字の発話にかかる時間
                self.before_pro_time = self.now_pro_time
                self.now_pro_time = self.char_per_sec[self.text_analyzer.num_char_text][0]
                self.switch_interval = (float(self.now_pro_time) - float(self.before_pro_time) - float(self.excess_time)) / 5
                #self.switch_interval = (float(self.now_pro_time) - float(self.before_pro_time)) / 5
            
            # 口形状の順序チェック                
            if self.dict_vowel_n[self.now_mouth] <= self.dict_vowel_n[self.next_mouth]:
                self.mouth_positive_order = True
                self.now_frame = 0
            else:
                self.mouth_positive_order = False
                self.now_frame = self.interp_frames
        # 特殊発音
        if self.text_analyzer.flag_stnyrzdj or self.text_analyzer.flag_bmpv or self.text_analyzer.flag_fw:

            # labファイルがある時
            if len(self.char_per_sec) != 0:
                if self.half:
                    frame_num = 2
                else:
                    frame_num = 3
                # 現在の発話文字の発話にかかる時間
                self.switch_interval = (float(self.now_pro_time) - float(self.before_pro_time) - float(self.excess_time)) / frame_num
                #self.switch_interval = (float(self.now_pro_time) - float(self.before_pro_time)) / frame_num

            # フレームが奇数番のとき
            if self.now_frame % 2 == 1:
                # フレームインクリメント
                if self.mouth_positive_order:
                    self.now_frame += 1
                else:
                    self.now_frame -= 1

            elif self.text_analyzer.half:
                # フレームインクリメント
                if self.mouth_positive_order and self.now_frame == 0:
                    self.now_frame += 2
                elif not self.mouth_positive_order and self.now_frame == self.interp_frames:
                    self.now_frame -= 2
                    
                # labファイルがある時
                if len(self.char_per_sec) != 0:
                    self.switch_interval = (float(self.now_pro_time) - float(self.before_pro_time) - float(self.excess_time)) / 2
                    #self.switch_interval = (float(self.now_pro_time) - float(self.before_pro_time)) / 2
                    self.half = True
        else:
            pass
                
        self.interp_model(self.now_mouth, self.next_mouth)
        self.last_switch_time = current_time
        
        allset_end = time.time()
        print('All Setted. '+str(allset_end-allset_start)+'sec.')

    #
    # モデル設定
    #
    def display(self, model):
        msg = 'Setting %s ...' % model.materials[0].tex
        print(msg)

        start = time.time()        
        #
        # 3次元データの読み込み
        # 
        
        data = []
        data1 = []
        data2 = []
        right = []
        left = []
        for j in range(self.cnt):
            data.append(model.landmark_nomalize[j])
            if j in self.delete_datalist:
                pass
            elif j in self.datalist1:
                # 2次元-3次元対応点(空間検出)に使われるランドマーク
                data1.append(model.landmark_nomalize[j])

        # 対応点を選択
        for j in range(self.cnt):  
            if j in self.delete_datalist:
                pass  
            if j in self.datalist2:
                data2.append(model.landmark_nomalize[j])
            if j in self.right_list:
                right.append(model.landmark_nomalize[j])
            if j in self.left_list:
                left.append(model.landmark_nomalize[j])   

        # 3次元データをアプリケーションにセット
        self.set_3D_point(data, self.datalist)
        self.set_3D_point_1(data1, self.datalist1)
        self.set_3D_point_2(data2, self.datalist2)
        self.set_3D_point_3(right, left, self.right_list, self.left_list)        
        self.set_mqo_model(model)

        end = time.time()
        print('Setted. '+str(end-start)+'sec.')

    #
    # モデル補間
    #
    def interp_model(self, before_mouth, after_mouth):
        print('\nInterpolating %s-%s-%d...' % (before_mouth, after_mouth, self.now_frame))  
        start = time.time()
        if before_mouth == after_mouth:
            # モデル読み込み
            model = self.parent_model[self.dict_vowel_n[before_mouth]]
        else:
            # 変化前モデルの読み込み
            model = self.parent_model[-1]

            # 口形状の前後
            if self.mouth_positive_order:
                mouth1 = before_mouth
                mouth2 = after_mouth
            else:
                mouth1 = after_mouth
                mouth2 = before_mouth
            
            # 頂点座標の変更
            point_before = self.parent_model[self.dict_vowel_n[before_mouth]].landmark_nomalize
            point_after = self.parent_model[self.dict_vowel_n[after_mouth]].landmark_nomalize

            npoints, dim = point_before.shape
            diff = 0
            split_diff = 0
            point = [0, 0, 0]

            for p in range(npoints):
                for i in range(3):
                    diff = point_after[p, i]-point_before[p, i]
                    split_diff = diff / self.interp_frames
                    if self.dict_vowel_n[before_mouth] < self.dict_vowel_n[after_mouth]:
                        split_diff *= self.now_frame
                    else:
                        split_diff *= (self.interp_frames - self.now_frame)
                    point[i] = point_before[p, i] + split_diff
                model.landmark_nomalize[p] = [point[0], point[1], point[2]]
                model.mesh.vertices[p].x = point[0]
                model.mesh.vertices[p].y = point[1]
                model.mesh.vertices[p].z = point[2]

            # UV座標の変更
            uv_before = self.parent_model[self.dict_vowel_n[before_mouth]].faces
            uv_after = self.parent_model[self.dict_vowel_n[after_mouth]].faces
            
            nmeshes, dim = self.mesh.shape
            diff = 0
            split_diff = 0
            
            for m in range(nmeshes):
                for i in range(3):
                    diff_u = uv_after[m].uvs[i].u - uv_before[m].uvs[i].u
                    diff_v = uv_after[m].uvs[i].v - uv_before[m].uvs[i].v
                    split_diff_u = diff_u / self.interp_frames
                    split_diff_v = diff_v / self.interp_frames
                    if self.dict_vowel_n[before_mouth] < self.dict_vowel_n[after_mouth]:
                        split_diff_u *= self.now_frame
                        split_diff_v *= self.now_frame
                    else:
                        split_diff_u *= (self.interp_frames - self.now_frame)
                        split_diff_v *= (self.interp_frames - self.now_frame)
                    model.faces[m].uvs[i].u = uv_before[m].uvs[i].u + split_diff_u
                    model.faces[m].uvs[i].v = uv_before[m].uvs[i].v + split_diff_v

            # テクスチャ名の決定
            if self.use_cut:
                texture_name = os.getcwd() + "/{}/{}-{}/cut_{}_{}-{}_{}-{}.png".format(self.data_folda, mouth1, mouth2, self.today, mouth1, mouth2, self.interp_frames, self.now_frame)
            else:
                texture_name = os.getcwd() + "/{}/{}-{}/{}_{}-{}_{}-{}.png".format(self.data_folda, mouth1, mouth2, self.today, mouth1, mouth2, self.interp_frames, self.now_frame)
            # テクスチャのセット
            model.materials[0].load_texture(texture_name, 1)
        
        # フレームインクリメント
        if self.mouth_positive_order:
            self.now_frame += 1
        else:
            self.now_frame -= 1
        
        end = time.time()
        print('Interpolated. '+str(end-start)+'sec.')      

        self.display(model)

    #
    # 画像を保存する関数
    #
    def save_image(self):
        filename = 'mqodata/output/images/image_{}-{}.png'.format(self.today, self.count_img)
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # バッファを読み込む(画面を読み込む)
        glReadBuffer(GL_FRONT)
        # ピクセルを読み込む
        glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, image.data)
        image = cv2.cvtColor (image, cv2.COLOR_BGR2RGB)
        image = cv2.flip (image, 0)
        if self.use_record:
            return image
        else:
            # 画像を保存
            cv2.imwrite(filename, image)
            print("保存が完了しました...({})".format(filename))
        
    #
    # 画面録画を保存する関数
    #
    def save_record(self):
        filename = 'mqodata/output/videos/video_{}-{}.mp4'.format(self.today, self.count_rec)
        video = self.camera.SaveRecord(filename)
        print("録画を開始します..." + filename)
        return video
    
    #
    # mediapipeで検出した顔のランドマーク座標を出力する関数
    #
    def save_landmarks(self, add = False, landmark = 0, txt = None):
        filename = 'output/landmarks/landmarks_{}_{}.dat'.format(self.today, self.count_img)
        output = open(filename, mode='w')
        if self.face_mesh.multi_face_landmarks:
            for landmarks in self.face_mesh.multi_face_landmarks:
                # enumerate()...オブジェクトの要素とインデックス番号を取得
                for idx, p in enumerate(landmarks.landmark):
                    # 座標のリストを指定
                    if idx in self.point_list:
                        text = str(idx) + ',' + str(p.x * self.width) + ',' + str(p.y * self.height) + ',' + str(p.z * self.width) + '\n'
                        # text = str(p.x * self.width) + ',' + str(p.y * self.height) + '\n'
                        output.write(text)
                        
        output.close()

    #
    # カメラ姿勢を計算する関数
    #
    def compute_camera_pose(self, point_2D, point_3D):
        point_2D = np.array(point_2D)
        point_3D = np.array(point_3D)
        # カメラ姿勢を計算
        # PoseEstimationクラスのcompute_camera_poseメソッドを実行
        success, R, t, r = self.estimator.compute_camera_pose(
            point_3D, point_2D, use_objpoint = True)
    
        if success:
            # 世界座標系に対するカメラ位置を計算
            # この位置を照明位置として使用
            if self.use_normal:
                # カメラ位置姿勢計算
                pos = -R.transpose().dot(t)
                self.camera_pos = np.array([pos[0], pos[1], pos[2], 1.0], dtype = "double")

            self.generate_modelview(R,t)
            
            # 顔の方向ベクトルを計算
            # PoseEstimationクラスのcompute_head_vectorメソッドを実行
            vector = self.estimator.compute_head_vector()
            # 顔のオイラー角を計算
            # PoseEstimationクラスのcompute_head_angleメソッドを実行
            angle = self.estimator.compute_head_angle(R, t)
            # 行列の値をファイルに保存
            if self.flag_save_matrix == 1:
                filename = 'output/images/matrix_{}_{}.dat'.format(self.today, self.count_img)
                output = open(filename, mode='a')
                output.write(str(np.linalg.norm(r)))
                output.write(",")
                output.write(str(np.linalg.norm(t)))
                output.write(",")
                output.write(str(vector))
                output.write(",")
                output.write(str(angle))
                output.write("\n")
                output.close()
            return success, vector, angle
            
        else:
            vector = None
            angle = None
            return success, vector, angle
    
    #
    # モデルビュー行列を生成
    #
    def generate_modelview(self, R, t):
        # OpenGLで使用するモデルビュー行列を生成
            self.modelview[0] = R[0][0]
            self.modelview[1] = R[1][0]
            self.modelview[2] = R[2][0]
            self.modelview[3] = 0.0
            self.modelview[4] = R[0][1]
            self.modelview[5] = R[1][1]
            self.modelview[6] = R[2][1]
            self.modelview[7] = 0.0
            self.modelview[8] = R[0][2]
            self.modelview[9] = R[1][2]
            self.modelview[10] = R[2][2]
            self.modelview[11] = 0.0
            self.modelview[12] = t[0]
            self.modelview[13] = t[1]
            self.modelview[14] = t[2]
            self.modelview[15] = 1.0      
      
    #
    # セッター
    #  
    # 親モデルをセット
    def set_parent_list(self, model_list):
        self.parent_model = model_list

    # 三次元データをセット(対応点全て)
    def set_3D_point(self, point_3D, point_list):
        self.point_3D = point_3D
        self.point_list = point_list
        self.estimator.ready = True
    
    # 三次元データをセット(一部の対応点)
    def set_3D_point_1(self, point_3D, point_list):
        self.point_3D1 = point_3D
        self.point_list1 = point_list       
    def set_3D_point_2(self, point_3D, point_list):
        self.point_3D2 = point_3D
        self.point_list2 = point_list 
    def set_3D_point_3(self, point_right, point_left, right_list, left_list):
        self.left = point_left
        self.right = point_right
        self.left_list = left_list
        self.right_list = right_list
    
    # 3次元データリストをセット
    def set_3D_list(self, cnt, delete_datalist, datalist, datalist1, datalist2, right_list, left_list, mesh):
        self.cnt = cnt
        self.delete_datalist = delete_datalist
        self.datalist = datalist
        self.datalist1 = datalist1
        self.datalist2 = datalist2
        self.right_list = right_list
        self.left_list = left_list
        self.mesh = mesh

    # ３次元モデルをセット
    def set_mqo_model(self, model):
        self.model = model
    
    # 入力画像をセット
    def set_image(self, image):
        image = cv2.cvtColor (image, cv2.COLOR_BGR2RGB)
        self.image = image