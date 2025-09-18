import sys
import numpy as np
### 追加
import cv2
import mediapipe as mp
import datetime
import math as m
import argparse
import time
#
# ３次元モデル生成クラス
#
class ChangeMQO:
    #
    # コンストラクタ
    #
    # @param texture : テクスチャ画像
    #
    def __init__(self, mouth, use_cut, interp_frames, output_original, use_AI):

        # 口形状解析
        if mouth == None:
            mouth_shape = "free"
        else:
            mouth_shape = mouth
            if len(mouth)==3:
                mouth_before, mouth_after = mouth.split('-')
        # 顔下部切り取りを行うか
        self.use_cut = use_cut
        # 補間前後の出力を行うか
        self.output_original = output_original
        # マスクモデル作成モード
        self.masked_face = False
        # 世界座標系を記述
        self.world_coordinate = False
        # AI生成を行うか
        self.use_AI = use_AI
        
        # ファイル名用日付
        self.today = str(datetime.date.today()).replace('-','')
        # ファイル作成
        self.mesh = np.loadtxt("mqodata/mesh.dat", dtype='int')
        self.mesh_cut = np.array([])

        # テクスチャ名
        texture_path_before = "mqodata/model/KohamaYujin/mouthShape_"+mouth_before+".png"
        texture_path_after = "mqodata/model/KohamaYujin/mouthShape_"+mouth_after+".png"
        frame_before = cv2.imread(texture_path_before)
        frame_after = cv2.imread(texture_path_after)
        interp_texture_dir = "mqodata/model/KohamaYujin/{}-{}/".format(mouth_before, mouth_after)
        
        # テクスチャ画像から特徴点、特徴点の正規化、新たなメッシュ情報を生成
        self.data = np.array([])
        self.datalist = []
        self.landmark = np.array([])
        self.landmark_nomalize = np.array([])
        # beforeデータ
        self.set_point(texture_path_before)
        self.data_before = self.data
        self.datalist_before = self.datalist
        self.landmark_before = self.landmark
        self.landmark_nomalize_before = self.landmark_nomalize
        # afterデータ    
        self.set_point(texture_path_after)
        self.data_after = self.data
        self.datalist_after = self.datalist
        self.landmark_after = self.landmark
        self.landmark_nomalize_after = self.landmark_nomalize
        
        interp_frames = int(interp_frames)+1
        # 出力フレーム数
        if self.output_original:
            frame_list = [i for i in range(interp_frames+1)]
        else:
            frame_list = [i for i in range(1, interp_frames)]

        for now_frame in frame_list:
            # フレーム補間
            interp_texture_filename = "{}_{}-{}_{}-{}.png".format(self.today, mouth_before, mouth_after, interp_frames, now_frame)
            if self.use_cut:
                interp_texture_filename = "cut_{}_{}-{}_{}-{}.png".format(self.today, mouth_before, mouth_after, interp_frames, now_frame)
            
            if not self.use_AI:                
                interp_texture = self.frame_interpolation(frame_before, frame_after, interp_frames, now_frame)
                cv2.imwrite(interp_texture_dir+interp_texture_filename, interp_texture)

            # ヘッダ出力
            self.outputs = []
            self.output_header(interp_texture_filename)

            # 3次元座標の出力
            self.output_3D_coord(self.landmark_nomalize_before, self.landmark_nomalize_after, interp_frames, now_frame)
            # 三角形メッシュの出力
            if self.use_cut:
                self.mesh = self.mesh_cut
            
            if self.use_AI:
                self.output_move_mesh_info(self.landmark_before, self.landmark_after, interp_frames, now_frame, self.mesh)
            else:
                self.output_normal_mesh_info(self.landmark_before, self.mesh)

            # フッダ出力
            self.outputs.append('}\nEof')        

            # ファイル出力
            self.model_filename = "mqodata/model/KohamaYujin/{}-{}/{}_{}-{}_{}-{}.mqo".format(mouth_before, mouth_after, self.today, mouth_before, mouth_after, interp_frames, now_frame)
            if self.use_cut:
                self.model_filename = "mqodata/model/KohamaYujin/{}-{}/cut_{}_{}-{}_{}-{}.mqo".format(mouth_before, mouth_after, self.today, mouth_before, mouth_after, interp_frames, now_frame)
            with open(self.model_filename, "w") as f:
                for output in self.outputs:
                    f.write(output)
    
    #
    # ヘッダを出力する関数
    #
    def output_header(self, interp_texture_filename):
        self.outputs.append('Metasequoia Document\n')
        self.outputs.append('Format Text Ver 1.1\n')
        self.outputs.append('\n')
        self.outputs.append('Scene {\n')
        self.outputs.append('\tpos 0 0 1000\n')
        self.outputs.append('\tlookat 0 0 0\n')
        self.outputs.append('\thead -1.5\n')
        self.outputs.append('\tpich 0.12\n')
        self.outputs.append('\tbank 0.0000\n')
        self.outputs.append('\tortho 0\n')
        self.outputs.append('\tzoom2 5.0000\n')
        self.outputs.append('\tamb 0.250 0.250 0.250\n')
        self.outputs.append('\tfrontclip 225.0\n')
        self.outputs.append('\tbackclip 45000\n')
        self.outputs.append('\tdirlights 1 {\n')
        self.outputs.append('\t\tlight {\n')
        self.outputs.append('\t\t\tdir 0.408 0.408 0.816\n')
        self.outputs.append('\t\t\tcolor 1.000 1.000 1.000\n')
        self.outputs.append('\t\t}\n')
        self.outputs.append('\t}\n')
        self.outputs.append('}\n')

        self.outputs.append('Material 1 {\n')
        
        if self.masked_face == True:
            self.outputs.append('\t"mat1" shader(3) col(0.176 1.000 0.000 0.500) dif(0.800) amb(0.600) emi(0.000) spc(0.000) power(5.00) tex("mask.png")\n')
        else:
            self.outputs.append('\t"mat1" shader(3) col(0.176 1.000 0.000 0.500) dif(0.800) amb(0.600) emi(0.000) spc(0.000) power(5.00) tex("%s")\n' % interp_texture_filename)
        self.outputs.append('}\n')

        self.outputs.append('Object "obj" {\n')
        self.outputs.append('\tdepth 0\n')
        self.outputs.append('\tfolding 0\n')
        self.outputs.append('\tscale 1 1 1\n')
        self.outputs.append('\trotation 0 0 0\n')
        self.outputs.append('\tvisible 15\n')
        self.outputs.append('\tlocking 0\n')
        self.outputs.append('\tshading 1\n')
        self.outputs.append('\tfacet 59.5\n')
        self.outputs.append('\tnormal_weight 1\n')
        self.outputs.append('\tcolor 0.898 0.498 0,698\n')
        self.outputs.append('\tcolor_type 0\n')
        
        
    #
    # ベクトル情報を出力する関数
    # @param point: ランドマークを世界座標系に変換した三次元座標
    #
    def output_3D_coord(self, point_before, point_after, interp_frames, now_frame):
        npoints, dim = point_before.shape
        self.outputs.append('\tvertex %d {\n' % npoints)
        diff = 0
        split_diff = 0
        point = [0, 0, 0]

        for p in range(npoints):
            for i in range(3):
                diff = point_after[p, i]-point_before[p, i]
                split_diff = diff / interp_frames
                split_diff *= now_frame
                point[i] = point_before[p, i] + split_diff
                int(point[i])

            self.outputs.append('\t\t%f %f %f\n' % (point[0], point[1], point[2]))
        self.outputs.append('\t}\n')

    #
    # メッシュ情報を出力する関数
    # @param uv : ランドマークの二次元座標 , mesh : メッシュ情報
    #   
    def output_normal_mesh_info(self, uv, mesh):
        nmeshes, dim = mesh.shape
        self.outputs.append('\tface %d {\n' % nmeshes)
        for m in range(nmeshes):
            self.outputs.append('\t\t3 V(%d %d %d) M(0) UV(%f %f %f %f %f %f)\n' % (mesh[m, 0], mesh[m, 1], mesh[m, 2],
                                                                    uv[mesh[m, 0], 0], uv[mesh[m, 0], 1],
                                                                    uv[mesh[m, 1], 0], uv[mesh[m, 1], 1],
                                                                    uv[mesh[m, 2], 0], uv[mesh[m, 2], 1]))
        self.outputs.append('\t}\n')

    def output_move_mesh_info(self, uv_before, uv_after, interp_frames, now_frame, mesh):
        nmeshes, dim = mesh.shape
        self.outputs.append('\tface %d {\n' % nmeshes)
        diff = 0
        split_diff = 0

        for m in range(nmeshes):
            uv = [
                [0, 0],
                [0, 0],
                [0, 0]
            ]
            for i in range(3):
                for j in range(2):
                    diff = uv_after[mesh[m, i],j]-uv_before[mesh[m, i], j]
                    split_diff = diff / interp_frames
                    split_diff *= now_frame
                    uv[i][j] = uv_before[mesh[m, i], j] + split_diff

            self.outputs.append('\t\t3 V(%d %d %d) M(0) UV(%f %f %f %f %f %f)\n' % (mesh[m, 0], mesh[m, 1], mesh[m, 2],
                                                                    uv[0][0], uv[0][1],
                                                                    uv[1][0], uv[1][1],
                                                                    uv[2][0], uv[2][1]))      
        self.outputs.append('\t}\n')

    #
    # テクスチャ画像の特徴点を検出し、uv座標、xyz座標、メッシュ情報を求める関数
    #
    def set_point(self, texture_filename):
        # MediaPipeのFaceMeshインスタンスを作成する
        face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5)
        drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            thickness=1, 
            circle_radius=1, 
            color=(0, 0, 255))

        # テクスチャファイル読み込み
        img = cv2.imread(texture_filename)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated_image = img.copy() 
        
        # FaceMeshを実行
        face_mesh = face.process(rgb_img)
        
        # 座標、メッシュ情報格納用リスト
        x = []
        y = []
        z = []
        landmark = []
        landmark_nomalize = []
        cut = []
        mesh_cut = []
        # ランドマークカウント用変数
        cnt = 0
        
        # ランドマークの導出及び描画
        for face_landmarks in face_mesh.multi_face_landmarks:
            # 画像上に描画
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image, 
                face_landmarks, 
                mp.solutions.face_mesh.FACEMESH_TESSELATION,
                drawing_spec,
                drawing_spec)
            # 座標を導出
            for idx, p in enumerate(face_landmarks.landmark):
                # ランドマークの2次元座標のリストを作成
                landmark.append(np.array([p.x, p.y, p.z])) 
                # 画像サイズに合わせて正規化   
                x.append(p.x * img.shape[1])
                y.append(p.y * img.shape[0])
                z.append(p.z * img.shape[1])
                cnt += 1
        
        # 三次元座標変換後の座標格納用リスト
        n1 = []
        n2 = []
        n3 = []
        n4 = []
        # 1.原点がランドマーク1となるように平行移動
        for i in range(cnt):
            n1.append(np.array([x[i]-x[1], y[i]-y[1], z[i]-z[1]]))
            
        # 2.実寸サイズにスケーリング
        # スケーリング倍率(両目の端の実際の距離/mediapipeで計測した距離)
        scale = 100 / abs(n1[263][0]-n1[33][0])
        for i in range(cnt):
            n2.append(np.array(n1[i] * scale))
            
        # 3.回転の補正
        v = np.array([n2[263][0]-n2[33][0], n2[263][1]-n2[33][1], n2[263][2]-n2[33][2]]) # 新しいX軸のベクトル
        v = v / np.linalg.norm(v) # 長さ1に正規化
        
        
        # ロドリゲスの定理による回転
        # 軸s/2、回転角π
        # https://ja.stackoverflow.com/questions/55865
        k = np.array([1,0,0])
        s = v + k
        r = np.array(2 * np.outer(s, s) / np.dot(s, s) - np.eye(3))
        for i in range(cnt):
            n3.append(np.dot(r, n2[i]))
        
        # 4.補正(特になし)
        for i in range(cnt):
            n4.append(np.array([n3[i][0], n3[i][1], n3[i][2]]))
        
        # ランドマークを世界座標系に変換した三次元座標のリストを作成
        landmark_nomalize = n4
        
        # カメラ位置計算用データ作成、メッシュ情報修正
        #for j in range(cnt):
            # カメラ姿勢計算に使う3次元データ
            # datalist = [6, 10, 33, 103, 133, 263, 332, 362]
            # if j in datalist:
            #     data.append(landmark_nomalize[j])
            # # 顔側面を削除
            # deletelist = [162, 127, 234, 93, 132, 58, 389, 356, 454, 323, 361, 288]
            # if j in deletelist:
            #     continue
            
            # # 顔下部のみ表示する(yがランドマーク197以上の点のみ)
            # if landmark_nomalize[j][1] >= landmark_nomalize[6][1]:
            #     cut.append(j)
            # else:
            #     datalist_new.append(j)
            #     data_new.append(landmark_nomalize[j])
        
        # 世界座標系記述
        if self.world_coordinate:
            vecx = np.array([200,0,0])
            vecy = np.array([0,200,0])
            vecz = np.array([0,0,200])
            vecx = np.dot(r,vecx)
            vecy = np.dot(r,vecy)
            vecz = np.dot(r,vecz)
            cv2.line(annotated_image,
                    pt1 = (int(x[1]),int(y[1])),
                    pt2 = (int(x[1]+vecx[0]),int(y[1]+vecx[1])),
                    color = (0,0,255),
                    thickness = 3,
            )
            cv2.line(annotated_image,
                    pt1 = (int(x[1]),int(y[1])),
                    pt2 = (int(x[1]+vecy[0]),int(y[1]+vecy[1])),
                    color = (0,255,0),
                    thickness = 3,
            )
            cv2.line(annotated_image,
                    pt1 = (int(x[1]),int(y[1])),
                    pt2 = (int(x[1]+vecz[0]),int(y[1]+vecz[1])),
                    color = (255,0,0),
                    thickness = 3,
            )
        
        #
        # 対応点データ(マスクに隠れない部分を対応点として使用)   
        # マスクに隠れる部分は三次元モデルを生成
        #
        data = []
        datalist = []
        data1 = []
        data2 = []
        right = []
        left = []
        
        # 顔の側面の対応点を削除
        delete_datalist = [234, 93, 132, 58, 172, 454, 323, 361, 288, 397]
        # カメラ位置姿勢推定に用いる対応点パターン
        # マスクに隠れない部分は対応点として使用し、マスクに隠れる部分はテクスチャとして使用する
        datalist1 = [6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 46, 52, 53, 
                    54, 55, 56, 63, 65, 66, 67, 68, 69, 70, 71, 103, 104, 105, 107, 108, 109, 
                    110, 112, 113, 122, 124, 127, 130, 133, 139, 143, 144, 145, 151, 153, 154, 155, 
                    156, 157, 158, 159, 160, 161, 162, 163, 168, 173, 189, 190, 193, 221, 222, 
                    223, 224, 225, 226, 243, 244, 245, 246, 247, 249, 251, 252, 253, 254, 255, 256, 257, 258, 
                    259, 260, 263, 264, 265, 276, 282, 283, 284, 285, 286, 293, 295, 296, 297, 
                    298, 299, 300, 301, 332, 333, 334, 336, 337, 338, 339, 341, 342, 351, 353, 356, 
                    359, 362, 368, 372, 373, 374, 380, 381, 382, 383, 384, 385, 386, 387, 388, 
                    389, 390, 398, 413, 414, 417, 441, 442, 443, 444, 445, 446, 463, 464, 465, 466, 467]
        datalist2 = [3, 5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33,
                    46, 47, 51, 52, 53, 54, 55, 56, 63, 65, 66, 67, 68, 69, 70, 71, 103, 104,
                    105, 107, 108, 109, 110, 112, 113, 114, 120, 121, 122, 124, 128, 130, 133,
                    144, 145, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 168,
                    173, 174, 188, 189, 190, 193, 195, 196, 197, 217, 221, 222, 223, 224,
                    225, 226, 228, 229, 230, 231, 232, 233, 236, 243, 244, 245, 246, 247, 248,
                    249, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 263, 276, 277,
                    281, 282, 283, 284, 285, 286, 293, 295, 296, 297, 298, 299, 300, 301, 332,
                    333, 334, 336, 337, 338, 339, 341, 342, 343, 349, 350, 351, 353, 357, 359,
                    362, 373, 374, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 398,
                    399, 412, 413, 414, 417, 419, 437, 441, 442, 443, 444, 445, 446, 448, 449,
                    450, 451, 452, 453, 456, 463, 464, 465, 466, 467]
        self.left_list = [6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 46, 52,
                          53, 54, 55, 56, 63, 65, 66, 67, 68, 69, 70, 71, 103, 104, 105, 107, 108, 109, 
                          110, 112, 113, 122, 124, 127, 130, 133, 139, 143, 144, 145, 151, 153, 154, 155, 
                          156, 157, 158, 159, 160, 161, 162, 163, 168, 173, 189, 190, 193, 221, 222, 223, 
                          224, 225, 226, 243, 244, 245, 246, 247]
        self.right_list = [6, 8, 9, 10, 151, 168, 249, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 263, 
                           264, 265, 276, 282, 283, 284, 285, 286, 293, 295, 296, 297, 298, 299, 300, 301, 332, 333, 
                           334, 336, 337, 338, 339, 341, 342, 351, 353, 356, 359, 362, 368, 372, 373, 374, 380, 381, 
                           382, 383, 384, 385, 386, 387, 388, 389, 390, 398, 413, 414, 417, 441, 442, 443, 444, 445, 
                           446, 463, 465, 464, 466, 467]
        # マスクのモデルを作成する際に用いる対応点
        self.mask_list = [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 32, 36, 37, 38, 39, 40, 
                          41, 42, 43, 44, 45, 48, 49, 50, 51, 57, 59, 60, 61, 62, 64, 72, 73, 74, 75, 76, 
                          77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 
                          98, 99, 100, 101, 102, 106, 115, 123, 125, 126, 129, 131, 134, 135, 136, 137, 138, 
                          140, 141, 142, 146, 147, 148, 149, 150, 152, 164, 165, 166, 167, 169, 170, 171, 175, 
                          176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 191, 192, 194, 195, 198, 
                          199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 
                          216, 218, 219, 220, 235, 236, 237, 238, 239, 240, 241, 242, 248, 250, 262, 266, 267, 
                          268, 269, 270, 271, 272, 273, 274, 275, 278, 279, 280, 281, 287, 289, 290, 291, 292, 
                          294, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 
                          318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 331, 335, 344, 352, 354, 
                          355, 358, 360, 363, 364, 365, 366, 367, 369, 370, 371, 375, 376, 377, 378, 379, 391, 
                          392, 393, 394, 395, 396, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 
                          415, 416, 418, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 
                          434, 435, 436, 438, 439, 440, 455, 456, 457, 458, 459, 460, 461, 462]
        
        for j in range(cnt):
            datalist.append(j)
            data.append(landmark_nomalize[j])
            if j in delete_datalist:
                pass
            elif j in datalist1:
                # 2次元-3次元対応点(空間検出)に使われるランドマーク
                data1.append(landmark_nomalize[j])
            else:
                if self.masked_face:
                    if j in self.mask_list:
                        cut.append(j)
                else:
                    # 三次元モデルに使われるランドマーク
                    cut.append(j)

        # 対応点を選択
        for j in range(cnt):  
            if j in delete_datalist:
                pass  
            if j in datalist2:
                data2.append(landmark_nomalize[j])
            if j in self.right_list:
                right.append(landmark_nomalize[j])
            if j in self.left_list:
                left.append(landmark_nomalize[j])

        # メッシュ情報の修正
        n, d = self.mesh.shape
        for i in range(n):
            if (self.mesh[i, 0] in cut) and (self.mesh[i, 1] in cut) and (self.mesh[i, 2] in cut):
                mesh_cut.append(np.array([self.mesh[i, 0],self.mesh[i, 1],self.mesh[i, 2]]))
    
        # ファイル出力
        np.savetxt("data/face_3D.dat", data)
        self.data = np.array(data)
        self.data1 = np.array(data1)
        self.data2 = np.array(data2)
        self.right = np.array(right)
        self.left = np.array(left)
        self.datalist = datalist
        self.datalist1 = datalist1
        self.datalist2 = datalist2
        np.savetxt("mqodata/landmark/landmark_"+self.today+".dat", landmark)
        self.landmark = np.array(landmark)
        np.savetxt("mqodata/landmark3d/landmark3d_"+self.today+".dat", landmark_nomalize)
        self.landmark_nomalize = np.array(landmark_nomalize)
        np.savetxt("mqodata/mesh/mesh_cut_"+self.today+".dat", mesh_cut, fmt='%d')
        self.mesh_cut = np.array(mesh_cut)
        cv2.imwrite("mqodata/output/face_mesh_"+self.today+".png", annotated_image)

        # インスタンスを終了させる
        face.close()

        cv2.destroyAllWindows()

    def frame_interpolation(self, _frame1, _frame2, interp_frames, now_frame):
        
        # FaceMeshインスタンスを作成
        face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5)
       
        # 鼻の位置を取得
        nose1, eye_x1, eye_y1, dis1 = self.get_face_properties(_frame1, face)
        nose2, eye_x2, eye_y2, dis2 = self.get_face_properties(_frame2, face)
        scale_factor = dis1 / dis2
        M_translate_scale = np.float32([[scale_factor, 0, eye_x1 - eye_x2 * scale_factor],
                                [0, scale_factor, eye_y1 - eye_y2 * scale_factor]])
        
        h2, w2, _ = _frame2.shape
        resize = cv2.warpAffine(_frame2, M_translate_scale, (w2, h2))
        nose2, _, _, _ = self.get_face_properties(resize, face)

        if nose1 and nose2:
            # 鼻の位置を基準に画像を移動
            dx = nose1[0] - nose2[0]
            dy = nose1[1] - nose2[1]

            # 移動後の画像を作成する
            M = cv2.getRotationMatrix2D((resize.shape[1] // 2, resize.shape[0] // 2), 0, 1)
            M[0, 2] += dx
            M[1, 2] += dy
            img2_aligned = cv2.warpAffine(resize, M, (resize.shape[1], resize.shape[0]))

        _beta = now_frame / interp_frames
        _alpha = 1.0 - _beta
        print("alpha:", _alpha, "beta:", _beta)
        return cv2.addWeighted(_frame1, _alpha, img2_aligned, _beta, 0.0)
    
    # 鼻の位置を取得する関数
    def get_face_properties(self, image, face_mesh):
        # 画像をRGBに変換
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 顔の特徴点を検出
        results = face_mesh.process(rgb_image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 4番目の特徴点（鼻の先端）
                nose_tip = face_landmarks.landmark[4]
                h, w, _ = image.shape
                nose_tip_coords = (int(nose_tip.x * w), int(nose_tip.y * h))
                
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                eye_x = int((left_eye.x + right_eye.x) * w // 2)
                eye_y = int((left_eye.y + right_eye.y) * h // 2)
                eye_distance = int(np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2) * w)

                return nose_tip_coords, eye_x, eye_y, eye_distance
        return None
    
if __name__ == '__main__':
    if len(sys.argv)<4:
        print("input argument.")
        sys.exit()
        
    start = time.time()
    parser = argparse.ArgumentParser(
        usage =  "python main.py <-f frames> <-m mouth_shape>",
        description = "description for commandline arguments",
        epilog = "end",
        add_help = True,
    )
    parser.add_argument('-c', action='store_true', help = "flag for use_cut")
    parser.add_argument('-m', "--mouth", help = "mouth_shape")
    parser.add_argument('-f', "--frames", help = "interp_frames")
    parser.add_argument('-o', action='store_true', help = "flag for output_original")
    parser.add_argument('-a', action='store_true', help = "flag for use_AI")
    args = parser.parse_args()

    ChangeMQO(args.mouth, args.c, args.frames, args.o, args.a)
    end = time.time()
    print('ALL. '+str(end-start)+'sec.\n')
