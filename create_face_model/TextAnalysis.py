import pykakasi
import MeCab
import re

#
# テキスト解析クラス
#
class TextAnalysis:

    #
    # コンストラクタ
    #
    def __init__(self, text, data_folda, today):
        self.kakasi = pykakasi.kakasi()
        self.vowel_list = ["a", "i", "u", "e", "o"]
        self.org_text = text
        self.text = text
        self.flag_stnyrzdj = False
        self.flag_bmpv = False
        self.flag_fw = False
        self.flag_tsu = False
        self.half = False

        self.data_folda = data_folda
        self.today = today

        # テキストの何文字目か
        self.num_char_text = 0

        print(self.text)
        # 発音カナに変換
        self.getProText()
        print(self.text)
        self.protext = self.text
        # 間違いがちな文字を変換
        self.convert_to_pronunciation()
        print(self.text)
        # ローマ字に変換
        self.convert_word_type_text('kunrei')
        self.kunrei_text = self.text

    # 発音カナに変換する関数
    def getProText(self):
        m = MeCab.Tagger("-d /var/lib/mecab/dic/ipadic-utf8")
        m_result = m.parse(self.text).splitlines() #mecabの解析結果の取得
        m_result = m_result[:-1] #最後の1行は不要な行なので除く
        
        pro_text = '' #発音文字列全体を格納する変数
        for v in m_result:
            #print(v)
            surface = re.split('[,\t]', v)[0] #表層形
            p = re.split('[,\t]', v)[-1] #発音カナ
            #発音が取得できていないときsurfaceで代用
            if p == '*':
                p = surface
            pro_text += p
        self.text = pro_text

        """
            p = v.split('\t')[1].split(',')[-1]#発音を取得したいとき
            #p = v.split('\t')[1].split(',')[-2] #ルビを取得したいとき

            #発音が取得できていないときsurfaceで代用
            if p == '':
                p = surface
            pro_text += p
        self.text = pro_text
        """
    # 間違いを引き起こす文字を、発音用に変換する関数
    def convert_to_pronunciation(self):
        nyanyunyo_vowel_dict =  {'ヲ':'オ', 'ウォ':'ヲ', 'っ':'q', 'ッ':'q', 'ン':'N', 'フ':'w', 'ー':''}
        # 'ニャ':'な', 'ニュ':'ウ', 'ニョ':'オ', 
        for key, value in nyanyunyo_vowel_dict.items():
            self.text = self.text.replace(key, value)

    # textをword_typeで指定した種類(ひらがな、ローマ字等)の文字列に変換して、空欄で単語ごとに区別したtextを返す関数
    def convert_word_type_text(self, word_type):
        # 単語ごとに分解されたリストを取得(単語は辞書型で色々な形式に変換された状態で格納されている)
        word_list = self.kakasi.convert(self.text)

        word_list_len = len(word_list)
        # word_typeで指定された種類(ひらがな、ローマ字等)の文字列のみを抽出
        word_type_word_list = [word_list[cnt][word_type] for cnt in range(word_list_len)]

        # 単語を連結する
        word_type_text = "".join(word_type_word_list)

        self.text = word_type_text

    # 発音を再現する母音を返す関数
    def extract_vowel_and_n(self, now_mouth):
        while True:
            if len(self.text) == 0:
                return
            
            self.pronunciation = True
            symbol = False

            # 折返しチェック
            if not self.half and (self.flag_stnyrzdj or self.flag_bmpv or self.flag_fw):
                self.half = True
            # フラグリセット
            elif self.half:
                self.half = False
                self.flag_stnyrzdj = False
                self.flag_bmpv = False
                self.flag_fw = False


            if self.text[0] == 's' or self.text[0] == 't' or self.text[0] == 'n' or self.text[0] == 'y' or self.text[0] == 'r' or self.text[0] == 'z'or self.text[0] == 'd' or self.text[0] == 'j':
                # stnyzdjで次がa, eの時→短く「い」を挟む
                if self.text[1] == 'a' or self.text[1] == 'e':
                    self.flag_stnyrzdj = True
                    self.input = 'i'
                # stnyzdjで次がa, eでない時→発音しない
                else:
                    self.input = self.text[0]
                    self.pronunciation = False
            # 破裂音の時→短く「ん」を挟む
            elif self.text[0] == 'b' or self.text[0] == 'm' or self.text[0] == 'p' or self.text[0] == 'v':
                self.flag_bmpv = True
                self.input = 'n'
            # 「ふぁふぃふぇふぉ」「わ」→短く「う」を挟む
            elif self.text[0] == 'f' or self.text[0] == 'w':
                self.flag_fw = True
                self.input = 'u'
            # 「ん」→発音
            elif self.text[0] == 'N':
                self.input = 'n'
            # 促音
            elif self.text[0] == 'q':
                # 促音で次がbmpvの時→「ん」を挟む
                if len(self.text) != 1 and (self.text[1] == 'b' or self.text[1] == 'm' or self.text[1] == 'p' or self.text[1] == 'v'):
                    self.input = 'n'
                    #self.text = self.text[1:]
                    #self.text = 'N' + self.text
                # 促音で今の発音が「あ、え」の時→「い」を挟む
                elif now_mouth == 'a' or now_mouth == 'e':
                    self.input = 'i'
                    #self.text = self.text[1:]
                    #self.text = 'i' + self.text
                # 促音で上記以外→今の発音を挟む
                else:
                    self.input = now_mouth
                    #self.text = self.text[1:]
                    #self.text = now_mouth + self.text
            # 「,」→今の発音で休止
            elif self.text[0] == ',':
                symbol = True
                self.input = now_mouth
                self.text = self.text[1:]
                if now_mouth == 'n':
                    now_mouth = 'N'
                self.text = now_mouth + self.text
            # 「.」「!」「？」→「ん」で休止
            elif self.text[0] == '.' or self.text[0] == '!' or self.text[0] == '？':
                symbol = True
                self.input = 'n'
                self.text = self.text[1:]
                self.text = 'N' + self.text
            else:
                self.input = self.text[0]
                # 母音のループ
                for vowel in self.vowel_list:
                    # 母音の時→発音
                    if self.text[0] == vowel:
                        self.pronunciation = True
                        break
                    else:
                        # 子音の時→発音しない
                        self.pronunciation = False

            # 記号以外のとき読み込みをインクリメント
            if not symbol:
                self.text = self.text[1:]
                self.num_char_text += 1

            # 拗音チェック
            if not self.pronunciation and self.text[0] == 'y':
                self.num_char_text -= 1

            # 発音口形状を返す
            if self.pronunciation:
                return self.input
    
    # ひらがなでテキスト保存する関数
    def save_hiraText(self):

        # 変換を実行
        result = self.kakasi.convert(self.protext)

        # 変換結果を表示
        hiratext = ''.join([item['hira'] for item in result])

        # テキストをひらがなで保存
        with open('{}/speech/{}_{}.txt'.format(self.data_folda, self.today, self.org_text), 'w', encoding='utf-8') as f:
            f.write(hiratext)

if __name__ == '__main__':
    text = "今日はやっと私の誕生日です"
    #text = "えーーーっ！？ウォーリーをさがそう。が木っ端微塵に爆発？"

    text_analyzer = TextAnalysis(text, './', 20)
    print(text_analyzer.text)
    