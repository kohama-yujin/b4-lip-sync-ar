import speech_recognition as sr
from pydub import AudioSegment
import wave
import pyaudio
import os
#
# 音声認識クラス
#
class SpeechRecognition:

    #
    # コンストラクタ
    #
    def __init__(self, data_folda, today):
        # 音声認識器の初期化
        self.recognizer = sr.Recognizer()
        self.data_folda = data_folda
        self.today = today

    def transcription(self):
        # マイクからの入力を使用
        with sr.Microphone() as source:

            print("音声認識を開始します")

            # バックグラウンドノイズに対応するために、環境ノイズを調整
            self.recognizer.adjust_for_ambient_noise(source)
            while True:
                try:
                    # ユーザーの声を記録
                    self.audio = self.recognizer.listen(source)
                    # GoogleのWeb Speech APIを使用して音声をテキストに変換
                    self.text = self.recognizer.recognize_google(self.audio, language="ja-JP")
                    # 空白削除
                    self.text = self.text.replace('　', '').replace('\t', '').replace('\n', '').replace(' ', '')
                    print("認識したテキスト："+ self.text)
                        
                    break
                
                    # "終わり"という単語でプログラムを終了
                    #if "終わり" in self.text or "おわり" in self.text:
                        #break

                except sr.UnknownValueError:
                    print("音声を認識できませんでした。")
                except sr.RequestError:
                    print("リクエストに失敗しました。インターネット接続を確認してください。")

    def play_wav(self, text):
        if os.path.isfile('{}/speech/{}_{}.wav'.format(self.data_folda, self.today, text)):
            wav_file = wave.open('{}/speech/{}_{}.wav'.format(self.data_folda, self.today, text), 'rb')
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wav_file.getsampwidth()),
                            channels=wav_file.getnchannels(),
                            rate=wav_file.getframerate(),
                            output=True)

            data = wav_file.readframes(1024)
            while data:
                stream.write(data)
                data = wav_file.readframes(1024)

            stream.stop_stream()
            stream.close()
            p.terminate()
        else:
            print("録音されていません")

if __name__ == "__main__":
    SpeechRecognition()