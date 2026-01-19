import os
from pathlib import Path
import numpy as np
from typing import Optional, Union
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder
from style_bert_vits2.nlp.bert_models import load_model, load_tokenizer
from style_bert_vits2.constants import Languages

class TTSProcessor:
    def __init__(
        self, 
        model_root_dir: Union[str, Path] = Path("/home/usr23/n_sakaguchi/graph_chatUI/models/assets")
    ):
        """
        TTSプロセッサーの初期化
        Args:
            model_root_dir: Style-Bert-VITS2のモデルが格納されているディレクトリ
        """
        self.model_root_dir = Path(model_root_dir)
        self.tts_model = TTSModelHolder(model_root_dir=self.model_root_dir, device="cuda")
        self.current_model_name = None
        self.current_model_path = None

        # BERTモデルを明示的にロード
        bert_path = "/home/usr23/n_sakaguchi/graph_chatUI/bert/deberta-v2-large-japanese-char-wwm"
        load_model(Languages.JP, pretrained_model_name_or_path=bert_path)
        load_tokenizer(Languages.JP, pretrained_model_name_or_path=bert_path)

        # デフォルトモデルのロードを試みる
        self._load_default_model()

    def _load_default_model(self):
        """デフォルトのモデルを読み込む"""
        print("DEBUG: _load_default_model() 開始")
        try:
            # モデル名とパスを取得（最初に見つかったものを使用）
            model_names = self.tts_model.model_names
            print("DEBUG: 見つかったモデル名 =", model_names)
            if not model_names:
                raise ValueError("モデルが見つかりません")

            model_name = model_names[0]
            model_files = [str(f) for f in self.tts_model.model_files_dict[model_name]]
            print(f"DEBUG: モデル '{model_name}' のファイル一覧 =", model_files)

            if not model_files:
                raise ValueError(f"モデル {model_name} のファイルが見つかりません")

            # モデルをロード
            print(f"DEBUG: モデル '{model_name}' をロードします -> {model_files[0]}")
            self.tts_model.get_model(
                model_name=model_name,
                model_path_str=model_files[0]
            ).load()

            self.current_model_name = model_name
            self.current_model_path = model_files[0]
            print(f"DEBUG: デフォルトモデルのロード完了: {self.current_model_name}, {self.current_model_path}")

        except Exception as e:
            print(f"デフォルトモデルのロードに失敗しました: {str(e)}")
            self.current_model_name = None
            self.current_model_path = None

    def synthesize(self, text: str) -> Optional[tuple]:
        """
        テキストから音声を合成
        Args:
            text: 合成するテキスト
        Returns:
            (サンプリングレート, 音声波形の numpy 配列) のタプル, 
            またはファイルパス (WAVなど) を返す実装の場合は文字列
            エラー時は None
        """
        print("DEBUG: synthesize() 呼び出し text=", repr(text))
        try:
            if not text:
                print("DEBUG: テキストが空なので合成しません。None を返します")
                return None

            if not self.current_model_name or not self.current_model_path:
                raise ValueError("モデルがロードされていません")

            print(f"DEBUG: 現在ロード中のモデル: {self.current_model_name}, {self.current_model_path}")
            print("DEBUG: 推論を開始します...")
            audio_data = self.tts_model.get_model(
                model_name=self.current_model_name,
                model_path_str=self.current_model_path
            ).infer(text=text)

            print("DEBUG: infer() の戻り値 audio_data =", audio_data)

            if audio_data is None:
                raise ValueError("音声合成に失敗しました")

            # もし (sr, wave_array) の形なら、型を確認してログ出力
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                sr, wave = audio_data
                print(f"DEBUG: 推論結果はタプルです -> sr={sr}, waveの型={type(wave)}")
            else:
                # タプル以外の場合、ファイルパスかもしれないのでログを出す
                print("DEBUG: 推論結果はタプル以外でした ->", type(audio_data))
                # ここで音声ファイルかディレクトリかをチェック
                if isinstance(audio_data, str):
                    print("DEBUG: 文字列として返却されました:", audio_data)
                    if os.path.isdir(audio_data):
                        print("WARNING: ディレクトリパスです。Gradio でエラーになる可能性があります。")
                else:
                    print("DEBUG: 文字列でもありません ->", audio_data)

            return audio_data

        except Exception as e:
            print(f"音声合成でエラーが発生しました: {str(e)}")
            return None

    def load_model(self, model_name: str, model_path: str) -> bool:
        """
        指定されたモデルをロード
        Args:
            model_name: モデル名
            model_path: モデルファイルのパス
        Returns:
            ロードが成功したかどうか
        """
        print(f"DEBUG: load_model() 呼び出し model_name={model_name}, model_path={model_path}")
        try:
            self.tts_model.get_model(
                model_name=model_name,
                model_path_str=model_path
            ).load()

            self.current_model_name = model_name
            self.current_model_path = model_path
            print(f"DEBUG: モデル {model_name} をロードしました -> {model_path}")
            return True

        except Exception as e:
            print(f"モデルのロードに失敗しました: {str(e)}")
            return False

    def get_available_models(self):
        """利用可能なモデルの一覧を取得"""
        print("DEBUG: get_available_models() 呼び出し")
        models_dict = {
            name: [str(f) for f in self.tts_model.model_files_dict[name]]
            for name in self.tts_model.model_names
        }
        print("DEBUG: 利用可能なモデル一覧 ->", models_dict)
        return models_dict


# 単独で実行したときのテストコード
if __name__ == "__main__":
    tts = TTSProcessor()
    test_text = "こんにちは、これはテストです。"
    result = tts.synthesize(test_text)
    print("DEBUG: 最終的な synthesize() の戻り値 =", result)
