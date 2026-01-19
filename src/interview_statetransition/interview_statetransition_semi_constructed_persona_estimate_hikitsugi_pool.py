# nohup python src/interview_statetransition/interview_statetransition_semi_constructed_persona_estimate_hikitsugi_pool.py > out/log/output_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 半構造化インタビューを行うプログラム

# --------------------------------------------------------------------------------------------------------------------------------------------
# 設定セクション
# --------------------------------------------------------------------------------------------------------------------------------------------

# osモジュールを先にインポート（パス設定で必要）
import os

# モデルの設定
MODEL_NAME = "gpt-4o-2024-11-20"
TEMPERATURE = 0

# 待機時間
WAIT_TIME = 5

# プロジェクトルートの取得（__file__を基準に）
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) if "__file__" in globals() else os.getcwd()

# Docker環境用のパス（/mnt/work/interview/）が存在する場合は優先
_docker_data_root = "/mnt/work/interview/data"
if os.path.exists(_docker_data_root):
    _data_root = _docker_data_root
else:
    # 通常のローカル環境ではプロジェクトルート基準
    _data_root = os.path.join(_project_root, "data")

# プロンプトの設定
PROMPT_IDLE_TALK_PATH = os.path.join(_data_root, "hashimoto-nakano/prompt_semi_const/proposed_method/prompt_idle_talk.txt")
PROMPT_FILL_SLOTS_PATH = os.path.join(_data_root, "hashimoto-nakano/prompt_semi_const/proposed_method/prompt_fill_slots.txt")
PROMPT_GENERATE_SLOTS_PATH = os.path.join(_data_root, "hashimoto-nakano/prompt_semi_const/proposed_method/prompt_generate_slots_fukabori.txt")
PROMPT_GENERATE_SLOTS_2_PATH = os.path.join(_data_root, "hashimoto-nakano/prompt_semi_const/proposed_method/prompt_generate_slots_2.txt")
PROMPT_GENERATE_QUESTIONS_PATH = os.path.join(_data_root, "hashimoto-nakano/prompt_semi_const/proposed_method/prompt_generating_question.txt")
PROMPT_USER_SIMULATOR_PATH = os.path.join(_data_root, "hashimoto-nakano/prompt_semi_const/proposed_method/prompt_user_simulator.txt")
PROMPT_CAREER_TOPIC_PATH = os.path.join(_data_root, "hashimoto-nakano/prompt_semi_const/proposed_method/prompt_career_topic.txt")
PROMPT_END_CONVERSATION_PATH = os.path.join(_data_root, "hashimoto-nakano/prompt_semi_const/proposed_method/prompt_end_conversation.txt")

PROMPT_ESTIMATE_PERSONA_PATH = os.path.join(_data_root, "hashimoto-nakano/prompt_semi_const/proposed_method/prompt_estimate_persona.txt")

# ユーザシミュレータのpersona設定
PERSONA_SETTINGS_PATH = os.path.join(_data_root, "hashimoto-nakano/persona_settings/hasegawa_data/hasegawa_p.txt")

# 自己評価アンケート
SELF_EVALUATION_PATH = os.path.join(_data_root, "hashimoto-nakano/questionnaire/endo_q.json")

# INTERVIEW_CONFIG（初期状態）の定義
# "value":\s*"[^"]*"   でスロットの値を抽出
# "value": None        でスロットの値を初期化
INTERVIEW_CONFIG = {
    "dialogue_history": [],
    "speak_count": {
        "total_count": 0,
        "interviewer_count": 0,
        "interviewee_count": 0,
        "interviewer_idle_talk_count": 0,
        "interviewee_idle_talk_count": 0,
        "interviewer_generate_question_count": 0,
        "interviewee_generate_answer_count": 0,
    },
    "max_total_count": 40,  # 40
    "min_total_count": 34,  # 30
    "estimate_persona": "リフレッシュ方法: 未  \nキャリア・職場: 未  \n悩みや不満点: 未  \n将来のキャリアプラン: 未  \n家族: 未  \n思い出・エピソード: 未",  # None
    "persona_attribute_candidates": [
        "性格",
        "SNS",
        "趣味",
        "個人の基本的情報",
        "学生時代の失敗エピソード",
        "昨日起きたとても悲しい出来事",
        "副業",
        "恋愛",
        "株式投資の経験",
        "起床後のルーティン",
        "パソコンを使う頻度",
        "行ってみたい国",
    ],
    "slots": {
        "現在のキャリア": {"question_priority": 1, "value": None},
        "悩みや不満点": {"question_priority": 1, "value": None},
        "昇進・転職": {"question_priority": 1, "value": None},
        "過去のキャリア": {"question_priority": 1, "value": None},
        "家族": {"question_priority": 1, "value": None},
        "思い出・エピソード": {"question_priority": 1, "value": None},
        "未来像": {"question_priority": 1, "value": None},
    },
    "slot_generation_count": 0,
    "branch": None,
}

# [
#         "性格",
#         "過去のキャリア",
#         "昇進・転職",
#         "SNS",
#         "趣味",
#         "家族",
#         "使用しているスマートフォンアプリ",
#         "個人の基本的情報",
#         "未来像",
#         "タスク管理",
#         "学生時代の失敗エピソード",
#         "服装の好み",
#         "昨日起きたとても悲しい出来事",
#         "思い出・エピソード",
#         "好きなゆるキャラ",
#         "好きな食べ物",
#         "好きな言葉・座右の銘",
#         "嫌いな食べ物",
#         "尊敬する人",
#         "好きな音楽",
#         "100年後の未来",
#         "時間管理で気をつけていること",
#         "副業",
#         "好きなお菓子",
#         "恋愛",
#         "人生で一番つらかった出来事",
#         "学生時代の得意科目",
#         "株式投資の経験",
#         "資産運用の考え方",
#         "人脈の広げ方",
#         "在宅ワーク",
#         "乗馬の経験",
#         "起床後のルーティン",
#         "パソコンを使う頻度",
#         "かかりつけの病院",
#         "作るのが得意な料理",
#         "断捨離や片付けのコツ",
#         "海外旅行の楽しみ方",
#         "語学学習のポイント",
#         "スポーツ観戦",
#         "スポーツジムの利用",
#         "飼っているペット",
#         "乗りたい車",
#         "好きな芸能人",
#         "行ってみたい国",
#         "テクノロジーとの付き合い方",
#         "通勤時の移動方法",
#         "オンライン会議における悩みごと",
#         "感情のコントロールで気をつけていること",
#         "金銭感覚の違い",
#     ],

#  "現在のキャリア": {
#             "question_priority": 1,
#             "value": None,
#         },
#         "悩みや不満点": {
#             "question_priority": 1,
#             "value": None,
#         },

# --------------------------------------------------------------------------------------------------------------------------------------------


import json
import datetime
import time
import requests
import copy
import random
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing import Dict, List, Optional
from typing_extensions import TypedDict
from icecream import ic
from pytz import timezone
from pydantic import BaseModel, Field, RootModel

# 必要な環境変数を設定
os.environ["LANGCHAIN_TRACING_V2"] = "false"


# ic.enable()
# ic.disable()

# デバッグ出力のフォーマットを設定
# ic.configureOutput(includeContext=True)

# モデルの定義
model = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=TEMPERATURE,
)

# model = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash-002",
#     temperature=TEMPERATURE,
# )


def load_file(file_path: str) -> str:
    """
    ファイルから情報を読み込む関数

    Args:
        file_path(str): ファイルパス
    Returns:
        str: ファイルの内容
    """
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return f.read()


# 雑談を行うプロンプト
prompt_idle_talk = load_file(PROMPT_IDLE_TALK_PATH)

# スロットを埋めるプロンプト
prompt_fill_slots = load_file(PROMPT_FILL_SLOTS_PATH)

# スロットを生成するプロンプト
prompt_generate_slots = load_file(PROMPT_GENERATE_SLOTS_PATH)

# スロットを生成するプロンプト（2回目）
prompt_generate_slots_2 = load_file(PROMPT_GENERATE_SLOTS_2_PATH)

# スロットから質問を生成するプロンプト
prompt_generate_questions = load_file(PROMPT_GENERATE_QUESTIONS_PATH)

# ユーザシミュレータのプロンプト
prompt_user_simulator = load_file(PROMPT_USER_SIMULATOR_PATH)

# キャリアの話題が出たかどうかを判定するプロンプト
prompt_career_topic = load_file(PROMPT_CAREER_TOPIC_PATH)

# 会話を終了するかどうかを判定するプロンプト
prompt_end_conversation = load_file(PROMPT_END_CONVERSATION_PATH)

# ペルソナ情報を推定するプロンプト
prompt_estimate_persona = load_file(PROMPT_ESTIMATE_PERSONA_PATH)

# ユーザシミュレータのpersona設定
persona_settings = load_file(PERSONA_SETTINGS_PATH)

# 自己評価アンケート
self_evaluation = load_file(SELF_EVALUATION_PATH)

# interview_config（初期状態）の定義
interview_config = copy.deepcopy(INTERVIEW_CONFIG)


class SpeakCount(TypedDict):
    total_count: int
    interviewer_count: int
    interviewee_count: int
    interviewer_idle_talk_count: int
    interviewee_idle_talk_count: int
    interviewer_generate_question_count: int
    interviewee_generate_answer_count: int


class State(TypedDict):
    """
    State：ノード間の遷移の際に保存される情報
    各ノードが参照、更新する

    Attributes:
        dialogue_history(List[str]): インタビューの対話履歴
        speak_count(SpeakCount): 発言回数
        max_total_count(int): 最大ターン数
        slots(Dict): スロット情報
    """

    dialogue_history: List[str]
    speak_count: SpeakCount
    max_total_count: Optional[int]
    min_total_count: Optional[int]
    estimate_persona: Optional[str]
    persona_attribute_candidates: List[str]
    slots: Dict
    slots_generation_count: int
    branch: Optional[str]


class Slot(BaseModel):
    question_priority: int = Field(
        0, title="過去のインタビューで何回情報を引き出したか（回数）"
    )
    value: Optional[str] = Field(None, title="スロットの値")


class SlotDict(RootModel[Dict[str, Slot]]):
    pass


slot_output_parser = PydanticOutputParser(pydantic_object=SlotDict)


class TargetSlot(BaseModel):
    question_priority: int = Field(
        0, title="過去のインタビューで何回情報を引き出したか（回数）"
    )
    value: Optional[str] = Field(None, title="スロットの値")


class QuestionOutput(BaseModel):
    Target_Slot: Dict[str, TargetSlot] = Field(..., title="スロット")
    Question: str = Field(..., title="質問")


question_output_parser = PydanticOutputParser(pydantic_object=QuestionOutput)


jst = timezone("Asia/Tokyo")
now = datetime.datetime.now(jst)
timestamp = now.strftime("%Y%m%d_%H%M%S")
formatted_timestamp = now.strftime("%Y/%m/%d %H:%M:%S")


# 保存フォルダの作成
def create_output_folder(base_folder="out/"):
    folder_path = os.path.join(base_folder, timestamp)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


# 保存するデータの初期化
execution_folder = create_output_folder()
info_path = os.path.join(execution_folder, "info.json")
graph_image_path = os.path.join(execution_folder, "graph.png")
readme_path = os.path.join(execution_folder, "_README.md")

# 空の情報を保存
with open(info_path, "w", encoding="utf-8-sig") as f:
    json.dump({}, f, indent=4, ensure_ascii=False)
with open(readme_path, "w", encoding="utf-8-sig") as f:
    f.write(
        f"- 開始時刻: {formatted_timestamp}\n- モデル: {MODEL_NAME}\n- ユーザシミュレータ: {PERSONA_SETTINGS_PATH}\n提案手法"
    )


# 保存される情報
execution_info = {
    "execution_time": formatted_timestamp,
    "command": __file__ if "__file__" in globals() else "<interactive>",
    "model_info": str(model),
    "persona_settings": persona_settings,
    "interview_info": interview_config,
    "new_state": {},
}


# 情報を保存する関数
def save_state_to_file(state: State, file_path: str):
    """
    現在の状態をファイルに保存する関数

    Args:
        state(State): 状態情報
        file_path(str): 保存先のファイルパス
    """
    execution_info["new_state"].update(state)

    try:
        with open(file_path, "w", encoding="utf-8-sig") as f:
            json.dump(execution_info, f, indent=4, ensure_ascii=False)
        print(f"!!状態を保存しました!!: {file_path}")
    except Exception as e:
        print(f"!!状態の保存に失敗しました!!: {e}")


save_state_to_file(interview_config, info_path)


def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = os.environ['LINE_TOKEN']
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers=headers, data=data)


# CLI でインタビュー対象者の入力を受け取るヘルパー
def prompt_user_input(label: str) -> str:
    user_input = input(f"{label} ").strip()
    if not user_input:
        user_input = "<no input>"
    return user_input


# ========================================
# -----ノードの定義-----
# ========================================


def interviewer_llm_idle_talk(state: State):
    """
    interviewer_llm_idle_talk: インタビュアーの雑談を生成する関数

    Args:
        state(State): 状態情報
    Returns:
        Dict: 更新された状態情報(インタビュアーの発言, 発言回数)
    """
    time.sleep(WAIT_TIME)
    new_state = copy.deepcopy(state)

    dialogue_history = new_state["dialogue_history"]
    sc = new_state["speak_count"]

    template = prompt_idle_talk
    prompt = PromptTemplate(
        template=template,
        input_variables=["dialogue_history_str"],
    )

    dialogue_history_str = "\n".join(dialogue_history)
    system_message = prompt.format(
        dialogue_history_str=dialogue_history_str,
    )

    human_message = "インタビュアー:"

    print(
        f"\n\n===============関数interviewer_llm_idle_talk===============\nSystemMessage=\n{system_message}\nHumanMessage=\n{human_message}\n"
    )

    try:
        response = model.invoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(content=human_message),
            ]
        )
        if response and hasattr(response, "content"):
            new_dialogue = f"インタビュアー: {response.content}"
        else:
            new_dialogue = (
                "インタビュアー: モデルの呼び出しに失敗しました。再試行してください。"
            )

    except Exception as e:
        print(f"Error occurred while invoking the model: {e}")
        new_dialogue = (
            "インタビュアー: モデルの呼び出しに失敗しました。再試行してください。"
        )

    print(f"new_dialogue: {new_dialogue}\n")

    # 発言回数を更新
    sc["total_count"] += 1
    sc["interviewer_count"] += 1
    sc["interviewer_idle_talk_count"] += 1

    new_state["dialogue_history"] = dialogue_history + [new_dialogue]
    new_state["speak_count"] = sc

    print(f"new_state: {new_state}\n")

    now = datetime.datetime.now(jst)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    node_name = "interviewer_llm_idle_talk"
    this_node_path = os.path.join(
        execution_folder, f"info_{timestamp}_{node_name}.json"
    )

    save_state_to_file(new_state, this_node_path)

    return new_state


# インタビュー対象者の雑談発話を生成する関数
def interviewee_llm_idle_talk(state: State):
    """
    interviewee_llm_idle_talk: インタビュー対象者の雑談を生成する関数

    Args:
        state(State): 状態情報
    Returns:
        Dict: 更新された状態情報(インタビュー対象者の発言, 発言回数)
    """
    time.sleep(WAIT_TIME)
    new_state = copy.deepcopy(state)

    dialogue_history = new_state["dialogue_history"]
    sc = new_state["speak_count"]
    user_text = prompt_user_input("インタビュー対象者(雑談)を入力してください:")
    new_dialogue = f"インタビュー対象者: {user_text}"
    print(f"new_dialogue: {new_dialogue}\n")

    # 発言回数を更新
    sc["total_count"] += 1
    sc["interviewee_count"] += 1
    sc["interviewee_idle_talk_count"] += 1

    new_state["dialogue_history"] = dialogue_history + [new_dialogue]
    new_state["speak_count"] = sc

    print(f"new_state: {new_state}\n")

    now = datetime.datetime.now(jst)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    node_name = "interviewee_llm_idle_talk"
    this_node_path = os.path.join(
        execution_folder, f"info_{timestamp}_{node_name}.json"
    )

    save_state_to_file(new_state, this_node_path)

    return new_state


def interviewer_llm_generate_question(state: State):
    """
    interviewer_llm_generate_question: インタビュアーの発言(質問)を生成する関数

    Args:
        state(State): 状態情報
    Returns:
        Dict: 更新された状態情報(インタビュアーの発言, 発言回数)
    """
    time.sleep(WAIT_TIME)
    new_state = copy.deepcopy(state)

    dialogue_history = new_state["dialogue_history"]
    sc = new_state["speak_count"]
    slots = new_state["slots"]
    estimate_persona = new_state["estimate_persona"]

    template = prompt_generate_questions
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "dialogue_history_str",
            "current_slots",
            "estimate_persona",
        ],
        partial_variables={
            "format_instructions": question_output_parser.get_format_instructions()
        },
    )

    dialogue_history_str = "\n".join(dialogue_history)
    system_message = prompt.format(
        dialogue_history_str=dialogue_history_str,
        current_slots=slots,
        estimate_persona=estimate_persona,
    )

    human_message = "出力:"

    print(
        f"\n\n===============関数interviewer_llm_generate_question===============\nSystemMessage=\n{system_message}\nHumanMessage=\n{human_message}\n"
    )

    try:
        response = model.invoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(content=human_message),
            ]
        )
        if response and hasattr(response, "content"):
            print(f"generated_question: {response.content}\n")
            try:
                parsed_output = question_output_parser.parse(response.content)
                question_text = parsed_output.Question
                new_dialogue = f"インタビュアー: {question_text}"

            except Exception as e:
                print(f"Error occurred while decoding the JSON: {e}")
                new_dialogue = "インタビュアー: モデルの呼び出しに失敗しました。再試行してください。"
        else:
            new_dialogue = (
                "インタビュアー: モデルの呼び出しに失敗しました。再試行してください。"
            )

    except Exception as e:
        print(f"Error occurred while invoking the model: {e}")
        new_dialogue = (
            "インタビュアー: モデルの呼び出しに失敗しました。再試行してください。"
        )

    print(f"new_dialogue: {new_dialogue}\n")

    # 発言回数を更新
    sc["total_count"] += 1
    sc["interviewer_count"] += 1
    sc["interviewer_generate_question_count"] += 1

    new_state["dialogue_history"] = dialogue_history + [new_dialogue]
    new_state["speak_count"] = sc

    print(f"new_state: {new_state}\n")

    now = datetime.datetime.now(jst)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    node_name = "interviewer_llm_generate_question"
    this_node_path = os.path.join(
        execution_folder, f"info_{timestamp}_{node_name}.json"
    )

    save_state_to_file(new_state, this_node_path)

    return new_state


# インタビュー対象者の発話(回答)を生成する関数
def interviewee_llm_generate_answer(state: State):
    """
    interviewee_llm_generate_answer: インタビュー対象者の発言(回答)を生成する関数

    Args:
        state(State): 状態情報
    Returns:
        Dict: 更新された状態情報(インタビュー対象者の発言, 発言回数)
    """
    time.sleep(WAIT_TIME)
    new_state = copy.deepcopy(state)

    dialogue_history = new_state["dialogue_history"]
    sc = new_state["speak_count"]
    user_text = prompt_user_input("インタビュー対象者(回答)を入力してください:")
    new_dialogue = f"インタビュー対象者: {user_text}"
    print(f"new_dialogue: {new_dialogue}\n")

    # 発言回数を更新
    sc["total_count"] += 1
    sc["interviewee_count"] += 1
    sc["interviewee_generate_answer_count"] += 1

    new_state["dialogue_history"] = dialogue_history + [new_dialogue]
    new_state["speak_count"] = sc

    print(f"new_state: {new_state}\n")

    now = datetime.datetime.now(jst)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    node_name = "interviewee_llm_generate_answer"
    this_node_path = os.path.join(
        execution_folder, f"info_{timestamp}_{node_name}.json"
    )

    save_state_to_file(new_state, this_node_path)

    return new_state


# スロットを埋める関数
def interviewer_llm_fill_slots(state: State):
    """
    interviewer_llm_fill_slots: インタビュアーがスロットを埋める関数

    Args:
        state(State): 状態情報
    Returns:
        Dict: 更新された状態情報(スロット情報)
    """
    time.sleep(WAIT_TIME)
    new_state = copy.deepcopy(state)

    dialogue_history = new_state["dialogue_history"]
    slots = new_state["slots"]

    template = prompt_fill_slots
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "dialogue_history_str",
            "current_slots",
        ],
        partial_variables={
            "format_instructions": slot_output_parser.get_format_instructions()
        },
    )

    dialogue_history_str = "\n".join(dialogue_history)
    system_message = prompt.format(
        dialogue_history_str=dialogue_history_str,
        current_slots=slots,
    )

    human_message = "出力:"

    print(
        f"\n\n===============関数interviewer_llm_fill_slots===============\nSystemMessage=\n{system_message}\nHumanMessage=\n{human_message}\n"
    )

    try:
        response = model.invoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(content=human_message),
            ]
        )
        if response and hasattr(response, "content"):
            print(f"new_slots_dict: {response.content}\n")
            try:
                new_slots_obj = slot_output_parser.parse(response.content)
                new_slot_dict = {
                    slot_name: slot_model.dict()
                    for slot_name, slot_model in new_slots_obj.root.items()
                }
                merged_slots = {**slots, **new_slot_dict}
            except Exception as e:
                print(f"Error occurred while decoding the JSON: {e}")
                merged_slots = slots
        else:
            merged_slots = slots

    except Exception as e:
        print(f"Error occurred while invoking the model: {e}")
        merged_slots = slots

    print(f"merged_slots: {merged_slots}\n")

    new_state["slots"] = merged_slots

    print(f"new_state: {new_state}\n")

    now = datetime.datetime.now(jst)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    node_name = "interviewer_llm_fill_slots"
    this_node_path = os.path.join(
        execution_folder, f"info_{timestamp}_{node_name}.json"
    )

    save_state_to_file(new_state, this_node_path)

    return new_state


# スロットを生成する関数
def interviewer_llm_generate_slots(state: State):
    """
    interviewer_llm_generate_slots: インタビュアーがスロットを生成する関数

    Args:
        state(State): 状態情報
    Returns:
        Dict: 更新された状態情報(スロット情報)
    """
    time.sleep(WAIT_TIME)
    new_state = copy.deepcopy(state)

    dialogue_history = new_state["dialogue_history"]
    slots = new_state["slots"]
    estimate_persona = new_state["estimate_persona"]

    template = prompt_generate_slots
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "dialogue_history_str",
            "current_slots",
            "estimate_persona",
        ],
        partial_variables={
            "format_instructions": slot_output_parser.get_format_instructions()
        },
    )

    dialogue_history_str = "\n".join(dialogue_history)
    system_message = prompt.format(
        dialogue_history_str=dialogue_history_str,
        current_slots=slots,
        estimate_persona=estimate_persona,
    )

    human_message = "出力:"

    print(
        f"\n\n===============関数interviewer_llm_generate_slots===============\nSystemMessage=\n{system_message}\nHumanMessage=\n{human_message}\n"
    )

    try:
        response = model.invoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(content=human_message),
            ]
        )
        if response and hasattr(response, "content"):
            content = response.content
            print(f"new_slots_dict: {content}\n")
            if content == "None":
                merged_slots = slots
            else:
                try:
                    new_slots_obj = slot_output_parser.parse(content)
                    new_slots_dict = {
                        slot_name: slot_model.dict()
                        for slot_name, slot_model in new_slots_obj.root.items()
                    }
                    merged_slots = {**slots, **new_slots_dict}
                except Exception as e:
                    print(f"Error occurred while decoding the JSON: {e}")
                    merged_slots = slots
        else:
            merged_slots = slots

    except Exception as e:
        print(f"Error occurred while invoking the model: {e}")
        merged_slots = slots

    print(f"merged_slots: {merged_slots}\n")

    new_state["slots"] = merged_slots

    print(f"new_state: {new_state}\n")

    now = datetime.datetime.now(jst)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    node_name = "interviewer_llm_generate_slots"
    this_node_path = os.path.join(
        execution_folder, f"info_{timestamp}_{node_name}.json"
    )

    save_state_to_file(new_state, this_node_path)

    return new_state


# ループの3回に1回、interviewer_llm_generate_slots関数の代わりに呼び出される関数
def interviewer_llm_generate_slots_2(state: State):
    """
    interviewer_llm_generate_slots_2: インタビュアーがスロットを生成する関数

    Args:
        state(State): 状態情報
    Returns:
        Dict: 更新された状態情報(スロット情報)
    """
    time.sleep(WAIT_TIME)
    new_state = copy.deepcopy(state)

    dialogue_history = new_state["dialogue_history"]
    slots = new_state["slots"]

    if "persona_attribute_candidates" not in new_state:
        new_state["persona_attribute_candidates"] = []

    if not new_state["persona_attribute_candidates"]:
        random_topic = "None"
    else:
        random_topic = random.choice(new_state["persona_attribute_candidates"])
        new_state["persona_attribute_candidates"].remove(random_topic)

    template = prompt_generate_slots_2
    prompt = PromptTemplate(
        template=template,
        input_variables=["dialogue_history_str", "current_slots", "topic"],
        partial_variables={
            "format_instructions": slot_output_parser.get_format_instructions()
        },
    )

    dialogue_history_str = "\n".join(dialogue_history)
    system_message = prompt.format(
        dialogue_history_str=dialogue_history_str,
        current_slots=slots,
        topic=random_topic,
    )

    human_message = "出力:"

    print(
        f"\n\n===============関数interviewer_llm_generate_slots_2===============\nSystemMessage=\n{system_message}\nHumanMessage=\n{human_message}\n"
    )

    try:
        response = model.invoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(content=human_message),
            ]
        )
        if response and hasattr(response, "content"):
            content = response.content
            print(f"new_slots_dict: {content}\n")
            if content == "None":
                merged_slots = slots
            else:
                try:
                    new_slots_obj = slot_output_parser.parse(content)
                    new_slots_dict = {
                        slot_name: slot_model.dict()
                        for slot_name, slot_model in new_slots_obj.root.items()
                    }
                    merged_slots = {**slots, **new_slots_dict}
                except Exception as e:
                    print(f"Error occurred while decoding the JSON: {e}")
                    merged_slots = slots
        else:
            merged_slots = slots

    except Exception as e:
        print(f"Error occurred while invoking the model: {e}")
        merged_slots = slots

    print(f"merged_slots: {merged_slots}\n")

    new_state["slots"] = merged_slots

    now = datetime.datetime.now(jst)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    node_name = "interviewer_llm_generate_slots_2"
    this_node_path = os.path.join(
        execution_folder, f"info_{timestamp}_{node_name}.json"
    )

    save_state_to_file(new_state, this_node_path)

    return new_state


# スロット生成の関数を選択するための関数
def select_generate_slots_node(state: State) -> State:
    """
    3回に1回はinterviewer_llm_generate_slots_2、それ以外はinterviewer_llm_generate_slotsを返す
    """
    new_state = copy.deepcopy(state)
    new_state["slots_generation_count"] = new_state.get("slots_generation_count", 0) + 1

    # 直近のインタビュー対象者の発話を取得
    last_interviewee_utternace = None
    for message in reversed(new_state["dialogue_history"]):
        if message.startswith("インタビュー対象者:"):
            last_interviewee_utternace = message
            break

    # 直前の回答が「わからない」場合
    if last_interviewee_utternace == "インタビュー対象者: わかりません。":
        new_state["branch"] = "interviewer_llm_generate_slots_2"
    # それ以外の回答の場合
    else:
        if random.random() < 0.5:
            new_state["branch"] = "interviewer_llm_generate_slots_2"
        else:
            new_state["branch"] = "interviewer_llm_generate_slots"

    return new_state


# インタビュー終了時の処理
def end_interview(state: State):
    """
    end_interview: インタビューを終了する関数

    Args:
        state(State): 状態情報
    Returns:
        Dict: 更新された状態情報(インタビュー終了)
    """
    time.sleep(WAIT_TIME)
    new_state = copy.deepcopy(state)

    dialogue_history = new_state["dialogue_history"]
    sc = new_state["speak_count"]

    new_dialogue = (
        "インタビュアー: それでは、インタビューを終了します。ありがとうございました。"
    )

    # 発言回数を更新
    sc["total_count"] += 1
    sc["interviewer_count"] += 1
    sc["interviewer_generate_question_count"] += 1

    new_state["dialogue_history"] = dialogue_history + [new_dialogue]
    new_state["speak_count"] = sc

    print("\n\n===============関数end_interview===============\n")
    print(f"new_dialogue: {new_dialogue}\n")
    print(f"new_state: {new_state}\n")

    now = datetime.datetime.now(jst)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    node_name = "end_interview"
    this_node_path = os.path.join(
        execution_folder, f"info_{timestamp}_{node_name}.json"
    )

    save_state_to_file(new_state, this_node_path)
    save_state_to_file(new_state, info_path)

    return new_state


# ========================================
# -----条件分岐判定関数-----
# ========================================


# キャリアの話題が出たかどうかを判定する関数
def has_career_topic(state: State) -> bool:
    """
    has_career_topic: キャリアの話題が出たかどうかを判定する関数

    Args:
        state(State): 状態情報
    Returns:
        bool: キャリアの話題が出たかどうか
    """
    time.sleep(WAIT_TIME)
    new_state = copy.deepcopy(state)

    dialogue_history = new_state["dialogue_history"]

    # キャリアの話題が出たかどうかを判定
    template = prompt_career_topic
    prompt = PromptTemplate(
        template=template,
        input_variables=["dialogue_history_str"],
    )

    dialogue_history_str = "\n".join(dialogue_history)
    system_message = prompt.format(dialogue_history_str=dialogue_history_str)

    human_message = "判定:"

    print(
        f"\n\n===============関数has_career_topic===============\nSystemMessage=\n{system_message}\nHumanMessage=\n{human_message}\n"
    )

    try:
        response = model.invoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(content=human_message),
            ]
        )
        if response and hasattr(response, "content"):
            # str型からbool型に変換(大文字小文字を無視)
            content = response.content.strip().lower()
            if content == "true":
                has_career_topic = True
            elif content == "false":
                has_career_topic = False
            else:
                # 判定が不明な場合はFalseとする
                print(f"トピック話題判定が不明です: {content}")
                has_career_topic = False
        else:
            has_career_topic = False

    except Exception as e:
        print(f"Error occurred while invoking the model: {e}")
        has_career_topic = False

    print(f"has_career_topic: {has_career_topic}\n")

    now = datetime.datetime.now(jst)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    node_name = "has_career_topic"
    this_node_path = os.path.join(
        execution_folder, f"info_{timestamp}_{node_name}.json"
    )

    save_state_to_file(new_state, this_node_path)

    return has_career_topic


# 対話の終了を判断する関数
def finish_interview(state: State) -> str:
    """
    finish_interview: 対話の終了を判断する関数

    Args:
        state(State): 状態情報
    Returns:
        str: 対話の終了条件
    """
    time.sleep(WAIT_TIME)
    new_state = copy.deepcopy(state)

    dialogue_history = new_state["dialogue_history"]
    max_total_count = new_state["max_total_count"]
    min_total_count = new_state["min_total_count"]
    sc = new_state["speak_count"]
    total_count = sc["total_count"]
    current_slots = new_state["slots"]

    if min_total_count is not None and total_count < min_total_count:
        output = "continue"
        return output

    # 最大ターン数に達したら会話を終了
    if max_total_count is not None and total_count >= max_total_count:
        output = "end"
        return output

    # 会話を続けるかどうかを判定
    template = prompt_end_conversation
    prompt = PromptTemplate(
        template=template,
        input_variables=["dialogue_history_str", "current_slots"],
    )

    dialogue_history_str = "\n".join(dialogue_history)
    system_message = prompt.format(
        dialogue_history_str=dialogue_history_str,
        current_slots=current_slots,
    )

    human_message = "判定:"

    print(
        f"\n\n===============関数finish_interview===============\nSystemMessage=\n{system_message}\nHumanMessage=\n{human_message}\n"
    )

    try:
        response = model.invoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(content=human_message),
            ]
        )
        if response and hasattr(response, "content"):
            # str型(大文字小文字を無視)
            content = response.content.strip().lower()
            if content == "end":
                output = "end"
            else:
                output = "continue"
        else:
            output = "continue"

    except Exception as e:
        print(f"Error occurred while invoking the model: {e}")
        output = "continue"

    print(f"output: {output}\n")

    now = datetime.datetime.now(jst)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    node_name = "finish_interview"
    this_node_path = os.path.join(
        execution_folder, f"info_{timestamp}_{node_name}.json"
    )

    save_state_to_file(new_state, this_node_path)

    return output


# ペルソナ情報の推定を行う関数
def interviewer_llm_estimate_persona(state: State) -> str:
    """
    interviewer_llm_estimate_persona: ペルソナ情報の推定を行う関数

    Args:
        state(State): 状態情報
    Returns:
        str: 推定されたペルソナ情報
    """
    time.sleep(WAIT_TIME)
    new_state = copy.deepcopy(state)

    dialogue_history = new_state["dialogue_history"]
    slots = new_state["slots"]
    estimate_persona = new_state["estimate_persona"]

    # ペルソナ情報の推定
    template = prompt_estimate_persona
    prompt = PromptTemplate(
        template=template,
        input_variables=["dialogue_history_str", "current_slots", "estimate_persona"],
    )

    dialogue_history_str = "\n".join(dialogue_history)
    system_message = prompt.format(
        dialogue_history_str=dialogue_history_str,
        current_slots=slots,
        estimate_persona=estimate_persona,
    )

    human_message = "出力:"

    print(
        f"\n\n===============関数interviewer_llm_estimate_persona===============\nSystemMessage=\n{system_message}\nHumanMessage=\n{human_message}\n"
    )

    try:
        response = model.invoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(content=human_message),
            ]
        )
        if response and hasattr(response, "content"):
            print(f"estimated_persona: {response.content}\n")
            # str型(大文字小文字を無視)
            estimated_persona = response.content.strip()
        else:
            estimated_persona = "ペルソナ情報の推定に失敗しました。"
    except Exception as e:
        print(f"Error occurred while invoking the model: {e}")
        estimated_persona = "ペルソナ情報の推定に失敗しました。"

    new_state["estimate_persona"] = estimated_persona

    now = datetime.datetime.now(jst)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    node_name = "interviewer_llm_estimate_persona"
    this_node_path = os.path.join(
        execution_folder, f"info_{timestamp}_{node_name}.json"
    )

    save_state_to_file(new_state, this_node_path)

    return new_state


# -----状態遷移図の作成-----

# Graph（状態遷移図）の作成
graph_builder = StateGraph(State)

# ノードの追加
graph_builder.add_node("interviewer_llm_idle_talk", interviewer_llm_idle_talk)
graph_builder.add_node(
    "interviewer_llm_generate_question", interviewer_llm_generate_question
)
graph_builder.add_node("interviewer_llm_fill_slots", interviewer_llm_fill_slots)
graph_builder.add_node(
    "interviewer_llm_estimate_persona", interviewer_llm_estimate_persona
)
graph_builder.add_node("interviewer_llm_generate_slots", interviewer_llm_generate_slots)
graph_builder.add_node(
    "interviewer_llm_generate_slots_2", interviewer_llm_generate_slots_2
)
graph_builder.add_node("select_generate_slots_node", select_generate_slots_node)
graph_builder.add_node("interviewee_llm_idle_talk", interviewee_llm_idle_talk)
graph_builder.add_node(
    "interviewee_llm_generate_answer", interviewee_llm_generate_answer
)
graph_builder.add_node("end_interview", end_interview)

# エッジの追加
# graph_builder.add_edge(START, "interviewer_start")
graph_builder.add_edge("interviewer_llm_idle_talk", "interviewee_llm_idle_talk")
graph_builder.add_conditional_edges(
    "interviewee_llm_idle_talk",
    has_career_topic,
    {
        True: "interviewer_llm_fill_slots",
        False: "interviewer_llm_idle_talk",
    },
)
graph_builder.add_edge("interviewer_llm_fill_slots", "interviewer_llm_estimate_persona")
graph_builder.add_conditional_edges(
    "interviewer_llm_estimate_persona",
    finish_interview,
    {
        "end": "end_interview",
        "continue": "select_generate_slots_node",
    },
)
graph_builder.add_conditional_edges(
    "select_generate_slots_node",
    lambda st: st["branch"],
    {
        "interviewer_llm_generate_slots": "interviewer_llm_generate_slots",
        "interviewer_llm_generate_slots_2": "interviewer_llm_generate_slots_2",
    },
)
graph_builder.add_conditional_edges(
    "interviewer_llm_generate_slots",
    finish_interview,
    {
        "end": "end_interview",
        "continue": "interviewer_llm_generate_question",
    },
)
graph_builder.add_conditional_edges(
    "interviewer_llm_generate_slots_2",
    finish_interview,
    {
        "end": "end_interview",
        "continue": "interviewer_llm_generate_question",
    },
)

graph_builder.add_edge(
    "interviewer_llm_generate_question", "interviewee_llm_generate_answer"
)
graph_builder.add_edge("interviewee_llm_generate_answer", "interviewer_llm_fill_slots")
# graph_builder.add_edge("interviewer", END)


# Graphの始点を宣言
graph_builder.set_entry_point("interviewer_llm_idle_talk")

# Graphの終点を宣言
graph_builder.set_finish_point("end_interview")

# Graphのコンパイル
graph = graph_builder.compile()


# CLIモードとUIモードを切り替え
RUN_MODE = os.getenv("RUN_MODE", "ui")  # "cli" または "ui"

if RUN_MODE == "cli":
    # CLIモード: グラフを自動実行
    new_state = graph.invoke(
        {
            "dialogue_history": interview_config["dialogue_history"],
            "speak_count": interview_config["speak_count"],
            "max_total_count": interview_config["max_total_count"],
            "min_total_count": interview_config["min_total_count"],
            "estimate_persona": interview_config["estimate_persona"],
            "persona_attribute_candidates": interview_config[
                "persona_attribute_candidates"
            ],
            "slots": interview_config["slots"],
            "slot_generation_count": interview_config["slot_generation_count"],
            "branch": interview_config["branch"],
        },
        config={
            "recursion_limit": 200,
        },
    )

    # Graphの可視化
    try:
        with open(graph_image_path, 'wb') as f:
            f.write(graph.get_graph(xray=True).draw_mermaid_png())
    except Exception as e:
        print(f"Graphの可視化に失敗しました: {e}")

    send_line_notify(
        f"インタビューが終了しました。結果は{execution_folder}に保存されました。"
    )

    print(f"インタビューが終了しました。結果は{execution_folder}に保存されました。")

else:
    # UIモード: Gradioインターフェースを起動
    from typing import Tuple
    import numpy as np
    import gradio as gr
    import sys
    from pathlib import Path
    
    # パスの設定
    project_root = Path(__file__).parent.parent.parent
    asr_path = project_root / "Realtime_ASR_FasterWhisper_Gradio-main"
    src_path = project_root / "src"
    
    # srcディレクトリをsys.pathに追加（core.tts_outputなどのインポート用）
    src_path_str = str(src_path.resolve())  # 絶対パスに変換
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)
    
    # Realtime_ASR_FasterWhisper_Gradio-mainのパスを追加
    if str(asr_path) not in sys.path:
        sys.path.insert(0, str(asr_path))
    
    # Realtime_ASR_FasterWhisper_Gradio-mainからインポート（明示的にパスを指定）
    import importlib.util
    processor_spec = importlib.util.spec_from_file_location(
        "asr_core_processor",
        asr_path / "core" / "processor.py"
    )
    asr_core_processor = importlib.util.module_from_spec(processor_spec)
    # sys.modulesに登録してからexec_module（dataclassデコレータが正しく動作するために必要）
    sys.modules["asr_core_processor"] = asr_core_processor
    processor_spec.loader.exec_module(asr_core_processor)
    
    config_spec = importlib.util.spec_from_file_location(
        "asr_core_config",
        asr_path / "core" / "config.py"
    )
    asr_core_config = importlib.util.module_from_spec(config_spec)
    # sys.modulesに登録してからexec_module
    sys.modules["asr_core_config"] = asr_core_config
    config_spec.loader.exec_module(asr_core_config)
    
    session_store_spec = importlib.util.spec_from_file_location(
        "asr_core_session_store",
        asr_path / "core" / "session_store.py"
    )
    asr_core_session_store = importlib.util.module_from_spec(session_store_spec)
    # sys.modulesに登録してからexec_module（dataclassデコレータが正しく動作するために必要）
    sys.modules["asr_core_session_store"] = asr_core_session_store
    session_store_spec.loader.exec_module(asr_core_session_store)
    
    StreamingAudioProcessor = asr_core_processor.StreamingAudioProcessor
    StreamingConfig = asr_core_config.StreamingConfig
    SessionStore = asr_core_session_store.SessionStore
    StreamingState = asr_core_session_store.StreamingState
    
    # TTSProcessorのインポート（動的インポート）- src/core/tts_output.pyから
    tts_output_path = src_path / "core" / "tts_output.py"
    if not tts_output_path.exists():
        raise FileNotFoundError(f"TTSProcessorモジュールが見つかりません: {tts_output_path}")
    
    import importlib.util
    tts_output_spec = importlib.util.spec_from_file_location(
        "core.tts_output",
        tts_output_path
    )
    tts_output_module = importlib.util.module_from_spec(tts_output_spec)
    # sys.modulesに登録してからexec_module（他のモジュールからの参照に対応）
    sys.modules["core.tts_output"] = tts_output_module
    tts_output_spec.loader.exec_module(tts_output_module)
    TTSProcessor = tts_output_module.TTSProcessor
    
    # TTSProcessorの初期化
    try:
        tts = TTSProcessor()
        print("TTSProcessorの初期化に成功しました")
    except Exception as e:
        print(f"TTSProcessorの初期化に失敗しました: {e}")
        # フォールバック: SimpleTTS
        class SimpleTTS:
            def __init__(self, sr: int = 22050):
                self.sr = sr
            def synthesize(self, text: str) -> Tuple[int, np.ndarray]:
                dur = 0.4
                n = int(self.sr * dur)
                return self.sr, np.zeros(n, dtype=np.float32)
        tts = SimpleTTS()
    
    # StreamingAudioProcessorの初期化
    try:
        streaming_config = StreamingConfig()
        
        # VAD設定を調整（必要に応じて変更可能）
        # 環境変数で設定されていない場合は、ここで直接変更できます
        # 注意: StreamingConfigはfrozen dataclassなので、以下のように環境変数を設定してから作成する必要があります
        if "VAD_THRESHOLD" not in os.environ:
            os.environ["VAD_THRESHOLD"] = "0.5"  # より低い閾値で発話を検出しやすく
        if "MIN_SPEECH_DURATION_MS" not in os.environ:
            os.environ["MIN_SPEECH_DURATION_MS"] = "250"  # より短い発話でも検出
        if "MIN_SILENCE_DURATION_MS" not in os.environ:
            os.environ["MIN_SILENCE_DURATION_MS"] = "400"  # より短い無音で確定
        if "STREAMING_AUDIO_DEBUG" not in os.environ:
            os.environ["STREAMING_AUDIO_DEBUG"] = "1"  # デバッグ出力を有効化
        
        # 設定を再読み込み
        streaming_config = StreamingConfig()
        
        print(f"[ASR INIT] StreamingConfig: whisper_model_id={streaming_config.whisper_model_id}, language={streaming_config.language}")
        print(f"[ASR INIT] VAD設定: vad_threshold={streaming_config.vad_threshold}, min_speech_duration_ms={streaming_config.min_speech_duration_ms}, min_silence_duration_ms={streaming_config.min_silence_duration_ms}")
        print("[ASR INIT] StreamingAudioProcessorの初期化を開始します（Whisperモデルの読み込みには時間がかかる場合があります）...")
        streaming_processor = StreamingAudioProcessor(streaming_config)
        print("StreamingAudioProcessorの初期化に成功しました")
    except Exception as e:
        print(f"StreamingAudioProcessorの初期化に失敗しました: {e}")
        import traceback
        traceback.print_exc()
        streaming_processor = None
    
    # セッションごとのStreamingState管理
    asr_session_store = SessionStore(ttl_seconds=3600)
    
    # 音声認識関数（Realtime_ASR_FasterWhisper_Gradio-mainを使用）
    def real_time_transcribe(new_chunk, asr_session_id=None):
        """Realtime_ASR_FasterWhisper_Gradio-mainを使用した音声認識"""
        try:
            if new_chunk is None:
                print("[ASR DEBUG] new_chunk is None")
                return ""
            
            if streaming_processor is None:
                print("[ASR DEBUG] streaming_processor is None")
                return ""

            sr, y = new_chunk
            print(f"[ASR DEBUG] 音声チャンク受信: sr={sr}, shape={y.shape if hasattr(y, 'shape') else type(y)}")
            
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = y.astype(np.float32)

            # StreamingAudioProcessorを使用
            asr_session_id, asr_state = asr_session_store.get_or_create(asr_session_id)
            asr_state, accumulated_text, is_final = streaming_processor.process_chunk(
                asr_state, y, int(sr)
            )
            
            print(f"[ASR DEBUG] process_chunk結果: accumulated_text='{accumulated_text}', is_final={is_final}, session_id={asr_session_id}")
            
            # 最終的なテキストが確定した場合のみ返す
            # セッションをリセットしない（次の発話でもセッションを継続使用）
            if is_final and accumulated_text:
                print(f"[ASR DEBUG] 認識結果を返します: '{accumulated_text.strip()}'")
                return accumulated_text.strip()
            
            # 中間結果は返さない（最終結果のみ）
            return ""
        except Exception as e:
            print(f"[ASR DEBUG] real_time_transcribe エラー: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    # セッション管理
    class Session:
        def __init__(self):
            self.chat_history: List[List[str]] = []
            self.state: State = {
                "dialogue_history": interview_config["dialogue_history"].copy(),
                "speak_count": copy.deepcopy(interview_config["speak_count"]),
                "max_total_count": interview_config["max_total_count"],
                "min_total_count": interview_config["min_total_count"],
                "estimate_persona": interview_config["estimate_persona"],
                "persona_attribute_candidates": interview_config["persona_attribute_candidates"].copy(),
                "slots": copy.deepcopy(interview_config["slots"]),
                "slots_generation_count": interview_config["slot_generation_count"],
                "branch": interview_config["branch"],
            }
            self.initiated = False
            self.current_node = "interviewer_llm_idle_talk"  # 現在のノード名
            self.asr_session_id = None  # ASRセッションID
    
    # UIアダプタ: グラフを段階的に実行
    class ChatInterface:
        _sessions: Dict[str, Session] = {}
        
        def __init__(self):
            self.tts = tts
        
        def get_or_create_session(self, cookie_id: str) -> Session:
            if cookie_id not in self._sessions:
                self._sessions[cookie_id] = Session()
            return self._sessions[cookie_id]
        
        def _pairs_from_history(self, dialogue_history: List[str]) -> List[List[str]]:
            """対話履歴を [人間, AI] のペアに変換"""
            pairs = []
            current_user = None
            current_ai = None
            
            for line in dialogue_history:
                if line.startswith("インタビュー対象者:"):
                    if current_ai:
                        pairs.append([current_user, current_ai])
                        current_user = None
                        current_ai = None
                    current_user = line.split(":", 1)[1].strip()
                elif line.startswith("インタビュアー:"):
                    current_ai = line.split(":", 1)[1].strip()
                    if current_user:
                        pairs.append([current_user, current_ai])
                        current_user = None
                        current_ai = None
            
            if current_user and current_ai:
                pairs.append([current_user, current_ai])
            elif current_ai:
                pairs.append(["", current_ai])
            
            return pairs
        
        def _get_last_interviewer_message(self, state: State) -> str:
            """最後のインタビュアーの発話を取得"""
            for line in reversed(state["dialogue_history"]):
                if line.startswith("インタビュアー:"):
                    return line.split(":", 1)[1].strip()
            return ""
        
        def _kickoff(self, session: Session) -> Optional[str]:
            """初回アイスブレイク"""
            if not session.state["dialogue_history"]:
                new_state = interviewer_llm_idle_talk(session.state)
                session.state = new_state
                session.current_node = "interviewee_llm_idle_talk"
                return self._get_last_interviewer_message(new_state)
            return None
        
        def _step_with_user_input(self, session: Session, user_text: str) -> str:
            """ユーザー入力を受け取って、グラフの次のステップを実行"""
            # ユーザー発話を履歴に追加（interviewee_llm_idle_talkまたはinterviewee_llm_generate_answerの代わり）
            session.state["dialogue_history"].append(f"インタビュー対象者: {user_text}")
            sc = session.state["speak_count"]
            sc["total_count"] += 1
            sc["interviewee_count"] += 1
            
            # 現在のノードに応じて次のステップを実行
            current_node = session.current_node
            
            # 発言回数の詳細を更新
            if current_node == "interviewee_llm_idle_talk":
                sc["interviewee_idle_talk_count"] += 1
            elif current_node == "interviewee_llm_generate_answer":
                sc["interviewee_generate_answer_count"] += 1
            
            if current_node == "interviewee_llm_idle_talk":
                # キャリア話題判定
                has_career = has_career_topic(session.state)
                if has_career:
                    session.state = interviewer_llm_fill_slots(session.state)
                    session.state = interviewer_llm_estimate_persona(session.state)
                    # 終了判定
                    finish_result = finish_interview(session.state)
                    if finish_result == "end":
                        session.state = end_interview(session.state)
                        return "インタビューを終了しました。"
                    # スロット生成ノード選択
                    session.state = select_generate_slots_node(session.state)
                    branch = session.state["branch"]
                    if branch == "interviewer_llm_generate_slots":
                        session.state = interviewer_llm_generate_slots(session.state)
                    else:
                        session.state = interviewer_llm_generate_slots_2(session.state)
                    # 終了判定
                    finish_result = finish_interview(session.state)
                    if finish_result == "end":
                        session.state = end_interview(session.state)
                        return "インタビューを終了しました。"
                    session.state = interviewer_llm_generate_question(session.state)
                    session.current_node = "interviewee_llm_generate_answer"
                else:
                    session.state = interviewer_llm_idle_talk(session.state)
                    session.current_node = "interviewee_llm_idle_talk"
                    
            elif current_node == "interviewee_llm_generate_answer":
                session.state = interviewer_llm_fill_slots(session.state)
                session.state = interviewer_llm_estimate_persona(session.state)
                # 終了判定
                finish_result = finish_interview(session.state)
                if finish_result == "end":
                    session.state = end_interview(session.state)
                    return "インタビューを終了しました。"
                # スロット生成ノード選択
                session.state = select_generate_slots_node(session.state)
                branch = session.state["branch"]
                if branch == "interviewer_llm_generate_slots":
                    session.state = interviewer_llm_generate_slots(session.state)
                else:
                    session.state = interviewer_llm_generate_slots_2(session.state)
                # 終了判定
                finish_result = finish_interview(session.state)
                if finish_result == "end":
                    session.state = end_interview(session.state)
                    return "インタビューを終了しました。"
                session.state = interviewer_llm_generate_question(session.state)
                session.current_node = "interviewee_llm_generate_answer"
            
            return self._get_last_interviewer_message(session.state)
        
        def process_message(
            self,
            message: str,
            chat_history: List[List[str]],
            cookie_id: str,
        ) -> Tuple[str, List[List[str]], Optional[Tuple[int, np.ndarray]]]:
            sess = self.get_or_create_session(cookie_id)
            
            # 初回: アイスブレイク
            if not sess.initiated:
                kickoff_text = self._kickoff(sess)
                pairs = self._pairs_from_history(sess.state["dialogue_history"])
                sess.chat_history = pairs
                sess.initiated = True
                if kickoff_text:
                    audio_data = self.tts.synthesize(kickoff_text)
                    return "", sess.chat_history, audio_data
                return "", sess.chat_history, None
            
            # 入力が空なら何もしない
            if not message:
                return "", chat_history, None
            
            # 1ステップ実行
            ai_text = self._step_with_user_input(sess, message)
            pairs = self._pairs_from_history(sess.state["dialogue_history"])
            sess.chat_history = pairs
            audio_data = self.tts.synthesize(ai_text) if ai_text else None
            return "", sess.chat_history, audio_data
    
    # Gradio UI
    def create_interface():
        with gr.Blocks() as demo:
            interface = ChatInterface()
            cookie_id = gr.State(value="")
            
            with gr.Tabs():
                with gr.Tab("チャットボット"):
                    chatbot = gr.Chatbot(label="チャットボット", height=600)
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            container=False,
                            show_label=False,
                            placeholder="メッセージを入力するか、マイクで録音してください",
                            scale=7
                        )
                        submit_btn = gr.Button("送信", variant="primary", scale=1, min_width=150)
                    
                    with gr.Row():
                        mic = gr.Audio(
                            label='マイク録音',
                            sources='microphone',
                            type='numpy',
                            streaming=True,  # ストリーミングモードを有効化
                        )
                    
                    audio_out = gr.Audio(label="AIの応答", autoplay=True)
                    
                    # 前回の音声認識結果を記録（自動送信の重複防止用）
                    last_transcribed_text = gr.State(value="")
                    asr_session_id_state = gr.State(value=None)  # ASRセッションIDを管理
                    
                    def transcribe_and_auto_submit(audio_input, current_msg, current_chat_history, current_cookie_id, last_text_state, asr_sess_id):
                        """音声認識して自動送信（ストリーミング処理）"""
                        print(f"[UI DEBUG] transcribe_and_auto_submit呼び出し: audio_input={audio_input is not None}, asr_sess_id={asr_sess_id}")
                        
                        if audio_input is None:
                            print("[UI DEBUG] audio_input is None - スキップ")
                            return current_msg, current_chat_history, None, last_text_state, asr_sess_id
                        
                        try:
                            # セッションを取得してASRセッションIDを管理
                            sess = interface.get_or_create_session(current_cookie_id if current_cookie_id else "")
                            # セッションIDがstateから渡されていない場合は、セッションから取得または新規作成
                            if asr_sess_id is None:
                                if sess.asr_session_id is None:
                                    sess.asr_session_id, _ = asr_session_store.get_or_create(None)
                                    print(f"[UI DEBUG] 新しいASRセッションを作成: {sess.asr_session_id}")
                                asr_sess_id = sess.asr_session_id
                            else:
                                # stateから渡されたセッションIDを更新
                                sess.asr_session_id = asr_sess_id
                            
                            # 音声認識（Realtime_ASR_FasterWhisper_Gradio-mainを使用）
                            # ストリーミング処理なので、音声チャンクごとに呼ばれる
                            new_text = real_time_transcribe(audio_input, asr_sess_id)
                            
                            print(f"[UI DEBUG] 認識結果: new_text='{new_text}', last_text_state='{last_text_state}', sess.initiated={sess.initiated}")
                            
                            # 認識結果があって、前回と異なる場合は自動送信（is_final=Trueのときのみ）
                            if new_text and new_text.strip() and new_text != last_text_state:
                                print(f"[UI DEBUG] 自動送信を実行します: '{new_text}'")
                                # 初回でない場合のみ自動送信
                                if sess.initiated:
                                    result_msg, result_chat, result_audio = interface.process_message(new_text, current_chat_history, current_cookie_id if current_cookie_id else "")
                                    return result_msg, result_chat, result_audio, new_text, sess.asr_session_id  # (msg, chatbot, audio_out, last_text, asr_session_id)
                                else:
                                    # 初回の場合はテキストボックスに反映するだけ
                                    return new_text, current_chat_history, None, new_text, sess.asr_session_id
                        except Exception as e:
                            print(f"[UI DEBUG] 音声認識・自動送信エラー: {e}")
                            import traceback
                            traceback.print_exc()
                        
                        # セッションIDを返して状態を更新
                        final_session_id = sess.asr_session_id if 'sess' in locals() and sess.asr_session_id else asr_sess_id
                        return current_msg, current_chat_history, None, last_text_state, final_session_id
                    
                    def initialize_chat():
                        """ページロード時にアイスブレイクを実行"""
                        empty_cookie = ""
                        sess = interface.get_or_create_session(empty_cookie)
                        if not sess.initiated:
                            result = interface.process_message("", [], empty_cookie)
                            return result  # (msg, chatbot, audio_out)
                        return "", [], None
                    
                    # 手動送信ボタン
                    submit_btn.click(
                        fn=interface.process_message,
                        inputs=[msg, chatbot, cookie_id],
                        outputs=[msg, chatbot, audio_out]
                    )
                    
                    # Enterキーでも送信
                    msg.submit(
                        fn=interface.process_message,
                        inputs=[msg, chatbot, cookie_id],
                        outputs=[msg, chatbot, audio_out]
                    )
                    
                    # 音声認識と自動送信（ストリーミング処理）
                    mic.stream(
                        fn=transcribe_and_auto_submit,
                        inputs=[mic, msg, chatbot, cookie_id, last_transcribed_text, asr_session_id_state],
                        outputs=[msg, chatbot, audio_out, last_transcribed_text, asr_session_id_state],
                        stream_every=0.3  # 0.3秒ごとに音声チャンクを処理
                    )
                    
                    # ページロード時に初期化（初回アイスブレイク）
                    demo.load(
                        fn=initialize_chat,
                        inputs=None,
                        outputs=[msg, chatbot, audio_out]
                    )
            
            return demo
    
    # UI起動
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", share=True)
