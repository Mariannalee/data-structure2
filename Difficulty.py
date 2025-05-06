import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from snownlp import SnowNLP

matplotlib.use('Agg')
matplotlib.rc('font', family='Microsoft JhengHei')

# ✅ JLPT 詞彙表（可自行擴充）
jlpt_vocab = {
    "N1": [],
    "N2": ["東奔西走", "縁談", "学務委員", "地主"],
    "N3": ["責任", "情熱", "無責任", "時間表", "文句", "いたずら", "却って", "安心"],
    "N4": ["先生", "子供", "校長", "主任", "事業", "音楽", "ソロバン", "必要", "可愛い", "理由", "甘える", "我がまま"],
    "N5": ["私", "人", "数", "こと", "も", "から", "ない", "は", "と", "だけ", "に", "が", "で", "する", "そういう", "ある", "本当"]
}

def get_jlpt_level(word):
    for level in jlpt_vocab:
        if word in jlpt_vocab[level]:
            return level
    return None  # ❌ 不在任何級數則跳過

def jlpt_level_to_score(level):
    return {"N1": 5, "N2": 4, "N3": 3, "N4": 2, "N5": 1}.get(level, None)

def compute_sentence_score(sentence):
    words = list(SnowNLP(sentence).words)
    scores = []
    for w in words:
        level = get_jlpt_level(w)
        score = jlpt_level_to_score(level)
        if score is not None:
            scores.append(score)
    return sum(scores) / len(scores) if scores else None  # 若沒有任何有效詞彙，就跳過不計 # ❗ 若無有效詞彙則回傳 None

def generate_jlpt_difficulty_plot(user_id, user_entries):
    output_dir = "static/jlpt_difficulty"
    os.makedirs(output_dir, exist_ok=True)

    user_entries["句數"] = pd.to_numeric(user_entries["句數"], errors="coerce")
    user_entries = user_entries.sort_values("句數")

    user_entries["平均難度"] = user_entries["內容"].apply(compute_sentence_score)

    # 移除無有效詞彙的句子
    filtered_entries = user_entries.dropna(subset=["平均難度"])
    avg_score = filtered_entries["平均難度"].mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x="句數", y="平均難度", data=filtered_entries, marker="o", color="purple", label="JLPT 難度")
    plt.axhline(y=avg_score, color='green', linestyle='--', label=f"全體平均 ({avg_score:.2f})")
    plt.xlabel("句數")
    plt.ylabel("平均難度分數 (N5=1 ~ N1=5)")
    plt.title(f"用戶 {user_id} 的句子難度趨勢圖")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.ylim(1, 5.2)

    output_path = os.path.join(output_dir, f"jlpt_trend_{user_id}.png")
    plt.savefig(output_path)
    plt.close()

    return output_path
