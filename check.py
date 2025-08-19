"""
检查两个文件答案之间的差异
"""
import json
from collections import defaultdict
import pandas as pd


def check(label_json_path:str, pred_json_path:str, dump:bool = False)-> pd.DataFrame:
    with open(label_json_path, "r", encoding="utf-8") as f:
        label_json = json.load(f)
    
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_json = json.load(f)
    label_df = pd.DataFrame(label_json)
    pred_df = pd.DataFrame(pred_json)
    pred_df.columns = list(pred_df.columns.map(lambda x: "pred_"+x))
    df = label_df.merge(pred_df, left_on="question", right_on="pred_question", how="left")
    
    def jaccard_similarity(s1:str, s2:str)->float:
        set1 = set(s1.split(""))
        set2 = set(s2.split(""))
        inter_len = len(set1 & set2)
        union_len = len(set1 | set2)
        return inter_len/union_len if union_len > 0 else 0.0
    df["score"] = df.apply(lambda x: jaccard_similarity(x["answer"], x["pred_answer"]), axis=1)
    if dump:
        file_noeq = df[df["filename"] != df["pred_filename"]]
        file_noeq.to_json("./result/file_noeq.json")
        file_eq = df[df["filename"] == df["pred_filename"]]
        page_noeq = file_eq[file_eq["page"] != file_eq["pred_page"]]
        page_noeq.to_json("./result/page_noeq.json")
        score_df = df.sort_values(by="score", ascending=True)
        score_df.to_json("./result/socre.json")

    return df