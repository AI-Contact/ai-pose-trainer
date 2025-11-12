import autorootcwd
from pathlib import Path

import click
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score


def find_threshold_by_f1(y_true, y_probs):
    """F1 score를 최대화하는 threshold 찾기"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


def find_threshold_by_roc(y_true, y_probs):
    """ROC AUC를 최대화하는 threshold 찾기 (Youden's J statistic)"""
    if len(np.unique(y_true)) < 2:
        return 0.5
    
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


@click.command()
@click.option("--exercise", default="push_up", type=click.Choice(["cross_lunge", "crunch", "plank", "push_up", "leg_raise"]), help="운돓 종류")
@click.option("--pred-csv", required=True, type=click.Path(path_type=Path), help="모델 예측 결과 CSV")
@click.option("--method", default="f1", type=click.Choice(["f1", "roc", "both"]), help="threshold 찾는 방법")
def main(exercise: str, pred_csv: Path, method: str):

    label_csv = Path(f"data/{exercise}/subject_labels.csv")    
    out_csv = Path(f"result/{exercise}/optimal_thresholds.csv")
    pred_df = pd.read_csv(pred_csv)
    label_df = pd.read_csv(label_csv)

    # sequence를 subject로 변환 (561/561-1-3-27-Z1_C -> 561)
    pred_df["subject"] = pred_df["sequence"].str.split("/").str[0].astype(str)
    
    # label_df의 subject 타입 일치시키기 (한 번만)
    label_df["subject"] = label_df["subject"].astype(str)
    
    # 조건 컬럼들 찾기 (첫 번째 컬럼 제외)
    condition_cols = [c for c in pred_df.columns if c not in ["sequence", "subject"]]
    
    results = []
    
    for cond in condition_cols:
        # subject별로 평균 예측 확률 계산
        pred_by_subject = pred_df.groupby("subject")[cond].mean()
        
        # 실제 라벨 (subject 기준)
        label_by_subject = label_df.set_index("subject")[cond]
        
        # 두 데이터프레임의 인덱스 교집합
        common_subjects = pred_by_subject.index.intersection(label_by_subject.index)
        
        if len(common_subjects) == 0:
            print(f"Skipping {cond}: no common subjects")
            print(f"  pred subjects (first 5): {list(pred_by_subject.index[:5])}")
            print(f"  label subjects (first 5): {list(label_by_subject.index[:5])}")
            continue
        
        y_true = label_by_subject.loc[common_subjects].astype(bool).astype(int).values
        y_probs = pred_by_subject.loc[common_subjects].values
        
        if len(np.unique(y_true)) < 2:
            print(f"Skipping {cond}: only one class in labels")
            continue
        
        thresh_f1 = find_threshold_by_f1(y_true, y_probs)
        thresh_roc = find_threshold_by_roc(y_true, y_probs)
        
        # threshold 적용해서 평가
        pred_f1 = (y_probs >= thresh_f1).astype(int)
        pred_roc = (y_probs >= thresh_roc).astype(int)
        
        f1_score_f1 = f1_score(y_true, pred_f1)
        f1_score_roc = f1_score(y_true, pred_roc)
        
        auc = roc_auc_score(y_true, y_probs)
        
        results.append({
            "condition": cond,
            "threshold_f1": thresh_f1,
            "threshold_roc": thresh_roc,
            "f1_score_f1": f1_score_f1,
            "f1_score_roc": f1_score_roc,
            "auc": auc,
        })
        
        print(f"{cond}:")
        print(f"  F1 method: threshold={thresh_f1:.4f}, F1={f1_score_f1:.4f}")
        print(f"  ROC method: threshold={thresh_roc:.4f}, F1={f1_score_roc:.4f}, AUC={auc:.4f}")
    
    if not results:
        print("\nNo valid results found. Check subject matching between pred_csv and label_csv.")
        return
    
    df_out = pd.DataFrame(results)
    
    if method == "f1":
        df_out["recommended_threshold"] = df_out["threshold_f1"]
    elif method == "roc":
        df_out["recommended_threshold"] = df_out["threshold_roc"]
    else:
        # F1이 더 좋으면 F1, ROC가 더 좋으면 ROC
        df_out["recommended_threshold"] = df_out.apply(
            lambda row: row["threshold_f1"] if row["f1_score_f1"] >= row["f1_score_roc"] else row["threshold_roc"],
            axis=1
        )
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()

