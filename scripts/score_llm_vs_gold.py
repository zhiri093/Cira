import argparse, pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def to01(x):
    if pd.isna(x): return np.nan
    if isinstance(x,str):
        t=x.strip().lower()
        if t in ("1","true","yes"): return 1
        if t in ("0","false","no"): return 0
    try: return int(float(x))
    except: return np.nan

ap = argparse.ArgumentParser(description="Score LLM predictions vs CiRA gold labels")
ap.add_argument("--label_col", default="Causal")
ap.add_argument("--overall_csv", default=r"annotated data sets\overall.csv")
ap.add_argument("--pred_csv", required=True)
ap.add_argument("--out_csv", default=r"outputs\merged_llm_gold.csv")
args = ap.parse_args()

gold = pd.read_csv(args.overall_csv).rename(columns={"Sentence":"text"})
pred = pd.read_csv(args.pred_csv)

merged = gold[["text", args.label_col]].merge(
    pred[["text","model_label"] + ([c for c in ["confidence"] if c in pred.columns])],
    on="text", how="inner"
)
merged["y_true"] = merged[args.label_col].apply(to01)
merged["y_pred"] = merged["model_label"].apply(to01)
merged = merged.dropna(subset=["y_true","y_pred"])

if len(merged)==0:
    raise SystemExit("No overlaps. Make sure 'text' matches exactly.")

y_true = merged["y_true"].astype(int).values
y_pred = merged["y_pred"].astype(int).values

acc = accuracy_score(y_true,y_pred)
prec,rec,f1,_ = precision_recall_fscore_support(y_true,y_pred,average="binary",zero_division=0)
tn,fp,fn,tp = confusion_matrix(y_true,y_pred).ravel()

print(f"N={len(merged)} | Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")
print(f"Confusion: [[{tn}, {fp}],[{fn}, {tp}]]")

merged.to_csv(args.out_csv, index=False)
print("Wrote", args.out_csv)
