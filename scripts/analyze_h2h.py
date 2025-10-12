import pandas as pd, numpy as np
from itertools import combinations

def kripp_alpha_nominal(X):
    X = np.asarray(X, dtype=float)
    mask = ~np.isnan(X)
    cats = np.unique(X[mask])
    # observed disagreement
    Do_num=0.0; Do_den=0.0
    for i in range(X.shape[0]):
        vals = X[i, mask[i]]
        m = len(vals)
        if m<=1: continue
        Do_den += m*(m-1)
        for c in cats:
            n = np.sum(vals==c)
            Do_num += n*(m-n)
    Do = Do_num/Do_den if Do_den>0 else np.nan
    # expected disagreement
    n_tot = np.sum(mask)
    if n_tot==0: return np.nan
    p = [np.sum((X==c)&mask)/n_tot for c in cats]
    De = 1.0 - np.sum(np.square(p))
    if De==0: return 1.0 if Do==0 else np.nan
    return 1.0 - Do/De

def main():
    path = r"annotated data sets\annotation_causal.csv"
    df = pd.read_csv(path)
    raters = [c for c in df.columns if c!="Sentence"]
    # normalize to floats with NaN for blanks
    df[raters] = df[raters].replace({"":np.nan,"nan":np.nan}).astype(float)
    rows=[]
    for a,b in combinations(raters,2):
        sub = df[[a,b]].dropna()
        if len(sub)>=2:
            alpha = kripp_alpha_nominal(sub.values)
            rows.append((a,b,len(sub),alpha))
    if not rows:
        print("No overlapping rater pairs with >=2 items.")
        return
    out = pd.DataFrame(rows, columns=["rater_a","rater_b","n_items","alpha"]).sort_values("alpha", ascending=False)
    out.to_csv(r"outputs\h2h_pairs.csv", index=False)
    print("Pairs:", len(out))
    print("Mean α:", out["alpha"].mean())
    print("Median α:", out["alpha"].median())
    print(out.head(10).to_string(index=False))

if __name__=="__main__":
    main()
