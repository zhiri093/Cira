import pandas as pd, os
src = r"annotated data sets\overall.csv"   # note the space in the folder name
df = pd.read_csv(src)
out = r"outputs\sentences_only.csv"
df[["Sentence"]].rename(columns={"Sentence":"text"}).to_csv(out, index=False)
print(f"Wrote {out} with", len(df), "rows")
