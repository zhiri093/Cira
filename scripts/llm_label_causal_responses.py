import os, json, time, argparse, sys, requests, pandas as pd

API_URL = "https://api.openai.com/v1/responses"
MODEL   = os.environ.get("OPENAI_MODEL", "gpt-5-nano")   # works with your test key

SYSTEM_PROMPT = """You are a careful annotator for causal relations in one sentence.
Return ONLY JSON: {"label": 0 or 1, "confidence": 0..1}
Label 1 if the sentence clearly states/implies cause→effect (because, due to, results in, leads to, if X then Y, etc.). Else 0.
Do NOT default to 0; in similar corpora 25–40% are causal."""

def parse_output(data: dict) -> str:
    """
    Responses API can return text in a few shapes; this tries them in order.
    Returns the JSON string we asked the model to produce.
    """
    # 1) direct convenience field (when present)
    if "output_text" in data and data["output_text"]:
        return data["output_text"]
    # 2) output.text (newer shape)
    try:
        ot = data.get("output", {})
        if isinstance(ot, dict) and "text" in ot and ot["text"]:
            return ot["text"]
    except: pass
    # 3) output.choices[0].message.content[0].text (older-like shape)
    try:
        choices = data.get("output", {}).get("choices", [])
        if choices:
            content = choices[0]["message"]["content"]
            if content and isinstance(content, list) and "text" in content[0]:
                return content[0]["text"]
    except: pass
    # 4) output (array of objects) → look for a message with content[].text
    try:
        out = data.get("output", [])
        for item in out:
            if item.get("type") == "message":
                content = item.get("content", [])
                for chunk in content:
                    if "text" in chunk:
                        return chunk["text"]
    except: pass
    raise ValueError("Could not find output text in response payload")

def call_responses(api_key, model, text, retries=3):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    payload = {
        "model": model,
        "input": f"{SYSTEM_PROMPT}\n\nSentence: \"{text.replace('\"','\\\"')}\"\nReturn JSON only.",
    }
    last = ""
    for a in range(retries):
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        last = r.text[:400]
        if r.status_code == 200:
            content = parse_output(r.json())
            obj = json.loads(content)
            y  = 1 if int(obj.get("label", 0)) == 1 else 0
            p  = float(obj.get("confidence", 0.5))
            return y, max(0.0, min(1.0, p))
        time.sleep(1.0*(a+1))
    raise RuntimeError(f"Responses API failed: {last}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key: sys.exit("Set OPENAI_API_KEY")
    df = pd.read_csv(args.in_csv)
    if "text" not in df.columns: sys.exit('Input must have "text" column')
    if args.limit > 0: df = df.head(args.limit)

    out = []
    for i, row in df.iterrows():
        t = str(row["text"]).strip()
        if not t: continue
        try:
            y, p = call_responses(key, MODEL, t)
            out.append({"text": t, "model_label": y, "confidence": p})
        except Exception as e:
            print("API error on row", i, "->", e)
            continue
        if args.sleep > 0: time.sleep(args.sleep)
        if (i+1) % 20 == 0: print(f"Labeled {i+1} sentences...")
    pd.DataFrame(out).to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(out)} rows.")

if __name__ == "__main__":
    main()
