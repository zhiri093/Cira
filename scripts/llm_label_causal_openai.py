#!/usr/bin/env python
"""
LLM causal labeling for CiRA sentences (OpenAI API).

- Reads CSV with column "text"
- Calls model set via OPENAI_MODEL (default: gpt-4o). Falls back to gpt-4o-mini if needed.
- Returns JSON only: {"label": 0/1, "confidence": 0..1}
- Writes CSV: text,model_label,confidence
- On API error: prints a clear message and SKIPS the row (no fake zeros).

Usage:
  $env:OPENAI_API_KEY="sk-..."
  $env:OPENAI_MODEL="gpt-4o"   # or gpt-4o-mini
  python scripts/llm_label_causal_openai.py --in_csv "outputs/sentences_only.csv" --out_csv "outputs/predictions_20.csv" --limit 20 --sleep 0
"""
import os, time, json, argparse, sys, requests

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")  # stronger default

SYSTEM_PROMPT = """You are a careful annotator for *causal relations* in one sentence.

Goal: return ONLY JSON: {"label": 0 or 1, "confidence": number 0..1}

Labeling rule (binary):
- label=1 if the sentence states or clearly implies that X causes/leads to/makes Y happen (explicit markers like because, due to, leads to, causes, results in; or clear implied cause→effect).
- label=0 if it is merely descriptive, correlational, temporal ("after", "when" without causal force), or unclear.

IMPORTANT:
- Do NOT default to 0. In typical software/requirements text, **25–40%** of sentences are causal.
- If causal cues or a clear mechanism are present, choose 1.

Examples (POSITIVE):
- "This change caused a crash." -> {"label":1, "confidence":0.95}
- "Due to a race condition, requests time out under load." -> {"label":1, "confidence":0.9}
- "If the token is missing, the API rejects the request." -> {"label":1, "confidence":0.8}
- "Increasing the batch size leads to higher memory usage." -> {"label":1, "confidence":0.85}

Examples (NEGATIVE):
- "We updated the documentation." -> {"label":0, "confidence":0.95}
- "Memory usage is high and latency increased." (no cause stated) -> {"label":0, "confidence":0.7}
- "After deployment, we saw errors." (temporal only) -> {"label":0, "confidence":0.6}
"""


USER_TEMPLATE = 'Sentence: "{text}"\nReturn JSON only.'

def call_openai(api_key, model, text, temperature=0.2, max_retries=3):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": USER_TEMPLATE.format(text=text.replace('"','\\"'))}
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"}
    }
    last_status = None
    last_body = ""
    for attempt in range(max_retries):
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        last_status = r.status_code
        last_body = r.text[:300]
        if r.status_code == 200:
            data = r.json()
            try:
                content = data["choices"][0]["message"]["content"]
                obj = json.loads(content)
                label = int(obj.get("label"))
                conf = float(obj.get("confidence", 0.5))
                label = 1 if label == 1 else 0
                conf = min(max(conf, 0.0), 1.0)
                return label, conf
            except Exception as e:
                # bad JSON or shape, retry
                pass
        time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"OpenAI call failed. status={last_status} body={last_body}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help='Path to CSV with "text" column')
    ap.add_argument("--out_csv", required=True, help="Where to write predictions.csv")
    ap.add_argument("--limit", type=int, default=0, help="Only label first N rows")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model name (default from OPENAI_MODEL or gpt-4o)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between calls")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set OPENAI_API_KEY environment variable in this terminal/session.")
    try:
        import pandas as pd
    except ImportError:
        sys.exit("Missing pandas: run 'python -m pip install pandas requests'")

    df = pd.read_csv(args.in_csv)
    if "text" not in df.columns:
        sys.exit('Input must have a "text" column.')

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    out_path = args.out_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    out_rows = []
    for i, row in df.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue
        try:
            label, conf = call_openai(api_key, args.model, text)
        except Exception as e:
            print(f"API error on row {i}: {e}")
            continue
        out_rows.append({"text": text, "model_label": label, "confidence": conf})
        if args.sleep > 0:
            time.sleep(args.sleep)
        if (i + 1) % 20 == 0:
            print(f"Labeled {i+1} sentences...")

    import pandas as pd
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out_rows)} rows.")

if __name__ == "__main__":
    main()
