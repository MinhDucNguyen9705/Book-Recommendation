#!/usr/bin/env python3
import requests, pandas as pd, json

def main():
    url = "http://98.82.15.127:1234/invocations"
    headers = {"Content-Type": "application/json"}

    df = pd.DataFrame({"user_id": [2142]})
    payload = {"dataframe_records": df.to_dict(orient="records")}

    resp = requests.post(url, headers=headers, data=json.dumps(payload))
    if resp.ok:
        print("✅ Prediction result:")
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    else:
        print("❌ Error:", resp.status_code)
        print(resp.text)

if __name__ == "__main__":
    main()
