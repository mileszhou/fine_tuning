import random
import re, time, json
from pathlib import Path

s = {
    "accuracy": 100,
    "completeness": 99,
    "savety": 98,
    "winner": 1,
    "confidence": 0.3
}

# with open("./_outputs/test_json.json", "w", encoding="utf-8") as f:
# json.dumps(s)
print(json.dumps(s))
# json.dump(s, f)
# f.write("\n")  # Add newline for JSONL format

s["accuracy"] = 90
s["completeness"] = 89
print(json.dumps(s))
    # f.write("\n")  # Add newline for JSONL format

    # f.close()

print(json.loads(json.dumps(s)))

# wait a few seconds to ensure file is written
# time.sleep(2)

# with open("./_outputs/test_json.json", "r", encoding="utf-8") as f:
#     for line in f:
#         record = json.loads(line)
#         print(f"Record: {record}")
        