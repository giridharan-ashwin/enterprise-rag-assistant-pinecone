import json


with open("results/retrieval_cache.json", "r", encoding="utf-8") as file:
    data = json.load(file)

print("Type:", type(data).__name__)

if isinstance(data, dict):
    print("Number of keys:", len(data))
    print("First keys:", list(data.keys())[:10])

    first_key = next(iter(data))

    print("\nFirst case:", first_key)
    print("\nFirst value:")
    print(json.dumps(data[first_key], indent=2))