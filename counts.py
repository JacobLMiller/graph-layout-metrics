import json 
from collections import Counter
with open("results/normalized-stress-results.json", "r") as fdata:
    results = json.load(fdata)

orderCounts = Counter([",".join(d["order"]) for d in results.values()])

print(orderCounts)