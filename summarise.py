import json
from pathlib import Path
from collections import defaultdict
from typing import Any



def summarise(input_path: Path | str, output_path: Path | str | None = None):
    input_path = Path(input_path)
    if output_path is not None:
        output_path = str(output_path)

    count = 0
    keys = ["progression", "total_cost", "num_steps"]
    acc_dict = defaultdict(list)

    for data_file in input_path.rglob("*.json"):
        data = json.load(open(data_file, "r"))
        count += 1
        for key in keys:
            if key in data:
                acc_dict[key].append(data[key])
            else:
                print(f"{key} not found in {data_file}")

    def avg(d: list) -> float | None:
        if len(d) == 0:
            return None
        else:
            return sum(d) / len(d)
    
    
    results: dict[str, Any] = {"count": count}
    for k in keys:
        results[f"{k}_total"] = sum(acc_dict[k])
        results[f"{k}_avg"] = avg(acc_dict[k])

    print(results)

    if output_path is not None:
        json.dump(results, open(output_path, "w"), indent=4)
    

if __name__ == '__main__':
    import fire
    fire.Fire(summarise)