import pandas as pd
import json
import os

def parse_md_table(content, section_title):
    try:
        section = content.split(section_title)[1].split("##")[0]
        lines = section.strip().split("\n")
        header = [h.strip() for h in lines[0].strip("| ").split("|")]
        data = []
        for line in lines[2:]:
            if "|" in line:
                row = [r.strip() for r in line.strip("| ").split("|")]
                data.append(row)
        df = pd.DataFrame(data, columns=header)
        for col in df.columns:
            try: df[col] = pd.to_numeric(df[col])
            except: pass
        return df
    except: return None

report_path = "swarmtorch/benchmarks/COMPREHENSIVE_EXPERIMENT_REPORT.md"
with open(report_path, "r") as f:
    content = f.read()

df_train = parse_md_table(content, "Detailed Results (Training Weight Optimization)")
df_hpo = parse_md_table(content, "Detailed Results (Hyperparameter Optimization)")

# Normalize names for HPO to match training names (stripping Search/HT)
df_hpo["Algorithm_Key"] = df_hpo["Algorithm"].str.replace("SearchHT", "").str.replace("Search", "").str.replace("HT", "")
df_train["Algorithm_Key"] = df_train["Algorithm"]

df_merged = pd.merge(df_train, df_hpo, left_on=["Algorithm_Key", "Category"], right_on=["Algorithm_Key", "Category"], suffixes=("_train", "_hpo"))

# Build Final Prompt Object
json_data = {
    "project": "SwarmTorch",
    "summary": {
        "total_algorithms_tested": 118,
        "categories": list(df_merged["Category"].unique()),
        "hpo_success_vs_random_baseline": "65.5%"
    },
    "baselines": {
        "training": {"Adam": 0.0065, "SGD": 0.0880},
        "hpo": {"RandomSearch": 0.9625}
    },
    "detailed_metrics": []
}

for _, row in df_merged.iterrows():
    json_data["detailed_metrics"].append({
        "name": row["Algorithm_Key"],
        "category": row["Category"],
        "metrics": {
            "training_loss_bce": round(float(row["Final Loss"]), 6),
            "hpo_validation_accuracy": round(float(row["Best Accuracy"]), 6)
        }
    })

print(json.dumps(json_data, indent=2))
