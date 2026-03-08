import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Ensure swarmtorch is findable if running from here
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Use a professional, clean style
plt.style.use("ggplot")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def parse_markdown_table(content, section_title):
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
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
        return df
    except:
        return None


# Load data from the report in the same folder
report_path = os.path.join(
    os.path.dirname(__file__), "COMPREHENSIVE_EXPERIMENT_REPORT.md"
)
with open(report_path, "r") as f:
    report_content = f.read()

df_train = parse_markdown_table(
    report_content, "Detailed Results (Training Weight Optimization)"
)
df_hpo = parse_markdown_table(
    report_content, "Detailed Results (Hyperparameter Optimization)"
)

# Professional Muted Palette
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
COLORS = {
    cat: PALETTE[i % len(PALETTE)]
    for i, cat in enumerate(df_train["Category"].unique())
}


def save_fig(name):
    plt.savefig(
        os.path.join(os.path.dirname(__file__), name), dpi=300, bbox_inches="tight"
    )
    plt.close()


# --- FIG 1: Category Reliability (Box Plot) ---
plt.figure(figsize=(10, 6))
categories = [c for c in df_train["Category"].unique() if c != "Gradient"]
data_to_plot = [
    df_train[df_train["Category"] == cat]["Final Loss"].values for cat in categories
]
labels = [c.replace("_", " ").title() for c in categories]

box = plt.boxplot(data_to_plot, patch_artist=True, tick_labels=labels)
for patch, cat in zip(box["boxes"], categories):
    patch.set_facecolor(COLORS.get(cat, "#555555"))
    patch.set_alpha(0.8)

plt.title("Benchmarking Category Reliability (Weight Optimization)")
plt.ylabel("Final Loss (BCE)")
plt.xticks(rotation=15)
save_fig("bench_category_dist.png")

# --- FIG 2: Generalist Map (Scatter) ---
# Normalize names for merging (e.g., 'PSOSearch' -> 'PSO')
df_hpo_norm = df_hpo.copy()
df_hpo_norm["Algorithm"] = (
    df_hpo_norm["Algorithm"]
    .str.replace("SearchHT", "")
    .str.replace("Search", "")
    .str.replace("HT", "")
)

# Ensure Training names also match (some might have 'Search' in them originally, but let's be safe)
df_train_norm = df_train.copy()
# df_train_norm["Algorithm"] remains as is for now as it's the base name

df_merged = pd.merge(df_train_norm, df_hpo_norm, on=["Algorithm", "Category"])

if df_merged.empty:
    print("Warning: df_merged is empty. Trying more aggressive name matching...")
    # Fallback: fuzzy matching or just stripping all common suffixes
    df_hpo_norm["Algorithm"] = df_hpo_norm["Algorithm"].apply(
        lambda x: x.replace("Search", "").replace("HT", "")
    )
    df_train_norm["Algorithm"] = df_train_norm["Algorithm"].apply(
        lambda x: x.replace("Search", "").replace("HT", "")
    )
    df_merged = pd.merge(df_train_norm, df_hpo_norm, on=["Algorithm", "Category"])

df_merged["Norm_Loss"] = 1.0 - (df_merged["Final Loss"] / df_merged["Final Loss"].max())
df_merged["Combined"] = (df_merged["Norm_Loss"] + df_merged["Best Accuracy"]) / 2

plt.figure(figsize=(10, 7))
for cat in categories:
    sub = df_merged[df_merged["Category"] == cat]
    plt.scatter(
        sub["Norm_Loss"],
        sub["Best Accuracy"],
        label=cat.replace("_", " ").title(),
        color=COLORS.get(cat),
        s=100,
        alpha=0.8,
        edgecolors="white",
        linewidth=0.5,
    )

top_gen = df_merged.sort_values(by="Combined", ascending=False).head(5)
for _, row in top_gen.iterrows():
    plt.annotate(
        row["Algorithm"],
        (row["Norm_Loss"], row["Best Accuracy"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        fontweight="medium",
    )

plt.title("Generalist Mapping: Training Robustness vs. HPO Performance")
plt.xlabel("Training Efficiency Score (Higher is Better)")
plt.ylabel("HPO Validation Accuracy")
plt.legend(loc="lower right", title="Inspiration")
save_fig("bench_generalist_map.png")

# --- FIG 3: Metaheuristic Success Rate ---
random_acc = df_hpo[df_hpo["Algorithm"] == "RandomSearchHT"]["Best Accuracy"].values
random_acc = random_acc[0] if len(random_acc) > 0 else 0.5
superior = len(df_hpo[df_hpo["Best Accuracy"] > random_acc])
total = len(df_hpo) - 1

plt.figure(figsize=(7, 7))
wedges, texts, autotexts = plt.pie(
    [superior, total - superior],
    labels=["Outperformed Random", "Underperformed Random"],
    autopct="%1.1f%%",
    startangle=140,
    colors=["#55A868", "#C44E52"],
    explode=(0.05, 0),
)
plt.setp(autotexts, size=10, weight="bold", color="white")
plt.title("Success Rate vs. Random Search Baseline (HPO)")
save_fig("bench_success_rate.png")

# --- FIG 4: Top 15 Overall Performers (Bar) ---
top_15 = df_hpo.sort_values(by="Best Accuracy", ascending=False).head(15)
plt.figure(figsize=(10, 6))
bars = plt.barh(
    top_15["Algorithm"],
    top_15["Best Accuracy"],
    color=[COLORS.get(c) for c in top_15["Category"]],
)
plt.gca().invert_yaxis()
plt.title("Top 15 High-Performance Metaheuristics (HPO)")
plt.xlabel("Validation Accuracy")
plt.xlim(0, 1.0)
save_fig("bench_top_15_hpo.png")

print("Upgraded production visuals generated in swarmtorch/benchmarks/")
