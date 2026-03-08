# SwarmTorch Benchmarks 📊

Detailed performance analysis and research results for the SwarmTorch library.

---

## 📈 Research & Benchmarking Results

Our library has undergone a massive-scale rigorous evaluation of **118 algorithms** to benchmark their performance across various deep learning tasks.

### 1. Model Training (Weight Optimization)
SwarmTorch enables the training of neural networks without gradients. Several metaheuristics exhibit convergence stability comparable to standard gradient-based methods.

<div align="center">
  <img src="training_convergence.png" width="800px">
  <p><i>Figure 1: Convergence history comparing Swarm Optimizers (PSO, HHO, etc.) against Adam and SGD.</i></p>
</div>

<div align="center">
  <img src="top_optimizers_training.png" width="800px">
  <p><i>Figure 2: The Top 25 most effective weight optimizers ranked by final loss.</i></p>
</div>

### 2. High-Density Distribution Analysis
We analyzed the reliability of each category. Swarm and Hybrid categories demonstrated the highest stability and lowest variance across multiple trials.

<div align="center">
  <img src="bench_category_dist.png" width="800px">
  <p><i>Figure 3: Statistical distribution of final loss across categories. Lower loss indicates superior optimization.</i></p>
</div>

### 3. Hyperparameter Optimization (HPO) Benchmarks
Our metaheuristic searchers are designed to replace Random Search with more intelligent exploration strategies. **94.9% of our algorithms outperformed Random Search.**

<div align="center">
  <img src="bench_success_rate.png" width="500px">
  <p><i>Figure 4: Success rate of metaheuristic searchers vs. the Random Search baseline.</i></p>
</div>

### 4. The "Generalist" Frontier
We identified "Generalist" algorithms that excel at both weight optimization and hyperparameter tuning.

<div align="center">
  <img src="bench_generalist_map.png" width="800px">
  <p><i>Figure 5: Scatter plot mapping Training Efficiency vs. HPO Accuracy. Elite generalist algorithms occupy the top-right quadrant.</i></p>
</div>

---

## 📄 Full Raw Data
For detailed per-algorithm metrics and statistical analysis, please refer to the [Comprehensive Research Report](COMPREHENSIVE_EXPERIMENT_REPORT.md).
