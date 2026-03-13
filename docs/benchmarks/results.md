# Benchmark Results

Summary of benchmark results for all 120 SwarmTorch algorithms.

## Model Training Results

### Top 10 Optimizers (Lower Loss = Better)

| Rank | Algorithm | Final Loss | Category |
|------|-----------|------------|----------|
| 1 | CA | 0.1317 | Evolutionary |
| 2 | HHO | 0.1668 | Swarm |
| 3 | CEM | 0.1995 | Evolutionary |
| 4 | SSA | 0.2201 | Swarm |
| 5 | GWO | 0.2462 | Swarm |
| 6 | WOA | 0.2510 | Swarm |
| 7 | DE | 0.2634 | Evolutionary |
| 8 | MFO | 0.2812 | Bio-Inspired |
| 9 | ALO | 0.2945 | Bio-Inspired |
| 10 | PSO | 0.3102 | Swarm |

### Category Performance

| Category | Best | Average Loss |
|----------|------|--------------|
| Evolutionary | CA (0.1317) | 0.28 |
| Swarm | HHO (0.1668) | 0.35 |
| Bio-Inspired | MFO (0.2812) | 0.38 |
| Physics | GSA (0.3521) | 0.41 |
| Human-Based | TLBO (0.3842) | 0.39 |
| Hybrid | SMA (0.3621) | 0.42 |

### Gradient Baseline Comparison

| Optimizer | Loss |
|-----------|------|
| Adam | 0.0330 |
| SGD | 0.0412 |
| Best SwarmTorch (CA) | 0.1317 |

<Note>
Gradient-based optimizers (Adam, SGD) still outperform metaheuristics on standard differentiable problems. SwarmTorch excels when gradients are unavailable or unreliable.
</Note>

---

## HPO Searcher Results

### Top 10 HPO Searchers (Higher Accuracy = Better)

| Rank | Searcher | Validation Accuracy | Category |
|------|----------|---------------------|----------|
| 1 | SASearch | 98.5% | Physics |
| 2 | DVBASearch | 98.5% | Swarm |
| 3 | PBILSearch | 98.0% | Evolutionary |
| 4 | CASearch | 97.9% | Evolutionary |
| 5 | HHOSearch | 97.5% | Swarm |
| 6 | SSASearch | 97.2% | Swarm |
| 7 | GWOSearch | 96.8% | Swarm |
| 8 | DE | 96.5% | Evolutionary |
| 9 | CEMSearch | 96.1% | Evolutionary |
| 10 | WOASearch | 95.8% | Swarm |

### Success Rate (% achieving >90% accuracy)

| Category | Success Rate |
|----------|--------------|
| Physics | 100% |
| Swarm | 72% |
| Evolutionary | 88% |
| Bio-Inspired | 60% |
| Human-Based | 50% |
| Hybrid | 60% |

### vs Random Search Baseline

- **65.5%** of searchers outperform Random Search
- **34.5%** perform similar to Random Search

---

## Key Findings

1. **Evolutionary algorithms excel** at model training (CA, CEM, DE)
2. **Swarm intelligence** is strong for HPO (HHO, SSA, GWO)
3. **Physics-based SA** performs surprisingly well for HPO
4. **Hybrid approaches** show promise but need tuning

## Visualizations

See the `benchmarks/` directory for:
- Convergence curves
- Category comparison charts
- Success rate visualizations

![Training Convergence](/benchmarks/training_convergence.png)
![HPO Comparison](/benchmarks/hpo_comparison.png)
