# swarmtorch — benchmark and dev workflow
#
# Don't have make on Windows? Every target is a one-line python invocation.
# Open the recipe and copy-paste it; or install make via scoop / chocolatey /
# MSYS2. The scripts in scripts/ are self-contained and work without make.
#
# Targets:
#
#   make test                Run the full pytest suite.
#   make lint                Run ruff lint + format check.
#   make install-bench       Install the benchmark dependencies.
#
#   make bench-synthetic     Stage 4.1 — synthetic CEC-style suite.
#   make bench-training-quick  Stage 4.2 — training, synthetic data only.
#   make bench-training      Stage 4.2 — full training, downloads MNIST/CIFAR.
#   make bench-hpo           Stage 4.3 — HPO comparison.
#   make ablation-init       Stage 4.4 — init-strategy ablation.
#   make ablation-swarm      Stage 4.4 — swarm-size ablation.
#
#   make report DIR=results/synthetic
#                            Re-aggregate JSONs in DIR into report.md.
#
#   make clean-results       Wipe ./results/.
#   make clean               Wipe results, build artefacts, caches.
#
# All bench-* targets accept overrides on the CLI:
#   make bench-synthetic SEEDS="0 1 2" MAX_FE=2000 DIMS="10 50"

PYTHON ?= python
RESULTS ?= results
SEEDS ?= 0 1 2 3 4
MAX_FE ?= 5000
DIMS ?= 10 50 200
SWARM_SIZE ?= 30


# --- Dev -----------------------------------------------------------------

.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -q

.PHONY: lint
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

.PHONY: format
format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

.PHONY: install-bench
install-bench:
	$(PYTHON) -m pip install -e ".[benchmark]"


# --- Stage 4 sweeps ------------------------------------------------------

.PHONY: bench-synthetic
bench-synthetic:
	$(PYTHON) scripts/run_synthetic.py \
		--output-dir $(RESULTS)/synthetic \
		--seeds $(SEEDS) \
		--max-fe $(MAX_FE) \
		--swarm-size $(SWARM_SIZE) \
		--dims $(DIMS)

.PHONY: bench-training-quick
bench-training-quick:
	$(PYTHON) scripts/run_training.py \
		--output-dir $(RESULTS)/training_quick \
		--seeds $(SEEDS) \
		--max-fe $(MAX_FE) \
		--swarm-size $(SWARM_SIZE) \
		--quick

.PHONY: bench-training
bench-training:
	$(PYTHON) scripts/run_training.py \
		--output-dir $(RESULTS)/training \
		--seeds $(SEEDS) \
		--max-fe $(MAX_FE) \
		--swarm-size $(SWARM_SIZE)

.PHONY: bench-hpo
bench-hpo:
	$(PYTHON) scripts/run_hpo.py \
		--output-dir $(RESULTS)/hpo \
		--seeds $(SEEDS) \
		--n-trials 20

.PHONY: ablation-init
ablation-init:
	$(PYTHON) scripts/run_ablations.py \
		--ablation init_strategy \
		--output-dir $(RESULTS)/ablations/init \
		--seeds $(SEEDS) \
		--max-fe $(MAX_FE) \
		--algorithms PSO CA TLBO CMAES

.PHONY: ablation-swarm
ablation-swarm:
	$(PYTHON) scripts/run_ablations.py \
		--ablation swarm_size \
		--output-dir $(RESULTS)/ablations/swarm \
		--seeds $(SEEDS) \
		--max-fe $(MAX_FE) \
		--algorithms PSO CMAES

.PHONY: bench-gpu
bench-gpu:
	$(PYTHON) scripts/run_gpu_vs_numpy.py \
		--output-dir $(RESULTS)/gpu_vs_numpy \
		--seeds $(SEEDS) \
		--max-fe $(MAX_FE)

.PHONY: bench-gpu-quick
bench-gpu-quick:
	$(PYTHON) scripts/run_gpu_vs_numpy.py --quick \
		--output-dir $(RESULTS)/gpu_quick


# --- Reporting -----------------------------------------------------------

DIR ?= $(RESULTS)/synthetic

.PHONY: report
report:
	$(PYTHON) -m swarmtorch.summarize_results $(DIR)


# --- Cleanup -------------------------------------------------------------

.PHONY: clean-results
clean-results:
	rm -rf $(RESULTS)

.PHONY: clean
clean: clean-results
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
