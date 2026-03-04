# ToolTree: Efficient LLM Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning

<p align="center">
  <a href="https://openreview.net/forum?id=Ef5O9gNNLE"><img src="https://img.shields.io/badge/ICLR-2026-blue.svg" alt="ICLR 2026"></a>
  <a href="https://openreview.net/pdf?id=Ef5O9gNNLE"><img src="https://img.shields.io/badge/Paper-PDF-red.svg" alt="Paper PDF"></a>
  <a href="https://github.com/SYang2000/ICLR_2026_ToolTree"><img src="https://img.shields.io/badge/GitHub-Code-black.svg" alt="GitHub"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

This is the **official implementation** for the paper:

> **ToolTree: Efficient LLM Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning**
>
> [Shuo Yang](https://github.com/SYang2000), Caren Han, Yihao Ding, Shuhe Wang, Eduard Hovy
>
> *Published as a conference paper at ICLR 2026*

---

## Overview

Large Language Model (LLM) agents are increasingly applied to complex, multi-step tasks that require interaction with diverse external tools across various domains. However, current LLM agent tool planning methods typically rely on greedy, reactive tool selection strategies that lack foresight and fail to account for inter-tool dependencies.

**ToolTree** is a novel Monte Carlo tree search-inspired planning paradigm for tool planning. It explores possible tool usage trajectories using a **dual-stage LLM evaluation** and **bidirectional pruning** mechanism that enables the agent to make informed, adaptive decisions over extended tool-use sequences while pruning less promising branches before and after the tool execution.

<p align="center">
  <img src="assets/comparison.png" width="100%">
  <br>
  <em>Figure 1: Comparison of ToolTree with greedy search and search-based tool planning. ToolTree chooses the optimal tool trajectory and answers correctly with bidirectional pruning.</em>
</p>

## Architecture

<p align="center">
  <img src="assets/architecture.png" width="100%">
  <br>
  <em>Figure 2: Architecture overview of ToolTree. An input query is processed sequentially via iterative dual evaluation-guided Monte Carlo Tree Search, including selection, pre-evaluation, expansion, execution, post-evaluation and backward-propagation.</em>
</p>

### Key Components

- **Pre-Evaluation**: A fast predictive signal that estimates the utility of a tool *before* execution, filtering schema- or slot-incompatible calls before expansion.
- **Post-Evaluation**: Assesses the actual contribution of a tool *after* execution based on observed outcomes, pruning unproductive branches using real feedback.
- **Bidirectional Pruning**: Combines pre- and post-evaluation to eliminate unpromising branches, concentrating computational budget on promising tool chains.
- **Answer Predictor**: Incorporates the tool trajectories with the highest reward found by the MCTS to produce the final prediction.

## Results

ToolTree achieves state-of-the-art performance across 4 benchmarks spanning both closed-set and open-set tool planning scenarios, with an average gain of ~10% over existing methods.

### Closed-Set Tool Planning (GTA & m&m)

| Method | GTA (GPT-4o) |  | m&m (GPT-4o) |  |
|:---|:---:|:---:|:---:|:---:|
| | Tool F1 | Avg | Plan F1 | Avg |
| Zero-shot | - | - | - | 80.51 |
| ReAct | - | - | - | - |
| CoT | - | - | - | - |
| ToT | - | - | - | - |
| LATS | - | - | - | - |
| **ToolTree (Ours)** | - | **66.95** | - | **88.61** |

### Open-Set Tool Planning (ToolBench & RestBench)

| Method | ToolBench (GPT-4o) |  | RestBench-TMDB (GPT-4o) |  |
|:---|:---:|:---:|:---:|:---:|
| | Pass Rate | AVG | Pass Rate | AVG |
| DFSDT | - | - | - | - |
| LATS | - | - | - | - |
| **ToolTree (Ours)** | **69.04** | **69.04** | - | **74.50** |

### Efficiency Analysis

<p align="center">
  <img src="assets/efficiency.png" width="100%">
  <br>
  <em>Figure 3: Progressive efficiency analysis across step limits. ToolTree achieves the highest efficiency (performance gain per second) compared with all baselines.</em>
</p>

### Model Scaling Analysis

<p align="center">
  <img src="assets/model_scaling.png" width="80%">
  <br>
  <em>Figure 4: Performance with respect to backbone model scale on GTA and ToolBench.</em>
</p>

### Ablation: Effect of Bidirectional Pruning

<p align="center">
  <img src="assets/ablation_pruning.png" width="80%">
  <br>
  <em>Figure 5: Ablation study on pruning strategies. Bidirectional pruning (both pre- and post-pruning) achieves the fewest rollouts and nodes, demonstrating the highest efficiency.</em>
</p>

### Case Study

<p align="center">
  <img src="assets/case_study.png" width="90%">
  <br>
  <em>Figure 6: A sample case of ToolTree on GTA. ToolTree progressively finds better tool trajectories guided by both pre-evaluation and post-evaluation rewards.</em>
</p>

## Installation

```bash
git clone https://github.com/SYang2000/ICLR_2026_ToolTree.git
cd ICLR_2026_ToolTree
pip install -r requirements.txt
```

## Quick Start

```bash
# Run on GTA benchmark
bash scripts/run_gta.sh

# Run on ToolBench benchmark
bash scripts/run_toolbench.sh

# Run on RestBench benchmark
bash scripts/run_restbench.sh

# Run on m&m benchmark
bash scripts/run_mm.sh
```

## Project Structure

```
ICLR_2026_ToolTree/
├── assets/                  # Figures and images
├── configs/                 # Configuration files
│   ├── default.yaml
│   ├── gta.yaml
│   └── toolbench.yaml
├── data/                    # Datasets (download separately)
├── scripts/                 # Experiment launch scripts
│   ├── run_gta.sh
│   ├── run_toolbench.sh
│   ├── run_restbench.sh
│   └── run_mm.sh
├── src/
│   ├── mcts/                # Monte Carlo Tree Search core
│   │   ├── tree_search.py   # Main MCTS algorithm
│   │   ├── node.py          # Tree node definition
│   │   └── pruning.py       # Bidirectional pruning
│   ├── agents/              # LLM agent implementations
│   │   ├── tooltree_agent.py
│   │   └── base_agent.py
│   ├── tools/               # Tool management
│   │   ├── tool_manager.py
│   │   └── tool_registry.py
│   └── evaluation/          # Evaluation metrics
│       ├── metrics.py
│       └── benchmarks.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{yang2026tooltree,
  title={ToolTree: Efficient LLM Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning},
  author={Yang, Shuo and Han, Caren and Ding, Yihao and Wang, Shuhe and Hovy, Eduard},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://openreview.net/forum?id=Ef5O9gNNLE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We thank the reviewers for their valuable feedback. This work was supported in part by computational resources from [your institution].
