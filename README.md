<h1 align="center">ToolTree: Efficient LLM Agent Tool Planning via<br>Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning</h1>

<h3 align="center">Accepted at ICLR 2026</h3>

<p align="center">
  Shuo Yang<sup>1</sup> &nbsp;&nbsp; Caren Han<sup>1</sup> &nbsp;&nbsp; Yihao Ding<sup>2</sup> &nbsp;&nbsp; Shuhe Wang<sup>1</sup> &nbsp;&nbsp; Eduard Hovy<sup>1</sup>
</p>
<p align="center">
  <sup>1</sup>The University of Melbourne &nbsp;&nbsp; <sup>2</sup>The University of Western Australia
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=Ef5O9gNNLE"><img src="https://img.shields.io/badge/ICLR-2026-blue.svg" alt="ICLR 2026"></a>
  <a href="https://openreview.net/pdf?id=Ef5O9gNNLE"><img src="https://img.shields.io/badge/Paper-PDF-red.svg" alt="Paper PDF"></a>
  <a href="https://github.com/SYang2000/ICLR_2026_ToolTree"><img src="https://img.shields.io/badge/GitHub-Code-black.svg" alt="GitHub"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

---

## Overview

**ToolTree** is a novel Monte Carlo tree search-inspired planning paradigm for LLM agent tool planning. It explores possible tool usage trajectories using a **dual-stage LLM evaluation** and **bidirectional pruning** mechanism that enables the agent to make informed, adaptive decisions over extended tool-use sequences while pruning less promising branches before and after the tool execution.








<p align="center">
  <img src="assets/comparison.png" width="100%">
  <br>
  <em>Comparison of ToolTree with greedy search and search-based tool planning. ToolTree chooses the optimal tool trajectory and answers correctly with bidirectional pruning.</em>
</p>

## Architecture

<p align="center">
  <img src="assets/architecture.png" width="100%">
  <br>
  <em>Architecture overview of ToolTree. An input query is processed sequentially via iterative dual evaluation-guided Monte Carlo Tree Search, including selection, pre-evaluation, expansion, execution, post-evaluation and backward-propagation.</em>
</p>

### Key Components

- **Pre-Evaluation**: A fast predictive signal that estimates the utility of a tool *before* execution, filtering schema- or slot-incompatible calls before expansion.
- **Post-Evaluation**: Assesses the actual contribution of a tool *after* execution based on observed outcomes, pruning unproductive branches using real feedback.
- **Bidirectional Pruning**: Combines pre- and post-evaluation to eliminate unpromising branches, concentrating computational budget on promising tool chains.
- **Answer Predictor**: Incorporates the tool trajectories with the highest reward found by the MCTS to produce the final prediction.

## Results

ToolTree achieves state-of-the-art performance across 4 benchmarks spanning both closed-set and open-set tool planning scenarios, with an average gain of ~10% over existing methods.


### Efficiency Analysis

<p align="center">
  <img src="assets/efficiency.png" width="100%">
  <br>
  <em>Figure 3: Progressive efficiency analysis across step limits. ToolTree achieves the highest efficiency (performance gain per second) compared with all baselines.</em>
</p>



### Case Study



<p align="center">
  <img src="assets/case_study_medical.png" width="90%">
  <br>
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
title={ToolTree: Efficient {LLM} Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning},
author={Shuo Yang and Caren Han and Yihao Ding and Shuhe Wang and Eduard Hovy},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Ef5O9gNNLE}
}
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We thank the reviewers for their valuable feedback. This work was supported in part by computational resources from Spartan at the University of Melbourne.
