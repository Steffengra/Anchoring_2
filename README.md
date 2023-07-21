

The code presented was used in the following publication ([preprint here](https://arxiv.org/abs/2304.12660)).

[1] S. Gracla, C. Bockelmann, A. Dekorsy,
"A Multi-Task Approach to Robust Deep Reinforcement Learning for Resource Allocation",
 International ITG 26th Workshop on Smart Antennas and 13th Conference on Systems, Communications, and Coding, Braunschweig, Germany, 27. February - 3. March 2023

The project structure is as follows:

```
/project/
├─ anchoring_2_imports/                     | python modules
├─ .gitignore                               | .gitignore
├─ config.py                                | contains configurable parameters
├─ README.md                                | this file
├─ requirements.txt                         | project dependencies
├─ runner.py                                | orchestrates training & testing EWC
├─ runner_GEM.py                            | orchestrates training & testing GEM
├─ test_anchoring_critical_allocation.py    | wrapper for testing different configurations of EWC
├─ test_gem.py                              | wrapper for training & testing different configurations of GEM
```