

The code presented was used in a publication submitted to WSASCC 2023.

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