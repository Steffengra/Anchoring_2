

The code presented was used in the following publication [(preprint here)](https://arxiv.org/abs/2304.12660).

[1] S. Gracla, C. Bockelmann, A. Dekorsy,
"A Multi-Task Approach to Robust Deep Reinforcement Learning for Resource Allocation",
International ITG 26th Workshop on Smart Antennas and 13th Conference on Systems, Communications, and Coding,
Braunschweig, Germany, 27. February - 3. March 2023

Email: {**gracla**, bockelmann, dekorsy}@ant.uni-bremen.de

The project structure is as follows:

```
/project/
|   .gitignore           | .gitignore
|   README.md            | this file
|   requirements.txt     | project dependencies
|   
+---data                 | intermediate generated data
+---models               | trained models
+---outputs              | generated results
+---references           | dev references
+---reports              | project reports
|   +---figures
+---src                  | python files
|   +---analysis         | generating results
|   |       test_anchoring_critical_allocation.py  | wrapper for testing different configurations of EWC
|   |       test_gem.py                            | wrapper for training & testing different configurations of GEM
|   |       
|   +---config           | configuration
|   +---data             | to generating data to learn from
|   +---models           | related to creating learned models
|   |       runner.py      | orchestrates training & testing EWC
|   |       runner_GEM.py  | orchestrates training & testing GEM
|   |       
|   +---plotting         | plotting results
|   +---tests            | automated tests
|   \---utils            | shared functions and code snippets
```