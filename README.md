# Robustness of large language models in moral judgments

This repository has three experiments.

1. Revised the data generation code in a balanced way.

2. Prompt variations for evaluating the robustness of the LLMs in dilemma situations `run-prompt.py` . 

3. Prompt variations for evaluating the robustness of the LLMs in non-dilemma situations `run-prompt-const.py`.



## Description of the data and file structure

### Requirements

`pip install -r requirements.txt`

### Run 

[Note] or proprietary models (e.g., GPT4), you need API keys that you can specify on chatapi.py.

`python run-prompt.py --model llama3 --prompt CC-original --nb``scenarios 50000 --random_seed`` 123`

## Sharing/Access information

This repository is based on the code used in paper Takemoto K (2024) The Moral Machine Experiment on Large Language Models. R. Soc. Open Sci. 11, 231393 the moral machine experiment on large language models. 

Data was derived from the following sources:

https://github.com/kztakemoto/mmllm/tree/main

