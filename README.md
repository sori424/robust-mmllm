# Robustness of large language models in moral judgments

This repository has three experiments.

1. Revised the data generation code in a balanced way.

2. Prompt variations for evaluating the robustness of the LLMs in dilemma situations.

3. Prompt variations for evaluating the robustness of the LLMs in non-dilemma situations.

Details are as belows.

We revised the data generation code in a balanced way and added prompt variations for evaluating the robustness of the LLMs. Moreover, we conducted further prompt variations for evaluating robustness of the large language models on value-laden tasks. We found limitations of LLMs in performing complex moral reasoning, particularly when required to simultaneously process multiple moral values (e.g., young (versus old) AND female (versus male) AND fit (versus large), etc.).

Logically, there could be different reasons for the inconsistency in model responses. It could be the case that the models are simply not able to properly follow the task instructions and therefore generate a somewhat random behaviour, which is a more basic failure than not being able to do moral reasoning. Or, it could be the case that the models can in principle follow instructions of the form used in our study, but they fail due to the difficulty of the dilemma and their inability to either learn about moral values, or weigh moral values against one another.

To tease apart these two situations, we additionally conduct experiments which uses a non-dilemma choice. The non-dilemma choice includes choosing between the death vs. sparing the characters where LLMs should reveal consistent behavior of always choosing sparing option as response.

## Description of the data and file structure

### Requirements

pip install -r requirements.txt

### Run 

[Note] or proprietary models (e.g., GPT4), you need API keys that you can specify on chatapi.py.

python run-prompt.py --model llama3 --prompt CC-original --nb``scenarios 50000 --random_seed`` 123

## Sharing/Access information

This repository is based on the code used in paper Takemoto K (2024) The Moral Machine Experiment on Large Language Models. R. Soc. Open Sci. 11, 231393 the moral machine experiment on large language models. 

Data was derived from the following sources:

https://github.com/kztakemoto/mmllm/tree/main

