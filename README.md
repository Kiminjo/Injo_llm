# Injo_llm

This github is a repository that I personally wrote down to use LLM. 

I have set the function according to my convenience. 

Langchain is convenient, but it has a limitation that it is difficult to use due to its slow speed. That's why I built my own custom library.

<br>

## Setup
This repository uses poetry to manage the version and dependencies of the library and operates under pyenv virtual environments. 

And the Python version is operating in the 3.10 environment.

If you want to use the repository, please install poetry and pyenv first.

1. Set python version using pyenv 
```shell
pyenv local 3.10
```

2. Setting Up a Virtual Environment with Python 3.10
```shell
poetry env use python
```

3. Install libraries 
```shell
poetry install 
```

4. Activate virtual Env 
```shell
poetry shell
```

<Br>

## Open AI API key settings
To use LLM model in this repository, create a `.txt` file under the `api/` directory and write down your own OpenAI API.

<br>

## Function
This repository implements an OpenAI API-based chat generation function and retrieval function. 

I used Meta's `FAISS` for vector data storage.

## Future plans 

In this repository, I plan to keep the various prompts I use in the form of a library. 

It also plans to implement a knowledge graph-based data search function.