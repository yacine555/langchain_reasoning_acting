# lanngchain_reasoning_acting

## Description

RAG setup with langchain

based on ReAct paper [link](https://arxiv.org/abs/2210.03629)

### Prerequisites and Dependencies

Before you begin, ensure you have the following installed:
- Python 3.10.10 or later. Note that this was only tested on 3.10.10
- [Pipenv](https://pipenv.pypa.io/en/latest/) 


Here are the PIP modules used:

- [**python-dotenv (1.0.0)**](https://pypi.org/project/python-dotenv/1.0.0/): Reads key-value pairs from a `.env` file and 
- [**tiktoken**](https://pypi.org/project/tiktoken/): tiktoken is a fast BPE (Byte pair encoding) tokeniser for use with OpenAI's models.
- [**Ollama**](https://ollama.ai/) for local LLMs

### Installation


Recommend using Install pipenv or other vitual environment tool. 

```bash
pipenv install
pipenv shell
pipenv --version
python --version
```


Clone the repository and install the required packages:

```bash
git clone https://github.com/yacine555/langchain_reasoning_acting.git
cd langchain_reasoning_acting
pipenv install -r requirements.txt
```

pipenv run pip install -r requirements.txt

to check the Lanchaing package:

```
pip list | grep  'langchain'
pipenv graph | grep langchain
```

```bash
pipenv run python main.py
pipenv run python ingestion.py
```

```bash
python main.py
python ingestion.py
```



### Download Web page resources

```
wget -mpEk https://python.langchain.com/docs/get_started
```