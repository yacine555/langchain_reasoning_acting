# langchain_reasoning_acting

## Description

RAG setup with langchain

based on ReAct paper [link](https://arxiv.org/abs/2210.03629)

### Prerequisites and Dependencies

Before you begin, ensure you have the following installed:
- Python 3.11.x or later. Note that this was only tested on 3.11.8
- [Pipenv](https://pipenv.pypa.io/en/latest/) 


```bash
python --version
pipenv --version
```

Here are the PIP modules used:

- [**python-dotenv**](https://pypi.org/project/python-dotenv/): Reads key-value pairs from a `.env` file and 
- [**tiktoken**](https://pypi.org/project/tiktoken/): tiktoken is a fast BPE (Byte pair encoding) tokeniser for use with OpenAI's models.
- [**Ollama**](https://ollama.ai/) for local LLMs
- see requirments.txt list


### Installation

Recommend using Install pipenv or other vitual environment tool. 

```bash
pipenv install
pipenv shell
```


Clone the repository and install the required packages:

```bash
git clone https://github.com/yacine555/langchain_reasoning_acting.git
cd langchain_reasoning_acting
pip install -r requirements.txt
```

outside of the virtual environment you can run:

```bash
pipenv run pip install -r requirements.txt
```

To check the Lanchain packages installed:

```
pip list | grep  'langchain'
pipenv graph | grep langchain
```

```bash 
pipenv run python main.py
pipenv run python ingestion.py
pipenv run python ./backend/agents.py
```

```bash
python main.py
python ingestion.py
python ingestion.py
python ./backend/agents.py
```

Run the the app streamlit
```bash
pipenv run streamlit run myApp.py
```

### Download Web page resources

```
wget -mpEk https://python.langchain.com/docs/get_started
```


### Debug Logging

The application is using Langsmith, Pezzo.ai and wandb to test the LLM application

#### Langsmith

[Go to LangSmith](https://smith.langchain.com)

#### Pezzo.ai

[Go to Pezzo](https://app.pezzo.ai/)

#### Wanbd

login to wandb

```
wandb login
```