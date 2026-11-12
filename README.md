# RAG_Eval

Python-based RAG evaluation framework using LangChain, OpenAI and DeepEval to measure retrieval quality, answer relevance, faithfulness and factual correctness through automated evaluations and experiment comparisons.

# UV Setup

uv --version

uv self update

uv init

uv venv --python 3.12

source .venv/bin/activate

uv add \
  "deepagents>=0.7.19" \
  "deepeval>=4.2.6" \
  "dotenv>=0.9.9" \
  "langchain>=1.4.2" \
  "langchain-openai>=1.6.6" \
  "langchain-text-splitters>=1.1.2" \
  "numpy>=2.5.3" \
  "pandas>=3.0.6" \
  "pymupdf>=1.28.2" \
  "streamlit>=1.64.0"

  uv sync

  uv lock

  