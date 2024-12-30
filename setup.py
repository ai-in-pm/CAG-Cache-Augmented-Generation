from setuptools import setup, find_packages

setup(
    name="cag-demonstrator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "rank-bm25>=0.2.2",
        "bert-score>=0.3.13",
        "python-dotenv>=1.0.0",
        "openai>=1.3.0",
        "anthropic>=0.5.0",
        "mistralai>=0.0.7",
        "httpx>=0.24.0",
        "google-generativeai>=0.3.0"
    ],
)
