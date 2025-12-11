"""
Setup configuration for Medea package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="medea-ai",
    version="1.0.0",
    author="Medea Team",
    author_email="",
    description="Multi-Agent Research Planning System for Single-Cell Analysis and Therapeutic Discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mims-harvard/Medea-1.0",
    packages=find_packages(exclude=["evaluation", "evaluation.*", "results", "results.*", "plots", "plots.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Core dependencies
        "agentlite-llm>=0.1.0",
        "pandas<2.0.0",
        "numpy>=1.24.0",
        
        # LLM and AI
        "openai>=1.0.0",
        "anthropic>=0.20.0",
        
        # Google AI/Gemini
        "google-generativeai>=0.8.4",
        "google-ai-generativelanguage>=0.6.15",
        "google-api-core>=2.24.2",
        "google-api-python-client>=2.166.0",
        "google-auth>=2.38.0",
        "google-auth-httplib2>=0.2.0",
        "googleapis-common-protos>=1.69.2",
        "grpcio>=1.71.0",
        "grpcio-status>=1.71.0",
        
        # NLP and embeddings
        "sentence-transformers>=2.2.0",
        "FlagEmbedding>=1.2.0",
        "transformers>=4.35.0",
        "tokenizers>=0.13.0",
        "torch>=2.0.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
        "spacy>=3.5.0",
        "spacy-legacy>=3.0.0",
        "spacy-loggers>=1.0.0",
        "nltk>=3.8.0",
        "keybert>=0.8.0",
        
        # Machine Learning
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "peft>=0.4.0",
        "umap-learn>=0.5.0",
        "ollama>=0.4.7",
        
        # Data processing
        "datasets>=2.0.0",
        "pyarrow>=10.0.0",
        "h5py>=3.0.0",
        "openpyxl>=3.0.0",
        "dill>=0.3.0",
        
        # Bioinformatics
        "biothings-client>=0.3.0",
        "mygene>=3.2.0",
        "gseapy>=1.0.0",
        "comut>=0.0.3",
        
        # Web and API
        "requests>=2.31.0",
        "requests-cache>=1.0.0",
        "beautifulsoup4>=4.12.0",
        "httpx>=0.24.0",
        "httpx-sse>=0.3.0",
        "aiohttp>=3.8.0",
        "lxml>=4.9.0",
        
        # Search and retrieval
        "duckduckgo-search>=3.0.0",
        "wikipedia>=1.4.0",
        "ir-datasets>=0.5.0",
        
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.0.0",
        "palettable>=3.3.0",
        
        # Monitoring and logging
        "wandb>=0.15.0",
        "rich>=13.0.0",
        "psutil>=5.9.0",
        
        # Utilities
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "click>=8.0.0",
        "typer>=0.9.0",
        "thefuzz>=0.20.0",
        "tqdm>=4.65.0",
        "retry-requests>=2.0.0",
        "tenacity>=8.0.0",
        "filelock>=3.12.0",
        "huggingface-hub>=0.16.0",
        "tiktoken>=0.4.0",
        "networkx>=3.0.0",
        "sympy>=1.12.0",
        "joblib>=1.3.0",
        "regex>=2023.0.0",
        "pyyaml>=6.0.0",
        "certifi>=2023.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medea-eval=medea.cli:main",
        ],
    },
    package_data={
        "medea": [
            "tool_space/*.json",
            "agents/*.py",
        ],
    },
    include_package_data=True,
)

