# CAG Demonstrator Agent

This project implements a demonstrator agent that compares the Cache-Augmented Generation (CAG) Framework with traditional Retrieval-Augmented Generation (RAG) using various LLMs.

## Features

- Implements both CAG and RAG frameworks for comparison
- Supports multiple LLM providers (OpenAI, Anthropic, Google, Mistral, Groq)
- Measures and compares performance metrics
- Generates detailed comparison reports
- Uses efficient caching mechanisms for improved performance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ai-in-pm/Cache-Augmented-Generation-CAG.git
cd Cache-Augmented-Generation-CAG
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Copy `.env.example` to `.env`
- Add your API keys for the LLM providers you want to use:
  ```
  OPENAI_API_KEY=your_key_here
  ANTHROPIC_API_KEY=your_key_here
  MISTRAL_API_KEY=your_key_here
  GROQ_API_KEY=your_key_here
  GOOGLE_API_KEY=your_key_here
  ```

## Usage

Run the demonstrator:
```bash
python demonstrator.py
```

The demonstrator will:
1. Initialize both CAG and RAG frameworks
2. Run a series of comparison queries
3. Generate metrics and save results to the `Results` directory

## Project Structure

```
CAG/
├── cag_demo/                  # Main package directory
│   ├── __init__.py
│   ├── cag_framework.py      # CAG implementation
│   ├── rag_framework.py      # RAG implementation
│   ├── llm_interface.py      # LLM API interface
│   └── config.py             # Configuration settings
├── Data/                     # Data directory
│   ├── Preloaded_Contexts/   # CAG knowledge base
│   └── Retrieved_Documents/  # RAG document store
├── Results/                  # Comparison results
├── demonstrator.py           # Main demonstration script
├── requirements.txt          # Project dependencies
└── .env.example             # Environment variables template
```

## Framework Comparison

The demonstrator compares two approaches:

1. **Cache-Augmented Generation (CAG)**:
   - Preloads and caches knowledge
   - Eliminates real-time retrieval steps
   - Reduces latency and improves response times
   - Uses efficient memory management

2. **Retrieval-Augmented Generation (RAG)**:
   - Traditional document retrieval approach
   - Real-time document fetching
   - Standard context processing

## Results

Results are saved in JSON format in the `Results` directory with the following information:
- Timestamp
- LLM model used
- Framework configurations
- Query responses
- Performance metrics
- Time comparisons

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors and the open-source community
- Inspired by advances in LLM architectures and retrieval techniques
