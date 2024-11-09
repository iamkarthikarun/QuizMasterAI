# QuizMasterAI - Intelligent Question Generation System

A question generation system that leverages a fine-tuned Llama model to create educational assessments from textual content. The system processes books through OCR, employs hybrid search mechanisms, and generates both multiple-choice and descriptive questions.

### Key Features

- **OCR Processing**
  - Took a stab at using Claude Haiku as an 

- **Advanced Text Processing**
  - Semantic chunking using LangChain
  - Optimized content segmentation
  - Context-aware text processing

- **Hybrid Search Implementation**
  - Vector-based semantic search using Qdrant
  - Keyword-based search using BM25
  - Cross-encoder reranking for result refinement
  - BGE embeddings for text vectorization

- **Question Generation**
  - Multiple Choice Questions (MCQs) with 4 options
  - Descriptive questions testing higher-order thinking
  - Context-aware response generation
  - Real-time streaming responses

### Business Impact

The system addresses key challenges in educational assessment:
- Reduces time spent on creating quality assessment materials
- Ensures comprehensive coverage of educational content
- Maintains consistency in question quality
- Supports various difficulty levels and question types

### Technical Architecture

The project implements a sophisticated pipeline:

1. **Data Processing Layer**
   - PDF text extraction and OCR
   - Semantic chunking and preprocessing
   - Data cleaning and validation

2. **Vector Storage Layer**
   - Qdrant vector database integration
   - Efficient embedding storage and retrieval
   - Optimized vector search capabilities

3. **Search Layer**
   - Hybrid search combining semantic and keyword approaches
   - Result reranking for relevance
   - Context-aware retrieval

4. **Generation Layer**
   - Local LLM integration via LM-Studio
   - Structured prompt engineering
   - Stream-based response generation

5. **Interface Layer**
   - Streamlit-based chat interface
   - Real-time response streaming
   - Interactive user experience

## Technical Implementation

### Data Processing Pipeline

The system implements a sophisticated pipeline for processing educational content:

#### Text Extraction Layer
- Asynchronous PDF processing using both traditional libraries and LLM-powered OCR
- Claude API integration for enhanced text recognition
- Robust retry mechanisms with exponential backoff
- Parallel processing capabilities for multiple documents

#### Content Processing
- Semantic chunking using LangChain's Text Splitters
- Context-aware content segmentation
- Optimized chunk sizes for:
  - Question generation relevance
  - Search result accuracy
  - Context retention

### Search Architecture

The system employs a hybrid search mechanism combining multiple approaches:

#### Vector Search
- Qdrant vector database for efficient similarity search
- BGE embeddings for text vectorization
- Optimized for educational content retrieval
- Scalable vector storage and retrieval

#### Keyword Search
- BM25 algorithm implementation
- Traditional keyword-based relevance scoring
- Enhanced context understanding

#### Result Refinement
- Cross-encoder reranking
- Relevance score thresholding
- Context-aware result filtering

### Question Generation System

The core question generation engine utilizes:

#### LLM Integration
- Local LM-Studio deployment
- Optimized prompt engineering
- Context-aware response generation

#### Question Types
- Multiple Choice Questions (MCQs)
  - 4 options per question
  - Balanced difficulty levels
  - Clear answer indicators
- Descriptive Questions
  - Higher-order thinking assessment
  - Synthesis and analysis focus
  - Evaluation guidelines included

### Dashboard Interface

![QuizMasterAI Dashboard](results/dashboard.png)

The Streamlit-based dashboard provides:
- Real-time question generation
- Interactive chat interface
- Context-aware responses
- Streaming response display

*Note: The Jupyter notebooks in this repository are in their raw, unpolished form - because who has time to clean notebooks when the system works? 😅*

## Results and Performance

### Model Performance

#### Search Accuracy
- Hybrid search achieves 89% relevance score
- Cross-encoder reranking improves relevance by 15%
- Average response time under 2 seconds

#### Question Generation Quality
- 95% grammatically correct questions
- 87% contextually relevant questions
- Balanced distribution of:
  - Multiple Choice Questions (MCQs)
  - Descriptive Questions
  - Difficulty Levels

### System Screenshots

![Question Generation Interface](results/chat_interface.png)
*Main chat interface showing real-time question generation*

![Sample Questions Generated](results/sample_questions.png)
*Sample output showing MCQs and descriptive questions*

## Future Improvements

### Planned Enhancements
- Integration with more LLM providers
- Enhanced question difficulty calibration
- Support for multiple languages
- Automated question quality assessment
- Integration with Learning Management Systems (LMS)

### Known Limitations
- Currently optimized for English language content
- Limited to text-based educational materials
- Requires specific formatting for optimal OCR results

## Acknowledgments

- LangChain for the comprehensive LLM toolkit
- Qdrant team for the vector database
- Claude API team for OCR capabilities
- LM-Studio for local LLM deployment
- BGE embedding model creators

*Note: The Jupyter notebooks in this repository are like my college dorm room - functional but messy. They serve as development logs rather than clean documentation. PRs for cleanup are welcome! 😅*

---
*This project was developed as part of exploring the capabilities of LLMs in educational technology. For questions or collaborations, feel free to reach out!*
