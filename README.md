# Medical Literature & Pandemic Trends Analyzer

A comprehensive Streamlit-based application for analyzing medical literature, extracting insights, and generating detailed research reports. This tool helps researchers, students, and healthcare professionals analyze academic papers, identify trends, and understand citation networks.

## Features

### Core Analysis Capabilities
- **Document Processing**: Support for PDF and TXT files with intelligent text extraction
- **Topic Modeling**: Advanced topic discovery using machine learning algorithms
- **Entity Extraction**: Automatic identification of medical terms, people, organizations, and locations
- **Citation Network Analysis**: Parse and analyze academic citations with visual networks
- **Temporal Trend Analysis**: Track medical terms and concepts over time
- **Sentiment Analysis**: Understand the overall tone and sentiment of research documents

### Advanced Features
- **Interactive Visualizations**: Word clouds, bar charts, timeline graphs, and network diagrams
- **Document Comparison**: Compare external papers with your corpus for similarity analysis
- **Comprehensive Report Generation**: Automated research insight reports with key findings
- **Data Export**: Download citation data and reports in CSV/Markdown formats
- **Real-time Processing**: Live analysis with progress indicators

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical-literature-analyzer.git
cd medical-literature-analyzer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

### Dependencies
The application requires the following Python packages:

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk>=3.8
spacy>=3.6.0
plotly>=5.15.0
wordcloud>=1.9.0
matplotlib>=3.7.0
textblob>=0.17.0
PyPDF2>=3.0.0
python-docx>=0.8.11
```

### Additional Setup
Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

Install spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Getting Started
1. Launch the application using `streamlit run main.py`
2. Open your browser to `http://localhost:8501`
3. Upload your medical literature files (PDF or TXT format)
4. Explore the different analysis tabs

### Navigation Tabs

#### Overview
- Document statistics and metrics
- Interactive word cloud visualization
- Quick insights into your corpus

#### Topics
- Advanced topic modeling with adjustable parameters
- Discover hidden themes in your literature
- Optimal topic number suggestions

#### Entities
- Named Entity Recognition (NER) for medical terms
- Automatic extraction of people, organizations, and locations
- Medical-specific entity detection

#### Citations
- Citation network analysis and visualization
- Most cited authors and papers
- Downloadable citation data
- Paper recommendation system
- Document similarity comparison

#### Trends
- Temporal pattern analysis
- Historical trend visualization
- Sentiment analysis across documents
- Keyword frequency tracking

#### Report
- Comprehensive research insight reports
- Automated document summarization
- Research gap identification
- Methodology extraction
- Interactive visualizations

### File Support
- **PDF Files**: Automatic text extraction from academic papers
- **TXT Files**: Plain text document processing
- **Multiple Files**: Batch processing of document collections

## Architecture

### Core Components
- **DocumentProcessor**: Handles file loading and text preprocessing
- **AdvancedAnalytics**: Performs NLP tasks and statistical analysis
- **ReportGenerator**: Creates comprehensive research reports
- **Visualization Engine**: Generates interactive charts and graphs

### Key Technologies
- **Streamlit**: Web application framework
- **NLTK & spaCy**: Natural language processing
- **Scikit-learn**: Machine learning algorithms
- **Plotly**: Interactive visualizations
- **NetworkX**: Graph analysis for citation networks

## Examples

### Analyzing COVID-19 Research
1. Upload a collection of COVID-19 research papers
2. Use topic modeling to identify research themes
3. Analyze citation networks to find influential papers
4. Generate trends report to track pandemic research evolution

### Literature Review Automation
1. Upload papers from your research domain
2. Extract key entities and concepts
3. Compare with external papers for similarity
4. Generate comprehensive literature review report


### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include unit tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

