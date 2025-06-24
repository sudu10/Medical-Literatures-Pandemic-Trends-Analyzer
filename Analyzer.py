import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt_tab')
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import fitz
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import plotly.express as px
import plotly.graph_objects as go
import time
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
import base64
from io import BytesIO
from matplotlib.figure import Figure
import random

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("The spaCy model 'en_core_web_sm' is not installed. Please install it by running: python -m spacy download en_core_web_sm")
    st.stop()


def load_css():
    st.markdown("""
        <style>
        .stApp { 
            max-width: 1400px; 
            margin: 0 auto; 
            font-family: 'Arial', sans-serif;
        }
        .main-header { 
            padding: 2rem; 
            border-radius: 10px; 
            text-align: center; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
            margin-bottom: 2rem;
        }
        .metric-card { 
            padding: 1.5rem; 
            border-radius: 10px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            margin: 1rem 0; 
            border-left: 4px solid #3b82f6;
        }
        .tab-content { 
            padding: 1.5rem; 
            border-radius: 10px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            margin: 1rem 0; 
        }
        .chart-container { 
            padding: 1.5rem; 
            border-radius: 10px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            margin: 1rem 0; 
        }
        .stButton>button { 
            color: white; 
            border-radius: 5px; 
        }
        .stProgress .st-bo { 
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 4px 4px 0 0;
        }
        .stTabs [aria-selected="true"] {
            color: white;
        }
        .entity-box {
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 10px;
            border-left: 3px solid #3b82f6;
        }
        .wordcloud-container {
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .keyword-tag {
            background-color: #f3f4f6;
            display: inline-block;
            color: #1e40af;
            padding: 5px 10px;
            margin: 3px;
            border-radius: 15px;
            font-size: 0.9em;
        }
        .sentiment-positive {
            color: #059669;
            font-weight: bold;
        }
        .sentiment-negative {
            color: #dc2626;
            font-weight: bold;
        }
        .sentiment-neutral {
            color: #6b7280;
            font-weight: bold;
        }
        .stMetric {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            background-color: #f8f9fa;
        }
        .stRecommendation {
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .similarity-metric {
            font-size: 2.5rem !important;
            font-weight: bold;
            color: #1e40af;
        }
        </style>
    """, unsafe_allow_html=True)


def create_loading_spinner(message="Processing..."):
    with st.spinner(message):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
    st.success(f"{message} Complete!")


class DocumentProcessor:
    def load_document(self, file):
        if file.type == "application/pdf":
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return " ".join([page.get_text() for page in doc])
        elif file.type == "text/plain":
            return file.read().decode("utf-8")
        return ""

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and len(token.text) > 2]
        return ' '.join(tokens)

class AdvancedAnalytics:

    def __init__(self):
        pass  # Add an __init__ method if needed

    def extract_medical_entities(self, text):
        doc = nlp(text)
        entities = {"PERSON": [], "ORG": [], "GPE": [], "DATE": []}
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        return entities

    def perform_topic_modeling(self, texts, n_topics=5):
        if not texts or len(texts) < 1:
            st.warning("At least one document is required for topic modeling. Please upload a file.")
            return None
        
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except Exception as e:
            st.error(f"Text vectorization failed: {str(e)}")
            return None

        max_topics = min(n_topics, len(texts), tfidf_matrix.shape[1] - 1)
        if max_topics < n_topics:
            st.info(f"Adjusted number of topics to {max_topics} due to limited data.")
            n_topics = max_topics
        if n_topics < 1:
            st.warning("Not enough unique words for topic modeling.")
            return None

        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method='online',
            max_iter=10
        )
        try:
            lda_model.fit(tfidf_matrix)
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda_model.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
                topics.append(f"Topic {topic_idx + 1}: {' '.join(top_words)}")
            return topics
        except Exception as e:
            st.error(f"Topic modeling with LDA failed: {str(e)}")
            return None

    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        return "Positive" if blob.sentiment.polarity > 0 else "Negative" if blob.sentiment.polarity < 0 else "Neutral"

    def extract_keywords(self, texts, top_n=10):
        if not texts or all(not text for text in texts):
            return []
            
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray().sum(axis=0)
            keyword_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)[:top_n]
            return [kw for kw, score in keyword_scores]
        except Exception as e:
            st.warning(f"Keyword extraction failed: {str(e)}")
            return None

    def analyze_citation_network(self, raw_texts: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Analyzes citations in medical research papers with improved pattern recognition.
        Returns a dictionary of citation info with author, year, count, and reference snippet.
        
        Args:
            raw_texts: List of raw document texts (PDF/TXT content)
        
        Returns:
            Dictionary of parsed citations with metadata:
            {
                "1": {
                    "author": "Smith et al.",
                    "year": "2020",
                    "count": 5,
                    "reference": "Smith J, et al. Study on...",
                },
                ...
            }
        """
        citations = Counter()
        reference_dict = {}
        citation_info = {}

        # Improved in-text citation patterns (AMA/Vancouver styles)
        citation_patterns = [
            r'\[(\d{1,3}(?:[,-]\s*\d{1,3})*)\]',  # [1], [1-3], [1,3,5]
            r'\((\d{1,3}(?:[,-]\s*\d{1,3})*)\)',  # (1), (1-3)
            r'(?<!\w)(\d{1,3})(?!\w)',            # Standalone numbers (e.g., "as shown in 1,2")
            r'\[\s*(\d+)\s*\]',                   # [ 1 ] with spaces
            r'[\(\[]\s*(\d+)\s*[\)\]]',           # ( 1 ) or [ 1 ] with spaces
        ]

        for raw_text in raw_texts:
            # Find all citations in text
            for pattern in citation_patterns:
                for match in re.finditer(pattern, raw_text):
                    citation = match.group(1)
                    
                    # Process citation ranges/splits
                    if "-" in citation:
                        start, end = map(int, citation.split("-"))
                        for num in range(start, end + 1):
                            citations[str(num)] += 1
                    elif "," in citation:
                        for num in citation.split(","):
                            citations[num.strip()] += 1
                    else:
                        citations[citation] += 1

            # Enhanced reference section detection
            reference_section = re.search(
                r'(?:References|Bibliography|Works Cited|Literature Cited)[\s\n:]*\n([\s\S]+?)(?=\n\s*(?:[A-Z][a-z]+\s*\n|\Z))',
                raw_text, 
                re.IGNORECASE
            )
            
            if reference_section:
                references_text = reference_section.group(1)
                
                # Multiple reference patterns to handle different formats
                reference_patterns = [
                    r'(?:^|\n)\s*(\d+)\s*\.\s*([\s\S]*?)(?=\n\s*\d+\.|\Z)',  # 1. Author et al. (2020)...
                    r'\[\s*(\d+)\s*\]\s*([\s\S]*?)(?=\[\s*\d+\s*\]|\Z)',      # [1] Author et al. (2020)...
                    r'(?:^|\n)\s*(\d+)\s*\.\s*([A-Z][^\n]+?\(\d{4}\)[^\n]*)', # 1. Author (2020) Title...
                    r'\[\s*(\d+)\s*\]\s*([A-Z][^\n]+?\(\d{4}\)[^\n]*)',       # [1] Author (2020) Title...
                ]
                
                for pattern in reference_patterns:
                    for ref_num, ref_text in re.findall(pattern, references_text):
                        reference_dict[ref_num.strip()] = ref_text.strip()

        # Enhanced author/year extraction
        for num, ref_text in reference_dict.items():
            # Extract year first (multiple patterns)
            year_match = re.search(
                r'(?:\((\d{4}[a-z]?)\)|(\d{4})(?=[^\d/]*$)|[\s,](\d{4})[\s,\.])', 
                ref_text
            )
            year = "Unknown Year"
            if year_match:
                year = next((g for g in year_match.groups() if g), "Unknown Year")
                if not (1900 <= int(year[:4]) <= 2100):  # Validate year range
                    year = "Unknown Year"

            # Improved author extraction (multiple patterns)
            author_patterns = [
                r'^([A-Z][a-z]+(?:,\s*[A-Z][a-zA-Z]*\.?-?)+)',  # Lastname, Firstname M.
                r'^([A-Z][a-z]+\s+[A-Z][a-zA-Z]*)',             # Firstname Lastname
                r'^([A-Z][a-z]+\s+(?:van|de|der|le)\s+[A-Z]\w+)',  # Names with prefixes
                r'^([A-Z]\w+,\s*[A-Z][\w-]+)',               # Lastname, Firstname
                r'^((?:[A-Z]\.\s*)+[A-Z][a-z]+)',               # Initials format
                r'^(International\s+(?:Committee|Agency|Society)\s+[A-Z]\w+)',  # Orgs
                r'^([A-Z][a-z]+\s+[A-Z]\w+\s+on\s+[A-Z]\w+)',   # Committee-style
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',           # Multiple names
            ]
            
            authors = "Unknown Author"
            for pattern in author_patterns:
                author_match = re.search(pattern, ref_text)
                if author_match:
                    authors = max(author_match.groups(''), key=len).strip()
                    # Clean up author strings
                    authors = re.sub(r'\s+', ' ', authors)  # Remove extra spaces
                    authors = re.sub(r'(?<=[a-z])\.', '', authors)  # Remove middle dots
                    authors = authors.split(';')[0].split(', and')[0]  # Take first author group
                    break

            # Fallback: Try to extract author before year if standard patterns fail
            if authors == "Unknown Author" and year != "Unknown Year":
                before_year = ref_text.split(year)[0]
                if before_year:
                    words = before_year.strip().split()
                    if len(words) >= 2:
                        potential_author = " ".join(words[-2:])
                        if re.match(r'^[A-Z]', potential_author):
                            authors = potential_author

            # Filter out invalid citations
            if (authors == "Unknown Author" or 
                year == "Unknown Year" or
                "abstract" in ref_text.lower() or
                "url" in ref_text.lower() or
                len(ref_text) < 20):
                continue

            citation_info[num] = {
                "author": authors,
                "year": year,
                "count": citations[num],
                "reference": ref_text[:150] + "..." if len(ref_text) > 150 else ref_text
            }

        return citation_info


    def analyze_temporal_patterns(self, raw_texts, top_n_keywords=10):
        """
        Extracts medical terms linked to years and identifies keyword trends.
        Uses raw text before preprocessing.
        """
        keyword_year_freq = defaultdict(Counter)

        year_pattern = r'\b(19|20)\d{2}\b'
        years_found = []
        references_text = ""

        for raw_text in raw_texts:
            years_found.extend(re.findall(year_pattern, raw_text))

            match = re.search(r"(References|Bibliography)(.*)", raw_text, re.DOTALL | re.IGNORECASE)
            if match:
                references_text += match.group(2) + "\n"

        ref_years = re.findall(year_pattern, references_text)
        years_found.extend(ref_years)

        med_terms = set()
        med_patterns = [
            r'\b[A-Z][a-z]+(itis|osis|emia|pathy)\b',  
            r'\b(COVID-19|SARS-CoV-2|pandemic|epidemic|outbreak)\b',  
            r'\b(vaccine|vaccination|immunization|treatment|therapy)\b',  
        ]
        
        for raw_text in raw_texts:
            for pattern in med_patterns:
                med_terms.update(re.findall(pattern, raw_text, re.IGNORECASE))

            doc = nlp(raw_text)
            for chunk in doc.noun_chunks:
                if len(chunk.text) > 3 and not any(token.is_stop for token in chunk):
                    med_terms.add(chunk.text.lower())

        sentences = []
        for raw_text in raw_texts:
            sentences.extend(sent_tokenize(raw_text))

        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=100)
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            top_indices = tfidf_scores.argsort()[-top_n_keywords:][::-1]
            top_keywords = [feature_names[i] for i in top_indices]
            med_terms.update(top_keywords)
        except:
            pass  
        for raw_text in raw_texts:
            for sentence in sent_tokenize(raw_text):
                found_years = re.findall(year_pattern, sentence)
                
                if found_years:
                    for term in med_terms:
                        if term.lower() in sentence.lower():
                            for year in found_years:
                                keyword_year_freq[term][year] += 1

        keyword_total_freq = {k: sum(v.values()) for k, v in keyword_year_freq.items()}
        top_keywords = sorted(keyword_total_freq.items(), key=lambda x: x[1], reverse=True)[:top_n_keywords]
        
        return {k: v for k, v in top_keywords}, keyword_year_freq

class ReportGenerator:
    def __init__(self, analytics):
        self.analytics = analytics
        self.processor = DocumentProcessor()
    
    def generate_report(self, citation_info, raw_texts, all_texts):
        """Generate a comprehensive report summarizing both the most cited paper and its most similar document"""
        if not citation_info:
            return "No citation information available for report generation."
        
        # Find most cited paper
        valid_citations = [info for info in citation_info.values()]
        if not valid_citations:
            return "No valid citations found for report generation."
        
        most_cited = max(valid_citations, key=lambda x: x["count"])
        
        # Find most similar document to the most cited paper
        matching_doc_idx = self._find_similar_document(most_cited, raw_texts)
        
        # Extract relevant sections from both documents
        most_cited_sections = self._extract_sections(most_cited["reference"])
        similar_doc_sections = self._extract_sections(raw_texts[matching_doc_idx])
        
        # Generate comprehensive summaries for both documents
        most_cited_summary = self._generate_extended_summary(most_cited["reference"], sentences=5)
        similar_doc_summary = self._generate_extended_summary(raw_texts[matching_doc_idx], sentences=5)
        
        # Extract key findings and methodology from both documents
        key_findings = self._extract_key_findings(raw_texts[matching_doc_idx])
        methodology = self._extract_methodology(raw_texts[matching_doc_idx])
        
        # Compare documents and extract similarities and differences
        comparison = self._compare_documents(most_cited["reference"], raw_texts[matching_doc_idx])
        
        # Extract top frequencies from most cited and similar document
        most_freq_words = self._get_frequent_words(all_texts[matching_doc_idx])
        
        # Create a plot for word frequencies
        word_freq_fig = self._create_word_freq_chart(most_freq_words)
        
        # Create sentiment analysis visualization
        sentiment_analysis = self._analyze_document_sentiment(raw_texts[matching_doc_idx])
        sentiment_fig = self._create_sentiment_gauge(sentiment_analysis)
        
        # Extract entities
        entities = self.analytics.extract_medical_entities(raw_texts[matching_doc_idx])
        
        # Create entity network visualization
        entity_fig = self._create_entity_network(entities)
        
        # Identify potential research gaps
        research_gaps = self._identify_research_gaps(most_cited["reference"], raw_texts[matching_doc_idx])
        
        # Format the report content
        report = {
            "title": f"Research Summary Report: {most_cited['author']} ({most_cited['year']})",
            "cited_paper": most_cited,
            "similar_doc_title": self._extract_title(raw_texts[matching_doc_idx]),
            "most_cited_summary": most_cited_summary,
            "similar_doc_summary": similar_doc_summary,
            "key_findings": key_findings,
            "methodology": methodology,
            "comparison": comparison,
            "research_gaps": research_gaps,
            "freq_words": most_freq_words,
            "word_freq_fig": word_freq_fig,
            "sentiment": sentiment_analysis,
            "sentiment_fig": sentiment_fig,
            "entities": entities,
            "entity_fig": entity_fig
        }
        
        return report
    
    def _find_similar_document(self, most_cited, raw_texts):
        """Find the most similar document to the most cited paper"""
        most_cited_text = most_cited["reference"]
        matching_doc_idx = 0
        
        # First try direct matching
        for idx, text in enumerate(raw_texts):
            if most_cited_text in text:
                matching_doc_idx = idx
                break
        else:
            # If exact match not found, use similarity based on keywords
            most_cited_keywords = re.findall(r'\b\w{3,}\b', most_cited_text.lower())
            highest_match = 0
            
            for idx, text in enumerate(raw_texts):
                text_lower = text.lower()
                match_count = sum(1 for word in most_cited_keywords if word in text_lower)
                if match_count > highest_match:
                    highest_match = match_count
                    matching_doc_idx = idx
        
        return matching_doc_idx
    
    def _extract_title(self, text):
        """Extract the title from a document"""
        # Try to find title from first few lines
        lines = text.split('\n')
        for i in range(min(5, len(lines))):
            line = lines[i].strip()
            if len(line) > 20 and len(line) < 200 and line.istitle() or line[0].isupper():
                return line
        
        # Fallback to first sentence
        sentences = sent_tokenize(text)
        if sentences:
            return sentences[0][:100] + "..." if len(sentences[0]) > 100 else sentences[0]
        
        return "Untitled Document"
    
    def _extract_sections(self, text):
        """Extract main sections from document text"""
        common_sections = [
            r'Abstract', r'Introduction', r'Methods', r'Methodology', 
            r'Results', r'Discussion', r'Conclusion', r'References'
        ]
        
        sections = {}
        current_section = "Unknown"
        section_text = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a section header
            is_header = False
            for section in common_sections:
                if re.match(rf'^{section}\b', line, re.IGNORECASE):
                    if section_text:
                        sections[current_section] = ' '.join(section_text)
                    
                    current_section = section
                    section_text = []
                    is_header = True
                    break
            
            if not is_header:
                section_text.append(line)
        
        # Add the last section
        if section_text:
            sections[current_section] = ' '.join(section_text)
        
        return sections
    
    def _generate_extended_summary(self, text, sentences=5):
        """Generate a comprehensive summary from the text"""
        try:
            blob = TextBlob(text)
            blob_sentences = blob.sentences
            
            if len(blob_sentences) <= sentences:
                return text
            
            # Extract first few sentences from abstract or introduction if available
            sections = self._extract_sections(text)
            
            summary = []
            # Try to add abstract
            if 'Abstract' in sections:
                abstract_blob = TextBlob(sections['Abstract'])
                abstract_sentences = abstract_blob.sentences[:3]
                summary.extend([str(s) for s in abstract_sentences])
            
            # Add introduction sentences
            if 'Introduction' in sections:
                intro_blob = TextBlob(sections['Introduction'])
                intro_sentences = intro_blob.sentences[:2]
                summary.extend([str(s) for s in intro_sentences])
            
            # Add conclusions or results
            if 'Conclusion' in sections:
                conclusion_blob = TextBlob(sections['Conclusion'])
                conclusion_sentences = conclusion_blob.sentences[:2]
                summary.extend([str(s) for s in conclusion_sentences])
            elif 'Results' in sections:
                results_blob = TextBlob(sections['Results'])
                results_sentences = results_blob.sentences[:2]
                summary.extend([str(s) for s in results_sentences])
            
            # If we still don't have enough sentences, add from the beginning of the document
            if len(summary) < sentences:
                remaining = sentences - len(summary)
                summary.extend([str(s) for s in blob_sentences[:remaining]])
            
            return " ".join(summary)
        except:
            # Fallback to first few sentences if sections extraction fails
            sentences_list = sent_tokenize(text)
            return " ".join(sentences_list[:sentences])
    
    def _extract_key_findings(self, text):
        """Extract key findings from the text"""
        findings = []
        sections = self._extract_sections(text)
        
        # Look for findings in results or discussion sections
        target_sections = ['Results', 'Discussion', 'Conclusion']
        combined_text = ""
        
        for section in target_sections:
            if section in sections:
                combined_text += sections[section] + " "
        
        # Look for sentences with indicator phrases
        indicator_phrases = [
            r'significant(\w+)?', r'find(ing|ings)?', r'show(s|ed|n)?',
            r'reveal(s|ed)?', r'demonstrate(d|s)?', r'highlight(s|ed)?',
            r'conclude(s|d)?', r'suggest(s|ed)?', r'indicate(s|d)?'
        ]
        
        sentences = sent_tokenize(combined_text)
        for sentence in sentences:
            for phrase in indicator_phrases:
                if re.search(phrase, sentence, re.IGNORECASE):
                    findings.append(sentence)
                    break
        
        # Limit to top 5 findings
        if not findings:
            # If no findings detected with indicators, take first sentences from Results
            if 'Results' in sections:
                findings = sent_tokenize(sections['Results'])[:3]
        
        return findings[:5]
    
    def _extract_methodology(self, text):
        """Extract methodology information from the text"""
        sections = self._extract_sections(text)
        
        # Check for Methods or Methodology section
        for section_name in ['Methods', 'Methodology', 'Materials and Methods']:
            if section_name in sections:
                method_text = sections[section_name]
                sentences = sent_tokenize(method_text)
                return sentences[:5]  # Return first 5 sentences of methodology
        
        # If no dedicated method section, look for methodology descriptions
        method_indicators = [
            r'data (was|were) collected', r'stud(y|ies) (was|were) conducted',
            r'participant(s)?', r'sample(s|d)?', r'analy(s|z)ed',
            r'method(s|ology)?', r'approach', r'technique', r'procedure'
        ]
        
        method_sentences = []
        text_sentences = sent_tokenize(text)
        
        for sentence in text_sentences:
            for indicator in method_indicators:
                if re.search(indicator, sentence, re.IGNORECASE):
                    method_sentences.append(sentence)
                    break
        
        return method_sentences[:5]  # Limit to 5 sentences
    
    def _compare_documents(self, doc1, doc2):
        """Compare two documents and identify similarities and differences"""
        # Get key sentences from both documents
        doc1_sentences = sent_tokenize(doc1)[:10]  # First 10 sentences
        doc2_sentences = sent_tokenize(doc2)[:20]  # First 20 sentences
        
        # Clean and tokenize sentences
        doc1_words = set(" ".join(doc1_sentences).lower().split())
        doc2_words = set(" ".join(doc2_sentences).lower().split())
        
        # Find common and unique keywords
        common_words = doc1_words.intersection(doc2_words)
        doc1_unique = doc1_words - doc2_words
        doc2_unique = doc2_words - doc1_words
        
        # Filter out stopwords
        stopwords = set(['the', 'and', 'or', 'a', 'an', 'in', 'on', 'at', 'of', 'to', 'for', 'with', 'by', 'as', 'is', 'are', 'was', 'were'])
        common_words = common_words - stopwords
        doc1_unique = doc1_unique - stopwords
        doc2_unique = doc2_unique - stopwords
        
        # Get top terms
        common_top = sorted(list(common_words))[:10]
        doc1_top = sorted(list(doc1_unique))[:5]
        doc2_top = sorted(list(doc2_unique))[:5]
        
        return {
            "common_themes": common_top,
            "cited_paper_unique": doc1_top,
            "similar_doc_unique": doc2_top
        }
    
    def _identify_research_gaps(self, cited_text, similar_text):
        """Identify potential research gaps based on document comparison"""
        # Look for phrases indicating limitations or future work
        gap_indicators = [
            r'future (work|research|studies|directions)', 
            r'limitation(s)?', 
            r'further (investigation|research|study|work)',
            r'recommend(ation|ations|ed|s)?',
            r'need(s|ed)? (for|to)',
            r'gap(s)?'
        ]
        
        # Combine both texts with emphasis on the end sections
        sections_cited = self._extract_sections(cited_text)
        sections_similar = self._extract_sections(similar_text)
        
        combined_text = ""
        
        # Focus on discussion and conclusion sections
        target_sections = ['Discussion', 'Conclusion', 'Future Work', 'Limitations']
        for section in target_sections:
            if section in sections_cited:
                combined_text += sections_cited[section] + " "
            if section in sections_similar:
                combined_text += sections_similar[section] + " "
        
        # If no specific sections found, use whole text
        if not combined_text:
            combined_text = cited_text + " " + similar_text
        
        # Find sentences containing gap indicators
        gap_sentences = []
        sentences = sent_tokenize(combined_text)
        
        for sentence in sentences:
            for indicator in gap_indicators:
                if re.search(indicator, sentence, re.IGNORECASE):
                    gap_sentences.append(sentence)
                    break
        
        return gap_sentences[:5]  # Limit to 5 gap statements
    
    def _get_frequent_words(self, text, top_n=10):
        """Extract most frequent words from text"""
        vectorizer = CountVectorizer(stop_words='english', max_features=top_n)
        X = vectorizer.fit_transform([text])
        words = vectorizer.get_feature_names_out()
        frequencies = X.toarray().sum(axis=0)
        return dict(zip(words, frequencies))
    
    def _create_word_freq_chart(self, word_freq_dict):
        """Create a horizontal bar chart for word frequencies"""
        words = list(word_freq_dict.keys())
        freqs = list(word_freq_dict.values())
        
        # Sort by frequency (descending)
        sorted_indices = sorted(range(len(freqs)), key=lambda i: freqs[i], reverse=True)
        words = [words[i] for i in sorted_indices]
        freqs = [freqs[i] for i in sorted_indices]
        
        fig = go.Figure(go.Bar(
            x=freqs,
            y=words,
            orientation='h',
            marker=dict(
                color='rgba(59, 130, 246, 0.8)',
                line=dict(color='rgba(59, 130, 246, 1.0)', width=1)
            )
        ))
        fig.update_layout(
            title='Most Frequent Terms',
            xaxis_title='Frequency',
            yaxis_title='Terms',
            height=400
        )
        return fig
    
    def _analyze_document_sentiment(self, text):
        """Analyze sentiment of document sections"""
        # Split text into sections (approx. paragraphs)
        sections = re.split(r'\n\s*\n', text)[:5]  # Use up to 5 sections
        
        results = []
        for section in sections:
            if len(section) < 50:  # Skip very short sections
                continue
            blob = TextBlob(section)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            sentiment = "Neutral"
            if polarity > 0.2:
                sentiment = "Positive"
            elif polarity < -0.2:
                sentiment = "Negative"
                
            results.append({
                "text": section[:100] + "...",
                "polarity": polarity,
                "subjectivity": subjectivity,
                "sentiment": sentiment
            })
        
        # Calculate overall sentiment
        if results:
            avg_polarity = sum(r["polarity"] for r in results) / len(results)
            overall = "Neutral"
            if avg_polarity > 0.2:
                overall = "Positive"
            elif avg_polarity < -0.2:
                overall = "Negative"
        else:
            overall = "Neutral"
            avg_polarity = 0
            
        return {
            "sections": results,
            "overall": overall,
            "avg_polarity": avg_polarity
        }
    
    def _create_sentiment_gauge(self, sentiment_data):
        """Create a gauge chart for sentiment"""
        value = sentiment_data["avg_polarity"]
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Analysis"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "rgba(59, 130, 246, 0.8)"},
                'steps': [
                    {'range': [-1, -0.2], 'color': "rgba(220, 38, 38, 0.5)"},
                    {'range': [-0.2, 0.2], 'color': "rgba(107, 114, 128, 0.5)"},
                    {'range': [0.2, 1], 'color': "rgba(5, 150, 105, 0.5)"},
                ]
            }
        ))
        fig.update_layout(height=300)
        return fig
    
    def _create_entity_network(self, entities):
        """Create a network visualization of entities"""
        # Collect nodes and edges
        nodes = []
        edges = []
        
        # Create center node
        center_node = {"id": "document", "label": "Document", "size": 20, "color": "#3b82f6"}
        nodes.append(center_node)
        
        # Add entity nodes and connect to center
        colors = {"PERSON": "#dc2626", "ORG": "#059669", "GPE": "#d97706", "DATE": "#7c3aed"}
        
        entity_count = 0
        for entity_type, items in entities.items():
            # Limit to top 5 items per category
            for i, item in enumerate(items[:5]):
                if len(item) < 3:  # Skip very short entities
                    continue
                    
                entity_id = f"{entity_type}_{i}"
                nodes.append({
                    "id": entity_id,
                    "label": item,
                    "group": entity_type,
                    "size": 10,
                    "color": colors.get(entity_type, "#6b7280")
                })
                
                edges.append({
                    "from": "document",
                    "to": entity_id,
                    "width": 2,
                    "label": entity_type
                })
                
                entity_count += 1
        
        # Create a simple force-directed layout using plotly
        fig = go.Figure()
        
        # Calculate node positions using a simple algorithm
        radius = 1
        angle = 2 * 3.14159 / entity_count if entity_count > 0 else 0
        
        # Center node
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            marker=dict(size=20, color=center_node["color"]),
            text=["Document"],
            textposition="bottom center",
            name="Document"
        ))
        
        # Entity nodes
        for i, node in enumerate(nodes[1:]):
            x = radius * np.cos(i * angle)
            y = radius * np.sin(i * angle)
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=10, color=node["color"]),
                text=[node["label"]],
                textposition="middle center",
                name=node["group"]
            ))
            
            # Draw edge
            fig.add_trace(go.Scatter(
                x=[0, x], y=[0, y],
                mode='lines',
                line=dict(width=1, color='rgba(107, 114, 128, 0.5)'),
                showlegend=False
            ))
        
        fig.update_layout(
            title='Entity Network',
            showlegend=True,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400
        )
        
        return fig


# Modify the main() function to include the new report feature
def main():
    st.set_page_config(page_title="Medical Literature Analysis", page_icon="📚", layout="wide")
    load_css()

    st.markdown('<div class="main-header"><h1>📚 Medical Literature & Pandemic Trends Analyzer</h1></div>', unsafe_allow_html=True)
    
    processor = DocumentProcessor()
    analytics = AdvancedAnalytics()
    report_generator = ReportGenerator(analytics)
    
    uploaded_files = st.file_uploader("Upload Medical Literature (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        create_loading_spinner("Loading Documents...")
        raw_texts = []
        all_texts = []

        for file in uploaded_files:
            try:
                text = processor.load_document(file)
                if text:
                    raw_texts.append(text) 
                    cleaned_text = processor.clean_text(text)
                    all_texts.append(cleaned_text)
            except Exception as e:
                st.error(f"Error processing file {file.name}: {str(e)}")

        if not all_texts:
            st.error("No valid text could be extracted from the uploaded files")
            return
            
        combined_text = " ".join(all_texts)
        
        # Add the new "Report" tab
        tabs = st.tabs(["Overview", "Topics", "Entities", "Citations", "Trends", "Report"])
        
        # Original tabs code remains the same...
        with tabs[0]:  
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-card"><h3>Total Documents</h3><h2>{len(uploaded_files)}</h2></div>', unsafe_allow_html=True)
            with col2:
                total_words = len(word_tokenize(combined_text))
                st.markdown(f'<div class="metric-card"><h3>Total Words</h3><h2>{total_words:,}</h2></div>', unsafe_allow_html=True)
            with col3:
                unique_words = len(set(word_tokenize(combined_text)))
                st.markdown(f'<div class="metric-card"><h3>Unique Words</h3><h2>{unique_words:,}</h2></div>', unsafe_allow_html=True)
            
            st.markdown('<div class="wordcloud-container">', unsafe_allow_html=True)
            st.subheader("Word Cloud")
            if combined_text:
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(combined_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.info("Not enough text to generate a word cloud")
            st.markdown('</div>', unsafe_allow_html=True)

        with tabs[1]:
            num_docs = len(all_texts)
            min_topics = 1 if num_docs == 1 else 2
            max_topics = min(50, num_docs) if num_docs > 2 else 3
            n_topics = st.slider("Number of Topics", min_topics, max_topics, min_topics, key="topic_slider")
            create_loading_spinner("Analyzing Topics...")
            topics = analytics.perform_topic_modeling(all_texts, n_topics)
            if topics:
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                for topic in topics:
                    st.write(f"**{topic}**")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Topic modeling could not be performed with the current data.")

        with tabs[2]:
            create_loading_spinner("Extracting Entities...")
            entities = analytics.extract_medical_entities(combined_text)
            has_entities = False
            for entity_type, items in entities.items():
                if items:
                    has_entities = True
                    st.subheader(entity_type)
                    st.markdown(f'<div class="entity-box">{", ".join(set(items))}</div>', unsafe_allow_html=True)
            if not has_entities:
                st.info("No entities were detected in the uploaded documents")

        with tabs[3]:  
            create_loading_spinner("Analyzing Citations...")
            
            citation_info = analytics.analyze_citation_network(raw_texts)
            
            if citation_info:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                
                citation_df = pd.DataFrame([
                    {
                        "Author": info["author"],
                        "Year": info["year"],
                        "Count": info["count"],
                        "Reference": info["reference"]
                    } for num, info in citation_info.items()
                ])
                
                st.dataframe(citation_df)
                
                csv = citation_df.to_csv(index=True)
                timestamp = time.strftime("%Y-%m-%dT%H-%M")
                st.download_button(
                    label="Download Citation Data as CSV",
                    data=csv,
                    file_name=f"{timestamp}_export.csv",
                    mime="text/csv",
                )
                
                author_counts = {}
                for _, info in citation_info.items():
                    author_key = f"{info['author']} ({info['year']})"
                    if author_key not in author_counts:
                        author_counts[author_key] = 0
                    author_counts[author_key] += info["count"]
                
                top_authors = dict(sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10])
                
                fig = px.bar(
                    x=list(top_authors.keys()),
                    y=list(top_authors.values()),
                    title="Most Cited Authors",
                    labels={"x": "Author (Year)", "y": "Citation Count"}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)
                
                st.markdown('</div>', unsafe_allow_html=True)

                # Keep the existing most cited paper recommendation
                valid_citations = [info for info in citation_info.values() ]
                if valid_citations:
                    # Show most cited paper recommendation
                    most_cited = max(valid_citations, key=lambda x: x["count"])
                    
                    st.markdown("---")
                    st.subheader("📄 Recommended Paper")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"""
                        **{most_cited['author']} ({most_cited['year']})**  
                        {most_cited['reference']}  
                        *Cited {most_cited['count']} times*  
                        """)
                    with col2:
                        st.markdown("""
                        **Next Steps:**  
                        🔍 Search for this paper on:  
                        [PubMed](https://pubmed.ncbi.nlm.nih.gov) | 
                        [Google Scholar](https://scholar.google.com)  
                        📥 Check institutional access for full text
                        """)

                    # Document comparison feature
                    st.markdown("---")
                    st.subheader("🔍 Compare External Paper")
                    compare_file = st.file_uploader("Upload a paper to analyze similarity", 
                                                type=["pdf", "txt"],
                                                help="Upload the recommended paper or similar research")
                    
                    if compare_file:
                        create_loading_spinner("Analyzing Content Similarity...")
                        try:
                            # Process comparison doc
                            compare_text = processor.load_document(compare_file)
                            cleaned_compare = processor.clean_text(compare_text)
                            
                            # Prepare documents
                            vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
                            tfidf_matrix = vectorizer.fit_transform(all_texts + [cleaned_compare])
                            
                            # Calculate similarity
                            similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
                            avg_similarity = np.mean(similarity) * 100
                            
                            # Visualize results
                            st.metric("Overall Similarity Score", f"{avg_similarity:.1f}%", 
                                    help="Percentage similarity with uploaded documents")
                            
                            fig = px.bar(
                                x=[f"Doc {i+1}" for i in range(len(similarity[0]))],
                                y=similarity[0]*100,
                                labels={"x": "Documents", "y": "Similarity %"},
                                color=similarity[0]*100,
                                color_continuous_scale="Blues"
                            )
                            fig.update_layout(title_text="Similarity with Each Document")
                            st.plotly_chart(fig)

                        except Exception as e:
                            st.error(f"Comparison failed: {str(e)}")
            else:
                st.info("No citations found in the uploaded documents. This analyzer works best with academic papers that use numbered citation style [1].")

        with tabs[4]: 
            create_loading_spinner("Analyzing Trends...")
            keyword_freq, keyword_year_data = analytics.analyze_temporal_patterns(raw_texts, top_n_keywords=10)

            if keyword_freq:
                df = pd.DataFrame(list(keyword_freq.items()), columns=["Keyword", "Frequency"])
                fig = px.bar(df, x="Keyword", y="Frequency", title="Top Medical Terms with Temporal References",
                            labels={"Keyword": "Medical Term", "Frequency": "Mentions with Years"},
                            color="Frequency", color_continuous_scale="Blues")
                st.plotly_chart(fig)
                timeline_data = []
                for keyword, year_counts in keyword_year_data.items():
                    for year, count in year_counts.items():
                        timeline_data.append({"Term": keyword, "Year": year, "Count": count})

                if timeline_data:
                    timeline_df = pd.DataFrame(timeline_data)
                    fig2 = px.line(timeline_df, x="Year", y="Count", color="Term",
                                title="Historical Trends of Top Medical Terms")
                    st.plotly_chart(fig2)

            else:
                st.info("No temporal patterns with keywords found in the documents.")
                
            sentiment = analytics.analyze_sentiment(combined_text)
            sentiment_class = f"sentiment-{sentiment.lower()}"
            st.markdown(f'<div class="tab-content"><h3>Document Sentiment</h3><p>Overall sentiment: <span class="{sentiment_class}">{sentiment}</span></p></div>', unsafe_allow_html=True)
            
            keywords = analytics.extract_keywords(all_texts)
            if keywords:
                st.markdown('<div class="tab-content"><h3>Top Keywords</h3>', unsafe_allow_html=True)
                keywords_html = ' '.join([f'<span class="keyword-tag">{kw}</span>' for kw in keywords])
                st.markdown(keywords_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # New Report tab
        with tabs[5]:  # Report tab
            if not (raw_texts and all_texts):
                st.info("Please upload documents to generate a report.")
            else:
                st.header("📊 Research Insight Report")
                
                citation_info = analytics.analyze_citation_network(raw_texts)
                
                if citation_info:
                    create_loading_spinner("Generating Comprehensive Research Report...")
                    
                    try:
                        report = report_generator.generate_report(citation_info, raw_texts, all_texts)
                        
                        if isinstance(report, str):
                            st.warning(report)
                        else:
                            # Display report title and citation
                            st.markdown(f"## {report['title']}")
                            
                            # Show summaries of both papers
                            st.markdown("### 📄 Key Research Papers")
                            
                            tabs1, tabs2 = st.tabs(["Most Cited Paper", "Similar Document"])
                            
                            with tabs1:
                                st.markdown(f"#### {report['cited_paper']['author']} ({report['cited_paper']['year']})")
                                st.markdown(f"*Cited {report['cited_paper']['count']} times in the analyzed literature*")
                                st.markdown(report['cited_paper']['reference'])
                                st.markdown("##### Summary")
                                st.markdown(report['most_cited_summary'])
                            
                            with tabs2:
                                st.markdown(f"#### {report['similar_doc_title']}")
                                st.markdown("##### Summary")
                                st.markdown(report['similar_doc_summary'])
                            
                            # Show key findings
                            st.markdown("### 🔍 Key Findings")
                            if report['key_findings']:
                                for i, finding in enumerate(report['key_findings']):
                                    st.markdown(f"**{i+1}.** {finding}")
                            else:
                                st.info("No specific findings could be extracted from the documents.")
                            
                            # Show methodology
                            st.markdown("### 🧪 Research Methodology")
                            if report['methodology']:
                                for method in report['methodology']:
                                    st.markdown(f"• {method}")
                            else:
                                st.info("No methodology information could be extracted.")
                            
                            # Show comparison between documents
                            st.markdown("### 📊 Document Comparison")
                            comparison = report['comparison']
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**Common Themes**")
                                if comparison['common_themes']:
                                    for theme in comparison['common_themes']:
                                        st.markdown(f"• {theme}")
                                else:
                                    st.info("No common themes found.")
                            
                            with col2:
                                st.markdown("**Unique to Cited Paper**")
                                if comparison['cited_paper_unique']:
                                    for theme in comparison['cited_paper_unique']:
                                        st.markdown(f"• {theme}")
                                else:
                                    st.info("No unique themes found.")
                            
                            with col3:
                                st.markdown("**Unique to Similar Document**")
                                if comparison['similar_doc_unique']:
                                    for theme in comparison['similar_doc_unique']:
                                        st.markdown(f"• {theme}")
                                else:
                                    st.info("No unique themes found.")
                            
                            # Show research gaps
                            st.markdown("### 🧩 Research Gaps & Future Directions")
                            if report['research_gaps']:
                                for gap in report['research_gaps']:
                                    st.markdown(f"• {gap}")
                            else:
                                st.info("No specific research gaps could be identified.")
                            
                            # Data visualization section
                            st.markdown("### 📈 Data Visualization")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.plotly_chart(report['word_freq_fig'], use_container_width=True)
                            
                            with col2:
                                sentiment_class = f"sentiment-{report['sentiment']['overall'].lower()}"
                                st.markdown(f"#### Sentiment Analysis: <span class='{sentiment_class}'>{report['sentiment']['overall']}</span>", unsafe_allow_html=True)
                                st.plotly_chart(report['sentiment_fig'], use_container_width=True)
                            
                            # Entity network visualization
                            st.markdown("### 🌐 Entity Relationship Network")
                            st.plotly_chart(report['entity_fig'], use_container_width=True)
                            
                            # Display entities
                            st.markdown("### 🏷️ Key Entities")
                            
                            entity_columns = st.columns(4)
                            with entity_columns[0]:
                                if report['entities'].get('PERSON'):
                                    st.markdown("**People:**")
                                    st.markdown("\n".join([f"• {person}" for person in list(set(report['entities']['PERSON']))[:5]]))
                            
                            with entity_columns[1]:
                                if report['entities'].get('ORG'):
                                    st.markdown("**Organizations:**")
                                    st.markdown("\n".join([f"• {org}" for org in list(set(report['entities']['ORG']))[:5]]))
                            
                            with entity_columns[2]:
                                if report['entities'].get('GPE'):
                                    st.markdown("**Locations:**")
                                    st.markdown("\n".join([f"• {loc}" for loc in list(set(report['entities']['GPE']))[:5]]))
                            
                            with entity_columns[3]:
                                if report['entities'].get('DATE'):
                                    st.markdown("**Dates:**")
                                    st.markdown("\n".join([f"• {date}" for date in list(set(report['entities']['DATE']))[:5]]))
                            
                            # Include a download button for the report
                            st.markdown("---")
                            
                            report_md = f"""
                            # {report['title']}
                            
                            ## Most Cited Paper
                            **{report['cited_paper']['author']} ({report['cited_paper']['year']})**  
                            {report['cited_paper']['reference']}  
                            *Cited {report['cited_paper']['count']} times*
                            
                            ### Summary
                            {report['most_cited_summary']}
                            
                            ## Similar Document
                            **{report['similar_doc_title']}**
                            
                            ### Summary
                            {report['similar_doc_summary']}
                            
                            ## Key Findings
                            {chr(10).join([f"{i+1}. {finding}" for i, finding in enumerate(report['key_findings'])])}
                            
                            ## Research Methodology
                            {chr(10).join([f"- {method}" for method in report['methodology']])}
                            
                            ## Document Comparison
                            
                            ### Common Themes
                            {chr(10).join([f"- {theme}" for theme in comparison['common_themes']])}
                            
                            ### Unique to Cited Paper
                            {chr(10).join([f"- {theme}" for theme in comparison['cited_paper_unique']])}
                            
                            ### Unique to Similar Document
                            {chr(10).join([f"- {theme}" for theme in comparison['similar_doc_unique']])}
                            
                            ## Research Gaps & Future Directions
                            {chr(10).join([f"- {gap}" for gap in report['research_gaps']])}
                            
                            ## Key Entities
                            - People: {', '.join(list(set(report['entities'].get('PERSON', [])))[:5])}
                            - Organizations: {', '.join(list(set(report['entities'].get('ORG', [])))[:5])}
                            - Locations: {', '.join(list(set(report['entities'].get('GPE', [])))[:5])}
                            - Dates: {', '.join(list(set(report['entities'].get('DATE', [])))[:5])}
                            
                            ## Most Frequent Terms
                            {', '.join([f"{term} ({freq})" for term, freq in report['freq_words'].items()])}
                            
                            ## Sentiment Analysis
                            Overall sentiment: {report['sentiment']['overall']}
                            """
                            
                            st.download_button(
                                label="💾 Download Comprehensive Report as Markdown",
                                data=report_md,
                                file_name=f"research_insight_report_{time.strftime('%Y-%m-%d')}.md",
                                mime="text/markdown",
                            )
                            
                        
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
                        st.info("Try uploading different documents or check the citation format.")
                else:
                    st.warning("No citation information available. The report generator works best with academic papers that use numbered citation style.")

if __name__ == "__main__":
    main()
