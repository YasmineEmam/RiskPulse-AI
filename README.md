#  RiskPulse AI

### AI-Powered Corporate Credit Risk Intelligence & Early Warning System

RiskPulse AI is an AI-powered **corporate credit risk monitoring system** designed to help credit analysts identify potential financial and operational risks affecting companies.

The system takes a **company name** as input, automatically collects relevant news from the past several years, extracts article content, analyzes each article using an LLM-based banking risk engine, and produces a **Composite Risk Score (CRS)** together with categorized risk events and supporting evidence.

---

##  Project Objective

RiskPulse AI aims to provide a lightweight decision-support system for corporate credit assessment by transforming unstructured news into structured risk intelligence.

Instead of manually reviewing large numbers of news articles, the system:

* Searches for relevant company news.
* Extracts and cleans article content.
* Resolves company names and aliases.
* Uses an LLM to identify material credit risks.
* Classifies risks into predefined banking risk domains.
* Considers source reliability and news recency.
* Calculates a Composite Risk Score.
* Presents the results through an interactive dashboard.

---

##  Key Features

###  Multi-Source News Aggregation

The system searches Google News RSS using multiple queries covering:

* Recent company news
* Company-specific news
* Financial and operational crisis indicators
* Alternative company names and aliases

The news retrieval process uses concurrent requests to improve performance.

###  Article Text Extraction

RiskPulse AI attempts to retrieve the full article text using multiple extraction strategies:

1. **Trafilatura**
2. **BeautifulSoup**
3. RSS summary as a fallback

A content-quality score is also assigned depending on the quality and amount of extracted text.

###  Corporate Entity Resolution

The system supports company aliases and Arabic/English variations.

It first performs exact matching and then applies **fuzzy matching** using RapidFuzz when an exact match is not found.

### 🇪🇬 Arabic Text Processing

The application includes Arabic normalization to improve company matching and text analysis.

Depending on the installed environment, it can use **CAMeL Tools** for:

* Unicode normalization
* Arabic diacritics removal
* Alef normalization
* Alef Maksura normalization
* Teh Marbuta normalization

A regex-based fallback is also available.

###  LLM-Based Credit Risk Analysis

Each article is analyzed by a banking-focused LLM engine.

The analysis distinguishes between:

**Potential material risks**

* Loan defaults
* Liquidity problems
* Debt restructuring
* Lawsuits
* Arbitration
* Production shutdowns
* Regulatory fines
* Fraud or corruption
* Credit-rating downgrades
* Significant profit declines
* Severe cost increases
* Supply-chain disruptions
* Relevant macroeconomic pressures

and non-risk information such as routine expansion, partnerships, investments, or positive company news.

###  Risk Domains

Detected risk events are categorized into five domains:

| Domain                    | Base Severity |
| ------------------------- | ------------: |
| Financial & Liquidity     |            10 |
| Operations & Supply Chain |             8 |
| Legal & Regulatory        |             9 |
| Market & Macroeconomic    |             6 |
| Reputation & Emergencies  |             7 |

---

##  Risk Scoring Methodology

RiskPulse AI does not rely only on the LLM's classification.

Each detected event receives a weighted contribution based on:

* **Severity**
* **Source reliability**
* **Entity confidence**
* **Recency**
* **Relevance**
* **Article content quality**

The event contribution is calculated as:

```text
Contribution =
Severity × Source Reliability × Entity Confidence
× Recency × Relevance × Content Quality
```

The individual event contributions are then aggregated into a **Composite Risk Score (CRS)** ranging from 0 to 100.

### Risk Levels

|     CRS | Risk Level | Decision Support         |
| ------: | ---------- | ------------------------ |
|    0–15 | Low        | Normal credit monitoring |
| 15.1–45 | Medium     | Enhanced credit review   |
|     >45 | High       | Detailed credit review   |

These thresholds are implemented directly in the application.

---

##  System Workflow

```text
User
  │
  ▼
Enter Company Name
  │
  ▼
Company Entity Resolution
  │
  ▼
Multi-Query News Search
  │
  ▼
Article Text Extraction
  │
  ▼
Arabic Text Processing
  │
  ▼
LLM-Based Risk Analysis
  │
  ├── Risk Detection
  ├── Risk Domain
  ├── Evidence
  └── Severity
  │
  ▼
Risk Validation & Weighting
  │
  ├── Source Reliability
  ├── Entity Confidence
  ├── Recency
  ├── Relevance
  └── Content Quality
  │
  ▼
Composite Risk Score
  │
  ▼
Streamlit Risk Dashboard
```

---

##  Dashboard

The Streamlit interface provides:

* Company selection
* Configurable number of target news articles
* Composite Risk Score
* Overall risk level
* Number of validated risk events
* Number of analyzed articles
* Risk distribution across domains
* Detailed risky articles
* Evidence supporting each detected risk
* Source and publication date
* Article URL
* Individual risk contribution metrics

The dashboard also separates analyzed articles into **detected-risk** and **no-risk** sections.

---

##  Technology Stack

### Frontend / Dashboard

* Streamlit

### AI / LLM

* Groq API
* `openai/gpt-oss-120b`

### Data & Numerical Processing

* Python
* NumPy
* Dataclasses

### News & Web Processing

* Google News RSS
* Feedparser
* Requests
* BeautifulSoup
* Trafilatura
* Newspaper

### NLP

* CAMeL Tools
* RapidFuzz
* Regular Expressions

### Concurrency

* Python `ThreadPoolExecutor`

---

##  Project Structure

```text
RiskPulse-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── groq_debug.log        # Generated when API/debug errors occur
└── llm_responses.log     # Generated LLM response log
```

> `groq_debug.log` and `llm_responses.log` are runtime-generated files and should generally not be committed to a public repository.

---

##  Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd RiskPulse-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not been created yet, the project requires the Python packages used by the application, including:

```text
streamlit
feedparser
numpy
requests
beautifulsoup4
rapidfuzz
trafilatura
newspaper4k
camel-tools
```

---

##  API Configuration

RiskPulse AI requires a **Groq API key**.

The current application accepts the API key securely through the Streamlit sidebar rather than hard-coding it into the source code.

You can obtain a key from:

https://console.groq.com/keys

**Never commit your API key to GitHub.**

For production deployment, use environment variables or a secure secrets manager.

---

##  Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

Then open the local Streamlit URL displayed in the terminal.

---

##  Example Usage

1. Enter a company name, for example:

```text
جهينة
```

2. Enter your Groq API key in the sidebar.

3. Select the desired number of news articles.

4. Click:

```text
 تحليل شامل
```

5. RiskPulse AI will:

```text
Resolve Company
       ↓
Collect News
       ↓
Extract Articles
       ↓
Analyze Articles
       ↓
Identify Risk Events
       ↓
Calculate Risk Score
       ↓
Display Results
```

---

##  Risk Evidence & Explainability

For every detected risk, the dashboard provides supporting information including:

* Risk domain
* Evidence sentence extracted from the article
* Severity
* Relevance
* Source reliability
* Entity confidence
* Risk contribution
* Original article source and URL

This allows the analyst to inspect **why an article contributed to the overall risk score** instead of receiving only a final numerical score.

---

##  Performance Considerations

The system uses parallel processing in several stages.

News queries and article extraction use `ThreadPoolExecutor`, while LLM analysis is intentionally limited to two concurrent workers to reduce the risk of hitting API rate limits.
The LLM API integration also includes:

* Retry attempts
* Rate-limit handling
* Request timeouts
* JSON response validation
* Debug logging

---

##  Limitations

RiskPulse AI is a **decision-support prototype**, not an automated credit approval system.

Current limitations include:

* News availability depends on external RSS/search sources.
* Some websites may block automated article extraction.
* Article content quality varies by source.
* Company entity resolution currently relies on a predefined company registry plus fuzzy matching.
* LLM outputs may contain errors and should be reviewed by a human analyst.
* Risk scores are analytical indicators and should not be interpreted as guaranteed predictions of future company performance.
* The current implementation focuses on news-based risk intelligence rather than complete financial-statement analysis.

---

##  Future Improvements

Potential future extensions include:

* Larger and dynamically managed company registry
* Additional financial-data sources
* Historical risk-score tracking
* Company risk dashboards over time
* More advanced source verification
* Automated report generation
* Persistent database storage
* Analyst feedback and model evaluation
* More sophisticated entity resolution
* Production-grade authentication and deployment

---

##  Project Status

**Current Status:** MVP / Prototype

RiskPulse AI currently provides an end-to-end workflow from:

**Company → News → Article Extraction → LLM Risk Analysis → Risk Scoring → Interactive Dashboard**

---

##  Authors

Developed as an AI/Data Science project focused on **corporate credit risk intelligence and early warning systems**.

---

##  License

This project is intended for educational and research purposes.

Add an appropriate open-source license such as MIT if the repository is intended for public reuse.
