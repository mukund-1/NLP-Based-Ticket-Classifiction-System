# Ticket Classification and Entity Extraction NLP Project

## Project Overview
This project builds an end-to-end NLP system to analyze customer support tickets. It includes:

- Issue Type Classification: Predicts the category of the ticket (e.g., Account Access, Payment Issues).

- Urgency Level Classification: Predicts the urgency of the ticket (Low, Medium, High).

- Entity Extraction: Extracts key entities such as product names, dates, and complaint keywords from the ticket text.

These models are integrated into a Gradio web app to provide an interactive interface for users to input ticket text and receive predictions along with extracted entities.

----------


## Features
- Custom NLP pipelines built using traditional NLP techniques and Word2Vec embeddings.

- Separate models for issue type and urgency level classification.

- Rule-based entity extraction using regex and product lists.

- Interactive and easy-to-use web app via Gradio.

- Modular pipeline structure for easy maintenance and extension.

-----------


## Getting Started

### Prerequisites
- Python 3.7+
- pip

### Installation
1. Clone the repository:
 ```bash
git clone https://github.com/mukund-1/NLP-Based-Ticket-Classifiction-System.git
cd NLP-Based-Ticket-Classifiction-System
  ```

3. Create a virtual environment by executing the following command in terminal
  ```bash
  python -m venv venv
  ```

  then activate the virtual environment
  ```bash
  ./venv/Scripts/activate 
  ```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the App
Run the Gradio app with:
```bash
python app.py
```

This will start a local web server, usually accessible at http://localhost:7860, where you can input ticket text and get predictions.


-----------


## Project Structure
```bash
.
├── app.py                   # Main Gradio app script
├── pipelines.py             # NLP pipelines: issue_type, urgency_level, entity_extraction
├── data/
│   └── datasets    # List of products used for entity extraction
├── models/
│   └── ...                  # Saved models (if any)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```



--------------------



## Usage
The app accepts raw ticket text and returns:

- Issue Type: Predicted category of the ticket.

- Urgency Level: Predicted urgency (Low, Medium, High).

- Extracted Entities: Dictionary with product names, dates, and complaint keywords found in the text.

Example input:
```text
I am unable to access my account since yesterday and this is urgent.    
```

Example output:

```json
Issue Type: Account Access
Urgency Level: High
Extracted Entities:
{
  "product_names": [],
  "dates": ["yesterday"],
  "complaint_keywords": ["unable", "access", "urgent"]
}
```

--------------



## Model Details
- Issue Type & Urgency Level: Trained using Word2Vec embeddings and classical classifiers (e.g., Logistic Regression or Random Forest).

- Entity Extraction: Rule-based using regex patterns and product keyword matching.


------------


## Notes
- Preprocessing steps include lowercasing, tokenization, and removing stopwords.

- You can update product_list.txt to customize entity extraction for your domain.

- For improved accuracy, consider fine-tuning models with more labeled data.
