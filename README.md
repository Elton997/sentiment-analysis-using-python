# Sentiment Analysis Project

This repository contains a Python-based sentiment analysis project. It includes data processing, machine learning model training, and evaluation scripts for sentiment classification of text data.

## Features

- **Data Processing**: Clean and preprocess text data for sentiment analysis.
- **Model Training**: Train machine learning models using natural language processing techniques.
- **Evaluation**: Evaluate model performance and generate metrics.

## Files Included

- `data_processing.py`: Script for preprocessing text data.
- `sentiment_model.py`: Script to train a sentiment analysis model.
- `sentiment_data.csv`: Sample dataset used for training and evaluation.
- `vectorized_data.csv`: Vectorized data generated from the dataset.
- `labels.csv`: Labels corresponding to the sentiment data.
- `vectorizer.pkl`: Pre-trained TF-IDF vectorizer for text data
- `sentiment_model.pkl`: Pre-trained sentiment analysis model
- `requirements.txt`: List of Python dependencies required to run the project.
- `.gitignore`: Specifies files and directories Git should ignore.

## Getting Started

### Prerequisites

- Python 3.x
- Pip (Python package installer)

### Installation

1. Clone the repository:
- git clone https://github.com/Elton997/sentiment-analysis-using-python.git
- cd sentiment-analysis-using-python

2. Install the required dependencies:
- pip install -r requirements.txt

## Implementation & Usage (Predefined data and without training model):

1. Run the Flask API:
- python app.py

2. Test the API using postman:
IMAGE:

## Implementation & Usage (New data and with training model):

1. Create a new sentiment_data.csv file using random_statement_generator.py where num_entries will define how much data you will be creating(which can be changed) :
- python random_statement_generator.py 

2. Process the data and create the vectorization data for model training:
- python data_processing.py
This would create vectorized_data.csv, labels.csv and vectorizer.pkl files respectively.

3. Train the sentiment model for new data:
- python sentiment_model.py

4. Run the Flask API:
- python app.py

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.