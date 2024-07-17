import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(0)

# Number of entries to generate
num_entries = 1000

# Generate random text entries
positive_texts = [
    "The movie was great!",
    "I love this product.",
    "The service was exceptional.",
    "The weather is perfect today.",
    "The team did an outstanding job.",
    "The food was delicious.",
    "I'm thrilled with the outcome.",
    "The presentation was impressive.",
    "I'm really happy with my purchase.",
    "The customer support was excellent."
]

negative_texts = [
    "I'm disappointed with the service.",
    "The product quality is poor.",
    "The movie was terrible.",
    "I regret buying this.",
    "The service was awful.",
    "I'm frustrated with the results.",
    "The experience was unpleasant.",
    "The software is full of bugs.",
    "The food was disappointing.",
    "The customer support was unhelpful."
]

neutral_texts = [
    "The event was average.",
    "I have no strong feelings about this.",
    "The movie was okay.",
    "The service was satisfactory.",
    "I'm neutral about the new feature.",
    "The product met my expectations.",
    "The experience was neither good nor bad.",
    "I'm indifferent towards this decision.",
    "The food was average.",
    "The service was neither impressive nor disappointing."
]

# Assign random sentiments
sentiments = np.random.choice(['positive', 'negative', 'neutral'], num_entries)

# Randomly sample from each sentiment category
positive_samples = np.random.choice(positive_texts, num_entries)
negative_samples = np.random.choice(negative_texts, num_entries)
neutral_samples = np.random.choice(neutral_texts, num_entries)

# Combine into a DataFrame
df_positive = pd.DataFrame({'text': positive_samples, 'sentiment': 'positive'})
df_negative = pd.DataFrame({'text': negative_samples, 'sentiment': 'negative'})
df_neutral = pd.DataFrame({'text': neutral_samples, 'sentiment': 'neutral'})

# Concatenate all dataframes
df = pd.concat([df_positive, df_negative, df_neutral], ignore_index=True)

# Shuffle the dataframe
df = df.sample(frac=1, random_state=0).reset_index(drop=True)

# Save to CSV
df.to_csv('sentiment_data.csv', index=False)
