# Intent-Classification-on-Real-Word-data
Here is an embedding method to classify intents which aims at improving chatbot responses. 

## Aim of the project
Dialogflow is a platform where we can create a chatbot by feeding it with sentences annotated with an intent. 
Example: "I want to eat an apple." and "I would like some chocolate." will be in "Food" intent, and an automatic response would be "Here is the list of available foods." Dialogflow uses a hybrid model (not publicly released) which combines Rule-Based (matching with regular expressions) and Machine Learning. Our main goal is to use Word Embedding to obtain an only Machine Learning model and to compare the results with Dialogflow's. 

## Data
For privacy reasons, we cannot share the dataset we used. However, we will show some results to compare a Dialogflow chatbot and our model.

## Getting started
The sentences are in French. In order to classify them, we use a French pre-trained Word2Vec model available on http://fauconnier.github.io/, then we used an SVM Classifier. 

### Installing
To get the required libraries, run: 
```
pip install -r requirements.txt
```

### Running the test
To get the results, run: 
```
python main.py
```

To predict a single sentence test, use the test function like in the main code. 
