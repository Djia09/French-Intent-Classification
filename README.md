# Intent-Classification-on-Real-Word-data
Here is an embedding method to classify intents which aims at improving chatbot responses. 

## Aim of the project
DialogFlow is a platform where we can create a chatbot by feeding it with sentences annotated with an intent. 

Example: "I want to eat an apple." and "I would like some chocolate." will be in "Food" intent, and an automatic response would be "Here is the list of available foods." 

DialogFlow uses a hybrid model (not publicly released) which combines Rule-Based (matching with regular expressions) and Machine Learning. 

Our main goal is to use Word Embedding to obtain an only Machine Learning model and to compare the results with DialogFlow's. 

## Data
For privacy reasons, we cannot share the dataset we used. However, we will show some results to compare a DialogFlow chatbot and our model.

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

## Results
The dataset contains 1900 sentences annotated in 34 potential classes. We split the dataset to have 80% of training and 20% of validation.


|Accuracy score|F1-score|
|--------------|--------|
|     0.94     |  0.94  |

However, these scores does not represent well real-world data, as the sentences in the dataset are semanticly close. A client can ask any kind of question to a chatbot. So let's display some random sentences prediction. 
```
sent1 = "J'ai besoin d'un service de nettoyage"
sent2 = "J'ai besoin de nettoyer mes vêtements"
sent3 = "J'ai besoin de repasser ma chemise"
sent4 = "J'ai besoin de garder mes enfants"

vec1, _ = test(sent1, model, clf, 0.3)
vec2, _ = test(sent2, model, clf, 0.3)
vec3, _ = test(sent3, model, clf, 0.3)
vec4, _ = test(sent4, model, clf, 0.3)
```

Output: 
```
Predicted intent for sent1: VETEMENT_NETTOYAGE
Predicted intent for sent2: VETEMENT_NETTOYAGE
Predicted intent for sent3: VETEMENT_NETTOYAGE
Predicted intent for sent4: ACTIVITES_ENFANTS
```

DialogFlow predicted the label 'VETEMENT_NETTOYAGE' for the 4 of them, while the last one should be ACTIVITES_ENFANTS.

Indeed, DialogFlow's model drawback is to match the structure of the training sentences. In the training sentences, we observe that in the class VETEMENT_NETTOYAGE, several sentences begin with "J'ai besoin de...". Hence, each sentence beginning with "J'ai besoin de..." will be classified in VETEMENT_NETTOYAGE category. 

Our model offers to train a model by extracting sentences' semantic, which DialogFlow model lacks.
