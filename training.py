import json
import numpy as np
from util import JSONParser
from util import Preprocess
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# load data
path = 'data/intents.json'
json_parser = JSONParser()
json_parser.parse(path)
df = json_parser.get_data_frame()

# preprocess
preprocess = Preprocess()

# data cleansing
df['text_input_prep'] = df.text_input.apply(preprocess.manipulation)

# modeling
pipeline = make_pipeline(CountVectorizer(), MultinomialNB())


def topic_prediction(sentence, pipeline):
    sentence_manipulation = preprocess.manipulation(sentence)

    # predict_topic = pipeline.predict([sentence])

    topic_probabilities = pipeline.predict_proba([sentence_manipulation])

    # get classes
    # classes = pipeline.classes_

    maximum_probability = max(topic_probabilities[0])
    if maximum_probability < 0.6:
        params = {
            'message': "Hmmm... I'm not sure I understand that :( I do my best but I'm still learning the human "
                       "language",
            'text_message': sentence
        }

        return json.dumps(params)
    else:
        # get maximum probability
        maximum_id = np.argmax(topic_probabilities[0])

        # get class
        predict_topic = pipeline.classes_[maximum_id]
        params = {
            'intents': {
                'topic': predict_topic,
                'threshold': maximum_probability
            },
            'text_message': sentence
        }

        return json.dumps(params)


# training data
print('[INFO] is processing training data ...')
pipeline.fit(df.text_input_prep, df.intents)
