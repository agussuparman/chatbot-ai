import json
import pandas as pd


class JSONParser:
    def __init__(self):
        self.data = None
        self.df = None
        self.texts = []
        self.intents = []

    def parse(self, json_path):
        with open(json_path) as data_file:
            self.data = json.load(data_file)

        for intent in self.data['intents']:
            for pattern in intent['patterns']:
                self.texts.append(pattern)
                self.intents.append(intent['topic'])

        self.df = pd.DataFrame({'text_input': self.texts, 'intents': self.intents})

        print(f"[INFO] data JSON converted to DataFrame with shape : {self.df.shape}")

    def get_data_frame(self):
        return self.df
