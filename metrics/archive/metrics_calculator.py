from collections import defaultdict
from dataclasses import dataclass

from sklearn.metrics import classification_report

from ..utils import load_index, load_json, load_jsonl, load_pickle


@dataclass
class Feature:
    text: str
    category: str

    def __repr__(self) -> str:
        return f"{self.text}"

    def __str__(self) -> str:
        return f"{self.text}"

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, i):
        return self.text[i]

    def __hash__(self) -> int:
        return hash(self.text)


class BaseMetrics:
    def __init__(self, data_path, gold_data_path, train_path):
        try:
            self.gen_data = load_json(data_path)
        except:
            self.gen_data = load_jsonl(data_path)
        self.gold_data = load_pickle(gold_data_path)
        self.train_index = load_index(train_path)
        try:
            self.item2feature, self.ui2feature = self.make_menu()
        except:
            self.item2feature, self.ui2feature = None, None
            print("Something is wrong with the gold data for menu creation")

    def make_menu(self):
        item2feature = defaultdict(set)
        ui2feature = defaultdict(set)
        for segment in self.gold_data:
            user = segment["user"]
            item = segment["item"]
            fea, adj, text, sco, category = segment["template"]
            for f in fea:
                item2feature[item].add(Feature(text, category))
                ui2feature[(user, item)].add(Feature(text, category))
        return item2feature, ui2feature

    def retrieval_dish_match_ratio(self):
        raise NotImplementedError

    def sentence_in_train_ratio(self):
        raise NotImplementedError


class RetrievedMetrics:
    def __init__(self, data_path, gold_data_path, train_path):
        try:
            self.gen_data = load_json(data_path)
        except:
            self.gen_data = load_jsonl(data_path)
        self.gold_data = load_pickle(gold_data_path)
        self.train_index = load_index(train_path)
        self.item2feature, self.ui2feature = self.make_menu()

    def make_menu(self):
        item2feature = defaultdict(set)
        ui2feature = defaultdict(set)
        for segment in self.gold_data:
            user = segment["user"]
            item = segment["item"]
            fea, adj, text, sco, category = segment["template"]
            for f in fea:
                item2feature[item].add(Feature(text, category))
                ui2feature[(user, item)].add(Feature(text, category))
        return item2feature, ui2feature

    def sentence_in_train_ratio(self):
        raise NotImplementedError


class RetrievedMetrics(BaseMetrics):
    def __init__(self, data_path, gold_data_path, train_path):
        super().__init__(data_path, gold_data_path, train_path)
        self.other_keys = [
            "user",
            "item",
            "user_name",
            "item_name",
            "gold_text",
            "gold_overall_rating",
        ]

    def retrieval_dish_match_ratio(self):
        match_count = 0

        for prediction in self.gen_data:
            user, item = prediction["user"], prediction["item"]
            pred_text = ""
            for key in prediction:
                if key not in self.other_keys:
                    pred_text += prediction[key]["predict_text"] + " "
            pred_text = pred_text.strip()
            gold_dishes = self.ui2feature[(user, item)]
            for gold_dish in gold_dishes:
                if str(gold_dish) in pred_text:
                    match_count += 1
                    break
        return match_count / len(self.gen_data)

    def sentence_in_train_ratio(self):
        # need to load the train index
        train_sentences = set(
            [self.gold_data[i]["template"][2] for i in self.train_index]
        )
        count = 0
        total_count = 0
        for prediction in self.gen_data:
            for key in prediction:
                if key not in self.other_keys:
                    total_count += 1
                    pred_text = prediction[key]["predict_text"]
                    if pred_text in train_sentences:
                        count += 1
        return count / total_count


class GeneratedMetrics(BaseMetrics):
    def __init__(self, data_path, gold_data_path, train_path):
        super().__init__(data_path, gold_data_path, train_path)

    def sentence_in_train_ratio(self):
        train_sentences = set(
            [self.gold_data[i]["template"][2] for i in self.train_index]
        )
        count = 0
        for x in self.gen_data:
            if x["fake"] in train_sentences:
                count += 1
        return count / len(self.gen_data)

    def __map_sentiment(self, sentiment: float) -> int:
        if sentiment < 3:
            return -1
        elif sentiment == 3:
            return 0
        else:
            return 1

    def sentiment_category_report(self):
        # f1, precision, recall
        golds = []
        preds = []

        for x in self.gen_data:

            gold_sentiment = self.__map_sentiment(x["gold_rating"])
            pred_sentiment = self.__map_sentiment(x["predicted_rating"])
            preds.append(pred_sentiment)
            golds.append(gold_sentiment)

        return classification_report(golds, preds, output_dict=True)
