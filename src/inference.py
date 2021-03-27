from farm.infer import Inferencer
from pprint import PrettyPrinter


basic_texts = [
    {"text": "기생충,,, 이 영화 정말 재밌네요."},
    {"text": "황정민 나오는 영화는 다 볼만한듯?"},
]

infer_model = Inferencer.load("./ckpt")

result = infer_model.inference_from_dicts(dicts=basic_texts)
PrettyPrinter().pprint(result)