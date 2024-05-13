from flask import Flask, request, jsonify
from distributional_models.scripts.create_model import create_model
from distributional_models.corpora.xAyBz import XAYBZ
from distributional_models.models.mlp import MLP
import params
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_path = "e2000.pth"
param2val = params.param2default
training_corpus = XAYBZ(sentence_sequence_rule=param2val['sentence_sequence_rule'],
                        random_seed=param2val['random_seed'],
                        ab_category_size=param2val['ab_category_size'],
                        num_ab_categories=param2val['num_ab_categories'],
                        num_omitted_ab_pairs=param2val['num_omitted_ab_pairs'])
missing_training_words = training_corpus.create_vocab()
the_model = create_model(training_corpus.vocab_list, param2val)
the_model.load_model(model_path)

# @app.route('/')
# def home():
#     return "Welcome to the Recurrent Neural Network Visualization API!"



@app.route('/model-summary', methods=['GET'])
def send_model_summary():
    # Assuming `the_model` is your PyTorch model instance
    summary = model_summary(the_model)
    return jsonify(summary)

def model_summary(model):
    summary = {
        "embedding_layer": {},
        "hidden_layer": {},
        "output_layer": {}
    }
    for name, module in model.named_modules():
        if name: 
            if "embedding" in name:
                summary["embedding_layer"] = {
                    "name": name,
                    "type": module.__class__.__name__,
                    "input_features": getattr(module, 'embedding_dim', None),
                    "output_features": getattr(module, 'num_embeddings', None)
                }
            elif "hidden" in name:
                summary["hidden_layer"] = {
                    "name": name,
                    "type": module.__class__.__name__,
                    "input_features": getattr(module, 'in_features', None),
                    "output_features": getattr(module, 'out_features', None)
                }
            elif "output" in name:
                summary["output_layer"] = {
                    "name": name,
                    "type": module.__class__.__name__,
                    "input_features": getattr(module, 'in_features', None),
                    "output_features": getattr(module, 'out_features', None)
                }
    return summary

if __name__ == '__main__':
    app.run(debug=True)
