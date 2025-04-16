import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from safetensors.torch import load_file
from sener import SeNER
import gradio as gr
import re

id2label = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC"
}

model_name = "arpit-tiwari/distilbert-finetuned-hindi-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SeNER(model_name, num_labels=7)

state_dict = load_file("SeNER/model.safetensors")
model.load_state_dict(state_dict)
model.to(device)


def predict_entities(model, tokenizer, text, id2label=None):
    inputs = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, padding=True)
    offset_mapping = inputs.pop("offset_mapping")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs["logits"]
    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    offset_mapping = offset_mapping[0].cpu().numpy()

    word_entities = []
    previous_word = None
    current_word = ""
    current_label = None

    for token, pred, offset in zip(tokens, predictions, offset_mapping):
        if offset[0] == 0 and offset[1] == 0:
            continue  

        start, end = offset
        word = text[start:end]

        if token.startswith("##"):
            current_word += word
        else:
            if current_word:
                label_name = id2label[current_label] if id2label else current_label
                word_entities.append((current_word, label_name))
            current_word = word
            current_label = pred

    if current_word:
        label_name = id2label[current_label] if id2label else current_label
        word_entities.append((current_word, label_name))

    return word_entities



def format_ner_predictions(text, predictions):
    word_to_label = {}
    for token, label in predictions:
        if label != "O":
            word_to_label[token] = label
    
    tokens = sorted(word_to_label.keys(), key=len, reverse=True)
    placeholder_map = {}
    temp_text = text
    for i, token in enumerate(tokens):
        placeholder = f"__PLACEHOLDER_{i}__"
        placeholder_map[placeholder] = f"[{token}: {word_to_label[token]}]"
        temp_text = temp_text.replace(token, placeholder)
    
    for placeholder, replacement in placeholder_map.items():
        temp_text = temp_text.replace(placeholder, replacement)

    color_map = {
        "B-PER": "red",
        "I-PER": "red",
        "B-ORG": "blue",
        "I-ORG": "blue",
        "B-LOC": "green",
        "I-LOC": "green"
    }
    
    highlighted_text = temp_text
    for label, color in color_map.items():
        pattern = f"\\[([^\\]]*): {label}\\]"
        replacement = f'<span style="color:{color};font-weight:bold">[\\1: {label}]</span>'
        highlighted_text = re.sub(pattern, replacement, highlighted_text)
    
    return highlighted_text

def predict_and_format(text):
    raw_predictions = predict_entities(model, tokenizer, text, id2label)
    highlighted_text = format_ner_predictions(text, raw_predictions)
    return highlighted_text

iface = gr.Interface(
    fn=predict_and_format,
    inputs=gr.Textbox(label="Input Text"),
    outputs=gr.HTML(label="Tagged Text"),
    title="Hindi Named Entity Recognition",
    description="Extracts named entities",
)

iface.launch(share=True,show_error=True)