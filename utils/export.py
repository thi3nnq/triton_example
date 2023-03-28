from transformers import RobertaForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer_qa = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
qa = RobertaForQuestionAnswering.from_pretrained(
    'deepset/roberta-base-squad2').eval()

text_input = tokenizer_qa('Hello xin chào mọi người', return_tensors='pt',
                          truncation=True, max_length=tokenizer_qa.model_max_length)

torch.onnx.export(
    qa,
    (text_input.input_ids.to(torch.int32),
     text_input.attention_mask.to(torch.int32)),
    "model_repository/question_answering/1/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["start_logits", "end_logits"],
    do_constant_folding=True,
    opset_version=14,
    dynamic_axes={
        'input_ids': {
            0: 'batch',
            1: 'length'
        },
        'attention_mask': {
            0: 'batch',
            1: 'length'
        },
        'start_logits': {
            0: 'batch',
            1: 'length'
        },
        'end_logits': {
            0: 'batch',
            1: 'length'
        },
    },
)

tokenizer_clf = AutoTokenizer.from_pretrained(
    "Jean-Baptiste/roberta-large-ner-english")
model_clf = AutoModelForTokenClassification.from_pretrained(
    "Jean-Baptiste/roberta-large-ner-english").eval()

text_input = tokenizer_clf("Apple was founded in 1976 by Steve Jobs, Steve Wozniak and Ronald Wayne to develop and sell Wozniak's Apple I personal computer",
                           return_tensors='pt', truncation=True, max_length=tokenizer_clf.model_max_length)

torch.onnx.export(
    model_clf,
    (text_input.input_ids.to(torch.int32),
     text_input.attention_mask.to(torch.int32)),
    "model_repository/NER/1/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    do_constant_folding=True,
    opset_version=14,
    dynamic_axes={
        'input_ids': {
            0: 'batch',
            1: 'length'
        },
        'attention_mask': {
            0: 'batch',
            1: 'length'
        },
        'logits':{
            0: 'batch',
            1: 'length'
        },
    },
)
