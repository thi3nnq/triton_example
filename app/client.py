import argparse
import gradio as gr
import numpy as np
from tritonclient import http as tritonhttpclient
from transformers import RobertaForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForTokenClassification
from scipy.special import log_softmax

gr.close_all()

parser = argparse.ArgumentParser()
parser.add_argument("--triton_url", default='103.119.132.171:2703')
args = parser.parse_args()

client = tritonhttpclient.InferenceServerClient(url=f"{args.triton_url}")

tokenizer_qa = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
tokenizer_clf = AutoTokenizer.from_pretrained(
    "Jean-Baptiste/roberta-large-ner-english")


def _classify(text):

    text_input = tokenizer_clf(text, return_tensors='np')
    _input_ids = text_input.input_ids.astype(np.int32)
    _attention_mask = text_input.attention_mask.astype(np.int32)

    input_ids = tritonhttpclient.InferInput('input_ids', shape=_input_ids.shape,
                                            datatype='INT32')
    input_ids.set_data_from_numpy(_input_ids, binary_data=False)

    attention_mask = tritonhttpclient.InferInput('attention_mask', shape=_attention_mask.shape,
                                                 datatype='INT32')
    attention_mask.set_data_from_numpy(_attention_mask, binary_data=False)

    req_output = tritonhttpclient.InferRequestedOutput('logits')

    response = client.infer(
        model_name='NER',
        inputs=[input_ids, attention_mask],
        outputs=[req_output]
    )

    logits = response.as_numpy("logits").astype(np.float32)

    log_probs = log_softmax(logits)
    print(str(log_probs.argmax(axis=-1)))
    return str(log_probs.argmax(axis=-1))


def _qa(text):

    text_input = tokenizer_qa(text, return_tensors='np')
    _input_ids = text_input.input_ids.astype(np.int32)
    _attention_mask = text_input.attention_mask.astype(np.int32)

    input_ids = tritonhttpclient.InferInput('input_ids', shape=_input_ids.shape,
                                            datatype="INT32")
    input_ids.set_data_from_numpy(_input_ids, binary_data=False)

    attention_mask = tritonhttpclient.InferInput('attention_mask', shape=_attention_mask.shape,
                                                 datatype='INT32')
    attention_mask.set_data_from_numpy(_attention_mask, binary_data=False)

    req_start = tritonhttpclient.InferRequestedOutput('start_logits')
    req_end = tritonhttpclient.InferRequestedOutput('end_logits')

    response = client.infer(
        model_name='question_answering',
        inputs=[input_ids, attention_mask],
        outputs=[req_start, req_end]
    )

    start_logits = response.as_numpy("start_logits").astype(np.float32)
    end_logits = response.as_numpy("end_logits").astype(np.float32)

    start_log_probs = log_softmax(start_logits)
    end_log_probs = log_softmax(end_logits)
    print(str(start_log_probs.argmax(axis=-1)))
    return str(start_log_probs.argmax(axis=-1)), str(end_log_probs.argmax(axis=-1))


def run_tasks(text):

    out_classify = _classify(text)
    start_index, end_index = _qa(text)

    return [out_classify, start_index, end_index]


demo = gr.Interface(
    fn=run_tasks,
    inputs=["text"],
    outputs=["text", "text", "text"],
    allow_flagging="never",
)

demo.launch(server_name='0.0.0.0', server_port=7860)
