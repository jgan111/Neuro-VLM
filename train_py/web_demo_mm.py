# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import re
from PIL import Image
from urllib.parse import urlparse
import os
from argparse import ArgumentParser
from threading import Thread

import gradio as gr
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextIteratorStreamer

DEFAULT_CKPT_PATH = '/home/zhangxw/Qwen2.5-VL-32B/'


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default=DEFAULT_CKPT_PATH,
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--cpu-only', action='store_true', help='Run demo with CPU only')

    parser.add_argument('--flash-attn2',
                        action='store_true',
                        default=False,
                        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=7860, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')

    args = parser.parse_args()
    return args


def _load_model_processor(args):
    if args.cpu_only:
        device_map = 'cpu'
    else:
        device_map = 'auto'

    # Check if flash-attn2 flag is enabled and load model accordingly
    if args.flash_attn2:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.checkpoint_path,
                                                                torch_dtype='auto',
                                                                attn_implementation='flash_attention_2',
                                                                device_map=device_map)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.checkpoint_path, device_map=device_map)

    processor = AutoProcessor.from_pretrained(args.checkpoint_path,use_fast=True) 
    if "<image>" not in processor.tokenizer.special_tokens_map.get("additional_special_tokens", []):
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        model.resize_token_embeddings(len(processor.tokenizer))
    processor.chat_template = """{% for message in messages %} 
<|im_start|>{{ message['role'] }} 
{% for item in message['content'] %} 
{% if item['type'] == 'image' %} 
<image> 
{% elif item['type'] == 'text' %} 
{{ item['text'] }} 
{% endif %} 
{% endfor %} 
<|im_end|> 
{% endfor %} 
<|im_start|>assistant 
"""
    return model, processor


def _parse_text(text):
    lines = text.split('\n')
    lines = [line for line in lines if line != '']
    count = 0
    for i, line in enumerate(lines):
        if '```' in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = '<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace('`', r'\`')
                    line = line.replace('<', '&lt;')
                    line = line.replace('>', '&gt;')
                    line = line.replace(' ', '&nbsp;')
                    line = line.replace('*', '&ast;')
                    line = line.replace('_', '&lowbar;')
                    line = line.replace('-', '&#45;')
                    line = line.replace('.', '&#46;')
                    line = line.replace('!', '&#33;')
                    line = line.replace('(', '&#40;')
                    line = line.replace(')', '&#41;')
                    line = line.replace('$', '&#36;')
                lines[i] = '<br>' + line
    text = ''.join(lines)
    return text


def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)


def _is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message['content']:
            if 'image' in item: 
                new_content.append({'type': 'image', 'image': item['image']})
                new_content.append({'type': 'text', 'text': '<image>'}) 
            elif 'text' in item:
                new_content.append({'type': 'text', 'text': item['text']})
            elif 'video' in item:
                new_content.append({'type': 'video', 'video': item['video']})

        new_message = {'role': message['role'], 'content': new_content}
        transformed_messages.append(new_message)

    return transformed_messages


def _launch_demo(args, model, processor):

    def call_local_model(model, processor, messages):
 
        image_inputs = []
        for message in messages:
            for part in message["content"]: 
                print("Inspecting part:", part)
                if isinstance(part, dict) and "image" in part:
                    image_url = part["image"]
                    if image_url.startswith("file://"):
                        local_path = urlparse(image_url).path
                    else: 
                        local_path = image_url 
                    try: 
                        img = Image.open(local_path).convert("RGB")
                        image_inputs.append(img)
                    except Exception as e: 
                        print(f"[Error] Could not load image: {local_path}, error: {e}") 
        if len(image_inputs) == 0:
            raise ValueError("No valid images were found in the input.")
        transformed_messages = _transform_messages(messages)
        text = processor.tokenizer.apply_chat_template(transformed_messages, tokenize=False, add_generation_prompt=True) 
        print(f"Generated input text:\n{text}")
        print(f"Tokenized '<image>': {processor.tokenizer.tokenize('<image>')}")
        inputs = processor(text=text, images=image_inputs, return_tensors="pt")
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict( 
            **inputs,
            streamer=streamer, 
            max_new_tokens=1024, 
            do_sample=True,
            temperature=0.7, 
            top_p=0.95,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def create_predict_fn():

        def predict(_chatbot, task_history):
            nonlocal model, processor
            chat_query = _chatbot[-1][0]
            query = task_history[-1][0]
            if len(chat_query) == 0:
                _chatbot.pop()
                task_history.pop()
                return _chatbot
            print('User: ' + _parse_text(query))
            history_cp = copy.deepcopy(task_history)
            full_response = ''
            messages = []
            content = []
            for q, a in history_cp:
                if isinstance(q, (tuple, list)):
                    if _is_video_file(q[0]):
                        content.append({'video': f'file://{q[0]}'})
                    else:
                        content.append({'image': f'file://{q[0]}'})
                else:
                    content.append({'text': q})
                    messages.append({'role': 'user', 'content': content})
                    messages.append({'role': 'assistant', 'content': [{'text': a}]})
                    content = []
            messages.pop()

            for response in call_local_model(model, processor, messages):
                _chatbot[-1] = (_parse_text(chat_query), _remove_image_special(_parse_text(response)))

                yield _chatbot 
                full_response += response

            task_history[-1] = (query, full_response)
            print('Qwen-VL-Chat: ' + _parse_text(full_response))
            yield _chatbot

        return predict

    def create_regenerate_fn():

        def regenerate(_chatbot, task_history):
            nonlocal model, processor
            if not task_history:
                return _chatbot
            item = task_history[-1]
            if item[1] is None:
                return _chatbot
            task_history[-1] = (item[0], None)
            chatbot_item = _chatbot.pop(-1)
            if chatbot_item[0] is None:
                _chatbot[-1] = (_chatbot[-1][0], None)
            else:
                _chatbot.append((chatbot_item[0], None))
            _chatbot_gen = predict(_chatbot, task_history)
            for _chatbot in _chatbot_gen:
                yield _chatbot

        return regenerate

    predict = create_predict_fn()
    regenerate = create_regenerate_fn()

    def add_text(history, task_history, text):
        task_text = text
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ''

    def add_file(history, task_history, file):
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def reset_user_input():
        return gr.update(value='')

    def reset_state(_chatbot, task_history):
        task_history.clear()
        _chatbot.clear()
        _gc()
        return []

    with gr.Blocks() as demo:
        gr.Markdown("""\
<p align="center"><img src="https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png" style="height: 80px"/><p>"""
                   )
        gr.Markdown("""<center><font size=8>Qwen2.5-VL</center>""")
        gr.Markdown("""\
<center><font size=3>This WebUI is based on Qwen2.5-VL, developed by Alibaba Cloud.</center>""")
        gr.Markdown("""<center><font size=3>本WebUI基于Qwen2.5-VL。</center>""")

        chatbot = gr.Chatbot(label='Qwen2.5-VL', elem_classes='control-height', height=500)
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            addfile_btn = gr.UploadButton('📁 Upload (上传文件)', file_types=['image', 'video'])
            submit_btn = gr.Button('🚀 Submit (发送)')
            regen_btn = gr.Button('🤔️ Regenerate (重试)')
            empty_bin = gr.Button('🧹 Clear History (清除历史)')

        submit_btn.click(add_text, [chatbot, task_history, query],
                         [chatbot, task_history, query]).then(predict, [chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [chatbot, task_history], [chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

        gr.Markdown("""\
<font size=2>Note: This demo is governed by the original license of Qwen2.5-VL. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Qwen2.5-VL的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue().launch(
        share=True,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    model, processor = _load_model_processor(args)
    _launch_demo(args, model, processor)


if __name__ == '__main__':
    main()
