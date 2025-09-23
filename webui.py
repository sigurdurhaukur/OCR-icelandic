import re
import time
from threading import Thread

import gradio as gr
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, TextIteratorStreamer

# import subprocess
# subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)


processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    # _attn_implementation="flash_attention_2"
).to("cuda")


def model_inference(
    input_dict,
    history,
    decoding_strategy,
    temperature,
    max_new_tokens,
    repetition_penalty,
    top_p,
):
    text = input_dict["text"]
    print(input_dict["files"])
    if len(input_dict["files"]) > 1:
        images = [Image.open(image).convert("RGB") for image in input_dict["files"]]
    elif len(input_dict["files"]) == 1:
        images = [Image.open(input_dict["files"][0]).convert("RGB")]
    else:
        images = []

    if text == "" and not images:
        gr.Error("Please input a query and optionally image(s).")

    if text == "" and images:
        gr.Error("Please input a text query along the image(s).")

    resulting_messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(len(images))]
            + [{"type": "text", "text": text}],
        }
    ]
    prompt = processor.apply_chat_template(
        resulting_messages, add_generation_prompt=True
    )
    inputs = processor(text=prompt, images=[images], return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }

    assert decoding_strategy in [
        "Greedy",
        "Top P Sampling",
    ]
    if decoding_strategy == "Greedy":
        generation_args["do_sample"] = False
    elif decoding_strategy == "Top P Sampling":
        generation_args["temperature"] = temperature
        generation_args["do_sample"] = True
        generation_args["top_p"] = top_p

    generation_args.update(inputs)
    # Generate
    streamer = TextIteratorStreamer(
        processor, skip_prompt=True, skip_special_tokens=True
    )
    generation_args = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
    generated_text = ""

    thread = Thread(target=model.generate, kwargs=generation_args)
    thread.start()

    yield "..."
    buffer = ""

    for new_text in streamer:
        buffer += new_text
        generated_text_without_prompt = buffer  # [len(ext_buffer):]
        time.sleep(0.01)
        yield buffer


examples = [
    [
        {
            "text": "What art era do these artpieces belong to?",
            "files": ["example_images/rococo.jpg", "example_images/rococo_1.jpg"],
        },
        "Greedy",
        0.4,
        512,
        1.2,
        0.8,
    ],
    [
        {
            "text": "I'm planning a visit to this temple, give me travel tips.",
            "files": ["example_images/examples_wat_arun.jpg"],
        },
        "Greedy",
        0.4,
        512,
        1.2,
        0.8,
    ],
    [
        {
            "text": "What is the due date and the invoice date?",
            "files": ["example_images/examples_invoice.png"],
        },
        "Greedy",
        0.4,
        512,
        1.2,
        0.8,
    ],
    [
        {"text": "What is this UI about?", "files": ["example_images/s2w_example.png"]},
        "Greedy",
        0.4,
        512,
        1.2,
        0.8,
    ],
    [
        {
            "text": "Where do the severe droughts happen according to this diagram?",
            "files": ["example_images/examples_weather_events.png"],
        },
        "Greedy",
        0.4,
        512,
        1.2,
        0.8,
    ],
]
demo = gr.ChatInterface(
    fn=model_inference,
    title="SmolVLM: Small yet Mighty 💫",
    description="Play with [HuggingFaceTB/SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) in this demo. To get started, upload an image and text or try one of the examples. This checkpoint works best with single turn conversations, so clear the conversation after a single turn.",
    examples=examples,
    textbox=gr.MultimodalTextbox(
        label="Query Input", file_types=["image"], file_count="multiple"
    ),
    stop_btn="Stop Generation",
    multimodal=True,
    additional_inputs=[
        gr.Radio(
            ["Top P Sampling", "Greedy"],
            value="Greedy",
            label="Decoding strategy",
            # interactive=True,
            info="Higher values is equivalent to sampling more low-probability tokens.",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=5.0,
            value=0.4,
            step=0.1,
            interactive=True,
            label="Sampling temperature",
            info="Higher values will produce more diverse outputs.",
        ),
        gr.Slider(
            minimum=8,
            maximum=1024,
            value=512,
            step=1,
            interactive=True,
            label="Maximum number of new tokens to generate",
        ),
        gr.Slider(
            minimum=0.01,
            maximum=5.0,
            value=1.2,
            step=0.01,
            interactive=True,
            label="Repetition penalty",
            info="1.0 is equivalent to no penalty",
        ),
        gr.Slider(
            minimum=0.01,
            maximum=0.99,
            value=0.8,
            step=0.01,
            interactive=True,
            label="Top P",
            info="Higher values is equivalent to sampling more low-probability tokens.",
        ),
    ],
    cache_examples=False,
)


demo.launch(debug=True, share=True)
