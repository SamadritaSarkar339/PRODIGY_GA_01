# 1. Install dependencies
!pip install -q gradio
!pip install -q git+https://github.com/huggingface/transformers.git

# 2. Import necessary libraries
import gradio as gr
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# 3. Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# 4. Define the text generation function
def generate_text(inp):
    input_ids = tokenizer.encode(inp, return_tensors='tf')
    beam_output = model.generate(
        input_ids,
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return " ".join(output.split(" ")[:-1])

# 5. Create Gradio interface
output_text = gr.outputs.Textbox()
gr.Interface(
    generate_text,
    "textbox",
    output_text,
    title="GPT-2",
    description="OpenAI’s GPT-2 is an unsupervised language model that can generate coherent text. Go ahead and input a sentence and see what it completes!"
).launch()
