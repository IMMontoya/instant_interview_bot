import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-trained model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"  # Replace with any Hugging Face model ID
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define chatbot function
def chatbot(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Create Gradio interface
interface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="Chatbot",
    description="A chatbot powered by GPT-2. Ask anything!"
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()
