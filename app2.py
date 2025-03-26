import gradio as gr

def chatbot_response(name):
    return "Hello, " + name

# Define the initial message
initial_message = [(
    None,
     "Hello, I am a bot designed to..."
)]

gr.ChatInterface(
    fn=chatbot_response,
    chatbot=gr.Chatbot(value=initial_message),  # Use gr.Chatbot to set the initial message
    title="Chatbot"
).launch()
