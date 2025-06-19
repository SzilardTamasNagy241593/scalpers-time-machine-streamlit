import gradio as gr

def greet(name):
    return f"Hello, {name}!"

gr.Interface(fn=greet, inputs="text", outputs="text").launch(
    server_name="127.0.0.1",
    server_port=7865,
    inbrowser=True,
    share=True
)