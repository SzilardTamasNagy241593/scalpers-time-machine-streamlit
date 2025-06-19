import gradio as gr

def greet(name):
    return f"Hello {name}! Welcome to Scalper’s Time Machine 🕰️"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()