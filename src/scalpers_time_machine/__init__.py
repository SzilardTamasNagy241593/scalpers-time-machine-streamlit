__version__ = "0.1.0"

from .model import train_model

def greet(name: str) -> str:
    return f"Hello, {name}! Welcome to MLOps."

def square(number: int) -> int:
    return number * number