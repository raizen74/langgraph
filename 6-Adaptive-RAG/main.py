from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from graph.graph import app


if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question":"How to make pizza?"}))
