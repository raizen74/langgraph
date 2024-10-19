from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from graph.graph import app


if __name__ == "__main__":
    print("Hello Advance RAG")
    print(app.invoke(input={"question":"what is agent memory?"}))
