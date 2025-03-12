from langgraph.graph import StateGraph, END, add_messages
from typing import TypedDict, Annotated, List
from langchain.schema import BaseMessage, HumanMessage
from agents.detail_detective import refine_description
from agents.word_weaver import create_post_content
from agents.customer_engagement_expert import review_and_improve_post
from model import generate_image_description
from PIL import Image
import io

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    image: Image.Image
    text_input: str
    initial_description: str
    refined_description: str
    post_content: str
    improved_post: str

def generate_initial_description(state: State) -> State:
    print("Generating initial description")
    initial_description = generate_image_description(state["image"])
    return {"messages": state["messages"], "image": state["image"], "text_input": state["text_input"], "initial_description": initial_description, "refined_description": None, "post_content": None, "improved_post": None}

def refine_initial_description(state: State) -> State:
    print("Refining initial description")
    refined_description = refine_description(state["initial_description"])
    return {"messages": state["messages"] + [HumanMessage(content=refined_description)], "image": state["image"], "text_input": state["text_input"], "initial_description": state["initial_description"], "refined_description": refined_description, "post_content": None, "improved_post": None}

def generate_post_content(state: State) -> State:
    print("Generating post content")
    post_content = create_post_content(state["refined_description"], state["text_input"])
    return {"messages": state["messages"] + [HumanMessage(content=post_content)], "image": state["image"], "text_input": state["text_input"], "initial_description": state["initial_description"], "refined_description": state["refined_description"], "post_content": post_content, "improved_post": None}

def review_and_improve(state: State) -> State:
    print("Reviewing and improving post content")
    improved_post = review_and_improve_post(state["post_content"])
    return {"messages": state["messages"] + [HumanMessage(content=improved_post)], "image": state["image"], "text_input": state["text_input"], "initial_description": state["initial_description"], "refined_description": state["refined_description"], "post_content": state["post_content"], "improved_post": improved_post}

def main_workflow(image: Image.Image, text_input: str):
    initial_state: State = {
        "messages": [HumanMessage(content=f"Image and text input received.")],
        "image": image,
        "text_input": text_input,
        "initial_description": None,
        "refined_description": None,
        "post_content": None,
        "improved_post": None,
    }

    graph = StateGraph(State)
    graph.add_node("generate_initial_description", generate_initial_description)
    graph.add_node("refine_initial_description", refine_initial_description)
    graph.add_node("generate_post_content", generate_post_content)
    graph.add_node("review_and_improve", review_and_improve)

    graph.set_entry_point("generate_initial_description")
    graph.add_edge("generate_initial_description", "refine_initial_description")
    graph.add_edge("refine_initial_description", "generate_post_content")
    graph.add_edge("generate_post_content", "review_and_improve")
    graph.add_edge("review_and_improve", END)

    chain = graph.compile()
    result = chain.invoke(initial_state)
    return result["improved_post"]