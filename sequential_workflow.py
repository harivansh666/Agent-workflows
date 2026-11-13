from llm import llm
import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

# This is sequential workflow


def get_text(response) -> str:
    """Handles AIMessage, string, or list of parts from LLM responses."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content).strip()


# define state
class pipelinestate(TypedDict):
    # this is going to be store my all nodes data.
    raw_input: str
    edited_text: str  # jdo Editor node vich jyuga ta. editor node kush na kush return kruga ta on data iss varible vich jyuga. ta  krke bnaya en varible
    script_writer: str
    hinglish: str
    final_output: str


def editor_node(state: pipelinestate) -> dict:
    """Stage 1: Clean up grammar, removes typos, and refines the tone."""
    print("\n-- [Stage 1] Execution Editor Node ---")

    prompt = (
        "you are an expert copyeditor. clean up the following raw text."
        "Fix any grammatical errors, spelling mistakes, and smooth out the transition flow"
        "while keeping the core message intact. Return only the edited text. \n\n "
        f"Text:\n {state['raw_input']}"
    )

    response = llm.invoke(prompt)

    print("Editor node response", get_text(response))
    return {"edited_text": get_text(response)}


def script_writer_node(state: pipelinestate) -> dict:
    """Stage 2: Formats the clean text into an engaging video script style."""
    print("\n-- [Stage 2] Execution Scriptwriter Node ---")

    prompt = (
        "You are a charismatic instagram content creator. take this edited text and transform"
        "it is a higher engaging, punchy, conversational video script hook. Make it sound"
        "like a real person speaking passionately. Return only the script content. \n\n"
        f"Edited Text: \n {state['edited_text']}"
    )
    response = llm.invoke(prompt)
    print("Script writer node response", get_text(response))

    return {"script_writer": get_text(response)}


def translator_node(state: pipelinestate) -> dict:
    """Stage 3: Translates the script into natural flowing Hinglish."""
    print("\n-- [Stage 3] Executing Hinglish Translator Node ---")

    prompt = (
        "You are an expert content localizer for the Indian market. Take the following script "
        "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence-by-sentence "
        "or repeat information. Alternating comfortably between Hindi and English, write it exactly how "
        "an intellectual tech educator would speak naturally on a live stream.\n"
        "Return only the final Hinglish text.\n\n"
        f"Script:\n{state['script_writer']}"
    )

    response = llm.invoke(prompt)

    text = get_text(response)

    print("translator writer node response", text)

    return {
        "hinglish": text,
        "final_output": text,
    }


# Create the graph
graph = StateGraph(pipelinestate)

# Added nodes in our graph
graph.add_node("editor", editor_node)
graph.add_node("scriptWriter", script_writer_node)
graph.add_node("hinglishConverter", translator_node)

# Add edges (sequential - one after other)
graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptWriter")
graph.add_edge("scriptWriter", "hinglishConverter")
graph.add_edge("hinglishConverter", END)

# Compile the graph
app = graph.compile()

response = app.invoke(
    {"raw_input": "make a script on load balancer for instagram reels "}
)

# Print Final Output
print("\n========== FINAL OUTPUT ==========")
print(response["final_output"])
