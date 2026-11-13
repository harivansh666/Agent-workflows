from llm import llm
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, START


def merge_score_dects(existing: dict, new_data: dict) -> dict:
    if existing is None:
        return new_data
    else:
        return {**existing, **new_data}


# existing → Existing dictionary ke saare key-value pairs unpack kiye.
# new_data → New dictionary ke saare key-value pairs unpack kiye.
# {} → In sabko ek new dictionary mein store kiya.


# Create a state
class AnalyzerState(TypedDict):
    raw_text: str
    safety_score: Annotated[dict[str, int], merge_score_dects]


# Nodes


def toxicity_node(state: AnalyzerState) -> dict:
    print("\n Branch 1 analyzing  toxicity and hate speech")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity"
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means higly toxic"
        "return only the plane integer number, nothing else \n\n"
        f"Text: \n {state["raw_text"]}"
    )

    response = llm.invoke(prompt)
    # try - except ek python ka error handling concept like in js  try catch
    try:
        score = int(response.text)
    except ValueError:
        score = 0

    return {"safety_score": {"toxicity_level": score}}


def copyright_node(state: AnalyzerState) -> dict:
    print("Branch 2 analizing copyright & Originality Rists...")

    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response = llm.invoke(prompt)

    try:
        score = int(response.text)
    except ValueError:
        score = 0

    return {"safety_score": {"copyright_risk": score}}


def culture_node(state: AnalyzerState) -> dict:
    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")

    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.text)
    except ValueError:
        score = 0

    return {"safety_score": {"calture_insensitivity": score}}


graph = StateGraph(AnalyzerState)

graph.add_node("toxicity_node", toxicity_node)
graph.add_node("copyright_check", copyright_node)
graph.add_node("culture_node", culture_node)


# Fan-Out
graph.add_edge(START, "toxicity_node")
graph.add_edge(START, "copyright_check")
graph.add_edge(START, "culture_node")

# Fan-in
# graph.add_edge(["toxicity_node", "copyright_check", "culture_node"], END)
#

# Iska matlab:
# - Toxicity node complete hona chahiye.
# - Copyright node complete hona chahiye.
# - Culture node complete hona chahiye.
# Jab teeno complete ho jayenge, tab graph END par jayega.

# use case mostly: ye tb use krna chaiya like jb mujhe 3no nodes ka result, mene kisi ek new node me send krna hai
# and vo 4th node he final result dega
# ====================================================
graph.add_edge("toxicity_node", END)
graph.add_edge("copyright_check", END)
graph.add_edge("culture_node", END)

# Iska matlab:
# - Har node ka apna alag edge hai END tak.
# - Koi bhi node complete ho, wo apne aap END ki taraf badh jata hai.
# - Ek node dusre ka wait nahi karta ("Jo aaya, wo gaya").
#
# Note: Teeno nodes parallel chal rahe hain, to graph tab tak chalta rahega
# jab tak teeno branches khatam na ho jayein. END pe final result same aata hai.
#
# Kab use karein:
# - Jab aage koi real node nahi hai, sirf graph khatam karna hai.
# - Jab results state me pehle se save ho rahe hain.
#
# Fan-in se difference:
# - Fan-in  -> teeno ka wait karta hai, phir ek baar aage badhta hai.
# - Separate -> koi wait nahi, har node apne aap aage badhta hai.
prompt = "asdf qwer zxcv 12345"

initialState = {"raw_text": prompt, "safety_score": {}}

app = graph.compile()
response = app.invoke(initialState)

print("\n========== FINAL OUTPUT ==========")
print(response["safety_score"])
