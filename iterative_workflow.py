from dotenv import load_dotenv

load_dotenv()  # imports se pehle, taaki llm.py ko env keys mil jayein

from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from llm import llm


def _extract_text(content) -> str:
    """Safely extract plain text from an LLM response or message content field,
    which may be a str or a list of content-block dicts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif hasattr(part, "text"):
                parts.append(getattr(part, "text"))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)

search_tool = TavilySearch(max_results=5, topic="general")
tools = [search_tool]
writer_llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    reviewer_feedback: str
    is_approved: bool
    attempts: int


WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)


def writer_node(state: State) -> dict:
    print("\nWriter Node: Generating draft...")

    topic = state["topic"]
    history = state["messages"]
    new_msgs = []

    if not history:
        # first attempt
        new_msgs.append(
            HumanMessage(f"Write a LinkedIn post about the topic: '{topic}'.")
        )
    elif isinstance(history[-1], AIMessage):
        # previous draft was rejected -> feedback do
        new_msgs.append(
            HumanMessage(
                f"Your previous draft on '{topic}' was rejected.\n"
                f"Reviewer feedback:\n{state['reviewer_feedback']}\n\n"
                "Write a new, improved draft that fixes every issue mentioned. "
                "Do not repeat the same mistakes."
            )
        )
    # else: last message ToolMessage hai -> search results ke saath aage badho

    response = writer_llm_with_tools.invoke(
        [SystemMessage(WRITER_SYSTEM_PROMPT)] + history + new_msgs
    )
    return {"messages": new_msgs + [response]}


tool_node = ToolNode(tools)


def extract_draft_node(state: State) -> dict:
    draft = _extract_text(state["messages"][-1].content)
    print(f"\n\nGenerated post:\n{draft}\n")
    return {"draft": draft}


REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim — uses short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
)


def reviewer_node(state: State) -> dict:
    prompt = (
        f"Review this LinkedIn post draft:\n\n{state['draft']}\n\nGive your review."
    )
    response = llm.invoke(
        [
            ("system", REVIEWER_SYSTEM_PROMPT),
            ("human", prompt),
        ]
    )

    review_text = _extract_text(response.content).strip()
    verdict_part = review_text.upper().split("FEEDBACK")[0]
    is_approved = "APPROVED" in verdict_part and "REJECTED" not in verdict_part

    print(f"\nReviewer: {'APPROVED' if is_approved else 'REJECTED'}\n{review_text}\n")

    return {
        "is_approved": is_approved,
        "reviewer_feedback": review_text,
        "attempts": state.get("attempts", 0) + 1,
    }


# routers
def should_use_tool(state: State):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "extract_draft"


def should_stop_looping(state: State):
    if state["is_approved"]:
        print("Post has been approved")
        return END
    if state["attempts"] >= 3:
        print("Reached max attempts")
        return END
    return "writer"


# build graph
graph = StateGraph(State)

graph.add_node("writer", writer_node)
graph.add_node("tools", tool_node)
graph.add_node("extract_draft", extract_draft_node)
graph.add_node("reviewer", reviewer_node)

graph.add_edge(START, "writer")
graph.add_conditional_edges("writer", should_use_tool, ["tools", "extract_draft"])
graph.add_edge("tools", "writer")  # search ke baad wapas writer
graph.add_edge("extract_draft", "reviewer")
graph.add_conditional_edges("reviewer", should_stop_looping, ["writer", END])

app = graph.compile()


print("=" * 55)
print("Welcome to the LinkedIn Post Generator")
print("=" * 55)
print("\nThis tool will draft a LinkedIn post for you, review it")
print("itself, and iterate until it's publish-ready.")
print("=" * 55)

topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

if not topic:
    print("\nNo topic given. Exiting.")
else:
    print("\nStarting generation...\n")

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "reviewer_feedback": "",
        "is_approved": False,
        "attempts": 0,
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(final_state["draft"])
    print("=" * 55)
    print(f"Total attempts: {final_state['attempts']}")
    print(f"Approved: {final_state['is_approved']}")
