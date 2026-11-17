from numba.core.types import none
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from typing import TypedDict

from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from pymupdf import get_text
from llm import llm

load_dotenv()
    
search_tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

tools = [search_tool]

writer_llm_with_tools = llm.bind_tools(tools) # add krte tools llm naal, Hum LLM ko bata rahe hain ki tumhare paas ye tools available hain.


class State(TypedDict):
  topic : str
  messages: Annotated[list, add_messages]
  draft: str # draft nu review kruga reviewer llm 
  reviewer_feedback: str # reviewer llm feedback
  is_approved: bool # reviewer llm feedback approve ya reject
  attempts: int # number of attempts made to improve the draft

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
    print("\n Writer Node: Generating draft...")
    
    attempt = state.get('attempts', 0) + 1
    topic = state["topic"]
    previous_feedback = state.get("reviewer_feedback", "")
    
    response = writer_llm_with_tools.invoke(WRITER_SYSTEM_PROMPT, state["messages"][-1].content)
    
    if attempt == 1:
      user_message =( f"Write a LinkedIn post about the topic: '{topic}'.",)
    else:
      user_message = (
        f"your prev draf on {topic} was rejected."
        f"Here is the reviewer's feedback: \n {previous_feedback}"
        f"Write a new, improved draft that fixes every issue mentiond"
        f"do not repeat the same mistake"
      )
    
    messages = [("system",WRITER_SYSTEM_PROMPT),("human",user_message)]
    response = writer_llm_with_tools.invoke(messages)
    return {
            "message": [("human", user_message), ("ai", response)], 
            "attempts": attempt,
        }
    
tool_nodes = ToolNode(tools) 

def extract_draft_node(state: State)->dict:
  last_message = state['messages'][-1]
  draft = last_message.content
  
  print(f"\n\n generated post \n {draft} \n ")
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

def reviewer_node(state:State):
  """Reviews the draft and decides: approve or reject with feedback."""
  draft = state["draft"]
  prompt = (
      f"review this LinkedIn post draft : \n"
      f"{draft}\n"
      f"give your reviews"
  )
  
  response = llm.invoke([("system": REVIEWER_SYSTEM_PROMPT),("human": prompt)])
  
  review_text = response.content.strip()

  is_approved = "APPROVED" in review_text.upper().split("FEEDBACK")[0]

# router function 

def should_use_tool(state:State):
  last_message = state['messages'][-1]
  if getattr(last_message, 'tool_calls', None): # last_message de attributes lvage and agar tools_calls de value hai ta return kr da ge "tools" agar false aa gya nahi hai aa gya ta "extract_draft" return kr do.   
    return "tools"
  return "extract_draft"

def should_stop_looping(state: State):
  if state['is_approved']:
    print("post has beed approved")
    return END
  elif state['attempts'] >= 3
    print("reached max attempts")
    return END
  return "writer"

# build the graph 

graph = StateGraph(State)

graph.add_node('writer',writer_node)
graph.add_node('review',reviewer_node)
graph.add_node('tools',tool_nodes)
graph.add_node('extract_draft',extract_draft_node)
graph.add_node('reviewer',reviewer_node)

