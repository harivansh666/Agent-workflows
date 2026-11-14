from typing import Annotated
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from typing import TypedDict

from llm import llm

load_dotenv()


def _extract_text(content) -> str:
    """Safely extract plain text from an LLM response content field,
    which may be a str or a list of content-block dicts (newer langchain-google-genai).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"] if isinstance(part, dict) and "text" in part else str(part)
            for part in content
        )
    return str(content)


# ---------------------------------------------------------------------------
embeddingsModel = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

################ [Step1] Building the RAG retrivers #########################


# ---- Lightweight PDF loader -----------------------------------------------
def build_retriver(file_path: str):

    loader = PyMuPDF4LLMLoader(file_path)
    # hle tk ek loader object create hoya aa jisnu pdf da path pta aa bus.

    documents = loader.load()
    # 10 page ki PDF → loader.load() → 10 Documents (1 page = 1 Document)
    # age chhota ho ya bada, ek page = ek Document

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    splitted_documents = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(
        documents=splitted_documents, embedding=embeddingsModel
    )

    return vector_store.as_retriever(search_kwargs={"k": 4})


acadmic_retever = build_retriver("academics_handbook.pdf")
fees_retever = build_retriver("fee_structure.pdf")

################ [Step2] State ##################################
class State(TypedDict):
    program: str  # can be a bcom, bca, bba
    messages: Annotated[list, add_messages]
    query_type: str  # fee| genereal question | acadmic basiclly kis lyi pushna aa
    retrieved_context: str  # jo db vicho lyanda aa


# [Step 3]  Nodes
# clasify node will decide which node to pass to acadmic, fees and other
# iss da kam ek aa ke user ka question kis da aa kon complete kruga task iss user da which node
def classifier_node(state: State) -> dict:
    """Look at the latest user message and decide which path to take."""

    last_message = _extract_text(state["messages"][-1].content)

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = _extract_text(response.content).strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}


def acadmic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academic handbook"""
    query = _extract_text(state["messages"][-1].content)

    docs = acadmic_retever.invoke(query)
    context = "\n \n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}
    

def fees_node(state: State):
    """Retrieves relevant chunks from the fee handbook"""
    query = _extract_text(state["messages"][-1].content)
    docs = fees_retever.invoke(query)
    context = "\n \n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state: State):
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """Generate the final answer, personalized using the students's program"""
    query = _extract_text(state["messages"][-1].content)
    program = state.get("program", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {program} student,"
            f"Answer this question using your own general knowledge: \n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {program} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {program} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )
    response = llm.invoke(prompt)
    return {"messages": [("ai", _extract_text(response.content).strip())]}


# [Step 4] - router function


def router_query(state: State):
    if state["query_type"] == "academic":
        return "academic_rag"
    elif state["query_type"] == "fee":
        return "fee_rag"
    else:
        return "general"


# [Step 5] Building the graph
graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", acadmic_rag_node)
graph.add_node("fee_rag", fees_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)


# [Step 6] Edges

graph.add_edge(
    START, "classifier"
)  ## jo ve question ayuga uss nu pehla me classifier node ko he send krunga
graph.add_conditional_edges(
    "classifier", router_query
)  # classifier response return kruga jis response de help naal asi node select krange using router_query function naal

# hun do ve conditional edge kisi ve node nu run kruga using router_query de help naal ta jehri node chalegi oh koi response ve return krugi. ta iss krke asi edges likhanga agar acadmic_node chali ta oh response node chlavegi, agar fee-node chali ta oh response_node run kregi
#
graph.add_edge("academic_rag", "response")
graph.add_edge("fee_rag", "response")
graph.add_edge("general", "response")
graph.add_edge("response", END)

app = graph.compile()


# [Step 6] Run the code

print("welcome to the College assistant \n\n")

print("which programe are you in ")
print("1. BCA")
print("2. BBA")
print("3. B.com (H)")

choice = input("\n Enter 1,2 or 3: ")


program_map = {"1": "BCA", "2": "BBA", "3": "B.Com (H)"}


student_program = program_map.get(choice, "BCA")

while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit"]:
        break
    else:
        result = app.invoke({"program": student_program, "messages": user_query})
        print(f"Assistant: {result['messages'][-1].content}")
