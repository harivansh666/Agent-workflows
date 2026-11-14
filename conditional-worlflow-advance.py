"""
College Course Query Agent - LangGraph conditional workflow + Advanced RAG

Flow:
  classify -> (chitchat | out_of_scope | rewrite)
  rewrite  -> retrieve (hybrid vector+BM25, RRF, metadata filter, cross-encoder rerank)
           -> grade_docs -> (generate | rewrite loop | escalate)
  generate -> check_grounding -> (END | regenerate | escalate)

pip install langgraph langchain langchain-anthropic langchain-chroma \
    langchain-huggingface langchain-community langchain-text-splitters \
    sentence-transformers rank_bm25 pypdf chromadb
export ANTHROPIC_API_KEY=...
Put course PDFs/txt in ./data/<COURSE_CODE>/*.pdf  (e.g. ./data/CS101/syllabus.pdf)
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

# ----------------------------- config ---------------------------------
DATA_DIR = Path("./data")
CHROMA_DIR = "./chroma_db"
MAX_REWRITES = 2  # retrieval retry loops
MAX_REGEN = 1  # regeneration after failed grounding check
TOP_K_FETCH = 8
TOP_K_FINAL = 4

llm = init_chat_model("claude-sonnet-4-5", model_provider="anthropic", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")


# ----------------------------- ingestion ------------------------------
def ingest() -> None:
    """Run once. Folder name = course_code metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks: List[Document] = []
    for course_dir in DATA_DIR.iterdir():
        if not course_dir.is_dir():
            continue
        for f in course_dir.iterdir():
            loader = PyPDFLoader(str(f)) if f.suffix == ".pdf" else TextLoader(str(f))
            for d in loader.load():
                d.metadata.update(course_code=course_dir.name.upper(), source=f.name)
                chunks.extend(splitter.split_documents([d]))
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)


@lru_cache(maxsize=1)
def vectorstore() -> Chroma:
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


@lru_cache(maxsize=64)
def bm25_for(course: Optional[str]) -> BM25Retriever:
    raw = vectorstore().get(where={"course_code": course} if course else None)
    docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(raw["documents"], raw["metadatas"])
    ]
    return BM25Retriever.from_documents(docs, k=TOP_K_FETCH)


# ----------------------------- schemas --------------------------------
class Intent(BaseModel):
    intent: Literal["course_info", "chitchat", "out_of_scope"] = Field(
        description="course_info = syllabus, schedule, credits, prerequisites, grading, "
        "faculty, exams, assignments, policies. chitchat = greetings/thanks. "
        "out_of_scope = unrelated to college courses."
    )
    course_code: Optional[str] = Field(
        None, description="e.g. CS101 if mentioned, else null"
    )


class Rewrites(BaseModel):
    queries: List[str] = Field(
        description="1 standalone question + 2 alternative phrasings"
    )


class DocGrade(BaseModel):
    relevant: bool


class Grounding(BaseModel):
    grounded: bool = Field(
        description="True only if every claim is supported by context"
    )


# ----------------------------- state ----------------------------------
class State(TypedDict, total=False):
    question: str
    history: List[str]
    intent: str
    course_code: Optional[str]
    queries: List[str]
    docs: List[Document]
    answer: str
    rewrites: int
    regens: int
    grounded: bool


# ----------------------------- nodes ----------------------------------
def classify(state: State) -> State:
    res = llm.with_structured_output(Intent).invoke(
        f"Chat history:\n{chr(10).join(state.get('history', [])[-6:])}\n\n"
        f"Student message: {state['question']}"
    )
    return {
        "intent": res.intent,
        "course_code": res.course_code.upper() if res.course_code else None,
        "rewrites": 0,
        "regens": 0,
    }


def chitchat(state: State) -> State:
    ans = llm.invoke(
        "You are a friendly college course assistant. Reply briefly and invite a "
        f"course-related question.\nStudent: {state['question']}"
    ).content
    return {
        "answer": ans,
        "history": state.get("history", []) + [f"S: {state['question']}", f"A: {ans}"],
    }


def refuse(state: State) -> State:
    ans = "I can only help with college course questions (syllabus, schedule, exams, grading, etc.)."
    return {"answer": ans}


def rewrite(state: State) -> State:
    retry = state.get("rewrites", 0)
    hint = "Use different keywords/synonyms than before." if retry else ""
    res = llm.with_structured_output(Rewrites).invoke(
        "Rewrite the student's latest question as a standalone search query using the "
        f"chat history, plus 2 alternative phrasings. {hint}\n"
        f"History:\n{chr(10).join(state.get('history', [])[-6:])}\n"
        f"Question: {state['question']}\nPrevious queries: {state.get('queries', [])}"
    )
    return {"queries": res.queries, "rewrites": retry + 1}


def _rrf(ranked_lists: List[List[Document]], k: int = 60) -> List[Document]:
    scores, by_key = {}, {}
    for lst in ranked_lists:
        for rank, d in enumerate(lst):
            key = d.page_content[:200]
            by_key[key] = d
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
    return [by_key[k_] for k_ in sorted(scores, key=scores.get, reverse=True)]


def retrieve(state: State) -> State:
    course = state.get("course_code")
    flt = {"course_code": course} if course else None
    lists: List[List[Document]] = []
    for q in state["queries"]:
        lists.append(vectorstore().similarity_search(q, k=TOP_K_FETCH, filter=flt))
        lists.append(bm25_for(course).invoke(q))
    fused = _rrf(lists)[:12]
    if not fused:
        return {"docs": []}
    scores = reranker.predict([(state["queries"][0], d.page_content) for d in fused])
    top = [d for _, d in sorted(zip(scores, fused), key=lambda x: x[0], reverse=True)]
    return {"docs": top[:TOP_K_FINAL]}


def grade_docs(state: State) -> State:
    docs = state["docs"]
    if not docs:
        return {"docs": []}
    grader = llm.with_structured_output(DocGrade)
    grades = grader.batch(
        [
            f"Question: {state['queries'][0]}\n\nChunk:\n{d.page_content}\n\nIs this chunk relevant?"
            for d in docs
        ]
    )
    return {"docs": [d for d, g in zip(docs, grades) if g.relevant]}


def _context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('course_code')} | {d.metadata.get('source')} p{d.metadata.get('page', '-')}]\n{d.page_content}"
        for d in docs
    )


def generate(state: State) -> State:
    strict = (
        "Be extra strict: include ONLY facts literally present in the context."
        if state.get("regens")
        else ""
    )
    ans = llm.invoke(
        "You are a college course assistant. Answer ONLY from the context. "
        "Cite sources like [CS101 | syllabus.pdf p2]. If something is missing, say so. "
        f"{strict}\n\nContext:\n{_context(state['docs'])}\n\nQuestion: {state['question']}"
    ).content
    return {"answer": ans}


def check_grounding(state: State) -> State:
    res = llm.with_structured_output(Grounding).invoke(
        f"Context:\n{_context(state['docs'])}\n\nAnswer:\n{state['answer']}\n\n"
        "Is every claim in the answer supported by the context?"
    )
    return {
        "grounded": res.grounded,
        "regens": state.get("regens", 0) + (0 if res.grounded else 1),
    }


def finalize(state: State) -> State:
    return {
        "history": state.get("history", [])
        + [f"S: {state['question']}", f"A: {state['answer']}"]
    }


def escalate(state: State) -> State:
    ans = (
        "I couldn't find a reliable answer in the course materials. Please contact the "
        "department office or the course instructor for confirmation."
    )
    return {
        "answer": ans,
        "history": state.get("history", []) + [f"S: {state['question']}", f"A: {ans}"],
    }


# ----------------------------- routers --------------------------------
def route_intent(state: State) -> Literal["chitchat", "refuse", "rewrite"]:
    return {"chitchat": "chitchat", "out_of_scope": "refuse"}.get(
        state["intent"], "rewrite"
    )


def route_after_grade(state: State) -> Literal["generate", "rewrite", "escalate"]:
    if state["docs"]:
        return "generate"
    return "rewrite" if state["rewrites"] <= MAX_REWRITES else "escalate"


def route_after_check(state: State) -> Literal["finalize", "generate", "escalate"]:
    if state["grounded"]:
        return "finalize"
    return "generate" if state["regens"] <= MAX_REGEN else "escalate"


# ----------------------------- graph ----------------------------------
def build_graph():
    g = StateGraph(State)
    for name, fn in [
        ("classify", classify),
        ("chitchat", chitchat),
        ("refuse", refuse),
        ("rewrite", rewrite),
        ("retrieve", retrieve),
        ("grade_docs", grade_docs),
        ("generate", generate),
        ("check_grounding", check_grounding),
        ("finalize", finalize),
        ("escalate", escalate),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_intent)
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade_docs")
    g.add_conditional_edges("grade_docs", route_after_grade)
    g.add_edge("generate", "check_grounding")
    g.add_conditional_edges("check_grounding", route_after_check)
    for end_node in ("chitchat", "refuse", "finalize", "escalate"):
        g.add_edge(end_node, END)
    return g.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    # ingest()  # run once after adding files in ./data
    app = build_graph()
    cfg = {"configurable": {"thread_id": "student-1"}}
    while True:
        q = input("Student: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        out = app.invoke({"question": q}, cfg)
        print("Bot:", out["answer"], "\n")
