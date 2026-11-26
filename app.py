import asyncio
import queue
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from agent import (
    get_agent,
    get_pending_interrupt,
    get_user_memories,
    load_conversation,
    retrieve_all_threads,
)
from db import get_runtime
from rag_store import ingest_pdf, thread_document_metadata, thread_has_document

runner, _, _, _ = get_runtime()
chatbot, loaded_tools = get_agent()

# ---------------------------------------------------------------------------
# Session Initialisation
# ---------------------------------------------------------------------------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state["user_id"] = "u1"
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

thread_key = str(st.session_state["thread_id"])
if thread_key not in st.session_state["chat_threads"]:
    st.session_state["chat_threads"].append(thread_key)

thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

if "pending_thread_load" in st.session_state:
    target = str(st.session_state.pop("pending_thread_load"))
    st.session_state["thread_id"] = target
    st.session_state["message_history"] = load_conversation(target)
    st.session_state["ingested_docs"].setdefault(target, {})
    st.rerun()

# ---------------------------------------------------------------------------
# Streaming Helper (works for both new turns and Command(resume=...))
# ---------------------------------------------------------------------------
def _run_and_stream(payload, config):
    with st.chat_message("assistant"):
        active_status = {"box": None}
        token_queue: queue.Queue = queue.Queue()

        async def _produce():
            try:
                async for chunk, metadata in chatbot.astream(
                    payload,
                    config=config,
                    stream_mode="messages",
                ):
                    token_queue.put(("item", chunk, metadata))
            except Exception as exc:
                token_queue.put(("error", exc, None))
            finally:
                token_queue.put(("done", None, None))

        asyncio.run_coroutine_threadsafe(_produce(), runner.loop)

        def _stream_ai():
            while True:
                kind, item, metadata = token_queue.get()
                if kind == "done":
                    break
                if kind == "error":
                    raise item

                if isinstance(item, ToolMessage):
                    tool_name = getattr(item, "name", "tool")
                    if active_status["box"] is None:
                        active_status["box"] = st.status(f"🔧 Using `{tool_name}`…", expanded=True)
                    else:
                        active_status["box"].update(label=f"🔧 Using `{tool_name}`…", state="running")

                if (
                    metadata.get("langgraph_node") == "chat_node"
                    and isinstance(item, AIMessage)
                    and isinstance(item.content, str)
                ):
                    yield item.content

        try:
            ai_reply = st.write_stream(_stream_ai())
        except Exception as exc:
            st.error(f"Agent error: {exc}")
            ai_reply = None

        if active_status["box"] is not None:
            active_status["box"].update(label="✅ Tool finished", state="complete", expanded=False)

    if ai_reply:
        st.session_state["message_history"].append({"role": "assistant", "content": ai_reply})


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("LangGraph RAG + MCP")
st.sidebar.text_input("User ID (LTM)", key="user_id")
user_id = st.session_state["user_id"].strip() or "u1"
st.sidebar.markdown(f"**Thread:** `{thread_key}`")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    new_id = str(uuid.uuid4())
    st.session_state["thread_id"] = new_id
    st.session_state["message_history"] = []
    st.session_state["chat_threads"].append(new_id)
    st.rerun()

with st.sidebar.expander("🧠 Stored Memories (Postgres)", expanded=False):
    memories = get_user_memories(user_id)
    if memories:
        for mem in memories:
            st.markdown(f"- {mem}")
    else:
        st.caption("No long-term memories saved yet.")

with st.sidebar.expander(f"🔌 Active Tools ({len(loaded_tools)})", expanded=False):
    for t_name in loaded_tools:
        st.markdown(f"- `{t_name}`")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"], key=f"pdf_{thread_key}")
if uploaded_pdf and (uploaded_pdf.name not in thread_docs or not thread_has_document(thread_key)):
    with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
        try:
            summary = ingest_pdf(uploaded_pdf.getvalue(), thread_id=thread_key, filename=uploaded_pdf.name)
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
        except Exception as exc:
            status_box.update(label=f"❌ Failed: {exc}", state="error")

st.sidebar.subheader("Past conversations")
for tid in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(tid), key=f"side-{tid}"):
        st.session_state["pending_thread_load"] = tid
        st.rerun()

# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------
st.title("🤖 Multi-Utility RAG & MCP Chatbot")

for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

doc_meta = thread_document_metadata(thread_key)
if doc_meta:
    st.caption(f"📄 {doc_meta['filename']} — {doc_meta['chunks']} chunks, {doc_meta['documents']} pages")

config = {
    "configurable": {"thread_id": thread_key, "user_id": user_id},
    "metadata": {"thread_id": thread_key, "user_id": user_id},
}

# ---------------------------------------------------------------------------
# Human-in-the-Loop (HITL) Approval Panel
# ---------------------------------------------------------------------------
pending = get_pending_interrupt(thread_key)
if pending and pending.get("action") == "send_email":
    with st.container(border=True):
        st.warning("⏸️ **Approval Required:** Review or edit this email before sending.")
        edited_to = st.text_input("To", value=pending.get("to", ""), key=f"hitl_to_{thread_key}")
        edited_subject = st.text_input("Subject", value=pending.get("subject", ""), key=f"hitl_sub_{thread_key}")
        edited_body = st.text_area("Body", value=pending.get("body", ""), height=150, key=f"hitl_body_{thread_key}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve & Send", type="primary", use_container_width=True):
                _run_and_stream(
                    Command(
                        resume={
                            "approved": True,
                            "to": edited_to,
                            "subject": edited_subject,
                            "body": edited_body,
                        }
                    ),
                    config,
                )
                st.rerun()
        with col2:
            if st.button("❌ Reject", use_container_width=True):
                _run_and_stream(Command(resume={"approved": False}), config)
                st.rerun()

# Disable chat input while waiting for approval
user_input = st.chat_input(
    "Ask about your PDF, check Gmail, schedule a Google Meet…",
    disabled=bool(pending),
)

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    _run_and_stream({"messages": [HumanMessage(content=user_input)]}, config)

    # If the graph paused at human_review, rerun immediately to display the approval box
    if get_pending_interrupt(thread_key):
        st.rerun()

