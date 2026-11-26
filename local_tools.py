import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from config import ALPHA_VANTAGE_KEY, REQUEST_TIMEOUT
from rag_store import get_retriever, thread_document_metadata

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform an arithmetic operation (add, sub, mul, div, pow, mod) on two numbers."""
    ops = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b if b != 0 else None,
        "pow": lambda a, b: a ** b if abs(b) <= 1000 else None,
        "mod": lambda a, b: a % b if b != 0 else None,
    }
    operation = operation.strip().lower()
    if operation not in ops:
        return {"error": f"Unsupported operation '{operation}'. Choose from: {list(ops)}"}
    try:
        result = ops[operation](first_num, second_num)
        if result is None:
            return {"error": "Division/modulo by zero or exponent too large."}
        return {"result": result}
    except Exception as exc:
        return {"error": str(exc)}


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price for a ticker symbol (e.g. 'AAPL', 'TSLA')."""
    if not ALPHA_VANTAGE_KEY:
        return {"error": "ALPHA_VANTAGE_KEY is not set in .env."}
    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol.strip().upper(), "apikey": ALPHA_VANTAGE_KEY},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # Alpha Vantage returns HTTP 200 with "Note"/"Information" when rate-limited or key is invalid
        if "Note" in data or "Information" in data or "Error Message" in data:
            return {"error": data.get("Note") or data.get("Information") or data.get("Error Message")}
        quote = data.get("Global Quote")
        if not quote:
            return {"error": f"No data found for symbol '{symbol}'."}
        return {
            "symbol": quote.get("01. symbol"),
            "price": quote.get("05. price"),
            "change": quote.get("09. change"),
            "change_percent": quote.get("10. change percent"),
            "latest_trading_day": quote.get("07. latest trading day"),
        }
    except (requests.RequestException, ValueError) as exc:
        return {"error": str(exc)}


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """Retrieve relevant passages from the PDF(s) uploaded for this chat thread."""
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    retriever = get_retriever(str(thread_id) if thread_id else None)
    if retriever is None:
        return {"error": "No document indexed for this chat. Please upload a PDF first."}
    try:
        results = retriever.invoke(query)
        return {
            "query": query,
            "context": [
                {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source"),
                    "page": (doc.metadata.get("page", -1) + 1) if isinstance(doc.metadata.get("page"), int) else None,
                }
                for doc in results
            ],
            "source_file": thread_document_metadata(str(thread_id)).get("filename"),
        }
    except Exception as exc:
        return {"error": f"Retrieval failed: {exc}"}


LOCAL_TOOLS = [search_tool, get_stock_price, calculator, rag_tool]
