import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import quote_plus
import os
import re
import time
import hashlib
import pandas as pd
import ast
from dotenv import load_dotenv
from difflib import SequenceMatcher

load_dotenv()

# Page Config
st.set_page_config(
    page_title="Text-to-SQL Chatbot",
    page_icon="database",
    layout="wide"
)

st.title("Text-to-SQL RAG Chatbot")
st.caption("Connect any MySQL database and ask questions in plain English.")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "semantic_cache" not in st.session_state:
    st.session_state.semantic_cache = {}
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_queries": 0,
        "successful_queries": 0,
        "failed_queries": 0,
        "cache_hits": 0,
        "self_healed": 0,
        "avg_response_time": 0.0,
        "response_times": []
    }


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def clean_sql(sql: str) -> str:
    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE)
    match = re.search(r"(SELECT|INSERT|UPDATE|DELETE|WITH)", sql, re.IGNORECASE)
    if match:
        sql = sql[match.start():]
    return sql.strip()


# ── INPUT VALIDATION ──────────────────────────────────────────────────────────

GREET_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|hiya|howdy|namaste|helo|helllo|sup|whatsup|what'?s up|good\s*(morning|evening|afternoon)|how are you|how r u)\s*[\?\!\.]*\s*$",
    re.IGNORECASE
)

GIBBERISH_PATTERN = re.compile(r"^[^a-zA-Z0-9\s]{3,}$")

def is_too_short(text: str) -> bool:
    return len(text.strip()) < 5

def is_greeting(text: str) -> bool:
    return bool(GREET_PATTERNS.match(text.strip()))

def is_gibberish(text: str) -> bool:
    words = text.strip().split()
    if len(words) == 1 and len(words[0]) < 3:
        return True
    if GIBBERISH_PATTERN.match(text.strip()):
        return True
    non_alpha = sum(1 for c in text if not c.isalnum() and c != ' ')
    if len(text) > 0 and (non_alpha / len(text)) > 0.6:
        return True
    return False

def is_off_topic(text: str) -> bool:
    off_topic_keywords = [
        "write a poem", "tell me a joke", "what is life", "weather",
        "news today", "stock price", "movie", "recipe", "translate",
        "capital of", "who is the president", "football", "cricket score",
        "what time is it", "calculate", "math problem"
    ]
    lower = text.lower()
    return any(kw in lower for kw in off_topic_keywords)

def validate_input(question: str) -> tuple[bool, str]:
    if is_too_short(question):
        return False, "Query is too short. Please be more specific, e.g: *'Show all customers'* or *'Total sales last month'*"

    if is_greeting(question):
        return False, "Hello! I am a SQL assistant. Ask something about your database, e.g:\n- *'Show top 10 customers'*\n- *'Total revenue this month'*\n- *'List all products with price > 500'*"

    if is_gibberish(question):
        return False, "Could not understand that. Please write your question clearly in English, e.g: *'How many orders were placed today?'*"

    if is_off_topic(question):
        return False, "This question does not seem related to your database. I can only answer database queries.\n\nExample: *'Show me all users who signed up this week'*"

    db_keywords = [
        "show", "list", "find", "get", "how many", "count", "total",
        "average", "avg", "max", "min", "top", "all", "where", "which",
        "who", "what", "when", "select", "fetch", "display", "give me",
        "tell me", "records", "rows", "data", "table", "column", "from",
        "between", "latest", "recent", "last", "first", "most", "least"
    ]
    lower = question.lower()
    has_db_intent = any(kw in lower for kw in db_keywords)

    if not has_db_intent:
        return False, "This does not look like a database query. Try asking for specific data, e.g:\n- *'Show all orders from last week'*\n- *'Which product has the highest sales?'*"

    return True, ""


# ── SEMANTIC CACHE ────────────────────────────────────────────────────────────

def get_question_hash(question: str) -> str:
    return hashlib.md5(question.strip().lower().encode()).hexdigest()

def semantic_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def find_cache_hit(question: str, threshold: float = 0.85) -> dict | None:
    cache = st.session_state.semantic_cache

    exact_key = get_question_hash(question)
    if exact_key in cache:
        return cache[exact_key]

    for cached_q, cached_entry in cache.items():
        if "original_question" in cached_entry:
            sim = semantic_similarity(question, cached_entry["original_question"])
            if sim >= threshold:
                return cached_entry

    return None

def add_to_cache(question: str, sql: str, result_df, result_str: str):
    key = get_question_hash(question)
    st.session_state.semantic_cache[key] = {
        "original_question": question,
        "sql": sql,
        "result_df": result_df,
        "result_str": result_str
    }


# ── SELF-HEALING SQL ──────────────────────────────────────────────────────────

def build_sql_chain(db, llm):
    template = """You are an expert MySQL data analyst.
Convert the question to a valid MySQL query.

=== DATABASE SCHEMA ===
{schema}

=== STRICT RULES ===
1. Return ONLY the raw SQL query — no explanation, no markdown
2. Single line only
3. Use backticks for column/table names with spaces or reserved words
4. Use LIMIT 100 unless user asks for all
5. Never use SELECT *
6. Only use tables and columns that exist in the schema above

Question: {question}
SQL Query:"""

    prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):
        return db.get_table_info()

    chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def self_healing_execute(db, llm, question: str, max_retries: int = 2):
    chain = build_sql_chain(db, llm)
    raw_sql = chain.invoke({"question": question})
    cleaned_sql = clean_sql(raw_sql)

    last_error = None
    healed = False

    for attempt in range(max_retries + 1):
        try:
            result = db.run(cleaned_sql)

            try:
                data = ast.literal_eval(result)
                df = pd.DataFrame(data)
                return cleaned_sql, df, None, healed, None
            except Exception:
                return cleaned_sql, None, str(result), healed, None

        except Exception as e:
            last_error = str(e)

            if attempt < max_retries:
                fix_template = """You are a MySQL expert. The following SQL query produced an error.
Fix the query and return ONLY the corrected SQL — no explanation, no markdown.

Original Question: {question}

Broken SQL:
{broken_sql}

Error:
{error}

Database Schema:
{schema}

Corrected SQL:"""

                fix_prompt = ChatPromptTemplate.from_template(fix_template)
                fix_chain = fix_prompt | llm | StrOutputParser()

                fixed_raw = fix_chain.invoke({
                    "question": question,
                    "broken_sql": cleaned_sql,
                    "error": last_error,
                    "schema": db.get_table_info()
                })
                cleaned_sql = clean_sql(fixed_raw)
                healed = True

    return cleaned_sql, None, None, healed, last_error


# ── METRICS ──────────────────────────────────────────────────────────────────

def update_metrics(success: bool, response_time: float, cache_hit: bool = False, healed: bool = False):
    m = st.session_state.metrics
    m["total_queries"] += 1
    if success:
        m["successful_queries"] += 1
    else:
        m["failed_queries"] += 1
    if cache_hit:
        m["cache_hits"] += 1
    if healed:
        m["self_healed"] += 1
    m["response_times"].append(response_time)
    m["avg_response_time"] = sum(m["response_times"]) / len(m["response_times"])


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Database Connection")

    host = st.text_input("Host", value="localhost")
    port = st.number_input("Port", value=3306)
    username = st.text_input("Username", value="root")
    password = st.text_input("Password", type="password")
    database = st.text_input("Database Name")
    groq_key = st.text_input("Groq API Key", type="password")

    connect_btn = st.button("Connect to Database", use_container_width=True)

    if connect_btn:
        if not all([host, username, password, database, groq_key]):
            st.error("Please fill all fields.")
        else:
            try:
                with st.spinner("Connecting..."):
                    encoded_password = quote_plus(password)
                    mysql_uri = f"mysql+pymysql://{username}:{encoded_password}@{host}:{int(port)}/{database}"
                    db = SQLDatabase.from_uri(mysql_uri, sample_rows_in_table_info=2)

                    llm = ChatGroq(
                        model="llama-3.1-8b-instant",
                        api_key=groq_key,
                        temperature=0,
                        max_tokens=1024,
                    )

                    st.session_state.db = db
                    st.session_state.llm = llm
                    st.session_state.connected = True
                    st.session_state.tables = db.get_usable_table_names()

                st.success("Connected successfully.")
                st.info(f"Tables: {', '.join(st.session_state.tables)}")

            except Exception as e:
                st.error(f"Connection failed: {str(e)}")

    # Metrics Dashboard
    if "connected" in st.session_state:
        st.divider()
        st.header("Query Metrics")

        m = st.session_state.metrics
        total = m["total_queries"]
        success_rate = (m["successful_queries"] / total * 100) if total > 0 else 0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", total)
            st.metric("Successful", m["successful_queries"])
            st.metric("Cache Hits", m["cache_hits"])
        with col2:
            st.metric("Success Rate", f"{success_rate:.0f}%")
            st.metric("Failed", m["failed_queries"])
            st.metric("Self-Healed", m["self_healed"])

        if total > 0:
            st.metric("Avg Response Time", f"{m['avg_response_time']:.2f}s")

        if st.button("Clear Cache", use_container_width=True):
            st.session_state.semantic_cache = {}
            st.success("Cache cleared.")

        if st.button("Reset Metrics", use_container_width=True):
            st.session_state.metrics = {
                "total_queries": 0, "successful_queries": 0,
                "failed_queries": 0, "cache_hits": 0,
                "self_healed": 0, "avg_response_time": 0.0,
                "response_times": []
            }
            st.success("Metrics reset.")


# ── MAIN CHAT AREA ────────────────────────────────────────────────────────────

if "connected" not in st.session_state:
    st.info("Please connect to a database using the sidebar.")
    st.stop()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sql" in msg:
            with st.expander("Generated SQL"):
                st.code(msg["sql"], language="sql")
        if "table" in msg and msg["table"] is not None:
            st.dataframe(msg["table"], use_container_width=True)
        if "badges" in msg:
            st.caption(msg["badges"])

# Chat input
question = st.chat_input("Ask anything about your data...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        # Step 1: Input Validation
        is_valid, validation_msg = validate_input(question)

        if not is_valid:
            st.warning(validation_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": validation_msg
            })

        else:
            start_time = time.time()

            # Step 2: Semantic Cache Check
            cache_entry = find_cache_hit(question)

            if cache_entry:
                elapsed = time.time() - start_time
                update_metrics(success=True, response_time=elapsed, cache_hit=True)

                st.success("Served from cache.")
                if cache_entry["result_df"] is not None:
                    st.dataframe(cache_entry["result_df"], use_container_width=True)
                else:
                    st.write(cache_entry["result_str"])

                with st.expander("Generated SQL"):
                    st.code(cache_entry["sql"], language="sql")

                st.caption("From cache")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Served from cache.",
                    "sql": cache_entry["sql"],
                    "table": cache_entry["result_df"],
                    "badges": "From cache"
                })

            else:
                # Step 3: Self-Healing SQL Execution
                with st.spinner("Generating SQL..."):
                    try:
                        db = st.session_state.db
                        llm = st.session_state.llm

                        sql, result_df, result_str, healed, error = self_healing_execute(
                            db, llm, question, max_retries=2
                        )

                        elapsed = time.time() - start_time

                        if error:
                            update_metrics(success=False, response_time=elapsed)
                            st.error(f"Query failed after retries.\n\nError: `{error}`")
                            st.info("Try rephrasing your question or check if the table and column names are correct.")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Query failed: {error}"
                            })

                        else:
                            update_metrics(success=True, response_time=elapsed, healed=healed)
                            add_to_cache(question, sql, result_df, result_str)

                            badges_list = [f"Response time: {elapsed:.2f}s"]
                            if healed:
                                badges_list.append("Auto-fixed by self-healing")
                            badges = "  |  ".join(badges_list)

                            if result_df is not None:
                                st.success(f"{len(result_df)} row(s) returned.")
                                st.dataframe(result_df, use_container_width=True)
                            else:
                                st.success("Query executed successfully.")
                                st.write(result_str)

                            with st.expander("Generated SQL"):
                                st.code(sql, language="sql")
                                if healed:
                                    st.info("This SQL was auto-fixed after an initial error.")

                            st.caption(badges)

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Query executed. {len(result_df) if result_df is not None else ''} row(s) returned.",
                                "sql": sql,
                                "table": result_df,
                                "badges": badges
                            })

                    except Exception as e:
                        elapsed = time.time() - start_time
                        update_metrics(success=False, response_time=elapsed)
                        st.error(f"Unexpected error: {str(e)}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {str(e)}"
                        })
