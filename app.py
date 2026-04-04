
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import quote_plus
import os
import re
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="Text-to-SQL Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Text-to-SQL RAG Chatbot")
st.caption("Connect any MySQL database and ask questions in plain English!")

# ─── Sidebar - DB Connection ───
with st.sidebar:
    st.header("🔌 Database Connection")
    
    host = st.text_input("Host", value="localhost")
    port = st.number_input("Port", value=3306)
    username = st.text_input("Username", value="root")
    password = st.text_input("Password", type="password")
    database = st.text_input("Database Name")
    groq_key = st.text_input("Groq API Key", type="password")
    
    connect_btn = st.button("🔌 Connect to Database", use_container_width=True)
    
    if connect_btn:
        if not all([host, username, password, database, groq_key]):
            st.error("❌ Please fill all fields!")
        else:
            try:
                with st.spinner("Connecting..."):
                    encoded_password = quote_plus(password)
                    mysql_uri = f"mysql+pymysql://{username}:{encoded_password}@{host}:{int(port)}/{database}"
                    db = SQLDatabase.from_uri(
                        mysql_uri,
                        sample_rows_in_table_info=2
                    )
                    
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
                    
                st.success("✅ Connected!")
                st.info(f"📊 Tables: {', '.join(st.session_state.tables)}")
                    
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")

# ─── Main Area ───
if "connected" not in st.session_state:
    st.info("👈 Please connect to a database first!")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sql" in msg:
            with st.expander("🔍 Generated SQL"):
                st.code(msg["sql"], language="sql")
        if "table" in msg:
            st.dataframe(msg["table"])

# Chat input
question = st.chat_input("Ask anything about your data...")

if question:
    # User message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                db = st.session_state.db
                llm = st.session_state.llm
                
                template = """You are an expert MySQL data analyst.
Convert the question to a MySQL query.

=== DATABASE SCHEMA ===
{schema}

=== STRICT RULES ===
1. Return ONLY the raw SQL query
2. Single line only
3. Use backticks for column names with spaces
4. Use LIMIT 100 unless asked for all
5. Never use SELECT *

Question: {question}
SQL Query:"""

                prompt = ChatPromptTemplate.from_template(template)
                
                def get_schema(_):
                    return db.get_table_info()
                
                def clean_sql(sql):
                    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE)
                    match = re.search(r"(SELECT|INSERT|UPDATE|DELETE|WITH)", sql, re.IGNORECASE)
                    if match:
                        sql = sql[match.start():]
                    return sql.strip()
                
                sql_chain = (
                    RunnablePassthrough.assign(schema=get_schema)
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                raw_sql = sql_chain.invoke({"question": question})
                cleaned_sql = clean_sql(raw_sql)
                result = db.run(cleaned_sql)
                
                import pandas as pd
                import ast
                
                try:
                    data = ast.literal_eval(result)
                    df = pd.DataFrame(data)
                    st.write("✅ Here are the results:")
                    st.dataframe(df, use_container_width=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "✅ Here are the results:",
                        "sql": cleaned_sql,
                        "table": df
                    })
                except:
                    st.write(f"✅ Result: {result}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Result: {result}",
                        "sql": cleaned_sql
                    })
                    
                with st.expander("🔍 Generated SQL"):
                    st.code(cleaned_sql, language="sql")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
