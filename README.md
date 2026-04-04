markdown# 🤖 Text-to-SQL RAG Chatbot

### _"Ask your database anything — in plain English!"_

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://ramandeep-text-to-sql.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.0-1C3C3C?style=for-the-badge)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F55036?style=for-the-badge)](https://groq.com)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=for-the-badge&logo=mysql)](https://mysql.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)

---

## 🔗 Live Demo

👉 **[https://ramandeep-text-to-sql.streamlit.app/](https://ramandeep-text-to-sql.streamlit.app/)**

Connect **any MySQL database**, ask questions in plain English, and get instant SQL-powered answers!

---

## 📌 What is this project?

A **production-level Generative AI application** that bridges natural language and SQL databases. Non-technical users can query any MySQL database without knowing SQL.

> _"Which customer has placed the most orders?"_
> _"What is the total revenue by sales channel?"_
> _"Show me the top 5 products by total sales."_

---

## 🏗️ Architecture
User Question
│
▼
Schema Fetch (db.get_table_info)
│
▼
Prompt Construction (ChatPromptTemplate)
│
▼
LLM Inference (Groq - LLaMA 3.1 8B)
│
▼
SQL Cleaner (regex parser)
│
▼
MySQL Execution
│
▼
Results → Streamlit UI

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **LLM** | Groq LLaMA 3.1 8B | Free, ultra-fast via LPU hardware |
| **Orchestration** | LangChain 0.3.0 | Industry-standard RAG framework |
| **Database** | MySQL + PyMySQL | Most widely used RDBMS |
| **UI** | Streamlit | Fastest ML app deployment |
| **Security** | python-dotenv | No hardcoded secrets |
| **SQL Parsing** | SQLAlchemy | Production-grade ORM |
| **Deployment** | Streamlit Cloud | Free, GitHub-integrated |

---

## ✨ Key Features

- 🔌 **Dynamic DB Connection** — Connect any MySQL database via sidebar
- 🧠 **Schema-Aware SQL Generation** — LLM gets full schema + sample rows
- 📝 **Production Prompt Engineering** — Strict rules + few-shot examples
- 🧹 **Robust SQL Cleaner** — Strips markdown, extracts clean SQL
- 📊 **Interactive Results** — Pandas DataFrame display
- 🔐 **Security Best Practices** — `.env`, password masking, `quote_plus()`

---

## 📁 Project Structure
Text-To-SQL/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env                # Credentials (NOT on GitHub)
├── .gitignore          # Excludes .env
└── README.md

---

## 🚀 Run Locally
```bash
# 1. Clone
git clone https://github.com/Ramandeep-Singh17/-text-to-sql-chatbot.git
cd -text-to-sql-chatbot

# 2. Install
pip install -r requirements.txt

# 3. Create .env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=your_database
GROQ_API_KEY=your_groq_api_key

# 4. Run
streamlit run app.py
```

---

## 💡 Example Queries

| Question | SQL Operation |
|----------|--------------|
| `Total budget for all products in 2017?` | SUM aggregation |
| `Customer with most orders?` | COUNT + GROUP BY + JOIN |
| `Top 5 products by total sales?` | JOIN + SUM + ORDER BY |
| `Total revenue by sales channel?` | GROUP BY channel |

---

## 🎯 Key Design Decisions

**Why Groq over OpenAI?** Free tier + 10x faster via custom LPU hardware.

**Why `temperature=0`?** SQL needs deterministic output — same question, same SQL every time.

**Why `quote_plus()`?** Special characters like `@` in passwords break MySQL URI — this URL-encodes them.

**Why `sample_rows_in_table_info=2`?** Sample rows give LLM concrete data format context, improving SQL accuracy significantly.

---

## 🗺️ Roadmap

- [ ] SQL Validator — Pre-execution syntax check
- [ ] Self-Healing Loop — Auto-retry on failed queries
- [ ] Multi-turn Memory — Conversation context
- [ ] ChromaDB RAG — Vector schema indexing
- [ ] RAGAS Evaluation — Automated quality metrics
- [ ] PostgreSQL Support

---

## 👨‍💻 Author

**Ramandeep Singh**
Built as a production-level GenAI portfolio project — end-to-end RAG pipeline, LLM integration, and cloud deployment.

---

## 📄 License

MIT License

---

_Built with LangChain + Groq + Streamlit_ 🚀
