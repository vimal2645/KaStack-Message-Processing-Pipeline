# ⚡ KaStack — Message Processing Pipeline

> A Kafka-inspired message processing pipeline built in Python. Handles high-throughput message queuing, parallel worker processing, and REST API for pipeline control and monitoring.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![Message Queue](https://img.shields.io/badge/Message-Queue-orange) ![REST API](https://img.shields.io/badge/REST-API-green) ![DevOps](https://img.shields.io/badge/DevOps-Pipeline-blue)

## 🎯 What It Does
KaStack is a lightweight message queue system that:
1. **Produces** messages via REST API
2. **Queues** them in an ordered topic-based queue
3. **Processes** with parallel Python workers
4. **Monitors** pipeline health via REST status endpoints

## 🔄 Architecture
```
Producer (REST API)
    ↓
Message Queue (Topic-based)
    ↓
Worker Pool (Parallel processors)
    ↓
Consumer / Output Store
    ↓
Monitor API (Status & metrics)
```

## 📁 Structure
```
KaStack-Message-Processing-Pipeline/
├── producer/           # Message producer (REST API)
├── consumer/           # Message consumer workers
├── queue/              # Core queue implementation
├── api/                # REST API for pipeline control
├── config/             # Pipeline configuration
└── requirements.txt
```

## ⚙️ Setup
```bash
pip install -r requirements.txt
python main.py --workers 4 --topic my-topic
```

## 🔑 Features
- ✅ Topic-based message routing
- ✅ Parallel worker processing
- ✅ REST API for produce/consume/monitor
- ✅ Dead-letter queue for failed messages
- ✅ Configurable worker pool size
- ✅ Message persistence

## 🛠️ Tech Stack
`Python` `FastAPI/Flask` `Threading` `REST API` `JSON`

## 💼 Use Cases
- ETL pipeline orchestration
- Background job processing
- Event-driven microservices
- Data ingestion automation

---
[LinkedIn](https://linkedin.com/in/vimalprakash26) | [GitHub](https://github.com/vimal2645)
