# 中文分词依赖于统计模型
python -m spacy download zh_core_web_lg

# 下载BERT中文基础模型
python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('shibing624/bert4ner-base-chinese')"

# 下载中文句向量模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('shibing624/text2vec-base-chinese')"



rag_name/
├── config.py                  # 配置文件，支持CPU/GPU切换
├── data_processing/
│   ├── text_parser.py         # 文本解析模块（去除了,采用llm代替了，直接集成在data_cleaning.py）
│   ├── knowledge_graph.py     # 知识图谱构建模块
│   └── data_cleaning.py       # 数据清洗与增强模块
├── vector_generation/
│   └── vector_db.py           # 向量数据库操作
├── retrieval_engine/
│   ├── retrieval_agent.py     # 检索与推理引擎
├── dialogue_management/
│   ├── dialogue_manager.py    # 对话管理模块
│   └── multi_agent.py         # 多代理协作模块（暂未加入）
│   └── prompt.py              # 提示词模板
├── evaluation/
│   └── metrics.py             # 评估指标计算
└── entrance.py                # 主程序示例
