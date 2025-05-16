import os
from data_processing.knowledge_graph import KnowledgeGraphBuilder, generate_graph_structure
from data_processing.data_cleaning import DataCleaner
from vector_generation.vector_db import ChromaDB
from retrieval_engine.retrieval_agent import RetrievalEngine
from dialogue_management.dialogue_manager import DialogueManager
from config import Config, init_llm


def main():
    print("正在初始化系统组件...")
    config = Config()
    llm = init_llm(config)

    chroma_path = os.path.join(config.CHROMA_PERSIST_PATH, 'chroma.sqlite3')
    kg_path = config.KG_PERSIST_PATH

    if not os.path.exists(chroma_path) and not os.path.exists(kg_path):
        # 1. 数据清洗与预处理
        print("\n正在加载和清洗数据...")
        script_path = os.path.join("utils", "story.txt")
        with open(script_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        data_cleaner = DataCleaner()
        cleaned_data = data_cleaner.clean_text(raw_text)

        # 2. 向量数据库操作
        print("\n正在构建向量库...")
        vector_db = ChromaDB()
        entities = vector_db.insert_vectors(cleaned_data)
        print(f"向量库构建完成，共有 {vector_db.get_collection_stats()} 个实体")

        # 3. 知识图谱构建
        print("\n正在构建知识图谱...")
        kg_builder = KnowledgeGraphBuilder()
        generate_graph_structure(kg_builder, entities)
    else:
        vector_db = ChromaDB()
        print(f"\n向量库已存在，读取\n{vector_db}")
        kg_builder = KnowledgeGraphBuilder()
        kg_builder.load_graph()
        print(f"\n知识图谱已存在，加载自 {kg_path}，"
              f"包含 {kg_builder.graph.number_of_nodes()} 个节点和 {kg_builder.graph.number_of_edges()} 条边")


    # 4. 示例查询处理
    retrieval_engine = RetrievalEngine(vector_db=vector_db)
    dialogue_manager = DialogueManager(retrieval_engine=retrieval_engine, llm_model=llm)
    print("\n正在处理示例查询...")
    user_queries = [
        {
            "entity_type": "narrative",
            "query": "小刘在故事中都做了什么？"
        },
        {
            "entity_type": "narrative",
            "query": "谁抱着电饭锅内胆，为什么？"
        },
        {
            "entity_type": "dialogue",
            "query": "小魏和谁聊了电路的事？"
        }
    ]

    for case in user_queries:
        print(f"\n处理查询: {case['query']}")
        retrieval_results = retrieval_engine.hybrid_retrieval(
            query_text=case["query"],
            entity_type=case["entity_type"],
            depth=2
        )

        key_entities = []
        for meta in retrieval_results["metadatas"]:
            if "character" in meta:
                key_entities.append(meta["character"])
            elif "scene" in meta:
                key_entities.append(meta["scene"])
        key_entities = list(set(key_entities))

        kg_context = []
        for entity in key_entities:
            entity_network = kg_builder.query_entity_network(
                entity_name=entity,
                depth=1,
                relation_types=["出现在", "情绪状态为", "向对方表示"]  # 匹配图谱关系
            )
            for node in entity_network.get("nodes", []):
                if node["id"] != entity:
                    kg_context.append(f"{entity} 与 {node['name']} 有关联")

        # 生成响应（传递知识图谱上下文）
        response = dialogue_manager.generate_response(
            user_query=case["query"],
            retrieval_results=retrieval_results,
            kg_context=kg_context
        )

        print(f"系统响应: {response}")
        print("支持证据:")
        for evidence in retrieval_results["documents"][:3]:
            print(f"- {evidence}")


if __name__ == "__main__":
    main()