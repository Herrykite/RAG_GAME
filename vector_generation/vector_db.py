import json
import os
import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from config import Config


class SentenceTransformerEmbedding:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, input):
        embeddings = self.model.encode(
            input,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        print(f"编码后嵌入形状：{embeddings.shape}")
        return embeddings

class ChromaDB:
    def __init__(self):
        config = Config()
        self.persist_path = config.CHROMA_PERSIST_PATH
        os.makedirs(self.persist_path, exist_ok=True)
        self.expected_dim = config.VECTOR_DIM
        print(f"[ChromaDB] 配置路径：{self.persist_path}")
        print(f"[ChromaDB] 期望维度：{self.expected_dim}")
        print(f"[ChromaDB] 使用模型：{config.SENTENCE_BERT}")
        self.collection_name = "embedded_results"
        self.embedding_model = self._init_embedding_model(config)
        self.client = self._init_client()
        self.collection = self._get_or_create_collection()
        print(f"[ChromaDB] 集合 '{self.collection_name}' 状态：")
        print(f"  - 文档数量：{self.collection.count()}")
        print(f"  - 元数据：{self.collection.metadata}")

    def _init_embedding_model(self, config):
        embedding_model = SentenceTransformerEmbedding(
            model_name=config.SENTENCE_BERT,
            device=config.DEVICE
        )
        test_embedding = embedding_model(["test"])[0]
        print(f"测试嵌入维度：{len(test_embedding)}")  # 新增打印
        if len(test_embedding) != self.expected_dim:
            raise ValueError(f"嵌入模型维度错误：期望{self.expected_dim}，实际{len(test_embedding)}")
        return embedding_model


    def _init_client(self):
        print(f"尝试加载ChromaDB路径: {self.persist_path}")
        try:
            client = chromadb.PersistentClient(path=self.persist_path)
            print(f"成功连接到ChromaDB，集合数量: {len(client.list_collections())}")
            return client
        except Exception as e:
            print(f"初始化失败: {e}")
            raise

    def _get_or_create_collection(self):
        """获取或创建ChromaDB集合"""
        test_embedding = self.embedding_model(["test"])[0]
        if len(test_embedding) != self.expected_dim:
            raise ValueError(f"嵌入模型维度不匹配，期望{self.expected_dim}，得到{len(test_embedding)}")

        try:
            # 获取现有集合
            collection = self.client.get_collection(self.collection_name)

            # 验证现有集合的维度
            if collection.metadata and "dim" in collection.metadata:
                collection_dim = collection.metadata["dim"]
                if collection_dim != self.expected_dim:
                    raise ValueError(f"集合维度不匹配，期望{self.expected_dim}，集合中为{collection_dim}")
            else:
                print("警告：集合未显式指定维度，可能存在不匹配风险")

            return collection

        except NotFoundError:
            print("不存在已有嵌入向量库，创建新的嵌入向量库")
            return self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_model,
                metadata={"hnsw:space": "cosine", "dim": self.expected_dim}
            )

    def _batch_insert(self, entities: List[Dict], batch_size: int = 100):
        """分批插入数据，避免内存溢出"""
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]

            ids = [str(e["entity_id"]) for e in batch]
            texts = [e["text"] for e in batch]
            metadatas = [e["metadata"] for e in batch]

            # 显式生成嵌入
            embeddings = self.embedding_model(texts)
            print(f"插入时嵌入维度：{embeddings.shape[1]}")  # 检查维度

            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings  # 显式传递嵌入
            )

    def insert_vectors(self, chapters: List[Dict]):
        if not chapters:
            return

        entities = []
        for chapter in chapters:
            chapter_id = chapter["chapter_id"]
            scene = chapter["scene"]
            chapter_metadata = chapter.get("metadata", {})  # 提取章节元数据

            for unit in chapter["text_units"]:
                print(f"Unit type: {unit['type']}")
                unit_metadata = unit.get("metadata", {})
                entity_id = f"{chapter_id}_{unit.get('id', len(entities))}"

                scene_objects = unit.get("scene_objects", [])
                scene_objects_str = json.dumps(scene_objects) if scene_objects else ""

                # 合并元数据，优先使用 unit 的值，若无则使用 chapter 的值
                metadata = {
                    "scene": scene,
                    "character": unit.get("character", "旁白"),
                    "source": unit_metadata.get("source", chapter_metadata.get("source", "未知来源")),
                    "emotion": unit.get("emotion", "neutral"),
                    "scene_objects": scene_objects_str,
                    "processing_time": unit_metadata.get("processing_time",
                                                         chapter_metadata.get("processing_time", "unknown")),
                    "model": unit_metadata.get("model", chapter_metadata.get("model", "unknown")),
                    "provider": unit_metadata.get("provider", chapter_metadata.get("provider", "unknown")),
                }

                # 添加 unit 中的其他字段
                if "cleaning_method" in unit:
                    metadata["cleaning_method"] = unit["cleaning_method"]
                if "text_type" in unit:
                    metadata["text_type"] = unit["text_type"]
                if "language" in unit:
                    metadata["language"] = unit["language"]

                entities.append({
                    "entity_id": entity_id,
                    "text": unit["content"],
                    "entity_type": unit["type"],
                    "metadata": metadata
                })

        if entities:
            self._batch_insert(entities)
        return entities

    def get_collection_stats(self):
        """获取集合统计信息（调试用）"""
        return self.collection.count() if self.collection else 0

    def search_vectors(self, query_text: str, top_k: int = 10):
        """向量检索（支持元数据过滤，返回完整信息）"""
        query_embedding = self.embedding_model([query_text])[0]
        print(f"查询文本嵌入维度：{len(query_embedding)}")  # 新增打印
        results = self.collection.query(
            query_embeddings=query_embedding,
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }