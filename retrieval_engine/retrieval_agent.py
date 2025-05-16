from typing import List, Dict, Any
from data_processing.knowledge_graph import KnowledgeGraphBuilder
from config import Config


config = Config()

class RetrievalEngine:
    def __init__(self, vector_db):
        # 权重配置
        self.vector_db = vector_db
        self.graph_relation_weights ={
            '出现在': 0.9,
            '有存在': 1.0,
            '参与': 0.6,
            '向对方表示': 0.6,
            '情绪状态为': 0.9,
            '后续章节': 1.0}
        self.graph_score_decay = 0.9

    def hybrid_retrieval(self, query_text: str, entity_type: str = None, depth: int = 2) -> Dict[str, Any]:
        # 1. 向量检索
        vector_results = self.vector_db.search_vectors(
            query_text,
            top_k=10
        )
        print(f"向量检索结果: {vector_results}")  # 检查是否有文档

        # 结构化处理
        parsed_results = [{
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "source": self._format_source(meta),
            "entity_name": meta.get("character", meta.get("scene"))
        } for doc, meta, dist, ent_name in zip(
            vector_results["documents"],
            vector_results["metadatas"],
            vector_results["distances"],
            [meta.get("character", meta.get("scene")) for meta in vector_results["metadatas"]]
        )]
        print(f"解析后的向量结果数量: {len(parsed_results)}")  # 检查解析后数据量

        # 2. 图谱增强
        graph_results = []
        if entity_type:
            entity_names = [
                item["entity_name"] for item in parsed_results
                if item["metadata"].get("entity_type") == entity_type and item["entity_name"]
            ]
            entity_names = list(set(entity_names))
            print(f"图谱查询实体: {entity_names}")  # 检查实体名是否正确

            for name in entity_names:
                subgraph = KnowledgeGraphBuilder().query_entity_network(
                    entity_name=name,
                    depth=depth,
                    relation_types=list(self.graph_relation_weights.keys())
                )
                graph_results.extend(self._format_graph_nodes(subgraph))
                print(f"图谱结果数量: {len(graph_results)}")  # 检查图谱数据量

        # 3. 智能融合
        combined = self._merge_results(
            parsed_results,
            graph_results,
            vector_weight=0.6,
            graph_weight=0.4
        )
        print(f"合并后结果数量: {len(combined)}")  # 检查合并后数据量

        return {
            "documents": [res["text"] for res in combined],
            "metadatas": [res["metadata"] for res in combined],
            "scores": [res["score"] for res in combined],
            "sources": [res["source"] for res in combined]
        }

    def _format_source(self, meta: dict) -> str:
        """标准化来源格式"""
        source_parts = []
        if meta.get("scene"):
            source_parts.append(f"场景:{meta['scene']}")
        if meta.get("chapter_id"):
            source_parts.append(f"章节:{meta['chapter_id']}")
        return " | ".join(source_parts) or "全局知识库"

    def _format_graph_nodes(self, subgraph: Dict) -> list:
        """图谱节点格式化（适配新结构）"""
        formatted = []
        for node in subgraph.get("nodes", []):
            formatted.append({
                "text": f"[知识图谱]{node.get('name')}({node.get('type')})",
                "metadata": {
                    "entity_id": node["id"],
                    "entity_type": node["type"],
                    **node.get("attributes", {})
                },
                "score": self._calculate_graph_score(node),
                "source": f"图谱路径:{self._format_path(node.get('paths', []))}",
                "distance": node.get("distance", 0)
            })
        return formatted

    def _merge_results(self, vector_res: List, graph_res: List, vector_weight: float, graph_weight: float) -> List:
        """智能融合算法（带归一化处理）"""
        # 向量结果处理（距离转分数）
        vec_scores = [1 - (d / 100) for d in [res["distance"] for res in vector_res]]  # 假设距离范围0-100
        vec_normalized = [s / max(vec_scores) if max(vec_scores) != 0 else 0 for s in vec_scores]

        # 图谱结果处理
        graph_normalized = [s / max([res["score"] for res in graph_res]) if graph_res else []
                            for s in [res["score"] for res in graph_res]]

        # 合并结果
        combined = []
        for i, res in enumerate(vector_res):
            combined.append({
                **res,
                "score": vec_normalized[i] * vector_weight,
                "source_type": "vector"
            })
        for i, res in enumerate(graph_res):
            combined.append({
                **res,
                "score": graph_normalized[i] * graph_weight,
                "source_type": "graph"
            })

        return sorted(combined, key=lambda x: x["score"], reverse=True)[:20]  # 限制结果数量


    def _calculate_graph_score(self, node: Dict) -> float:
        """图谱节点得分计算（优化公式）"""
        distance = node.get("distance", 1)
        decay = self.graph_score_decay ** distance

        relation_score = sum(
            self.graph_relation_weights.get(rel["type"], 0.5) * rel["weight"]
            for path in node.get("paths", [{}])
            for rel in path.get("relations", [])
        )

        type_bonus = {
            "人物": 0.4,
            "事件": 0.3,
            "地点": 0.3
        }.get(node.get("type", "其他"), 0.2)

        return round((relation_score * 0.7 + type_bonus * 0.3) * decay, 2)


    def _format_path(self, paths: List[Dict]) -> str:
        """路径格式化（显示关键关系）"""
        if not paths:
            return "直接关联"
        best_path = max(paths, key=lambda x: x["weight"])
        path_steps = []
        for i, node in enumerate(best_path["path"]):
            if i % 2 == 0:
                path_steps.append(f"{node}")
            else:
                path_steps.append(f"[{node}]")
        return " → ".join(path_steps)



