import ast
import networkx as nx
import os
import torch
from typing import List, Dict, Optional
from transformers import BertTokenizer, BertForTokenClassification
from config import Config


class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.graph_path = os.path.join(Config().KG_PERSIST_PATH, "knowledge_graph.gml")


    def _validate_graph(self):
        """验证图谱完整性"""
        # 确保所有节点都有type属性
        for node in list(self.graph.nodes()):
            if 'type' not in self.graph.nodes[node]:
                self.graph.nodes[node]['type'] = "未知"

    def load_graph(self):
        try:
            self.graph = nx.read_gml(self.graph_path)
            self._validate_graph()  # 验证图谱完整性
        except FileNotFoundError:
            self.graph = nx.MultiDiGraph()


    def build(self, entities: List[Dict], relations: List[Dict]):
        """构建知识图谱（支持批量操作）"""
        # 添加实体节点
        for entity in entities:
            self.graph.add_node(
                entity["name"],
                type=entity["type"],
                **entity.get("attributes", {})
            )

        # 添加关系边（支持权重和类型）
        for rel in relations:
            self.graph.add_edge(
                rel["subject"],
                rel["object"],
                type=rel["predicate"],
                weight=rel.get("weight", 1.0),
                **rel.get("attributes", {})
            )

    def persist(self):
        """持久化图谱到文件"""
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        nx.write_gml(self.graph, self.graph_path)


    def build_from_entities(self, entities: List[Dict], relations: List[Dict]):
        """直接从实体和关系列表构建图谱"""
        # 添加实体节点
        for entity in entities:
            self.graph.add_node(
                entity["name"],
                type=entity["type"],
                **entity.get("attributes", {})
            )

        # 添加关系边
        for rel in relations:
            # 确保关系的两端都存在于图中
            if rel["subject"] not in self.graph:
                self.graph.add_node(rel["subject"], type="未知")
            if rel["object"] not in self.graph:
                self.graph.add_node(rel["object"], type="未知")

            self.graph.add_edge(
                rel["subject"],
                rel["object"],
                type=rel["predicate"],
                weight=rel.get("weight", 1.0),
                **rel.get("attributes", {})
            )


    def query_entity_network(self, entity_name: str, depth: int = 2,
                             relation_types: Optional[List[str]] = None) -> Dict:
        """查询实体关联网络，增加防御性检查"""
        subgraph = {"nodes": [], "paths": {}}

        # 检查实体是否存在于图中
        if entity_name not in self.graph:
            print(f"警告: 实体 '{entity_name}' 不存在于知识图谱中")
            return subgraph

        visited = {entity_name: {"distance": 0, "paths": []}}
        queue = [(entity_name, 0, [])]

        while queue and depth >= 0:
            current_node, distance, path = queue.pop(0)
            if distance > depth:
                continue

            # 处理出边
            for neighbor, _, edge_data in self.graph.out_edges(current_node, data=True):
                if relation_types and edge_data["type"] not in relation_types:
                    continue

                # 检查邻居节点是否存在于图中
                if neighbor not in self.graph:
                    print(f"警告: 节点 '{neighbor}' 不存在于图中，但在边中被引用")
                    continue

                new_distance = distance + 1
                new_path = path + [(current_node, edge_data["type"], neighbor)]
                edge_weight = edge_data.get("weight", 1.0)

                if neighbor not in visited or new_distance < visited[neighbor]["distance"]:
                    visited[neighbor] = {
                        "distance": new_distance,
                        "paths": [{"steps": new_path, "weight": edge_weight}]
                    }
                    queue.append((neighbor, new_distance, new_path))
                elif new_distance == visited[neighbor]["distance"]:
                    visited[neighbor]["paths"].append({"steps": new_path, "weight": edge_weight})

            # 处理入边
            for neighbor, _, edge_data in self.graph.in_edges(current_node, data=True):
                if relation_types and edge_data["type"] not in relation_types:
                    continue

                # 检查邻居节点是否存在于图中
                if neighbor not in self.graph:
                    print(f"警告: 节点 '{neighbor}' 不存在于图中，但在边中被引用")
                    continue

                new_distance = distance + 1
                new_path = path + [(neighbor, edge_data["type"], current_node)]
                edge_weight = edge_data.get("weight", 1.0)

                if neighbor not in visited or new_distance < visited[neighbor]["distance"]:
                    visited[neighbor] = {
                        "distance": new_distance,
                        "paths": [{"steps": new_path, "weight": edge_weight}]
                    }
                    queue.append((neighbor, new_distance, new_path))
                elif new_distance == visited[neighbor]["distance"]:
                    visited[neighbor]["paths"].append({"steps": new_path, "weight": edge_weight})

        # 构建节点数据（增加节点存在性检查）
        for node, data in visited.items():
            if node in self.graph:
                node_type = self.graph.nodes[node].get("type", "未知")
                subgraph["nodes"].append({
                    "id": node,
                    "name": node,
                    "type": node_type,
                    "distance": data["distance"],
                    "paths": data["paths"][:3]  # 最多保留3条路径
                })
            else:
                print(f"警告: 节点 '{node}' 不在图谱中，但在遍历结果中")

        return subgraph


def extract_entities(unit_content: str) -> list:
    """使用BERT模型从文本中提取实体"""
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForTokenClassification.from_pretrained("bert-base-chinese")

    inputs = tokenizer(unit_content, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=2)[0].tolist()

    id2label = {
        0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
        5: "B-LOC", 6: "I-LOC", 7: "B-PRO", 8: "I-PRO", 9: "B-TIME", 10: "I-TIME"
    }

    entities = []
    current_entity = None
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    for token, pred in zip(tokens, predictions):
        if token in ["[CLS]", "[SEP]", "[PAD]"]: continue
        label = id2label[pred]
        if label.startswith("B-"):
            if current_entity is not None: entities.append(current_entity)
            entity_type = label.split("-")[1]
            current_entity = {
                "name": token.replace("##", ""),
                "type": entity_type,
                "attributes": {"confidence": 0.9}
            }
        elif label.startswith("I-") and current_entity:
            entity_type = label.split("-")[1]
            if entity_type == current_entity["type"]:
                current_entity["name"] += token.replace("##", "")
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
    if current_entity is not None: entities.append(current_entity)

    merged_entities = []
    for entity in entities:
        if merged_entities and entity["type"] == merged_entities[-1]["type"]:
            merged_entities[-1]["name"] += " " + entity["name"]
        else:
            merged_entities.append(entity)
    return merged_entities


def generate_graph_structure_with_bert(cleaned_data, kg_builder):
    entities, relations = [], []
    entity_names = set()  # 用于跟踪已添加的实体名称

    # 处理章节级实体和关系
    for chapter in cleaned_data:
        chapter_id = chapter["chapter_id"]
        scene = chapter["scene"]

        # 添加章节实体
        entities.append({
            "name": chapter_id,
            "type": "章节",
            "attributes": {"scene": scene}
        })
        entity_names.add(chapter_id)

        # 添加章节-场景关系
        relations.append({
            "subject": chapter_id,
            "object": scene,
            "predicate": "有存在",
            "weight": 1.0
        })

    # 处理文本单元级实体和关系
    for chapter in cleaned_data:
        scene = chapter["scene"]
        for unit in chapter["text_units"]:
            unit_content = unit.get("content", "")

            # 提取文本中的实体（使用BERT或其他方法）
            unit_entities = extract_entities(unit_content)

            # 处理scene_objects
            scene_objects = unit.get("scene_objects", [])
            for obj in scene_objects:
                if obj not in entity_names:
                    entities.append({
                        "name": obj,
                        "type": "实体",
                        "attributes": {"source": "scene_objects"}
                    })
                    entity_names.add(obj)

            # 处理角色
            character = unit.get("character", "未知角色")
            if character != "未知角色" and character not in entity_names:
                entities.append({
                    "name": character,
                    "type": "人物",
                    "attributes": {"source": "character"}
                })
                entity_names.add(character)

            # 建立关系：角色-场景
            if character != "未知角色":
                relations.append({
                    "subject": character,
                    "object": scene,
                    "predicate": "出现在",
                    "weight": 0.9
                })

            # 建立关系：场景对象-场景
            for obj in scene_objects:
                relations.append({
                    "subject": obj,
                    "object": scene,
                    "predicate": "出现在",
                    "weight": 0.9
                })

            # 建立关系：角色-情感
            if "emotion" in unit and character != "未知角色":
                emotion = unit["emotion"]
                if emotion not in entity_names:
                    entities.append({
                        "name": emotion,
                        "type": "情感",
                        "attributes": {"source": "emotion"}
                    })
                    entity_names.add(emotion)
                relations.append({
                    "subject": character,
                    "object": emotion,
                    "predicate": "情绪状态为",
                    "weight": 0.9
                })

    # 建立章节顺序关系
    chapters_sorted = sorted(cleaned_data, key=lambda x: int(x["chapter_id"]))
    for i in range(len(chapters_sorted) - 1):
        current_id = chapters_sorted[i]["chapter_id"]
        next_id = chapters_sorted[i + 1]["chapter_id"]
        relations.append({
            "subject": current_id,
            "object": next_id,
            "predicate": "后续章节",
            "weight": 1.0
        })

    # 建立同一场景角色对话关系
    scene_character_map = {}
    for chapter in cleaned_data:
        scene = chapter["scene"]
        characters = set()
        for unit in chapter["text_units"]:
            character = unit.get("character", "旁白")
            if character != "旁白":
                characters.add(character)
        if scene in scene_character_map:
            scene_character_map[scene].update(characters)
        else:
            scene_character_map[scene] = characters

    for scene, characters in scene_character_map.items():
        characters = list(characters)
        for i in range(len(characters)):
            for j in range(i + 1, len(characters)):
                # 双向对话关系
                relations.append({
                    "subject": characters[i],
                    "object": characters[j],
                    "predicate": "向对方表示",
                    "weight": 0.6
                })
                relations.append({
                    "subject": characters[j],
                    "object": characters[i],
                    "predicate": "向对方表示",
                    "weight": 0.6
                })

    # 构建并持久化图谱
    kg_builder.build(entities, relations)
    kg_builder.persist()
    print(f"知识图谱构建完成，包含 {len(entities)} 个实体和 {len(relations)} 个关系")


def generate_graph_structure(kg_builder, entities):
    relations = []
    entity_names = set()  # 跟踪已添加的实体

    for metadata in entities:
        # 处理人物实体
        character = metadata['metadata'].get("character")
        if character and character not in entity_names:
            entity_names.add(character)

        # 处理场景实体
        scene = metadata['metadata'].get("scene")
        if scene and scene not in entity_names:
            entity_names.add(scene)

        # 处理场景物体
        scene_objects_str = metadata['metadata'].get("scene_objects", "[]")
        scene_objects = [obj.strip('"').strip() for obj in scene_objects_str.strip('[]').split(',') if obj.strip()]
        for obj in scene_objects:
            if obj and obj not in entity_names:
                entity_names.add(obj)

    entities_data = []
    # 再次遍历构建实体和关系
    for metadata in entities:
        character = metadata['metadata'].get("character")
        scene = metadata['metadata'].get("scene")

        # 构建实体

        if character and character not in [e["name"] for e in entities_data if e.get("type") == "人物"]:
            entities_data.append({
                "name": character,
                "type": "人物",
                "attributes": {k: v for k, v in metadata['metadata'].items() if k != "character"}
            })
        if scene and scene not in [e["name"] for e in entities_data if e.get("type") == "场景"]:
            entities_data.append({
                "name": scene,
                "type": "场景",
                "attributes": {k: v for k, v in metadata['metadata'].items() if k != "scene"}
            })
        scene_objects_str = metadata['metadata'].get("scene_objects", "[]")
        scene_objects = [obj.strip('"').strip() for obj in scene_objects_str.strip('[]').split(',') if obj.strip()]
        for obj in scene_objects:
            if obj and obj not in [e["name"] for e in entities_data if e.get("type") == "物体"]:
                entities_data.append({
                    "name": obj,
                    "type": "物体",
                    "attributes": {k: v for k, v in metadata['metadata'].items() if k != "scene_objects"}
                })

        # 构建关系：人物 - 场景
        if character and scene:
            relations.append({
                "subject": character,
                "object": scene,
                "predicate": "出现在",
                "weight": 0.9
            })

    # 构建图谱
    kg_builder.build_from_entities(entities_data, relations)
    kg_builder.persist()
    print(f"知识图谱构建完成，包含 {len(entities_data)} 个实体和 {len(relations)} 个关系")
