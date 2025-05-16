from dialogue_management.prompt import final_query_prompt


class DialogueManager:
    def __init__(self, retrieval_engine, llm_model):
        self.retrieval_engine = retrieval_engine
        self.llm_model = llm_model
        self.context = []  # 对话历史

    def generate_response(self, user_query, retrieval_results, kg_context=None):
        """
        结合检索结果和知识图谱上下文，生成最终回答

        Args:
            user_query: 用户的原始问题
            retrieval_results: 混合检索结果
            kg_context: 知识图谱上下文信息

        Returns:
            格式化后的回答，包含LLM分析结果和来源信息
        """
        # 提取检索结果中的关键实体
        key_entities = self._extract_key_entities(retrieval_results)

        # 调用LLM生成回答
        chain = final_query_prompt | self.llm_model
        llm_response = chain.invoke({
                "user_query": user_query,
                "document_text": retrieval_results,
                "kg_text": kg_context
            })

        # 更新对话历史
        self.context.append({
            "user_query": user_query,
            "response": llm_response,
            "key_entities": key_entities
        })

        return llm_response.content


    def _extract_key_entities(self, retrieval_results):
        """从检索结果中提取关键实体"""
        key_entities = set()
        for meta in retrieval_results.get("metadatas", [{}]):
            if "character" in meta:
                key_entities.add(meta["character"])
            elif "scene" in meta:
                key_entities.add(meta["scene"])
        return list(key_entities)
