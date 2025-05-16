from langchain_core.prompts import PromptTemplate


# 1. 知识图谱构建Prompt
kg_construction_prompt = PromptTemplate(
    input_variables=["query", "context_info"],
    template="""
    你是一位知识图谱构建专家。根据以下上下文信息和用户问题，提取实体和关系，构建知识图谱。

    上下文信息: {context_info}
    用户问题: {query}

    请以JSON格式返回以下信息:
    1. 实体列表（包括角色、物品、地点等）
    2. 实体之间的关系
    3. 关键事件

    格式:
    {{
        "entities": ["实体1", "实体2", "实体3"],
        "relations": [
            {{
                "subject": "实体1",
                "predicate": "关系类型",
                "object": "实体2"
            }}
        ],
        "events": ["事件1", "事件2"]
    }}
    """
)

# 2. 章节提取Prompt
chapter_extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    请从以下文本中提取所有章节信息，包括章节编号、章节标题和章节内容。
    如果文本没有明确的章节划分，请按逻辑段落进行拆分。

    输出格式（JSON数组）：
    [
        {{
            "chapter_id": "1",
            "scene": "章节标题",
            "content": "章节内容..."
        }},
        ...
    ]

    文本：
    {text}
    """
)

# 3. 章节内容处理Prompt
content_processing_prompt = PromptTemplate(
    input_variables=["content", "scene"],
    template="""
    分析以下场景中的文本内容，提取所有对话和叙述信息：
    场景：{scene}
    文本：{content}

    输出格式（JSON数组）：
    [
        {{
            "type": "dialogue",
            "character": "角色名",
            "content": "对话内容",
            "emotion": "情感状态",
            "scene_objects": ["场景中提到的物体"]
        }},
        {{
            "type": "narrative",
            "content": "叙述内容",
            "emotion": "情感状态",
            "scene_objects": ["场景中提到的物体"]
        }},
        ...
    ]

    注意：
    - 从对话中推断角色情感（如：愤怒、喜悦、担忧等）
    - 识别场景中提到的所有物体（参考常见物体列表：{common_objects}）
    - 对话格式通常都会带有双引号，如”对话内容“
    - 叙述内容中不可以包含对话内容，在此前提下尽量保留长一些的文段（建议50-100字），以确保故事背景相对充分
    """
)

# 4. 对话分析Prompt
dialogue_analysis_prompt = PromptTemplate(
    input_variables=["query", "scene_description"],
    template="""
    你是一位专业的剧本分析专家。根据以下场景描述和用户问题，分析对话内容。

    场景描述: {scene_description}
    用户问题: {query}

    请提供以下信息:
    1. 角色名称
    2. 主导情感（如愤怒、喜悦、悲伤、恐惧等）
    3. 情感强度（1-10分）
    4. 可能的情感触发因素
    5. 场景中提到的物体

    格式:
    {{
        "character": "角色名称",
        "emotion": "主导情感",
        "intensity": 情感强度,
        "triggers": ["触发因素1", "触发因素2"],
        "scene_objects": ["物体1", "物体2"]
    }}
    """
)

# 5. 物品提取Prompt
object_extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    任务：从给定文本中精准提取所有实体物品，包括但不限于家具、工具、自然元素、人造物品、食物等。
    
    具体要求：
    1. 输出格式必须为JSON数组，如 ["椅子", "桌子", "水杯"]
    2. 排除纯抽象概念（如"幸福"、"时间"）和动作描述
    3. 优先使用具体名称（如"红色沙发"而非"家具"）
    4. 合并同类物品（如"三本书"应输出为["书"]）
    
    文本内容：
    ---
    {text}
    ---
    """
)

# 6. RAG后提问
final_query_prompt = PromptTemplate(
    input_variables=["user_query", "document_text", "kg_text"],
    template="""
    任务：基于以下信息，回答用户的问题。
    
    具体要求：如果没有足够信息，直接说明发现信息不足，直接回答“信息不足”即可。
    
    用户问题: {user_query}
    
    相关文档内容:
    {document_text}
    
    相关知识图谱信息:
    {kg_text}
    """
)