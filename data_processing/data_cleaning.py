from typing import List, Dict
from config import Config, init_llm
from dialogue_management.prompt import (chapter_extraction_prompt, content_processing_prompt,
                                        dialogue_analysis_prompt, object_extraction_prompt)
import json
from datetime import datetime
import re


config = Config()

class DataCleaner:
    def __init__(self):
        self.llm = init_llm(config)
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.dialogue_prompt = dialogue_analysis_prompt
        self.object_extraction_chain = object_extraction_prompt | self.llm

    def _basic_clean(self, raw_text: str) -> str:
        """基础文本清洗：仅保留必要的格式标准化"""
        # 去除多余空白行
        cleaned = re.sub(r'\n\s*\n', '\n\n', raw_text.strip())
        # 处理智能引号
        cleaned = cleaned.replace('“', '"').replace('”', '"')
        return cleaned


    def _extract_chapters_with_llm(self, text: str) -> List[Dict]:
        """使用LLM提取章节结构"""
        config = Config()
        config.LLM_MODEL.max_tokens = 4096
        llm = init_llm(config)
        chain = chapter_extraction_prompt | llm
        response = chain.invoke({"text": text})
        response_text = response.content if hasattr(response, 'content') else str(response)
        return json.loads(response_text)

    def _extract_common_objects_with_llm(self, text: str, retries: int = 0) -> List[str]:
        """使用LLM从全文中提取常见物体，最多尝试2次"""
        try:
            response = self.object_extraction_chain.invoke({"text": text})
            return json.loads(response.content)
        except Exception as e:
            if retries >= 5:
                raise RuntimeError(f"LLM实体提取失败，{retries}次尝试均未成功: {str(e)}")
            return self._extract_common_objects_with_llm(text, retries + 1)

    def _process_chapter_with_llm(self, content: str, scene: str, common_objects: List[str]) -> List[Dict]:
        """使用LLM处理章节内容（包含common_objects参数）"""
        config = Config()
        config.LLM_MODEL.max_tokens = 4096
        llm = init_llm(config)
        chain = content_processing_prompt | llm
        try:
            response = chain.invoke({
                "content": content,
                "scene": scene,
                "common_objects": common_objects
            })
            units = json.loads(response.content)
            for unit in units:
                unit["scene"] = scene
            print(f"{scene}内容提取成功")
            return units
        except json.JSONDecodeError:
            print(f"LLM响应格式错误, 退返重来")
            return self._process_chapter_with_llm(content, scene, common_objects)

    def _enrich_metadata(self, chapters: List[Dict]) -> List[Dict]:
        """为章节添加全局元数据"""
        for chapter in chapters:
            chapter["metadata"] = {
                "processing_time": self.current_time,
                "source": "story.txt",
                "model": config.LLM_MODEL.model,
                "provider": config.LLM_MODEL.provider
            }
            for unit in chapter["text_units"]:
                unit.update(self._get_common_metadata())
        return chapters

    def _get_common_metadata(self) -> Dict:
        """获取通用元数据"""
        return {
            "cleaning_method": "llm_based",
            "text_type": "narrative_dialogue_mixed",
            "language": "chinese_simplified"
        }

    def clean_text(self, raw_text: str) -> List[Dict]:
        """使用LLM解析文本结构"""
        print(f"开始清洗文本，长度: {len(raw_text)}")
        # 基础清洗
        cleaned_text = self._basic_clean(raw_text)
        print(f"清洗后文本长度: {len(cleaned_text)}")

        # 提取章节结构
        chapters = self._extract_chapters_with_llm(cleaned_text)
        print(f"LLM解析出 {len(chapters)} 个章节")

        processed_chapters = []
        for i, chapter in enumerate(chapters):
            # 为每个章节单独提取物体
            chapter_objects = self._extract_common_objects_with_llm(chapter["content"])
            print(f"第{i+1}章节提取实体：\n{chapter_objects}")
            units = self._process_chapter_with_llm(
                content=chapter["content"],
                scene=chapter["scene"],
                common_objects=chapter_objects  # 使用章节特定的物体列表
            )
            processed_chapters.append({
                "chapter_id": chapter["chapter_id"],
                "scene": chapter["scene"],
                "text_units": units
            })
        # 添加元数据
        return self._enrich_metadata(processed_chapters)

