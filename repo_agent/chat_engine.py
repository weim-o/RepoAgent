from llama_index.core.base.llms.types import ChatMessage

from repo_agent.doc_meta_info import DocItem
from repo_agent.log import logger
from repo_agent.prompt import chat_template
from repo_agent.settings import SettingsManager
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
from pathlib import Path
import os



class LocalModelWrapper:
    def __init__(self, model_name: str):
        """
        初始化时加载 transformers 模型和分词器
        """
        self.model_name = model_name
        self.generator = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def chat(self, messages: list[ChatMessage]):
        """
        实现 chat 方法，接收 List[ChatMessage]，返回类似 OpenAI 接口的响应格式

        参数:
            messages: List[ChatMessage] —— 包含对话历史，每条消息包含 role 与 content 属性

        返回:
            一个字典，包含生成的文本和 token 使用信息
        """
        # 拼接对话历史为一个 prompt 字符串
        prompt = "".join([f"{msg.role}: {msg.content}\n" for msg in messages])

        logger.debug(f"Combined prompt: {prompt}")

        try:
            # 计算 prompt token 数量
            response, history = self.generator.chat(self.tokenizer, prompt, history=None)
            return response
        except Exception as e:
            logger.error(f"Error in chat call: {e}")
            raise

class ChatEngine:
    """
    ChatEngine is used to generate the doc of functions or classes.
    """

    def __init__(self, project_manager):
        setting = SettingsManager.get_setting()

        self.llm = LocalModelWrapper(setting.chat_completion.model)

    def build_prompt(self, doc_item: DocItem):
        """Builds and returns the system and user prompts based on the DocItem."""
        setting = SettingsManager.get_setting()

        code_info = doc_item.content
        referenced = len(doc_item.who_reference_me) > 0

        code_type = code_info["type"]
        code_name = code_info["name"]
        code_content = code_info["code_content"]
        have_return = code_info["have_return"]
        file_path = doc_item.get_full_name()

        def get_referenced_prompt(doc_item: DocItem) -> str:
            if len(doc_item.reference_who) == 0:
                return ""
            prompt = [
                """As you can see, the code calls the following objects, their code and docs are as following:"""
            ]
            for reference_item in doc_item.reference_who:
                instance_prompt = (
                    f"""obj: {reference_item.get_full_name()}\nDocument: \n{reference_item.md_content[-1] if len(reference_item.md_content) > 0 else 'None'}\nRaw code:```\n{reference_item.content['code_content'] if 'code_content' in reference_item.content.keys() else ''}\n```"""
                    + "=" * 10
                )
                prompt.append(instance_prompt)
            return "\n".join(prompt)

        def get_referencer_prompt(doc_item: DocItem) -> str:
            if len(doc_item.who_reference_me) == 0:
                return ""
            prompt = [
                """Also, the code has been called by the following objects, their code and docs are as following:"""
            ]
            for referencer_item in doc_item.who_reference_me:
                instance_prompt = (
                    f"""obj: {referencer_item.get_full_name()}\nDocument: \n{referencer_item.md_content[-1] if len(referencer_item.md_content) > 0 else 'None'}\nRaw code:```\n{referencer_item.content['code_content'] if 'code_content' in referencer_item.content.keys() else 'None'}\n```"""
                    + "=" * 10
                )
                prompt.append(instance_prompt)
            return "\n".join(prompt)

        def get_relationship_description(referencer_content, reference_letter):
            if referencer_content and reference_letter:
                return "And please include the reference relationship with its callers and callees in the project from a functional perspective"
            elif referencer_content:
                return "And please include the relationship with its callers in the project from a functional perspective."
            elif reference_letter:
                return "And please include the relationship with its callees in the project from a functional perspective."
            else:
                return ""

        code_type_tell = "Class" if code_type == "ClassDef" else "Function"
        parameters_or_attribute = (
            "attributes" if code_type == "ClassDef" else "parameters"
        )
        have_return_tell = (
            "**Output Example**: Mock up a possible appearance of the code's return value."
            if have_return
            else ""
        )
        combine_ref_situation = (
            "and combine it with its calling situation in the project,"
            if referenced
            else ""
        )

        referencer_content = get_referencer_prompt(doc_item)
        reference_letter = get_referenced_prompt(doc_item)
        has_relationship = get_relationship_description(
            referencer_content, reference_letter
        )

        project_structure_prefix = ", and the related hierarchical structure of this project is as follows (The current object is marked with an *):"

        return chat_template.format_messages(
            combine_ref_situation=combine_ref_situation,
            file_path=file_path,
            project_structure_prefix=project_structure_prefix,
            code_type_tell=code_type_tell,
            code_name=code_name,
            code_content=code_content,
            have_return_tell=have_return_tell,
            has_relationship=has_relationship,
            reference_letter=reference_letter,
            referencer_content=referencer_content,
            parameters_or_attribute=parameters_or_attribute,
            language=setting.project.language,
        )

    def generate_doc(self, doc_item: DocItem):
        """Generates documentation for a given DocItem."""
        messages = self.build_prompt(doc_item)

        try:
            response = self.llm.chat(messages)
            return response
        except Exception as e:
            logger.error(f"Error in llamaindex chat call: {e}")
            raise
