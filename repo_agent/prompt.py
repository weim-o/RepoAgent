from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

doc_generation_instruction = (
    "你是一名AI文档助手，你的任务是根据给定对象的代码生成相应的文档。"
    "该文档的目的是帮助开发者和初学者理解代码的功能和具体用法。\n\n"
    "当前，你正在一个项目中{project_structure_prefix}\n"
    "{project_structure}\n\n"
    "你需要在该项目中生成文档的路径是 {file_path}。\n"
    '现在，你需要为一个 {code_type_tell} 生成文档，其名称为 "{code_name}"。\n\n'
    "该代码的内容如下:\n"
    "{code_content}\n\n"
    "{reference_letter}\n"
    "{referencer_content}\n\n"
    "请根据该目标对象的代码 {combine_ref_situation} 生成一份详细的解释文档。\n\n"
    "请用**加粗的纯文本**写出该 {code_type_tell} 的功能描述，并在后续内容中使用普通文本进行详细分析 "
    "(包括所有细节)，使用 {language} 作为该部分代码的文档语言。\n\n"
    "标准格式如下:\n\n"
    "**{code_name}**: {code_name} 的功能是 XXX。（仅需代码名称和一句话功能描述）\n"
    "**{parameters_or_attribute}**: 该 {code_type_tell} 的 {parameters_or_attribute}。\n"
    "· 参数1: XXX\n"
    "· 参数2: XXX\n"
    "· ...\n"
    "**代码描述**: 该 {code_type_tell} 的详细描述。\n"
    "(详细且准确的代码分析和说明...{has_relationship})\n"
    "**注意事项**: 该代码的使用注意点\n"
    "{have_return_tell}\n\n"
    "请注意:\n"
    "- 你生成的内容不应包含Markdown的层级标题和分隔符语法。\n"
    "- 主要使用目标语言编写。如有必要，你可以在分析和描述中保留部分英文词汇，以增强文档的可读性，"
    "但无需翻译函数名或变量名。\n"
)

documentation_guideline = (
    "请牢记，你的读者是文档使用者，因此请使用确定性的语气生成精准的内容，"
    "不要让他们察觉到你是基于代码片段和文档生成的。避免任何推测和不准确的描述！"
    "现在，请以专业的方式使用 {language} 生成目标对象的文档。"
)

message_templates = [
    ChatMessage(content=doc_generation_instruction, role=MessageRole.SYSTEM),
    ChatMessage(
        content=documentation_guideline,
        role=MessageRole.USER,
    ),
]

chat_template = ChatPromptTemplate(message_templates=message_templates)
