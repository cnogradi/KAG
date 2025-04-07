from typing import List
from kag.interface import PromptABC


@PromptABC.register("atomic_question_extraction_prompt")
class AtomicQuestionExtractionPrompt(PromptABC):
    template_en = """
        # Task
        Your task is to extract as many questions as possible that are relevant and can be answered by the given content. Please try to be diverse and avoid extracting duplicated or similar questions. Make sure your question contain necessary entity names and avoid to use pronouns like it, he, she, they, the company, the person etc.
        
        # Output Format
        Output your answers line by line, with each question on a new line, without itemized symbols or numbers.
        
        # Content
        $content
        
        # Output:
        """.strip()

    template_zh = """
        # Task
        请从给定内容中提取尽可能多的相关问题，确保这些问题能够通过原文内容得到解答。要求问题具有多样性，避免提取重复或相似度高的提问。每个问题必须包含具体的实体名称，禁止使用"它、他、她、他们、该公司、此人"等代词。
        
        # Output Format
        将每个问题单独成行输出，不使用项目符号或编号标记。
        
        # Content
        $content
        
        # Output:
        """.strip()

    @property
    def template_variables(self) -> List[str]:
        return ["content"]

    def parse_response(self, response: str, **kwargs):
        questions = response.split("\n")
        questions = [question.strip() for question in questions if len(question.strip()) > 0]
        return questions
