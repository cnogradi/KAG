from typing import List

from kag.common.llm.llm_response_parser import parse_json
from kag.interface import PromptABC


@PromptABC.register("atomic_query_decomposition_prompt")
class AtomicQueryDecompositionPrompt(PromptABC):
    template_en = """
        # Task
        Your task is to analyse the providing context then raise atomic sub-questions for the knowledge that can help you answer the question better. Think in different ways and raise as many diverse questions as possible.
        
        # Output Format
        Please output in following JSON format:
        {{
            "thinking": <A string. Your thinking for this task, including analysis to the question and the given context.>,
            "sub_questions": <A list of string. The sub-questions indicating what you need.>
        }}
        
        # Context
        The context we already have:
        $chosen_context
        
        # Question
        $content
        
        # Your Output:
        """.strip()

    template_zh = template_en

    @property
    def template_variables(self) -> List[str]:
        return ["content", "chosen_context"]

    def parse_response(self, response: str, **kwargs):
        try:
            output = parse_json(response)

            thinking: str = output["thinking"]
            sub_questions = output["sub_questions"]
            return len(sub_questions) > 0, thinking, sub_questions
        except Exception as e:
            print(f"[QuestionDecompositionParser] content to decode: {response}")
            print(f"Exception: {e}")
            return False, "", []
