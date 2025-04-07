import json
from typing import List

from kag.common.llm.llm_response_parser import parse_json
from kag.interface import PromptABC
from kag.solver.executor.retriever.local_knowledge_base.kag_retriever.kag_atomic_executor import AtomRetrievalInfo


@PromptABC.register("atomic_question_selection_prompt")
class AtomicQuestionSelectionPrompt(PromptABC):
    template_en = """
        # Task
        Your task is to analyse the providing context then decide which sub-questions may be useful to be answered before you can answer the given question. Select a most relevant sub-question from the given question list, avoid selecting sub-question that can already be answered with the given context or with your own knowledge.
        
        # Output Format
        Please output in following JSON format:
        {{
            "thinking": <A string. Your thinking for this selection task.>,
            "question_idx": <An integer, indicating a sub-question index from 1 to {num_atoms}.>
        }}
        
        # Context
        The context we already have:
        $chosen_context
        
        # Sub-Questions You Can Choose From
        $atom_list_str
        
        # Question
        $content
        
        # Your output:
        """.strip()

    template_zh = template_en

    @property
    def template_variables(self) -> List[str]:
        return ["content", "num_atoms", "chosen_context", "atom_list_str"]

    def parse_response(self, response: str, **kwargs):
        try:
            output = parse_json(response)
            return output
        except Exception as e:
            print(f"[AtomQuestionSelectionParser] content to decode: {response}")
            print(f"Exception: {e}")
            return ""

    def parse_response_with_candidates(self, response: json, atom_info_candidates:List[AtomRetrievalInfo], **kwargs):
        try:
            thinking: str = response["thinking"]
            question_idx = response["question_idx"]
            if question_idx is not None and question_idx > 0 and question_idx <= len(atom_info_candidates):
                chosen_info = atom_info_candidates[question_idx - 1]
                return True, thinking, chosen_info
            else:
                return False, thinking, None
        except Exception as e:
            print(f"[AtomQuestionSelectionParser] content to decode: {response}")
            print(f"Exception: {e}")
            return False, "", None
