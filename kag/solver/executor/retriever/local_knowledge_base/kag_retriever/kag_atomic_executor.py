import json
import logging
import uuid

from typing import List, Any

from kag.common.conf import KAG_PROJECT_CONF
from kag.common.config import get_default_chat_llm_config
from kag.common.graphstore.neo4j_graph_store import Neo4jClient
from kag.common.utils import generate_hash_id
from kag.interface import ExecutorABC, ExecutorResponse, LLMClient, Context, VectorizeModelABC, PromptABC
from kag.interface.solver.model.one_hop_graph import (
    ChunkData,
    RetrievedData,
    KgGraph, AtomRetrievalInfo,
)
from kag.solver.executor.retriever.local_knowledge_base.kag_retriever.kag_hybrid_executor import KAGRetrievedResponse
from kag.solver.utils import init_prompt_with_fallback
from kag.tools.graph_api.graph_api_abc import GraphApiABC
from kag.tools.search_api.search_api_abc import SearchApiABC

logger = logging.getLogger()

def store_results(task, response: KAGRetrievedResponse):
    """Store final results in task context

    Args:
        task: Task configuration object
        response (KAGRetrievedResponse): Processed results
    """
    task.update_memory("response", response)
    task.update_memory("chunks", response.chunk_datas)
    task.update_result(response)

def run_cypher_query(database, cypher_query):
    neo4j_client = Neo4jClient(
        uri="neo4j://127.0.0.1:7687",
        user="neo4j",
        password=None,
        database=database
    )
    neo4j_response = neo4j_client.run_cypher_query(database=database, query=cypher_query)
    return neo4j_response

@ExecutorABC.register("kag_atomic_executor")
class KagAtomicExecutor(ExecutorABC):
    """
        atomic query executor:
            1、retrieve atomic questions and corresponding Chunks
            2、select best match atomic question
    """

    def __init__(
            self,
            search_client: SearchApiABC = None,
            graph_client: GraphApiABC = None,
            vectorizer: VectorizeModelABC = None,
            llm_client: LLMClient = None,
            query_decomposition_prompt: PromptABC = None,
            atomic_question_selection_prompt: PromptABC = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.search_client = search_client or SearchApiABC.from_config(
            {"type": "openspg_search_api"}
        )

        self.graph_client = graph_client
        self.vectorizer = vectorizer
        self.retrieve_atomic_question_k = 16
        self.query_decomposition_prompt = query_decomposition_prompt
        self.atomic_question_selection_prompt = atomic_question_selection_prompt
        self.solve_question_without_spo_prompt = init_prompt_with_fallback(
            "summary_question", KAG_PROJECT_CONF.biz_scene
        )
        self.llm_client = llm_client or LLMClient.from_config(
            get_default_chat_llm_config()
        )

    @property
    def output_types(self):
        """Output type specification for executor responses"""
        return KAGRetrievedResponse

    def gen_atomic_recall(self, atomic_question, neo4jresponse):
        if len(neo4jresponse) < 1 or len(neo4jresponse[0]) != 3:
            return None

        atomic_recall_res = {}
        atomic_recall_res['atomic_question_id'] = atomic_question.get("node").get('id')
        atomic_recall_res['atomic_question_name'] = atomic_question.get("node").get('name')
        atomic_recall_res['atomic_question_content'] = atomic_question.get("node").get('content')

        atomic_recall_res['chunk_id'] = neo4jresponse[0][2]._properties.get('id')
        atomic_recall_res['chunk_title'] = neo4jresponse[0][2]._properties.get('name')
        atomic_recall_res['chunk_content'] = neo4jresponse[0][2]._properties.get('content')

        atomic_recall_res['score'] = atomic_question.get('score')

        return atomic_recall_res

    def parse_chosen_atom_infos(self, context: Context):
        chunks = []
        for task in context.gen_task(False):
            if isinstance(task.result, KAGRetrievedResponse):
                chunks.extend(task.result.chunk_datas)
        return chunks

    def rewrite_query(self, query: str, context: Context):
        chosen_atom_infos = self.parse_chosen_atom_infos(context)
        i_decomposed, thinking, sub_questions = self.llm_client.invoke(
            {"content": query, "chosen_context": chosen_atom_infos}, self.query_decomposition_prompt, with_except=False,
            with_json_parse=False
        )

        rewritten_queries = sub_questions
        return rewritten_queries

    def recall_atomic_chunk(self, atomic_query, label):
        # get current namespace
        query_emb = self.vectorizer.vectorize(atomic_query)
        atomic_questions = self.search_client.search_vector(label=label, property_key="content",
                                                            query_vector=query_emb, topk=16)
        chunks = []
        for atomic_question in atomic_questions:
            s_name = atomic_question.get("node").get('name')
            s_id = generate_hash_id(s_name)

            spg_gql = f"""
                MATCH (s:`MusiquePike.AtomicQuestion`)-[p:source]->(o:`MusiquePike.Chunk`)
                WHERE s.id='{s_id}'
                RETURN s,p,o
                """
            neo4jresponse = run_cypher_query("musiquepike", spg_gql)
            chunks.append(self.gen_atomic_recall(atomic_question, neo4jresponse))

        return chunks

    def to_kag_retrieved_response(self, chosen_info:AtomRetrievalInfo):
        kag_retrieved_response = KAGRetrievedResponse()
        kag_retrieved_response.chunk_datas.append(chosen_info)
        return kag_retrieved_response

    def atom_infos_to_context_string(self, chosen_atom_infos: List[AtomRetrievalInfo], limit: int=80000) -> str:
        context: str = ""
        chunk_id_set = set()
        for info in chosen_atom_infos:
            if info.chunk_id in chunk_id_set:
                continue
            chunk_id_set.add(info.chunk_id)

            if info.title is not None:
                context += f"\nTitle: {info.title}. Content: {info.content}\n"
            else:
                context += f"\n{info.content}\n"

            if len(context) >= limit:
                break

        context = context.strip()
        return context

    def select_suitable_chunk(self, ori_query, atom_info_candidates, context: Context) -> KAGRetrievedResponse:
        chosen_atom_infos = self.parse_chosen_atom_infos(context)
        chosen_context = self.atom_infos_to_context_string(chosen_atom_infos)

        atom_list_str = ""
        for i, info in enumerate(atom_info_candidates):
            atom_list_str += f"Question {i + 1}: {info.atom}\n"

        num_atoms = len(atom_info_candidates)
        response = self.llm_client.invoke(
            {"content": ori_query, "num_atoms":num_atoms, "chosen_context": chosen_context, "atom_list_str": atom_list_str},
            self.atomic_question_selection_prompt, with_except=False,
            with_json_parse=False
        )

        i_selected, thinking, chosen_info = self.atomic_question_selection_prompt.parse_response_with_candidates(
            response, atom_info_candidates)

        kag_retrieved_response = self.to_kag_retrieved_response(chosen_info)
        return kag_retrieved_response

    def store_results(self, task, response: KAGRetrievedResponse):
        """Store final results in task context

        Args:
            task: Task configuration object
            response (KAGRetrievedResponse): Processed results
        """
        task.update_memory("response", response)
        task.update_memory("chunks", response.chunk_datas)
        task.update_result(response)

    def invoke(self, query: str, task: Any, context: Context, **kwargs):
        namespace = "MusiquePike"

        # step1：Rewrite query
        rewritten_queries = self.rewrite_query(query, context)

        # step2, recall atomic Questions for each rewritten query
        atom_info_candidates: list = []
        for rewritten_query in rewritten_queries:
            label = f"{namespace}.AtomicQuestion"
            atomic_chunks = self.recall_atomic_chunk(rewritten_query, label)

            for atomic_chunk in atomic_chunks:
                atom_info_candidate = AtomRetrievalInfo(
                    atom_query = rewritten_query,
                    atom = atomic_chunk['atomic_question_name'],
                    chunk_id= atomic_chunk['chunk_id'],
                    content= atomic_chunk['chunk_content'],
                    title= atomic_chunk['chunk_title'],
                    score= atomic_chunk['score'],
                )
                atom_info_candidates.append(atom_info_candidate)

        # step3, select suitable atomic_chunk
        retrievedResponse = self.select_suitable_chunk(query, atom_info_candidates, context)

        # step4, store result for current task
        self.store_results(task, retrievedResponse)
        return

    def schema(self) -> dict:
        """Function schema definition for OpenAI Function Calling

        Returns:
            dict: Schema definition in OpenAI Function format
        """
        return {
            "name": "Retriever",
            "description": "Retrieve relevant knowledge from the local knowledge base.",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "User-provided query for retrieval.",
                    "optional": False,
                },
            },
        }
