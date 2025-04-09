import json
import logging
import uuid

from typing import List, Any

from kag.common.conf import KAG_PROJECT_CONF
from kag.common.config import get_default_chat_llm_config
from kag.common.graphstore.neo4j_graph_store import Neo4jClient
from kag.common.utils import generate_hash_id
from kag.interface import ExecutorABC, ExecutorResponse, LLMClient, Context, VectorizeModelABC, PromptABC, Task
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


def run_text_query(database, label, text_query, topk = 5):
    neo4j_client = Neo4jClient(
        uri="neo4j://127.0.0.1:7687",
        user="neo4j",
        password=None,
        database=database
    )
    neo4j_response = neo4j_client.text_search(query_string=text_query, label_constraints=label, topk = topk)
    atomic_recall_res = {}
    if neo4j_response and len(neo4j_response) > 0:
        atomic_recall_res['chunk_id'] = neo4j_response[0][2]._properties.get('id')
        atomic_recall_res['chunk_title'] = neo4j_response[0][2]._properties.get('name')
        atomic_recall_res['chunk_content'] = neo4j_response[0][2]._properties.get('content')
    return atomic_recall_res

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
            neo4j_response = run_cypher_query("musiquepike", spg_gql)
            chunks.append(self.gen_atomic_recall(atomic_question, neo4j_response))

        # neo4j_response = run_text_query("musiquepike", "MusiquePike.Chunk", atomic_query)
        # chunks.append(neo4j_response)

        return chunks

    def to_kag_retrieved_response(self, chosen_info: AtomRetrievalInfo):
        kag_retrieved_response = KAGRetrievedResponse()
        if chosen_info:
            kag_retrieved_response.chunk_datas.append(chosen_info)
        return kag_retrieved_response

    def atom_infos_to_context_string(self, ori_query, chosen_atom_infos: List[AtomRetrievalInfo],
                                     limit: int = 80000) -> str:
        context: str = ""
        chunk_id_set = set()
        for info in chosen_atom_infos:
            if not info:
                print(f"ori_query:{ori_query}, chosen_atom_infos:{chosen_atom_infos}")
                continue
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
        chosen_context = self.atom_infos_to_context_string(ori_query, chosen_atom_infos)

        atom_info_candidates_dedup_dict = {}
        for atom_info_candidate in atom_info_candidates:
            atom = atom_info_candidate.atom
            if atom not in atom_info_candidates_dedup_dict or atom_info_candidates_dedup_dict[
                atom].score <= atom_info_candidate.score:
                atom_info_candidates_dedup_dict[atom] = atom_info_candidate

        atom_info_candidates_dedup_list = [value for value in atom_info_candidates_dedup_dict.values()]

        atom_list_str = ""
        for i, info in enumerate(atom_info_candidates_dedup_list):
            atom_list_str += f"Question {i + 1}: {info.atom}\n"

        num_atoms = len(atom_info_candidates_dedup_list)
        response = self.llm_client.invoke(
            {"content": ori_query, "num_atoms": num_atoms, "chosen_context": chosen_context,
             "atom_list_str": atom_list_str},
            self.atomic_question_selection_prompt, with_except=False,
            with_json_parse=False
        )

        i_selected, thinking, chosen_info = self.atomic_question_selection_prompt.parse_response_with_candidates(
            response, atom_info_candidates_dedup_list)

        kag_retrieved_response = self.to_kag_retrieved_response(chosen_info)
        return kag_retrieved_response

    def update_reponse(self, task:Task, atom_info_candidates:List[AtomRetrievalInfo]):
        atom_info_candidates_dedup_dict = {}
        for atom_info_candidate in atom_info_candidates:
            chunk_id = atom_info_candidate.chunk_id
            if chunk_id not in atom_info_candidates_dedup_dict:
                atom_info_candidates_dedup_dict[chunk_id] = atom_info_candidate

        chunk_list_str = ""
        chunk_datas = []
        for atom_info_candidate in atom_info_candidates_dedup_dict.values():
            tmp_list = atom_info_candidate.content.split("\n")
            tmp_content = " ".join(tmp_list[1:])
            chunk_list_str += f"{tmp_content}\n\n"
            chunk_data = ChunkData(
                content = tmp_content,
                title = atom_info_candidate.title,
                chunk_id= atom_info_candidate.chunk_id,
                score = atom_info_candidate.score
            )
            chunk_datas.append(chunk_data)

        system_instruction = """
As an adept specialist in resolving intricate multi-hop questions, I require your assistance in addressing a multi-hop question. The question has been segmented into multiple straightforward single-hop inquiries, wherein each question may depend on the responses to preceding questions, i.e., the question body may contain content such as "{{i.output}}", which means the answer of ith sub-question. I will furnish you with insights on how to address these preliminary questions, or the answers themselves, which are essential for accurate resolution. Furthermore, I will provide textual excerpts pertinent to the current question, which you are advised to peruse and comprehend thoroughly. Begin your reply with "Thought: ", where you'll outline the step-by-step thought process that leads to your conclusion. End with "Answer: " to deliver a clear and precise response without any extra commentary.
        
Docs:
Sylvester
Sylvester is a name derived from the Latin adjective silvestris meaning ``wooded ''or`` wild'', which derives from the noun silva meaning ``woodland ''. Classical Latin spells this with i. In Classical Latin y represented a separate sound distinct from i, not a native Latin sound but one used in transcriptions of foreign words. After the Classical period y came to be pronounced as i. Spellings with Sylv - in place of Silv - date from after the Classical period.

Charlemagne
Charlemagne or Charles the Great (2 April 742 -- 28 January 814), numbered Charles I, was King of the Franks from 768, King of the Lombards from 774, and Holy Roman Emperor from 800. He united much of western and central Europe during the early Middle Ages. He was the first recognized emperor to rule from western Europe since the fall of the Western Roman Empire three centuries earlier. The expanded Frankish state that Charlemagne founded is called the Carolingian Empire. He was later invalidly canonized by the antipope Paschal III.

Middle Ages
Charlemagne's court in Aachen was the centre of the cultural revival sometimes referred to as the \"Carolingian Renaissance\". Literacy increased, as did development in the arts, architecture and jurisprudence, as well as liturgical and scriptural studies. The English monk Alcuin (d. 804) was invited to Aachen and brought the education available in the monasteries of Northumbria. Charlemagne's chancery\u2014or writing office\u2014made use of a new script today known as Carolingian minuscule,[M] allowing a common writing style that advanced communication across much of Europe. Charlemagne sponsored changes in church liturgy, imposing the Roman form of church service on his domains, as well as the Gregorian chant in liturgical music for the churches. An important activity for scholars during this period was the copying, correcting, and dissemination of basic works on religious and secular topics, with the aim of encouraging learning. New works on religious topics and schoolbooks were also produced. Grammarians of the period modified the Latin language, changing it from the Classical Latin of the Roman Empire into a more flexible form to fit the needs of the church and government. By the reign of Charlemagne, the language had so diverged from the classical that it was later called Medieval Latin.

Questions:
0: Who was crowned emperor of the west in 800 CE?
Thought: One of the provided passage on Charlemagne indicates that he was crowned Holy Roman Emperor in 800. Answer: Charlemagne.

1: What was {{0.output}} later known as?
Thought: To determine what {{0.oputput}} (Charlemagne) was later known as, I need to review the provided passage about Charlemagne. The passage indicates that Charlemagne was also known as "Charles the Great." Answer: Charles the Great

2: What was the language from which the last name Sylvester originated during {{0.output}} era?
Thought: The question asks about the origin of the last name Sylvester during the time of the person {{0.output}}, which was Charlemagne, whose reign was in the Early Middle Ages. The passage about the name Sylvester states that it is derived from Latin. Answer: Latin
"""
        query = f"{task.id}: {task.arguments['query']}"
        subqa = []
        for pt in task.parents:
            subq = f"{pt.id}: {pt.arguments['query']}"
            suba = str(pt.result.summary)
            subqa.append(f"{subq}\n{suba}")
        subqa = "\n\n".join(subqa)
        request = f"{system_instruction}\nDocs:\n{chunk_list_str}\nQuestions:\n{subqa}\n{query}"

        result = self.llm_client.__call__(request)
        task.update_memory("response", result)
        task.update_memory("chunks", chunk_list_str)

        kag_retrieved_response = KAGRetrievedResponse()
        kag_retrieved_response.summary = result
        kag_retrieved_response.retrieved_task = query
        kag_retrieved_response.chunk_datas = chunk_datas
        # task.result = json.dumps(
        #     {"query": task.arguments["query"], "response": result}, ensure_ascii=False
        # )
        task.result = kag_retrieved_response

        return

    def invoke(self, query: str, task: Any, context: Context, **kwargs):
        namespace = "MusiquePike"

        task_query = task.arguments["query"]
        # step1：Rewrite query
        rewritten_queries = self.rewrite_query(task_query, context)
        rewritten_queries.append(task_query)

        # step2, recall atomic Questions for each rewritten query
        atom_info_candidates: list = []
        for rewritten_query in rewritten_queries:
            label = f"{namespace}.AtomicQuestion"
            atomic_chunks = self.recall_atomic_chunk(rewritten_query, label)

            for atomic_chunk in atomic_chunks:
                atom_info_candidate = AtomRetrievalInfo(
                    atom_query=rewritten_query,
                    atom=atomic_chunk['atomic_question_name'],
                    chunk_id=atomic_chunk['chunk_id'],
                    content=atomic_chunk['chunk_content'],
                    title=atomic_chunk['chunk_title'],
                    score=atomic_chunk['score'],
                )
                atom_info_candidates.append(atom_info_candidate)

        # step3, select suitable atomic_chunk
        # retrievedResponse = self.select_suitable_chunk(task_query, atom_info_candidates, context)

        # step4, store result for current task
        self.update_reponse(task, atom_info_candidates)
        return

    def schema(self) -> dict:
        """Function schema definition for OpenAI Function Calling

        Returns:
            dict: Schema definition in OpenAI Function format
        """
        return {
            "name": "Reasoner",
            "description": "Retrieve relevant knowledge from the local knowledge base.",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "User-provided query for retrieval.",
                    "optional": False,
                },
            },
        }
