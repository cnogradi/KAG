import asyncio
import logging
from kag.common.conf import KAG_CONFIG
from kag.common.registry import import_modules_from_path
from kag.interface import SolverPipelineABC
from kag.solver.reporter.trace_log_reporter import TraceLogReporter

logger = logging.getLogger(__name__)


class MedicineDemo:
    """
    init for kag client
    """

    async def qa(self, query):
        reporter: TraceLogReporter = TraceLogReporter()
        resp = SolverPipelineABC.from_config(KAG_CONFIG.all_config["solver_pipeline"])
        answer = await resp.ainvoke(query, reporter=reporter)

        logger.info(f"\n\nso the answer for '{query}' is: {answer}\n\n")

        info, status = reporter.generate_report_data()
        logger.info(f"trace log info: {info.to_dict()}")
        return answer


if __name__ == "__main__":
    import_modules_from_path("./prompt")

    demo = MedicineDemo()
    #query = "Who said: 'I have overcome the world'"
    #query = "Where was Jesus Christ betrayed and by whom'"
    #query = "what is Jesus say about his body?"
    #query = "Were there sibllings among Jesus' disciples?"
    #query = "Did Jesus teach about the holy spirit?"
    #query = "Did Jesus claim to be God?"
    #query = "Did Jesus teach that men are born spiritually blind?"
    query = "what was the greatest accusation made against Jesus?"
    answer = asyncio.run(demo.qa(query))
    print(f"Question: {query}")
    print(f"Answer: {answer}")
