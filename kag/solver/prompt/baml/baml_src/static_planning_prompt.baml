// Defining a data model.
class SPPArgument {
  name string @description(#"
    The name of the argument to pass to the executor
  "#)
  instruction string @description(#"
    The value of the argument to pass to the executor
  "#)
}

class STask {
  task_id int @description(#"
    The id of this task which will be used for dependent tasks to refer to as a dependency
  "#)
  executor string @description(#"
    One of the tools provided in the executors list
  "#)
  dependent_task_ids int[] @description(#"
    A list of task_ids that this task depends on
  "#)
  args SPPArgument[] @description(#"
    A list of arguments to pass to the executor
  "#)
}

class STasks {
  tasks STask[] @description(#"
    A list of tasks generated from the query provided in the form of a DAG
  "#)
}

// Create a function to extract the resume from a string.
function StaticPlanningPrompt(executors: string, query: string)-> STasks {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    instruction: 

      You are a problem-solving planner. Your task is to analyze complex problems provided by users and, through your analysis and reasoning, provide specific step-by-step plans for solving the problem using available tools. The plan should be represented as a Directed Acyclic Graph (DAG). Note that you do not need to answer the question directly but instead produce the steps required to answer the question. The user's question is provided in the query field, and the available tools are listed in the executors field.

      Your reasoning should follow these steps:
      1. Analyze the request to understand the scope of the task.
      2. Break down the original complex problem into several finest-grained atomic problem sets and define their dependencies in DAG format.
      3. Use the tools defined in executors to create clear and actionable plans to drive the task forward and make substantial progress.

    Note:
      1. Please return your planning result in JSON format, following the example in the output field of the example section.

    example:
     {
        "query": "Which movies have Jacky Cheung and Andy Lau starred in together?",
        "executors": [
            {
                "name": "Retriever",
                "description": "Retrieve relevant knowledge from the local knowledge base.",
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "User-provided query for retrieval.",
                        "optional": False,
                    },
                },
            },
            {
                "name": "Code",
                "description": "Used to address users' math or computational problems.",
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "The computable problem derived from the user's input question, retaining the essential information for the calculation target and dependencies.",
                        "optional": False,
                    }
                },
            },
        ],
        "output": {
            "0": {
                "executor": "Retriever",
                "dependent_task_ids": [],
                "arguments": {
                    "query": "List of movies Jacky Cheung has starred in"
                },
            },
            "1": {
                "executor": "Retriever",
                "dependent_task_ids": [],
                "arguments": {"query": "List of movies Andy Lau has starred in"},
            },
            "2": {
                "executor": "Code",
                "dependent_task_ids": ["0", "1"],
                "arguments": {
                    "query": "Please write Python code to find the common elements in the following two lists:\nJacky Cheung movie list: \{\{0.output\}\}\nAndy Lau movie list: \{\{1.output\}\}"
                },
            },
        },
      }
    
    query:
    
    {{ query }}

    executors:

    {{ executors }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [StaticPlanningPrompt]
  args {
    resume #"
      Vaibhav Gupta
      vbv@boundaryml.com

      Experience:
      - Founder at BoundaryML
      - CV Engineer at Google
      - CV Engineer at Microsoft

      Skills:
      - Rust
      - C++
    "#
  }
}
