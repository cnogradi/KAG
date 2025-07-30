

// Create a function to extract the resume from a string.
function RetrieverStaticPlanning(executors: string, query: string) -> STasks {
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
      "query": "Which year was Zhou Xingchi's first movie released?",
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
      ],
      "output": {
          "0": {
              "executor": "Retriever",
              "dependent_task_ids": [],
              "arguments": {"query": "What was Zhou Xingchi's first movie?"},
          },
          "1": {
              "executor": "Retriever",
              "dependent_task_ids": ["0"],
              "arguments": {"query": "Which year was \{\{0.output\}\} released?"},
          },
      },
    }

    executors:

    {{executors}}

    query:

    {{query}}
    
    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [RetrieverStaticPlanning]
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
