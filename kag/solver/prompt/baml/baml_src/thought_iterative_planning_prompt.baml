// Defining a data model.
class ExecutorArgument {
  name string @description(#"
    The name of the argument to pass to the executor
  "#)
  instruction string @description(#"
    The value of the argument to pass to the executor
  "#)
}

class Executor {
  name string
  arguments ExecutorArgument[]
  thought string
}

// Create a function to extract the resume from a string.
function ToughtIterativePlanning(example_executors: string, query: string, context: string, executors: string[])-> Executor {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
  instruction:

    You are a problem-solving planner. Your task is to analyze the complex problem provided by the user along with the problem-solving context (including historical planning steps and execution results). Based on your own reasoning, you should **step-by-step** plan specific actions to solve the problem using the available tools. The user's problem is provided in the "query" field, the available tools are listed in the "executors" field, and the problem-solving context, which includes executed tool calls and their results, is given in the "context" field.

    Your reasoning should follow these steps:
    1. Analyze the request to understand the scope of the task.
    2. Read and analyze the context to understand the progress made in problem-solving and determine the next actions to execute. If the context is empty, start planning from scratch.
    3. Use the tools defined in the "executors" field to create a clear, actionable plan that drives substantive progress on the task.

    Important considerations:
    1. Return your planning results in JSON format, following the example provided in the "output" field of the "example" section.
    2. If you determine from the context that the task has been completed, return a "Finish" tool call to indicate no further actions are required.  

  examples:
    {
      "example1": {
            "query": "How old was Albert Einstein when he won the Nobel Prize?",
            "context": [],
            "executors": {{example_executors}},
            "output": {
                "executor": {
                    "name": "Retriever",
                    "arguments": {"query": "When was Albert Einstein born?"},
                    "thought": "Initial requirement: To calculate Einstein's age at the time of the award, we must first obtain his birth year. This foundational data is critical for all subsequent calculations.",
                }
            },
        },
        "example2": {
            "query": "How old was Albert Einstein when he won the Nobel Prize?",
            "context": [
                {
                    "executor": "Retriever",
                    "arguments": {"query": "When was Albert Einstein born?"},
                    "result": "1879",
                }
            ],
            "executors": {{example_executors}},
            "output": {
                "executor": {
                    "name": "Retriever",
                    "arguments": {
                        "query": "What year did Albert Einstein win the Nobel Prize?"
                    },
                    "thought": "Confirmed birth year: 1879. Next critical data point: Nobel Prize year. Age calculation requires both temporal markers. Priority: Retrieve award year.",
                }
            },
        },
        "example3": {
            "query": "How old was Albert Einstein when he won the Nobel Prize?",
            "context": [
                {
                    "executor": "Retriever",
                    "arguments": {"query": "When was Albert Einstein born?"},
                    "result": "1879",
                },
                {
                    "executor": "Retriever",
                    "arguments": {
                        "query": "What year did Albert Einstein win the Nobel Prize?"
                    },
                    "result": "1921",
                },
            ],
            "executors": {{example_executors}},
            "output": {
                "executor": {
                    "name": "Math",
                    "arguments": {"query": "Subtract 1879 from 1921 to get his age"},
                    "thought": "Essential data acquired: Birth year (1879) and award year (1921). Execution formula: Award year - Birth year = Age. Mathematical verification required.",
                }
            },
        },
        "example4": {
            "query": "How old was Albert Einstein when he won the Nobel Prize?",
            "context": [
                {
                    "executor": "Retriever",
                    "arguments": {"query": "When was Albert Einstein born?"},
                    "result": "1879",
                },
                {
                    "executor": "Retriever",
                    "arguments": {
                        "query": "What year did Albert Einstein win the Nobel Prize?"
                    },
                    "result": "1921",
                },
                {
                    "executor": "Math",
                    "arguments": {"query": "Subtract 1879 from 1921 to get his age"},
                    "result": "42",
                },
            ],
            "executors": {{example_executors}},
            "output": {
                "executor": {
                    "name": "Deduce",
                    "arguments": [{
                        "name":"query", "instruction": "Combine birth year and award year to confirm his age"
                    }],
                    "thought": "Calculation result: 42 years old. Validation checklist: 1. Consistency with historical records 2. Potential timeline anomalies (e.g., award delays). Context synthesis needed for final confirmation.",
                }
            },
        },
    }
  
  context:

    {{context}}
  
  executors:

    {{executors}}
  
  query:

    {{query}}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [ToughtIterativePlanning]
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
