// Defining a data model.
class SolveNoDocs {
  answer string
}

// Create a function to extract the resume from a string.
function SolveQuestionNoDocs(question: string, history: string, knowledge_graph: string) -> SolveNoDocs {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    Please answer the question below based on the retrieved knowledge graph, combine historical information for comprehensive analysis. 

    Requirement:
      1. Answer the question as directly as possible, without including any other information.
      2. Do not repeat the content of the question.
      3. Generate answers based on the provided information. If multiple answers are possible, generate all of them.
      4. If there is no suitable answer, answer 'I don't know'.
      5. Provide the answer and also provide the reason.

    the question to answer:

    {{ question }}

    history:

    {{ history }}

    knowledge graph to use to answer question:
    
    {{ knowledge_graph }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [SolveQuestionNoDocs]
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
