// Defining a data model.
class SolveNoSPO {
  answer string
}

// Create a function to extract the resume from a string.
function SolveQuestionNoSPO(question: string, history: string, docs: string) -> SolveNoSPO {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    Please answer the question below based on the retrieved relevant documents, and combine historical information for comprehensive analysis.
    
    Requirements in ansering question:
        1. Answer the question as directly as possible, without including any other information.
        2. Do not repeat the content of the question.
        3. Generate answers based on the provided information. If multiple answers are possible, generate all of them.
        4. If there is no suitable answer, answer 'I don't know'.
        5. Provide the answer and also provide the reason.

    the question to answer:

    {{ question }}
  
    history:
    
    {{ history }}

    relevant documents to use to answer the question:

    {{ docs }}


    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [SolveQuestionNoSPO]
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
