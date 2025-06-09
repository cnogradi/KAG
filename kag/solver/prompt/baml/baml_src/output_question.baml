// Defining a data model.
class AnswerQuestion {
  answer string
}

// Create a function to extract the resume from a string.
function OutputQuestion(context: string, question: string) -> AnswerQuestion {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    Please answer the question {{ question }} based on the retrieved relevant documents and combine it with historical information for comprehensive analysis. The requirements are as follows:
    1. Do not repeat the content of the question.
    2. Generate answers based on the provided information. If there are multiple possible answers, generate all of them.
    3、If there is no suitable answer, analyze relevant content based on the document information.
    4. If the context is empty, respond based on the model's internal knowledge.
    5. When outputting, use a sentence structure similar to this: "Based on the question, I first considered xxx, then yyy, and finally zzz. In conclusion, therefore mmmm."
    6. The tone of the response should be smooth and natural, avoiding any mechanical feel.

    Context: 

    {{ context }}
    
    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [OutputQuestion]
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
