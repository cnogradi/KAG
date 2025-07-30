// Defining a data model.
class QuestionSummaryAnswer {
  answer string
}

// Create a function to extract the resume from a string.
function QuestionSummary(history: string, question: string, docs: string[])-> QuestionSummaryAnswer {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    Please answer the question `{{question}}` based on the retrieved relevant documents, and combine historical information for comprehensive analysis.
    
    Requirement:
    1.Do not repeat the content of the question.
    2.Generate answers strictly based on provided information. If multiple answers are possible, list all plausible answers.
    3.If no suitable answer exists, analyze related content based on document information.
    4.Provide the answer along with the reasoning.
    5.Output format should not have line breaks.

    history:
    {{history}}

    docs:
    {{docs}}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [QuestionSummary]
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
