class Answer {
  answer string @description(#"
    A few paragraphs in markdown format answering the provided questions using the provided data
  "#)
}

// Create a function to extract the resume from a string.
function AnswerQuestion(memory: string, questions: string) -> Answer {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
  You are a helpful assistant responding to questions about the provided data. Generate a response of the target length and format that responds to the user's question, summarizing all information in the provided data appropriate for the response length and format, and incorporating any relevant general knowledge. If you don't know the answer, just say so. Do not make anything up. Do not include information where the supporting evidence for it is not provided. Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
  
  Provided data:

    {{ memory }}

  Questions about provided data:

    {{ questions }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [AnswerQuestion]
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
