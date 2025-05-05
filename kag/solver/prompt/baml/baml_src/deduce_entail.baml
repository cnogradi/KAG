// Defining a data model.
class Entail {
  entail string
  no_available_information bool
}

// Create a function to extract the resume from a string.
function DeduceEntail(instruction: string, memory: string) -> Entail {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    Based on the provided information, first determine whether you can directly respond to the instruction '{{ instruction }}'. If you can directly answer, reply with the answer without any explanation; if you cannot answer directly but there is related information, summarize the key information related to the instruction '{{ instruction }}' and clearly explain why it is related; if there is no relevant information, simply set the no_available_information boolean in your response without explanation.
    
    [Information]: '{{ memory }}'

    Ensure that the information provided comes directly and accurately from the retrieved document, without any speculation.

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [DeduceEntail]
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
