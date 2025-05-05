// Defining a data model.
class Answer {
  answer string
}

// Create a function to extract the resume from a string.
function RespGenerator(memory: string, instruction: string) -> Answer {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    Answer the question based on the given reference. Give me the answer and why. 
    
    The following are given reference:
    
    {{ memory }}
    
    Question: 
    
    {{ instruction }}
 
    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [RespGenerator]
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
