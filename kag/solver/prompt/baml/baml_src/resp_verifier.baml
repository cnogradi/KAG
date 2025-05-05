// Defining a data model.
class Verifier {
  answer string
  answered bool
}

// Create a function to extract the resume from a string.
function RespVerifier(sub_instruction: string, supporting_fact: string) -> Verifier {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    Judging based solely on the current known information and without allowing for inference, are you able to respond completely and accurately to the question '{{ sub_instruction }}'?
    If yes, please set the answered boolean to true, and set the answer field with an accurate response to the question '{{ sub_instruction }}', without restating the question; if no, please set answered boolean to false directly.
 
    Known information: 
    -----
    {{ supporting_fact }} 
    -----
    
    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [RespVerifier]
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
