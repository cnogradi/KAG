// Defining a data model.
class Reflection {
  thought_questions string[]
}

// Create a function to extract the resume from a string.
function RespReflector(memory: string, instruction: string) -> Reflection {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. Please understand the information gap between the currently known information and the target problem. Your task is to generate one thought in the form of question for next retrieval step directly. DON'T generate the whole thoughts at once!
    
    Known information: 
    
    {{ memory }}
    
    Target question:
    
    {{ instruction }} 
 
    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [RespReflector]
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
