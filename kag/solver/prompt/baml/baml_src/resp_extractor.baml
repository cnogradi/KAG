// Defining a data model.
class Passage {
  passage string
}

// Create a function to extract the resume from a string.
function RespExtractor(supporting_fact: string, instruction: string) -> Passage {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"  
    Your job is to act as a professional writer. You will write a good-quality passage that can support the given prediction about the question only based on the information in the provided supporting passages. Now, let's start.
    
    Question:
    
    {{ instruction }}
 
    Known information:
    
    {{ supporting_fact }}
 
    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [RespExtractor]
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
