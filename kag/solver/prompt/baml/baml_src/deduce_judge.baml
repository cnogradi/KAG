// Defining a data model.
class Judge {
  assessable bool
  no_available_information bool
}

// Create a function to extract the resume from a string.
function DeduceJudge(instruction: string, memory: string) -> Judge {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    Based on the provided information, first determine if the question '{{ instruction }}' can be directly assessed. If it can be directly answered, simply respond with true or false if assessable based on the provided information, no explanation needed; If there is no relevant information, simply set the boolean no_available_information in your response without explanation.
    
    Information: 
    
    {{ memory }}
        
    Ensure that the information provided comes directly and accurately from the retrieved document, without any speculation.
      
    Question: '{{ instruction }}'

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [DeduceJudge]
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
