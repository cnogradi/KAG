// Defining a data model.
class Choice {
  choice string
  no_available_information bool
}

// Create a function to extract the resume from a string.
function DeduceChoice(instruction: string, memory: string) -> Choice {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    Based on the provided options and related answers, choose one option to respond to the question '{{ instruction }}'. No explanation is needed; If there are no available options, simply set the no_available_information boolean to true in the output without any other explanation otherwise set it to false. Ensure that the information provided comes directly and accurately from the retrieved document, without any speculation.
    
    Information: 
    
    {{ memory }}
       
    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [DeduceChoice]
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
