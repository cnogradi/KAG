// Defining a data model.
class QNer {
  name string
  category string
}

class QNers {
    output QNer[]
}

// Create a function to extract the resume from a string.
function QuestionNER(schema: string[], input: string) -> QNers {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    You are an expert in named entity recognition. Please extract entities and that match the schema definition from the input. Please respond in the format of a JSON string. You can refer to the example for extraction.
    
    schema to use: 
    
    {{ schema }}

    example:
    
      input: "Which magazine was started first, Arthur's Magazine or First for Women?"
      
      output:
      
      [
          {
              "name": "First for Women",
              "category": "Works"
          },
          {
              "name": "Arthur's Magazine",
              "category": "Works"
          }
       ]

    
    input:
    
    {{ input }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [QuestionNER]
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
