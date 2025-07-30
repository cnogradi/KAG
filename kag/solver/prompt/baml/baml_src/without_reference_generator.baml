// Defining a data model.
class WRG {
  answer string
}

// Create a function to extract the resume from a string.
function WithoutRefGenerator(now: string, content: string, query: string)-> WRG {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    You are an information analysis expert, today is {{now}}.
    Answer the question based on the given context. The tone of the response should be smooth and natural, avoiding any mechanical feel, and the language of the output should match the language of the question.  Do not repeat the references when outputting the answer.

    Example 1:
      Task Process Context: 'After calculation by the calculator, 9.2 is greater than 9.1.'
      Given references: ''
      Question: 'Which is larger, 9.1 or 9.2?'
      Answer: 9.2 is larger.

    Task process contextual information:
    
    {{content}}
    
    Question: 
    
    {{query}}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [WithoutRefGenerator]
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
