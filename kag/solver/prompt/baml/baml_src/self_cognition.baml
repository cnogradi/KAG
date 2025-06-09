// Defining a data model.
class Cognition {
  is_cognition_question bool @description(#"
    The question is a self-awareness question"
  "#)
}

// Create a function to extract the resume from a string.
function SelfCognition(question: string) -> Cognition {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    You are an AI assistant. Your task is to determine whether the input question “{{ question }}” is a self-awareness question.
    Extended Requirements: Questions related to self-recognition are also considered, such as "What is your website?"
 
    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [SelfCognition]
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
