// Defining a data model.
class RewriteQuery {
  rewritten_question bool @description(#"
    rewriten question as complete as possible without any additional information.
  "#)
}

// Create a function to extract the resume from a string.
function RewriteSubQuery(history_qa: string, question: string) -> RewriteQuery {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    instruction: 
    
      Please rewrite the current question based on historical Q&A information and complete the entities referred to in the question.
    
    cases: 
    
    [
        {
            "history_qa": "Who is the director of God'S Gift To Women?  \nanswer: Michael Curtiz\nWho is the director of Aldri Annet Enn Bråk?  \nanswer: Edith Carlmar",
            "question": "What is the birth year of the director of God'S Gift To Women? "
            "answer": "What is the birth year of Michael Curtiz"
        },
        {
            "history_qa": "Who is the director of God'S Gift To Women?  \nanswer: Michael Curtiz\nWhat is the birth year of the director of God'S Gift To Women?  \nanswer: 1886\nWho is the director of Aldri Annet Enn Bråk?  \nanswer: Edith Carlmar",
            "question": "What is the birth year of the director of Aldri Annet Enn Bråk? "
            "answer": "What is the birth year of Edith Carlmar"
        }
    ]

    history_qa: 
    
      {{ history_qa }}

    question:
  
      {{ question }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [RewriteSubQuery]
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
