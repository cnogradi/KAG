// Defining a data model.
class Rewrite {
  rewritten_question string @description(#"
    rewriten question as complete as possible without any additional information.
  "#)
}

// Create a function to extract the resume from a string.
function RewriteSubTaskQuery(content: string, input: string) -> Rewrite {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    instruction: 
    
      Based on the context information, replace the referential entities in the current question, and rewrite the current question.
    
    cases: 
      [
        {
            "content": {
                "target question": "What are the representative works of Mr. X's wife?",
                "history_qa": [
                    "Step1: Who is Mr. X's wife?\nanswer: Ms. Y"
                ]
            },
            "question": "What are the representative works of the person mentioned in Step1's answer?",
            "rewrite question": "What are the representative works of Ms. Y?"
        },
        {
            "content": {
                "target question": "In which school do Mr. A's brothers study?",
                "history_qa": [
                    "Step1: Who are Mr. A's brothers?\nanswer: Mr. A's brothers are Mr. B, Mr. C, and Mr. D"
                ]
            },
            "question": "Where do the people retrieved in Step1 receive education?",
            "rewrite question": "Where do Mr. B, Mr. C, and Mr. D receive education?"
        }
    ]

    content:
    
    {{ content }}

    question: 
    
    {{ input }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [RewriteSubTaskQuery]
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
