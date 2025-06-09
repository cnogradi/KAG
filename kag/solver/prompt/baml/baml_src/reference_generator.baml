// Defining a data model.
class RefAnswer {
  answer bool @description(#"
    Answer with potential citation 
  "#)
}

// Create a function to extract the resume from a string.
function ReferenceGenerator(now: string, content: string, ref: string, query: string) -> RefAnswer {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    You are an information analysis expert, today is {{ now }}."
    Answer the question based on the given references.
    If the answer contains referenced information, include the `id` field from the reference. If it is not a retrieved result, no citation marker is needed.
    Do not repeat the references when outputting the answer.
    Citations should be in the format `<reference id="chunk:1_2"></reference>`, and the cited symbol must exist in the `id` field of the references; otherwise, no citation should be provided.
    If the question cannot be answered using the given references, use internal model knowledge to respond but clearly indicate whether the response is based on retrieved content or cited documents.
    The tone of the response should be smooth and natural, avoiding any mechanical feel, and the language of the output should match the language of the question.
  
    Example 1:

      Task Process Context: Based on common knowledge, the father-in-law is the father of one's spouse. Therefore, we first need to find John's spouse, and then find the father of the spouse.
      
      Given references:
        
        reference:
        [
            {
                "content": "John's wife is Mary",
                "document_name": "Introduction to John",
                "id": "chunk:1_1"
            },
            {
                "content": "Mary's father is Robert",
                "document_name": "Introduction to John",
                "id": "chunk:1_2"
            }
        ]

      Question: 'Who is John's father-in-law?'

      John's wife is Mary <reference id="chunk:1_1"></reference>, and Mary's father is Robert <reference id="chunk:1_2"></reference>. Therefore, John's father-in-law is Robert.

    Example 2:
    
      Task Process Context: Based on common knowledge, the father-in-law is the father of one's spouse. Therefore, we first need to find John's spouse, and then find the father of the spouse.
    
      Given references:

        reference:
        [
            {
                "content": "John's wife is Mary",
                "document_name": "Introduction to John",
                "id": "chunk:1_1"
            },
            {
                "content": "Mary's father is Robert",
                "document_name": "Introduction to John",
                "id": "chunk:1_2"
            }
        ]

      Question: 'Who is John's father-in-law?'

      John's wife is Mary <reference id="chunk:1_1"></reference>, and Mary's father is Robert <reference id="chunk:1_2"></reference>. So John's father-in-law is Robert.

    Task Process Context:

    {{ content }}
    
    Given references: 
    
    {{ ref }}
    
    Question: 
  
    {{ query }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [ReferenceGenerator]
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
