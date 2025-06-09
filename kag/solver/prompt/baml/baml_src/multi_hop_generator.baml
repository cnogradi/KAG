// Defining a data model.
class MultiAnswer {
  thought string
  answer string
}

// Create a function to extract the resume from a string.
function MultiHopGenerator(context: string, query: string) -> MultiAnswer {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    As an advanced reading comprehension assistant, your task is to answer complex multi-hop questions based on the context I provide. The context I offer includes two parts: a set of documents that are helpful for answering the question, and a step-by-step breakdown of the question along with an analytical thought process. Please combine these two parts of the context to answer the question. Your response start after "Thought: ", where you will methodically break down the reasoning process step by step, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.
    
    NOTE:
    1. Directly quote verbatim when the answer exists in Docs.
    2. Inferred answers must be based on document content so you must bit fabricate information.
    3. I hope your answer matches the answer exactly, so ENSURE that the answer following "Answer:" is concise, such as 14 May, 1832 or yes. THE SHORTER, THE BETTER!!
    4. If the answer is a date, please provide the full date as much as possible, such as 18 May, 1932.3. Pay attention to the differences in part of speech, such as "Japan" and "Japanese," and provide the accurate format according to the question.
    5. If you believe the provided documents cannot answer the question, response with Answer: UNKNOWN.
    6. output format use json

    Context:

    {{ context }}

    Query:
   
    {{ query }}
    
    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [MultiHopGenerator]
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
