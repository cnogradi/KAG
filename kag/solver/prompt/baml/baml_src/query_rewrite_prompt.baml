// Defining a data model.
class RewrittenPrompt {
  query bool @description(#"
    The reformulated question string
  "#)
}

// Create a function to extract the resume from a string.
function QueryRewritePrompt(context: string, query: string) -> RewrittenPrompt {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    instruction: 
    
      You are an expert skilled in restructuring multi-step queries. The original multi-step query has been broken down into several straightforward single-step query, each of which may explicitly (in the form of "\{\{i.output\}\}") or implicitly (such as "the person," "the city") rely on the answers to preceding queries. Please replace the references to the answers of previous queries in the provided query with the actual answers, based on the thought by the preceding queries, and rephrase the question accordingly. Refer to the example for the output format.
      The example field provides a simple example for reference. Please return only the reformulated question string IN JSON FORMAT, follow the format of  output field in the example.
        
      example:
      
      {
          "input": {
              "query": "\{\{0.output\}\} has won awards, which ones have not been obtained by /{/{1.output/}/}?"
          },
          "context": {
              "0": {
                  "query": "Who played Sun Wukong in Journey to the West?",
                  "output": [
                      "Liu Xiao Ling Tong was born in 1959 and hails from Shaoxing, Zhejiang. He is a National Class-A Actor and has served as an adjunct professor at the School of Humanities, Zhejiang University. During the filming of the TV series 'Journey to the West' in 1982, he played the role of Sun Wukong throughout, winning the Best Actor Award at the 6th China Golden Eagle Awards for his performance. Answer: Liu Xiao Ling Tong"
                  ],
              },
              "1": {
                  "query": "Who sing the song 'Blue and White Porcelain'?",
                  "output": [
                      "The song 'Blue and White Porcelain' was written by Vincent Fang, composed by Jay Chou, arranged by Roger Chung, and included in Jay Chou’s eighth studio album 'I'm Busy', released on November 2, 2007. Answer: JayChou"
                  ],
              },
          },
          "output": {
              "query": "Among the awards won by Liu Xiao Ling Tong, which ones have not been won by Jay Chou?"
          },
        },
      }

    context:

    {{ context }}

    query:

    {{ query }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [QueryRewritePrompt]
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
