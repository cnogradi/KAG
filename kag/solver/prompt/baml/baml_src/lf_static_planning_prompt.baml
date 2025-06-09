// Defining a data model.
class LFStaticPlanning {
  python_code string @description(#"
    The generated python code"
  "#)
  not_known bool @description(#"
    The answer to the question is not known"
  "#)
}

class LFStaticPlan {
  idea string @description(#"
    Before outputting each step's Step and Action, you can add some ideas, such as 'First... Second... Finally...'
  "#)
  step string
  action string
}

class LFStaticPlans {
    plans LFStaticPlan[]
}

// Create a function to extract the resume from a string.
function LFStaticPlanningPrompt(now: string, case: string, query: string,) -> LFStaticPlans {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    Today is {{ now }}

    instruction:

      You are a planning expert who designs plans based on the operators in the function.
    
    function_description:

       functionName is operator name; the function format is functionName(arg_name1=arg_value1,[args_name2=arg_value2, args_name3=arg_value3]). The parameters in brackets are optional parameters, and those not in brackets are mandatory parameters.
     
    function:

      [
          {
              "functionName": "Retrieval",
              "function_declaration": "Retrieval(s=s_alias:type[name], p=p_alias:edge, o=o_alias:type[name])",
               "description": "For text information retrieval, retrieve information based on spo. The elements s, p, and o cannot appear multiple times in the same expression. Constrained queries can be performed on s, p, and o. Multi-hop retrieval involves performing multiple retrievals. When referencing a variable from the current context to a prior one, the variable name must match the referenced variable name exactly, and only the variable name needs to be provided; the entity type and name are given only when first referenced."
          },
          {
              "functionName": "Math",
              "function_declaration": "Math(content=[`XXX` or `o_alias/s_alias`], target=`XXX`)->math_alias",
              "description": "Perform calculations, which include set operations such as numerical calculations or sorting and counting. Content provides input information, which can be text or a referenced variable name. The target is the computational objective, usually the current subproblem. Math_alia is a variable name that represents its calculation result and can be referenced in subsequent actions."
          },
          {
              "functionName": "Deduce",
              "function_declaration": "Deduce(op=judgement|entailment|extract|choice|multiChoice, content=[`XXX` or `o_alias/s_alias`], target=`XXX`)->deduce_alias",
              "description": "Inference refers to the process of inferring search or calculation results to answer questions. op=judgement | entailment | rule | choice | multiChoice respectively represents true or false questions, implication reasoning (such as sentencing), fragment extraction, multiple choice questions, and multiple-choice questions. Content refers to questions, historical conversations, or search results, which can be text fragments or referred to by variable names. The target is the inference objective."
          },
          {
              "functionName": "Output",
              "function_declaration": "Output(A,B,...)",
              "description": "Directly output A, B, ... as answers, where A and B are variable names referring to previous retrieval or calculation results."
          }
        ]
    
    cases: 
    
      {
        {{ case }}
      }
    
    output_format: 
    
      Use text format to output, only output the answer content in the case, do not output '```markdown'" at the beginning
    
    tips: 
      [
        Before outputting each step's Step and Action, you can add some ideas, such as 'First... Second... Finally...',
        Each `Step` must contain exactly one `Action`,
        Each step is an indivisible atomic question, please re-split accordingly,
        Output also needs to be a separate step,
        Step and Action content needs to be output using code style
      ]
    
    query:
    
      {{ query }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [LFStaticPlanningPrompt]
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
