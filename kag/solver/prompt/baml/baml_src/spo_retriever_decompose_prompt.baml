// Defining a data model.
class SPODecompose {
  step string
  action string
}

class SPODecomposes {
    spos SPODecompose[]
}

// Create a function to extract the resume from a string.
function SPORetrievalDecompose(question: string)-> SPODecomposes {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #" {
        "instruction": "You are a task decomposer. Your job is to break down a complex task into several atomic questions, where each sub-question corresponds to only one Action.",
        "function_description": "functionName is operator name; the function format is functionName(arg_name1=arg_value1,[args_name2=arg_value2, args_name3=arg_value3]),括号中为参数，被[]包含的参数为可选参数，未被[]包含的为必选参数",
        "function": [
          {
              "functionName": "Retrieval",
              "function_declaration": "Retrieval(s=s_alias:type[name], p=p_alias:edge, o=o_alias:type[name], p.prop=value, s.prop=value, o.prop=value)",
               "description": "Retrieval information according to SPO. 's' represents the subject, 'o' represents the object, and they are denoted as variable_name:entity_type[entity_name]. The entity name is an optional parameter and should be provided when there is a specific entity to query. 'p' represents the predicate, which can be a relationship or attribute, denoted as variable_name:edge_type_or_attribute_type. Each variable is assigned a unique variable name, which is used for reference in subsequent mentions. Note that 's', 'p', and 'o' should not appear repeatedly within the same expression; only one set of SPO should be queried at a time. When a variable is a reference to a previously mentioned variable name, the variable name must match the previously mentioned variable name, and only the variable name needs to be provided; the entity type is only given when it is first introduced. And 's.prop', 'o.prop', 'p.prop' represent the properties of subject, object and edge. the subject (s), predicate (p), and object (o) should not repeatedly appear multiple times within the same expression.Constraints can be applied specifically to the predicate (p) for more targeted querying. For multi-hop queries, each hop necessitates a separate retrieval operation. When a current variable references a previously mentioned variable, the variable name must be identical to the one it represents, and only the variable name needs to be stated. The entity type and name should be provided only upon the first mention of the variable; Note Use camelCase for types. Enclose strings in square brackets with backticks to avoid conflicts with keywords."
          }
        ],
        "cases": [
            {
                "query": "Which sports team for which Cristiano Ronaldo played in 2011 was founded last ?",
                "answer": [
                            { "step": "Which Sports Teams Cristiano Ronaldo Played for in 2011 ?",
                              "action": "Retrieval(s=s1:Player[`Cristiano Ronaldo`],p=p1:PlayedForIn2011Year,o=o1:SportsTeam)
                            },
                            {"step": "In which year were these teams established?",
                             "action": "Retrieval(s=o1,p=p2:FoundationYear,o=o2:Year)"
                            }
                          ]
            },
            {
                "query": "Who was the first president of the association which published Journal of Psychotherapy Integration?",
                "answer": [
                            { "step": "Which association that publishes the Journal of Psychotherapy Integration ?",
                              "action": "Retrieval(s=s1:Player[`Psychotherapy Integration`],p=p1:Publish,o=o1:Association)"
                            },
                            { "step": "Who was the first president of that specific association?",
                              "action": "Retrieval(s=o1,p=p2:FirstPresident,o=o2:Person)"
                            }
                          ]
            },
            {
                "query": "When did the state where Pocahontas Mounds is located become part of the United States?",
                "answer": [
                            { "step": "Which State Where Pocahontas Mounds is Located?",
                              "action": "Retrieval(s=s1:HistoricalSite[`Pocahontas Mounds`], p=p1:LocatedIn, o=o1:State)"
                            },
                            { "step": "When did this state become a part of the United States？",
                              "action": "Retrieval(s=o1, p=p2:YearOfBecamingPartofTheUnitedStates, o=o2:Date)"
                            }
                        ]
            },
            {
                "query": "Which of the two tornado outbreaks killed the most people?",
                "answer": [
                            { "step": "Which is the first tornado outbreaks?",
                              "action": "Retrieval(s=s1:Event[`Tornado Outbreak`], p=p1:TheFirst, o=o1:Event)"
                            },
                            { "step": "Which is the second tornado outbreaks?",
                              "action": "Retrieval(s=s2:Event[`Tornado Outbreak`], p=p2:TheSecond, o=o2:Event)"
                            },
                            { "step": "How many people died in the first tornado outbreak?"
                              "action": "Retrieval(s=s1, p=p3:KilledPeopleNumber, o=o3:Number)"
                            },
                            { "step": "How many people died in the second tornado outbreak?",
                              "action": "Retrieval(s=s2, p=p4:KilledPeopleNumber, o=o4:Number)"
                            }
                        ]
            },
            {
                "query": "Which film was released first, Aas Ka Panchhi or Phoolwari?",
                "answer": [
                            { "step": "When was Aas Ka Panchhi released?",
                              "action": "Retrieval(s=s1:Work[`Aas Ka Panchhi`], p=p1:ReleaseTime, o=o1:Date)"
                            },
                            { "step": "When was Phoolwari released?",
                              "action": "Retrieval(s=s2:Work[`Phoolwari`], p=p2:ReleaseTime, o=o2:Date)"
                            }
                          ]
            }
        ],
        "tips": [
            "Each `Step` must contain exactly one `Action`",
            "Each step is an indivisible atomic question, please re-split accordingly."
        ],
        "query": "{{ question }}"
        } 

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [SPORetrievalDecompose]
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
