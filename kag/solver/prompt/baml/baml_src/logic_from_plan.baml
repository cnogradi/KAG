// Defining a data model.
class Plan {
  step string
  action string
}

class Plans {
    plans Plan[]
}

// Create a function to extract the resume from a string.
function LogicFromPlan(question: string) -> Plans {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
  {
  "instruction": "Decompose the query below into its individual logical steps in the form of groups of simple question (Step) and resulting action (Action) following the provided examples. Make sure the Step is a question and respond in JSON.",
  "function_description": "functionName is operator name; the function format is functionName(arg_name1=arg_value1,[args_name2=arg_value2, args_name3=arg_value3]), The parameters in parentheses are the arguments. Parameters enclosed in [] are optional, while those not enclosed in [] are required.",
    "function": [
      {
          "functionName": "get_spo",
          "function_declaration": "get_spo(s=s_alias:entity_type[entity_name], p=p_alias:edge_type, o=o_alias:entity_type[entity_name])",
          "description": "Find SPO information. 's' represents the subject, 'o' represents the object, and they are denoted as variable_name:entity_type[entity_name]. The entity name is an optional parameter and should be provided when there is a specific entity to query. 'p' represents the predicate, which can be a relationship or attribute, denoted as variable_name:edge_type_or_attribute_type. Each variable is assigned a unique variable name, which is used for reference in subsequent mentions. Note that 's', 'p', and 'o' should not appear repeatedly within the same expression; only one set of SPO should be queried at a time. When a variable is a reference to a previously mentioned variable name, the variable name must match the previously mentioned variable name, and only the variable name needs to be provided; the entity type is only given when it is first introduced."
      },
      {
          "functionName": "count",
          "function_declaration": "count(alias)->count_alias",
          "description": "Count the number of nodes. The parameter should be a specified set of nodes to count, and it can only be variable names that appear in the get_spo query. The variable name 'count_alias' represents the counting result, which must be of int type, and this variable name can be used for reference in subsequent mentions."
      },
      {
          "functionName": "sum",
          "function_declaration": "sum(alias, num1, num2, ...)->sum_alias",
          "description": "Calculate the sum of data. The parameter should be a specified set to sum, which can be either numbers or variable names mentioned earlier, and its content must be of numeric type. The variable name 'sum_alias' represents the result of the calculation, which must be of numeric type, and this variable name can be used for reference in subsequent mentions."      },
      {
          "functionName": "sort",
          "function_declaration": "sort(set=alias, orderby=o_alias or count_alias or sum_alias, direction=min or max, limit=N)",
          "description": "Sort a set of nodes. The 'set' parameter specifies the set of nodes to be sorted and can only be variable names that appear in the get_spo query. The 'orderby' parameter specifies the basis for sorting, which can be the relationship or attribute name of the nodes. If it has been mentioned earlier, an alias should be used. The 'direction' parameter specifies the sorting order, which can only be 'min' (ascending) or 'max' (descending). The 'limit' parameter specifies the limit on the number of output results and must be of int type. The sorted result can be used as the final output."      },
      {
          "functionName": "compare",
          "function_declaration": "compare(set=[alias1, alias2, ...], op=min|max)",
          "description": "Compare nodes or numeric values. The 'set' parameter specifies the set of nodes or values to be compared, which can be variable names that appear in the get_spo query or constants. The 'op' parameter specifies the comparison operation: 'min' to find the smallest and 'max' to find the largest."
      },
      {
          "functionName": "get",
          "function_decl:aration": "get(alias)",
          "description": "Return the information represented by a specified alias. This can be an entity, a relationship path, or an attribute value obtained in the get_spo query. It can be used as the final output result."
      }
    ],
    "examples": [
        {
            "query": "Which sports team for which Cristiano Ronaldo played in 2011 was founded last ?",
            "answer": [ {"Step":"Which Sports Teams Cristiano Ronaldo Played for in 2011 ?",
                         "Action" : "get_spo(s=s1:Player[Cristiano Ronaldo],p=p1:PlayedForIn2011Year,o=o1:SportsTeam)"},
                        {"Step": "In which year were these teams established ?",
                         "Action": "get_spo(s=o1,p=p2:FoundationYear,o=o2:Year)"},
                        {"Step": "Which team was founded last ?",
                         "Action": "sort(set=o1, orderby=o2, direction=max, limit=1)"}
                      ]
        },
        {
            "query": "Who was the first president of the association which published Journal of Psychotherapy Integration?",
            "answer": [ {"Step": "Which association that publishes the Journal of Psychotherapy Integration ?",
                         "Action":"Journal(s=s1:Player[Psychotherapy Integration],p=p1:Publish,o=o1:Association)"},
                        {"Step": "Who was the first president of that specific association?",
                        "Action":"get_spo(s=o1,p=p2:FirstPresident,o=o2:Person)"}
                      ]
        },
        {
            "query": "When did the state where Pocahontas Mounds is located become part of the United States?",
            "answer": [ {"Step": "Which State Where Pocahontas Mounds is Located ?",
                         "Action": "get_spo(s=s1:HistoricalSite[Pocahontas Mounds], p=p1:LocatedIn, o=o1:State)"},
                        {"Step":"When did this state become a part of the United States ？",
                         "Action": "get_spo(s=o1, p=p2:YearOfBecamingPartofTheUnitedStates, o=o2:Date)"}
                      ]
        },
        {
            "query": "Which of the two tornado outbreaks killed the most people?",
            "answer": [ {"Step": "Which is the first tornado outbreaks ?",
                         "Action": "get_spo(s=s1:Event[Tornado Outbreak], p=p1:TheFirst, o=o1:Event)"},
                         {"Step": "Which is the second tornado outbreaks ?",
                          "Action": "get_spo(s=s2:Event[Tornado Outbreak], p=p2:TheSecond, o=o2:Event)"},
                         { "Step": "How many people died in the first tornado outbreak ?",
                           "Action": "get_spo(s=s1, p=p3:KilledPeopleNumber, o=o3:Number)"},
                          { "Step": "How many people died in the second tornado outbreak ?",
                           " Action": "get_spo(s=s2, p=p4:KilledPeopleNumber, o=o4:Number)"},
                           { "Step": "To compare the death toll between two tornado outbreaks to determine which one had more fatalities.",
                             "Action": "compare(set=[o3,o4], op=max)" }
                      }
        }
    ],
    "output_format": "Only output words in answer, for examples: `Step`, `Action` content",
    "query": {{ question }}
    }     

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [LogicFromPlan]
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
