// Defining a data model.
class NamedEntity {
  name string
  category string
  official_name string
}

class NamedEntities {
  entities NamedEntity[]
}

// Create a function to extract the resume from a string.
function ExtractNamedEntities(chunk: string, named_entities: string) -> NamedEntities {
  // Specify a client as provider/model-names
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
  The `input` field contains a user provided context. The `named_entities` field contains extracted named entities from the context, which may be unclear abbreviations, aliases, or slang. To eliminate ambiguity, please attempt to provide the official names of these entities based on the context and your own knowledge. Note that entities with the same meaning can only have ONE official name. Please respond as shown in the `output` field of the provided example.",
    
  Example named entities:
  
    [
        {"name": "American", "category": "GeographicLocation"},
        {"name": "political party", "category": "Organization"},
        {"name": "southern states", "category": "GeographicLocation"},
        {"name": "Civil War", "category": "Keyword"},
        {"name": "House", "category": "Organization"},
        {"name": "Republicans", "category": "Organization"},
        {"name": "chamber", "category": "Organization"},
        {"name": "2006 midterm elections", "category": "Date"}
    ]

  Example input:

    American History.When did the political party that favored harsh punishment of southern states after the Civil War, gain control of the House? Republicans regained control of the chamber they had lost in the 2006 midterm elections.
  
  Example output:

    [
        {
            "name": "American",
            "category": "GeographicLocation",
            "official_name": "United States of America"
        },
        {
            "name": "political party",
            "category": "Organization",
            "official_name": "Radical Republicans"
        },
        {
            "name": "southern states",
            "category": "GeographicLocation",
            "official_name": "Confederacy"
        },
        {
            "name": "Civil War",
            "category": "Keyword",
            "official_name": "American Civil War"
        },
        {
            "name": "House",
            "category": "Organization",
            "official_name": "United States House of Representatives"
        },
        {
            "name": "Republicans",
            "category": "Organization",
            "official_name": "Republican Party"
        },
        {
            "name": "chamber",
            "category": "Organization",
            "official_name": "United States House of Representatives"
        },
        {
            "name": "midterm elections",
            "category": "Date",
            "official_name": "United States midterm elections"
        }
    ]  

    The entities to standardize:

    {{ named_entities }}

    Standardize the above entities from this content:

    {{ chunk }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test haibhav_resume {
  functions [ExtractNamedEntities]
  args {
    resume #"
      Vaibhav Gupta
      vbv@boundaryml.com
ExtractTEntities
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
