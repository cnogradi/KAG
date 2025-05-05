// Defining a data model.
class Entity {
  name string
  type string
  category string
  description string
}

class Entities {
  entities Entity[]
}

// Create a function to extract the resume from a string.
function ExtractEntities(chunk: string, schema: string) -> Entities {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    You're a very effective entity extraction system. Please extract all the entities that are important for knowledge building and questioning, along with type, category and a brief description of the entity. The description of the entity is based on your OWN KNOWLEDGE AND UNDERSTANDING and does not need to be limited to the context. the entity's category belongs taxonomically to one of the items defined by schema, please also output the category. Note: Type refers to a specific, well-defined classification, such as Professor, Actor, while category is a broader group or class that may contain more than one type, such as Person, Works. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string. You can refer to the example for extraction:
    
    Schema to use:

    {{ schema }} 

    Example input:

    The Rezort is a 2015 British zombie horror film directed by Steve Barker and written by Paul Gerstenberger. It stars Dougray Scott, Jessica De Gouw and Martin McCann. After humanity wins a devastating war against zombies, the few remaining undead are kept on a secure island, where they are hunted for sport. When something goes wrong with the island's security, the guests must face the possibility of a new outbreak.",
            
    Example output: 
    
    [
        {
            "name": "The Rezort",
            "type": "Movie",
            "category": "Works",
            "description": "A 2015 British zombie horror film directed by Steve Barker and written by Paul Gerstenberger."
        },
        {
            "name": "2015",
            "type": "Year",
            "category": "Date",
            "description": "The year the movie 'The Rezort' was released."
        },
        {
            "name": "British",
            "type": "Nationality",
            "category": "GeographicLocation",
            "description": "Great Britain, the island that includes England, Scotland, and Wales."
        },
        {
            "name": "Steve Barker",
            "type": "Director",
            "category": "Person",
            "description": "Steve Barker is an English film director and screenwriter."
        },
        {
            "name": "Paul Gerstenberger",
            "type": "Writer",
            "category": "Person",
            "description": "Paul is a writer and producer, known for The Rezort (2015), Primeval (2007) and House of Anubis (2011)."
        },
        {
            "name": "Dougray Scott",
            "type": "Actor",
            "category": "Person",
            "description": "Stephen Dougray Scott (born 26 November 1965) is a Scottish actor."
        },
        {
            "name": "Jessica De Gouw",
            "type": "Actor",
            "category": "Person",
            "description": "Jessica Elise De Gouw (born 15 February 1988) is an Australian actress. "
        },
        {
            "name": "Martin McCann",
            "type": "Actor",
            "category": "Person",
            "description": "Martin McCann is an actor from Northern Ireland. In 2020, he was listed as number 48 on The Irish Times list of Ireland's greatest film actors"
        }
    ]

    Extract from this content:

    {{ chunk }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [ExtractEntities]
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
