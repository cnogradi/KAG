// Defining a data model.
class Triple {
  subject string
  predicate string
  object string
}

class Triples {
  triples Triple[]
}

// Create a function to extract the resume from a string.
function ExtractTriples(chunk: string, entities: string) -> Triples {
  // Specify a client as provider/model-names
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    You are an expert specializing in carrying out open information extraction (OpenIE). Please extract any possible relations (including subject, predicate, object) from the given text, and list them following the json format in the example output. If there are none, do not list them. Pay attention to the following requirements:- Each triple should contain at least one, but preferably two, of the named entities in the entity_list.- Clearly resolve pronouns to their specific names to maintain clarity.

    Example entity list:
     
        [
            {
                "name": "The Rezort",
                "category": "Works"
            },
            {
                "name": "2015",
                "category": "Others"
            },
            {
                "name": "British",
                "category": "GeographicLocation"
            },
            {
                "name": "Steve Barker",
                "category": "Person"
            },
            {
                "name": "Paul Gerstenberger",
                "category": "Person"
            },
            {
                "name": "Dougray Scott",
                "category": "Person"
            },
            {
                "name": "Jessica De Gouw",
                "category": "Person"
            },
            {
                "name": "Martin McCann",
                "category": "Person"
            },
            {
                "name": "zombies",
                "category": "Creature"
            },
            {
                "name": "zombie horror film",
                "category": "Concept"
            },
            {
                "name": "humanity",
                "category": "Concept"
            },
            {
                "name": "secure island",
                "category": "GeographicLocation"
            }
        ]

    Example input:

        The Rezort The Rezort is a 2015 British zombie horror film directed by Steve Barker and written by Paul Gerstenberger. It stars Dougray Scott, Jessica De Gouw and Martin McCann. After humanity wins a devastating war against zombies, the few remaining undead are kept on a secure island, where they are hunted for sport. When something goes wrong with the island's security, the guests must face the possibility of a new outbreak.

    Example output: 

        [
            {
                "subject": "The Rezort",
                "predicate": "is",
                "object": "zombie horror film"
            },
            {
                "subject": "The Rezort",
                "predicate": "publish at",
                "object": "2015"
            },
            {
                "subject": "The Rezort",
                "predicate": "released",
                "object": "British"
            },
            {
                "subject": "The Rezort",
                "predicate": "is directed by",
                "object": "Steve Barker"
            },
            {
                "subject": "The Rezort",
                "predicate": "is written by",
                "object": "Paul Gerstenberger"
            },
            {
                "subject": "The Rezort",
                "predicate": "stars",
                "object": "Dougray Scott"
            },
            {
                "subject": "The Rezort",
                "predicate": "stars",
                "object": "Jessica De Gouw"
            },
            {
                "subject": "The Rezort",
                "predicate": "stars",
                "object": "Martin McCann"
            },
            {
                "subject": "humanity",
                "predicate": "wins",
                "object": "a devastating war against zombies"
            },
            {
                "subject": "the few remaining undead",
                "predicate": "are kept on",
                "object": "a secure island"
            },
            {
                "subject": "they",
                "predicate": "are hunted for",
                "object": "sport"
            },
            {
                "subject": "something",
                "predicate": "goes wrong with",
                "object": "the island's security"
            },
            {
                "subject": "the guests",
                "predicate": "must face",
                "object": "the possibility of a new outbreak"
            }
        ]      

    Entity list to use for extraction:

       {{ entities }}

    Extract from this content:

    {{ chunk }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test haibhav_resume {
  functions [ExtractTriples]
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
