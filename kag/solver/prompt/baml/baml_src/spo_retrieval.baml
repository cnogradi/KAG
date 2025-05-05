// Defining a data model.
class SPOs {
  spos string[]
}

// Create a function to extract the resume from a string.
function SPORetrieval(question: string, mention: string, candis: string[])-> SPOs {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    You are a language expert. Your task is to select the correct SPO (Subject Predicate Object) text from the given SPO candidates according to the following rules to answer the given question. Ensure that the selected SPO matches the SPO mention or appropriately answers the question. Provide a JSON list of the selected SPO candidates that best answer the question based on the spo mention.

    requirements: 

    1. The output must be selected from the SPO candidates and remain consistent with their content, presented in a list format.
    2. If there is no suitable answer in the SPO candidates, output an empty list. Ensure that the output is highly relevant to the question or SPO mention.
    3. If there are multiple correct answers in the SPO candidates, output all matching SPOs in a JSON list format.
  
    example1: 

        question: Is the Woman's Viewpoint a British publication?
        spo_mention: Publication[Woman's Viewpoint] Nationality Country
        spo_candidates: 
        
        [
            "the woman s viewpoint ranFrom 1923 to 1927",
            "the woman s viewpoint publishedBy florence m sterling",
            "the woman s viewpoint foundedIn 1923",
            "the woman s viewpoint foundedIn texas",
            "the woman s viewpoint was a woman s magazine",
            "rolandos liatsos starredIn woman in mind",
            "rolandos liatsos starredIn woman in mind"
        ]

        analysis: The question seeks the nationality of the publication "Woman's Viewpoint." Among the SPO candidates, "the woman s viewpoint foundedIn texas" indicates the location of its founding, which relates to its nationality.

        output:

        { spos: [
              "the woman s viewpoint foundedIn texas"
          ]
        }
  
    example2:

      question: Who is the German musician whose hand the manuscript for Flute Sonata in C major, BWV 1033 is in?
      spo_mention: Entity[Flute Sonata in C major, BWV 1033] InHandOf Entity
      spo_candidates: 
      
        [
              "flute sonata in c major bwv 1033 isAttributedTo johann sebastian bach",
              "flute sonata in c major bwv 1033 isFor flute or recorder and basso continuo",
              "flute sonata in c major bwv 1033 is a sonata in 4 movements"
        ]

      analysis: The question aims to identify who holds the manuscript of "Flute Sonata in C major, BWV 1033." The SPO candidate "flute sonata in c major bwv 1033 isAttributedTo johann sebastian bach" implies ownership, indicating that Johann Sebastian Bach holds the manuscript.
      output:

      { spo: [
            "flute sonata in c major bwv 1033 isAttributedTo johann sebastian bach"
        ]
      }

    question: 
    
    {{ question }}

    spo_mention: 
    
    {{ mention }}

    spo_candidates: 
    
    {{ candis }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [SPORetrieval]
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
