class Segment {
  summary string @description(#"
    A brief summary of the section text"
  "#)
  starting_point string @description(#"
    The starting point of the summarized section in the original text, limited to about 4 words exactly from the original. Do not use your own knowledge.  Do not correct the original in any way. This segmentation point will be used to split the original text, so it must be found in the original text!
  "#)
}

class Segments {
  segments Segment[]
}

// Create a function to extract the resume from a string.
function SemanticSegmentClarify(chunk: string, segment: string) -> Segment {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    The segment provided was extracted from the passage below, please correct the segment to match the passage exactly word for word. Do not use your own knowledge.

    Segment:

    {{ segment }}

    Passage:

    {{ chunk }}

    {{ ctx.output_format }}
  "#
}


// Create a function to extract the resume from a string.
function SemanticSegment(chunk: string) -> Segments {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
  Please understand the content of the text in the input field, recognize the structure and components of the text, and determine the segmentation points according to the semantic theme, dividing it into several non-overlapping sections. If the article has recognizable structural information such as chapters, please divide it according to the top-level structure. Please return the results according to the schema definition, including summaries and starting points of the sections. The format must be a JSON string. Please follow the examples given in the example field.

  Examples:

    [
        {
            "input": "Jay Chou (Jay Chou), born on January 18, 1979, in Xinbei City, Taiwan Province, originally from Yongchun County, Fujian Province, is a Mandopop male singer, musician, actor, director, screenwriter, and a graduate of Tamkang Senior High School.\nIn 2000, recommended by Yang Junrong, Jay Chou started singing his own compositions.",
            "output": [
                {
                    "summary": "Personal Introduction",
                    "starting_point": "Jay Chou (Jay Chou), born on January 18"
                },
                {
                    "summary": "Career Start",
                    "starting_point": "\nIn 2000, recommended by Yang Junrong"
                }
            ]
        },
        {
            "input": "Hangzhou Flexible Employment Personnel Housing Provident Fund Management Measures (Trial)\nTo expand the benefits of the housing provident fund system and support flexible employment personnel to solve housing problems, according to the State Council's 'Housing Provident Fund Management Regulations', 'Zhejiang Province Housing Provident Fund Regulations' and the relevant provisions and requirements of the Ministry of Housing and Urban-Rural Development and the Zhejiang Provincial Department of Housing and Urban-Rural Development on flexible employment personnel participating in the housing provident fund system, combined with the actual situation in Hangzhou, this method is formulated.\n This method applies to the voluntary deposit, use, and management of the housing provident fund for flexible employment personnel within the administrative region of this city.\nThe flexible employment personnel referred to in this method are those who are within the administrative region of this city, aged 16 and above, and males under 60 and females under 55, with full civil capacity, and employed in a flexible manner such as part-time, self-employed, or in new forms of employment.\nFlexible employment personnel applying to deposit the housing provident fund should apply to the Hangzhou Housing Provident Fund Management Center (hereinafter referred to as the Provident Fund Center) for deposit registration procedures and set up personal accounts.",
            "output": [
                {
                    "summary": "Background and Basis for Formulating the Management Measures",
                    "starting_point": "To expand the benefits of the housing provident fund system"
                },
                {
                    "summary": "Scope of Application of the Management Measures",
                    "starting_point": "This method applies to the voluntary deposit"
                },
                {
                    "summary": "Definition of Flexible Employment Personnel",
                    "starting_point": "The flexible employment personnel referred to in this method"
                },
                {
                    "summary": "Procedures for Flexible Employment Personnel to Register for Deposit",
                    "starting_point": "Flexible employment personnel applying to deposit the housing provident fund"
                }
            ]
        }
    ]   

    Extract from this content:

    {{ chunk }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [SemanticSegment]
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
