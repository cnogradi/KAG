// Defining a data model.
class Builder {
  python_code string @description(#"
    The generated python code"
  "#)
  not_known bool @description(#"
    The answer to the question is not known"
  "#)
}

// Create a function to extract the resume from a string.
function ExpressionBuilder(now: string, question: string, context: string, error: string) -> Builder {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "Ollama" // Set OPENAI_API_KEY to use this client.
  prompt #"
    Today is {{ now }}

    instruction:
    
        Generate Python code based on the given question and context data to output the result. 
        Show extracted data from context, intermediate calculations and final results for clarity.
        Strictly use input content without making assumptions (e.g., if disability grade isn't mentioned, assume no disability).
        Set not_known flag in response if question cannot be answered.

    output format:

        Output only Python 3.8 code without any additional information

    examples:
        ## Example 1
            ### input
                #### question
                    Calculate interest for 47,000元 at 0.015% daily rate over 612 days
            ### output
                ```python
                # Initial principal
                principal = 47000

                # Daily rate (0.015‰)
                rate = 1.5 / 10000

                # Days
                days = 612

                # Calculate annual rate
                annual_rate = rate * 365

                # Calculate interest
                interest = principal * (annual_rate / 365) * days

                # Calculate total
                total = principal + interest

                print(f"Total amount: {total:.2f} yuan")
                ```

        ## Example 2
            ### input
                #### question
                    Calculate 2019 game revenue growth rate and predict 2020 revenue based on 2018-2019 data
                #### context
                    2018 game revenue: 13M, 2019 game revenue: 15.8M
            ### output
                ```python
                # 2018-2019 revenue (million)
                revenue_2018 = 1300
                revenue_2019 = 1580

                # Calculate growth rate
                growth_rate = (revenue_2019 - revenue_2018) / revenue_2018
                print(f"Growth rate: {growth_rate*100:.2f}%")

                # Predict 2020 revenue
                revenue_2020 = revenue_2019 * (1 + growth_rate)
                print(f"2020 projected revenue: {revenue_2020:.2f} million")
                ```

  # input
    ## question
        {{ question }}
    ## context
        {{ context }}
    ## error
        {{ error }}

    {{ ctx.output_format }}
  "#
}

// Test the function with a sample resume. Open the VSCode playground to run this.
test laibhav_resume {
  functions [ExpressionBuilder]
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
