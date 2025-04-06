from langchain_openai import OpenAI
from langchain_core.runnables import RunnableSequence
from langchain.prompts import  PromptTemplate
from dotenv import load_dotenv

load_dotenv();

def generate_pet_name(animal_type, pet_color):
    llm =OpenAI(temperature=0.7)

    # Explanation of PromptTemplate:
    # PromptTemplate is a class from LangChain that helps create structured prompts for language models.
    # It allows us to define a template string with placeholders (like {animal_type} and {pet_color})
    # that will be filled in with actual values when the prompt is used.
    # 
    # Benefits of using PromptTemplate:
    # 1. Reusability - The same template can be used multiple times with different inputs
    # 2. Consistency - Ensures prompts follow the same structure each time
    # 3. Readability - Makes the code more understandable by separating prompt structure from logic
    # 4. Dynamic content - Easily insert variables into specific places in your prompt
    #
    # When we use the template below, the placeholders will be replaced with the actual
    # animal type and color provided by the user to create a complete prompt for the LLM.

    prompt_template_name = PromptTemplate(
        input_variables=["animal_type", "pet_color"],
        template="I have a {animal_type} and I want a name for it. Its {pet_color} in color. Suggest me five cool names for my pet."
    )
    
    # Explanation of the chain:
    # In LangChain, a chain is a sequence of components that process data in order.
    # Below, we're creating a simple chain that connects our prompt template to the LLM.
    # This pattern follows the concept of "composability" where complex workflows
    # can be built by connecting simpler components together.
    
    # This line creates a processing chain called 'name_chain'
    # The '|' operator in LangChain connects components together in a sequence
    # Here's what happens when this chain is invoked:
    # 1. prompt_template_name takes variables and formats them into a prompt string
    # 2. llm receives this formatted prompt and generates a text completion
    # So when we call name_chain.invoke() later, it will:
    #   - Format our animal_type and pet_color into the template
    #   - Send that to the OpenAI model
    #   - Return the model's response with pet name suggestions
    name_chain = prompt_template_name | llm
    
    response  = name_chain.invoke({"animal_type":animal_type, "pet_color":pet_color})
    return response

if __name__ == "__main__":
    print(generate_pet_name("cat", "Brown"))