import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()
from langchain_groq import ChatGroq
#lets create the state first

class State(TypedDict):
    raw_input:str
    edited_text:str
    script_text:str
    final_text:str

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

def editor_node(state:State)->dict:
    prompt=(
        "You are an expert copyeditor. You will be given a raw input text. Your task is to edit the text for clarity, grammar, and style while preserving the original meaning. Please provide the edited version of the text."
        f"\n\nRaw Input:\n{state['raw_input']}"
    )
    response=llm.invoke(prompt)

    return {"edited_text":response.content.strip()

}

def script_node(state:State)->dict:
    """  stage 2: formats the edited text into a script format. """
    print("executing script node...")
    prompt= (
        "you are a charismatic youtube content creator. You will be given an edited text. Your task is to format the text into a script format suitable for a youtube video. Please provide the script version of the text." \
        f"\n\nEdited Text:\n{state['edited_text']}"
    )
    response=llm.invoke(prompt)
    return {"script_text":response.content.strip()}

def translator_node(state:State)->dict:
    """ stage 3: translates the script into a transcript format. """
    print("executing translator node...")
    prompt= (
       " you are expert content localizer for the indain market.take the script and convert into the natural ,follwing hinglish style of speaking, and make it sound like a natural spoken transcript. Please provide the transcript version of the script." \
        f"\n\nScript Text:\n{state['script_text']}"
    )
    response=llm.invoke(prompt)
    return {"final_text": response.content.strip()}


#now the states and nodes are created and we have to connect it
#and for that we have to create the graph and so we use the edges 
#edges are very important because they connect the nodes and the states and they are the ones that define the flow of the graph




#create the graph
graph=StateGraph(State)

#add the start and end nodes
graph.add_node("editor",editor_node)
graph.add_node("script",script_node)
graph.add_node("translator",translator_node)

#add edges(seq one after another)

graph.add_edge(START,"editor")
graph.add_edge("editor","script")
graph.add_edge("script","translator")
graph.add_edge("translator",END)

#complie the graph
app = graph.compile()

result=app.invoke({"raw_input":"Hello, I am a software engineer and I love to code. I have been working in the industry for 5 years and have experience in various programming languages. I am passionate about learning new technologies and improving my skills."})


#output
print("YOUR RESULT IS READY: \n\n")
print(result["final_text"])


