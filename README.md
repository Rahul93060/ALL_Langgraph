# LangGraph Projects

A collection of small projects built with LangGraph, each one using a different workflow pattern. The common idea across all of them is the same: define a shared state, split the work into nodes, and connect the nodes with edges so the flow is easy to follow and control.

## Projects

### Sequential workflow
Basic graph where nodes run one after another and each node updates the shared state. This is the starting point for everything else in the repo.

### Parallel workflow
Multiple nodes run at the same time on the same input. Reducers are used to merge their outputs back into a single state without overwriting each other.

### Conditional workflow
The graph decides the next step based on the current state, so different inputs take different paths through the nodes.

### PDF RAG with LangGraph
Upload a PDF and ask questions about it. The PDF is split into chunks, embedded and stored in a vector store. When a question comes in, the graph retrieves the relevant chunks and passes them to the LLM to generate the answer. Each stage (load, chunk, retrieve, answer) is a separate node.

### Iterative workflow
A loop where the output is generated, checked, and sent back for improvement until it meets the condition or hits the retry limit.

### Human in the loop
The graph pauses at a chosen step and waits for human input or approval before continuing.

## Setup

1. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the root folder and add your API key:
   ```
   OPENAI_API_KEY=your_key_here
   ```
   `.env` is in `.gitignore`, so it is not pushed to GitHub.

