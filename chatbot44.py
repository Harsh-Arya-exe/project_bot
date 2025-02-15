from typing import TypedDict, List, Union
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver 
from dotenv import load_dotenv
import os
import certifi
import logging
from google import genai


client = genai.Client(api_key="AIzaSyBP1QgYKOnW0Ywbz-6R1Y4IByjw1nW3nok")

chat = client.chats.create(model="gemini-2.0-flash")

# Configure logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Set SSL certificate path and load environment
load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()


class AgentState(TypedDict):
    messages: List[Union[AIMessage, HumanMessage]]
    should_continue: bool

class ChatBot:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.retriever = self._init_rag()
        self.workflow = self._init_workflow()

        
        
    def _init_rag(self):
        rag_urls = [
            "https://arxiv.org/abs/2303.18223",
            "https://huyenchip.com/2023/04/11/llm-engineering.html",
            "https://lilianweng.github.io/posts/2023-06-23-agent/"
        ]
        loader = WebBaseLoader(rag_urls)
        docs = loader.load()
        vector_store = FAISS.from_documents(docs, self.embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.4})

    def _init_workflow(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes – each node returns the full state.
        

        workflow.add_node("router", self.router)
        workflow.add_node("general_agent", self.general_agent)
        workflow.add_node("rag_agent", self.rag_agent)

        # Configure conditional edge from router.
        # The routing function now returns a string.
        workflow.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "general_agent": "general_agent",
                "rag_agent": "rag_agent",
                END: END
            }
        )

        # Both agent nodes connect directly to END.
        workflow.add_edge("general_agent", END)
        workflow.add_edge("rag_agent", END)
        workflow.set_entry_point("router")

        return workflow.compile()

    # The routing function now returns a string key.
    def route_decision(self, state: AgentState):
        return state.get("next_node", "general_agent")

    def rag_agent(self, state: AgentState):
        try:
            user_query = next(msg.content for msg in state["messages"] 
                              if isinstance(msg, HumanMessage))
            
            # Retrieve relevant context
            context = self.retriever.invoke(user_query)
            if not context:
                return {**state,
                        "messages": state["messages"] + [
                            AIMessage(content="No technical docs found. Switching to web search...")
                        ],
                        "should_continue": True,
                        "next_node": "general_agent"}
            
            # Generate technical response
            prompt = ChatPromptTemplate.from_template("""
                Technical Context: {context}
                User Question: {query}
                Provide a detailed, technical answer using the context:
            """)
            response = (prompt | self.llm).invoke({
                "context": context,
                "query": user_query
            })
            return {**state,
                    "messages": state["messages"] + [
                        AIMessage(content=response.content)
                    ],
                    "should_continue": False,
                    "next_node": END}  # Mark termination explicitly

        except Exception as e:
            logging.error(f"RAG Error: {str(e)}")
            return {**state,
                    "messages": state["messages"] + [
                        AIMessage(content="Technical system error. Trying web search...")
                    ],
                    "should_continue": True,
                    "next_node": "general_agent"}

    def general_agent(self, state: AgentState):
        try:
            user_query = next(msg.content for msg in state["messages"] 
                              if isinstance(msg, HumanMessage))
            
            # Perform web search
            search = DuckDuckGoSearchResults(max_results=3)
            results = search.invoke(user_query)
            
            # Generate general response
            prompt = ChatPromptTemplate.from_template("""
                Web Results: {context}
                User Question: {query}
                Provide a comprehensive answer using available information:
            """)
            response = (prompt | self.llm).invoke({
                "context": results,
                "query": user_query
            })
            return {**state,
                    "messages": state["messages"] + [
                        AIMessage(content=response.content)
                    ],
                    "should_continue": False,
                    "next_node": END}  # Mark termination explicitly

        except Exception as e:
            logging.error(f"General Agent Error: {str(e)}")
            return {**state,
                    "messages": state["messages"] + [
                        AIMessage(content="Couldn't retrieve information. Please try again.")
                    ],
                    "should_continue": False,
                    "next_node": END}

    def router(self, state: AgentState):
        try:
            if not state["should_continue"]:
                return {**state, "next_node": END}
            
            user_query = next(msg.content for msg in state["messages"] 
                              if isinstance(msg, HumanMessage))
            
            # Check for technical terms.
            technical_terms = {
                'llm', 'transformer', 'attention', 'embedding',
                'vector', 'fine-tuning', 'architecture', 'agent',
                'rag', 'tokenizer', 'pretrained', 'generation'
            }
            is_technical = any(term in user_query.lower() for term in technical_terms)
            next_node = "rag_agent" if is_technical else "general_agent"
            return {**state, "next_node": next_node}

        except Exception as e:
            logging.error(f"Router Error: {str(e)}")
            return {**state, "next_node": "general_agent"}
    
    def get_grok_response(self, user_input: str):
        try:
            # Ensure the Grok model is initialized
            if not self.llm:
                raise Exception("Grok model is not initialized properly.")

            # Use a simple prompt structure for Grok model
            response = self.llm.invoke(user_input)

            # Return the response content if available, else fallback message
            return response.content if hasattr(response, 'content') else "Error: No content returned from Grok model."

        except Exception as e:
            logging.error(f"Grok model error: {str(e)}")
            return f"System error: {str(e)}"

    def run_chat(self, user_input: str):
        try:
            response = chat.send_message(user_input)

            print("Testing: ",response)
            return response


        except Exception as e:
            logging.error(f"Chat Error: {str(e)}")
            return f"System error: {str(e)}", str(e)
