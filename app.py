import sys
from dataclasses import dataclass
from typing import List, Literal

import qdrant_client

# print(transformers.__path__)
import retrieval
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.docstore.document import Document
from langchain.schema.retriever import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

QDRANT_HOST = st.secrets["QDRANT_HOST"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]



print(sys.path)

st.set_page_config(page_title="Chat QGen - a chatbot designed to assist you in generating questions")
st.title("EduTechBot")
st.write("This is a chatbot for our custom knowledge base - books")

# Defining message class
@dataclass
class Message :
    """Class for keepiong track of chat Message."""
    origin : Literal["Customer","elsa"]
    Message : "str"


# laodinf styles.css
def load_css():
    with open("static/styles.css", "r")  as f:
        css = f"<style>{f.read()} </style>"
        # st.write(css)
        st.markdown(css, unsafe_allow_html = True)

def load_db():

    class CustomRetriever(BaseRetriever):
    
        def _get_relevant_documents(self, query: str) -> List[Document]:
            """
            _get_relevant_documents is function of BaseRetriever implemented here

            :param query: String value of the query

            """
            result_docs = list()
            # Call the search function from retrieval.py
            retrieved_strings = retrieval.search(query)

            # Convert retrieved strings to LangChain Documents
            for string in retrieved_strings:
                doc =  Document(page_content=string)  # Assuming text field is relevant
                result_docs.append(doc)

            return result_docs
    
    retriever = CustomRetriever()
    return retriever

def initialize_session_state() :
    retriever = load_db()

    if "initial_message_sent" not in st.session_state:
        st.session_state.initial_message_sent = False
    
    if "input_value" not in st.session_state:
        st.session_state.input_value = ""

    if "history" not in st.session_state:
        st.session_state.history = []

    if "chain" not in st.session_state :          
        
        prompt_template = """
You are a question-maker agent tasked with creating a comprehensive questionnaire from the following input text out of a book, for use in an academic setting with learners of a higher intelligence level

Your goal is to create a comprehensive questionnaire based on the input text that will test learners' understanding across a range of intelligence levels. The questionnaire should include an equal mix of multiple choice questions (MCQs) and descriptive answer questions.

Here is the step-by-step process to follow:
<scratchpad>
1. Read through the entire input text carefully. Identify the main topics and sections covered.

2. For each main topic or section that you identified:
- Come up with one or more multiple choice questions (MCQs) depending on the size of the text. These MCQs should span a range of difficulties
- For each MCQ, provide 4 answer options and bold the correct answer.
- Generate a description based question for which the learner would have to write a medium-long answer. Try to test them on their capabilities like synthesis, cause/effect, interpretation etc., 
- For each descriptive question, generate a short answer based on the input text that an evaluator can make use of. It doesn't have to be the exact answer. Just points that an evaluator can use to grade the learner.

3. After generating questions for each main topic, review your questionnaire as a whole. Check that
you have:
- Good coverage of all the important points from the input text
- A relatively even distribution of easy, moderate, and challenging questions
- Add in any additional questions needed to improve the balance and coverage

4. Before submitting your final questionnaire, proofread it carefully to check for any errors or opportunities for improvement.
</scratchpad>

Remember, your fundamental goal is to create a questionnaire that comprehensively tests understanding of the input text, for learners with a higher levels. Strive to make your questions clear, insightful, and appropriately challenging. The quality and thoughtfulness of your questions is more important than the quantity.

Do not output the above scratchpad.
Now, Your response should only be a questionnaire .

You will be also be given a history of the conversation made so far followed by the text,  give the answer to the question using the history and the text. Never Hallucinate

History: {context}

Text: {question}

Response:"""

        PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

        llm = OpenAI(base_url="http://localhost:8189/v1", api_key="lm-studio", max_tokens=1500)

        chain_type_kwargs = { "prompt" : PROMPT }

        # model_id = "philschmid/flan-t5-base-samsum"
        # summarizer = transformers.pipeline("summarization", model=model_id)
        # hf = HuggingFacePipeline(pipeline=summarizer)

        template = (
                """Combine the chat history and follow up question into 
                a standalone question. 
                If chat hsitory is empty, use the follow up question as it is.
                Chat History: {chat_history}
                Follow up question: {question}"""
            )
        prompt = PromptTemplate.from_template(template)

        st.session_state.chain = ConversationalRetrievalChain.from_llm(     
                                                                        llm = llm,
                                                                        chain_type = "stuff",
                                                                        memory = ConversationSummaryMemory(llm = llm, memory_key='chat_history', input_key='question', output_key= 'answer', return_messages=True),
                                                                        retriever = retriever,
                                                                        condense_question_prompt = prompt,
                                                                        return_source_documents=False,
                                                                        combine_docs_chain_kwargs=chain_type_kwargs)
        
        "Generate a questionnaire on Robert Beverley's comments on slavery"
        
def on_click_callback():

    load_css()
    customer_prompt = st.session_state.customer_prompt

    if customer_prompt:
        
        st.session_state.input_value = ""
        st.session_state.initial_message_sent = True

        with st.spinner('Generating response...'):

            llm_response = st.session_state.chain(
                {"context": st.session_state.chain.memory.buffer, "question": customer_prompt}, return_only_outputs=True)
            
         

    st.session_state.history.append(
        Message("customer", customer_prompt)
    )
    st.session_state.history.append(
        Message("AI", llm_response)
    )

def main():

    initialize_session_state()
    chat_placeholder = st.container()

    with chat_placeholder:
        for chat in st.session_state.history:
            if type(chat.Message) is dict:
                msg = chat.Message['answer']
            else:
                msg = chat.Message 
            div = f"""
            <div class = "chatRow 
            {'' if chat.origin == 'AI' else 'rowReverse'}">
                <img class="chatIcon" src = "app/static/{'elsa.png' if chat.origin == 'AI' else 'admin.png'}" width=32 height=32>
                <div class = "chatBubble {'adminBubble' if chat.origin == 'AI' else 'humanBubble'}">&#8203; {msg}</div>
            </div>"""
            st.markdown(div, unsafe_allow_html=True)

    with st.form(key="chat_form"):
        cols = st.columns((6, 1))
        
        # Display the initial message if it hasn't been sent yet
        if not st.session_state.initial_message_sent:
            cols[0].text_input(
                "Chat",
                placeholder="Hello, how can I assist you?",
                label_visibility="collapsed",
                key="customer_prompt",
            )  
        else:
            cols[0].text_input(
                "Chat",
                value=st.session_state.input_value,
                label_visibility="collapsed",
                key="customer_prompt",
            )

        cols[1].form_submit_button(
            "Ask",
            type="secondary",
            on_click=on_click_callback,
        )

    st.session_state.input_value = cols[0].text_input


if __name__ == "__main__":
    main()
