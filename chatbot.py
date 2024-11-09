"""Streamlit-based chatbot for educational question generation."""

from dataclasses import dataclass
from typing import List, Literal

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.docstore.document import Document
from langchain.schema.retriever import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

import retrieval

@dataclass
class Message:
    """Chat message structure."""
    origin: Literal["customer", "AI"]
    message: str

def load_db() -> BaseRetriever:
    """Initialize custom retriever for document search."""
    class CustomRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str) -> List[Document]:
            retrieved_strings = retrieval.search(query)
            return [Document(page_content=string) for string in retrieved_strings]
    
    return CustomRetriever()

def initialize_chain(retriever: BaseRetriever) -> ConversationalRetrievalChain:
    """Initialize the conversation chain with LLM and memory."""
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
    
    llm = OpenAI(base_url="http://localhost:8189/v1", api_key="lm-studio", max_tokens=4096)
    
    question_prompt = PromptTemplate.from_template(
        "Combine the chat history and follow up question into a standalone question. "
        "If chat history is empty, use the follow up question as it is. "
        "Chat History: {chat_history} "
        "Follow up question: {question}"
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        memory=ConversationSummaryMemory(
            llm=llm,
            memory_key='chat_history',
            input_key='question',
            output_key='answer',
            return_messages=True
        ),
        retriever=retriever,
        condense_question_prompt=question_prompt,
        return_source_documents=False,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        }
    )

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "chain" not in st.session_state:
        st.session_state.chain = initialize_chain(load_db())
    
    for key in ["initial_message_sent", "input_value", "history"]:
        if key not in st.session_state:
            st.session_state[key] = False if key == "initial_message_sent" else "" if key == "input_value" else []

def handle_user_input():
    """Process user input and generate response."""
    prompt = st.session_state.customer_prompt
    if prompt:
        st.session_state.input_value = ""
        st.session_state.initial_message_sent = True
        
        with st.spinner('Generating response...'):
            response = st.session_state.chain(
                {
                    "context": st.session_state.chain.memory.buffer,
                    "question": prompt
                },
                return_only_outputs=True
            )
            
            st.session_state.history.extend([
                Message("customer", prompt),
                Message("AI", response)
            ])

def main():
    """Main application function."""
    st.set_page_config(page_title="QuizMasterAI")
    st.title("QuizMasterAI")
    st.write("An intelligent chatbot for generating educational questions from books")
    
    initialize_session_state()
    
    chat_placeholder = st.container()
    with chat_placeholder:
        for chat in st.session_state.history:
            message = chat.message['answer'] if isinstance(chat.message, dict) else chat.message
            st.markdown(f"**{'AI' if chat.origin == 'AI' else 'User'}:** {message}")
    
    with st.form(key="chat_form"):
        cols = st.columns((6, 1))
        cols[0].text_input(
            "Chat",
            placeholder="How can I help you?" if not st.session_state.initial_message_sent else "",
            value=st.session_state.input_value,
            label_visibility="collapsed",
            key="customer_prompt"
        )
        cols[1].form_submit_button("Ask", type="secondary", on_click=handle_user_input)

if __name__ == "__main__":
    main()
