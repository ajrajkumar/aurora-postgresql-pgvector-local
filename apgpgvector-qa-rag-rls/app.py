# Import libraries
import traceback
from typing import List
import json
import boto3
import psycopg
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.llms import Bedrock
from langchain.prompts import PromptTemplate
from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document, HumanMessage
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA
from PIL import Image
from pgvector.psycopg import register_vector
from htmlTemplates import css

user_list = ('Generic','User1','User2','User3')

class CustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        result_docs = list()
        for result in similarity_search(query):
            doc = Document(page_content=result)
            result_docs.append(doc)
        return result_docs


# This function takes the query parameter convert it to embeddings and does the similarity search in the qa_rag_rls table
# Based on the username, it sets the session role to enfoce the Row Level Security in Aurora PostgreSQL database
def similarity_search(query):

    query_embedding = bedrock_embeddings.embed_query(query)
    dbconn = psycopg.connect(conninfo=dbconnstring, connect_timeout=10)
    register_vector(dbconn)

    if 'username' in st.session_state:
        dbconn.execute("set role={}".format(st.session_state.username.lower()))

    result = dbconn.execute("""SELECT document FROM qa_rag_rls ORDER BY
                            embedding <=> %s limit 3;""",(np.array(query_embedding),)).fetchall()
    dbconn.close()
    return [ i[0] for i in result ]


# This function takes a list of PDF documents as input and extracts the text from them using PdfReader.
# It concatenates the extracted text and returns it.
def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Given the extracted text, this function splits it into smaller chunks using the RecursiveCharacterTextSplitter module.
# The chunk size, overlap, and other parameters are configured to optimize processing efficiency.
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
     )

    chunks = text_splitter.split_text(text)
    return chunks

# This function takes the text chunks and username as input and creates a vector store using Bedrock Embeddings (Titan) and pgvector.
# The vector store stores the vector representations of the text chunks, enabling efficient retrieval based on semantic similarity.

def store_vector(username,chunks):

    dbconn = psycopg.connect(conninfo=dbconnstring,autocommit=True)
    register_vector(dbconn)

    ee = bedrock_embeddings.embed_documents(chunks)

    for idx, x in enumerate(chunks):
        dbconn.execute("""INSERT INTO qa_rag_rls (username, document, embedding)
                          VALUES(%s, %s, %s);""", (username.lower(), x, ee[idx]))
    dbconn.close()


# Here, a conversation chain is created using the conversational AI model (Anthropic's Claude v2),
# vector store (created in the previous function)
# This chain allows the Gen AI app to engage in q&A interactions.
def get_qa_chain():

    # Define model_id, client and model keyword arguments for Anthropic Claude v2
    llm = Bedrock(model_id="anthropic.claude-v2:1", client=BEDROCK_CLIENT)
    llm.model_kwargs = {"temperature": 0.5, "max_tokens_to_sample": 8191}

    # The text that you give Claude is designed to elicit, or "prompt", a relevant output.
    # A prompt is usually in the form of a question or instructions.
    # When prompting Claude through the API, it is very important to use the correct `\n\nHuman:` and `\n\nAssistant:` formatting.
    # Claude was trained as a conversational agent using these special tokens to mark who is speaking.
    # The `\n\nHuman:` (you) asks a question or gives instructions, and the`\n\nAssistant:` (Claude) responds.
    prompt_template = """Human: You are a helpful assistant that answers questions directly and only using the information provided in the context below.
    Guidance for answers:
        - Begin your answers with "Based on the context provided: "
        - Simply answer the question clearly and with lots of detail using only the relevant details from the information below.
        - Use bullet-points and provide as much detail as possible in your answer. 
        - Always provide a summary at the end of your answer.
        - If the context is not relevant, please answer the question by using your own knowledge about the topic
        - If the context is not given, treat it as a standalone question
        
    Now read this context below and answer the question at the bottom.
    
    Context: {context}

    Question: {question}
    
    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    retriever_r = CustomRetriever()

    qa_chain = RetrievalQA.from_chain_type(
         llm = llm,
         retriever = retriever_r,
         return_source_documents=True,
         chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

# This function is responsible for processing the user's input question and generating a response from the QA bot
def handle_userinput(user_question, username):

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        response = st.session_state.conversation({'query': user_question})

    except ValueError:
        st.write("Sorry, I didn't understand that. Could you rephrase your question?")
        print(traceback.format_exc())
        return

    chat_history = st.session_state.chat_history
    chat_history.append({
                        "query" : "{}".format(response['query']),
                        "result" : response['result'],
                        "username": st.session_state.username
                        })

    st.session_state.chat_history = chat_history


def write_chat_history():

    if len(st.session_state.chat_history) == 0:
        return

    for chat in reversed(st.session_state.chat_history):
        if st.session_state.username == chat['username'] :
            st.success(chat['query'], icon="🤔")
            st.write(chat['result'])

def main():

    # Set the page configuration for the Streamlit application, including the page title and icon.
    st.set_page_config(page_title="Generative AI Q&A with Amazon Bedrock, Aurora PostgreSQL and pgvector",
                       layout="wide",
                       page_icon=":books::parrot:")

    st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}
            </style>""".format(padding_top=1, padding_bottom=1), unsafe_allow_html=True,)

    st.write(css, unsafe_allow_html=True)

    logo_url = "static/Powered-By_logo-stack_RGB_REV.png"
    st.sidebar.image(logo_url, width=150)

    st.sidebar.markdown(
    """
    ### Instructions:
    1. Choose the user from the drop down 
    2. Browse and upload PDF files
    3. Click Process
    4. Type your question in the search bar to get more insights
    """
    )

    # Check if the conversation and chat history are not present in the session state and initialize them to None.
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_qa_chain()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # A header with the text appears at the top of the Streamlit application.
    st.header("Generative AI Q&A with Amazon Bedrock, Aurora PostgreSQL and pgvector :books::parrot:")
    subheader = '<p style="font-family:Calibri (Body); color:Grey; font-size: 16px;">Leverage Foundational Models from <a href="https://aws.amazon.com/bedrock/">Amazon Bedrock</a> and <a href="https://github.com/pgvector/pgvector">pgvector</a> as Vector Engine</p>'

    # Write the CSS style to the Streamlit application, allowing you to customize the appearance.
    st.markdown(subheader, unsafe_allow_html=True)
    image = Image.open("static/rag-rls-apg.png")
    st.image(image, caption='Generative AI Q&A with Amazon Bedrock, Aurora PostgreSQL and pgvector')

    # Create a text input box where you can ask questions about your documents.
    username = st.selectbox( '##### Documents pertained to the user', user_list)
    st.session_state.username = username
    user_question = st.text_input("##### Ask a question about your documents:", key="question_input", placeholder="What is Amazon Aurora?")

    # Define a Go button for user action
    go_button = st.button("Submit", type="secondary")

    # If the go button is pressed or the user enters a question, it calls the handle_userinput() function to process the user's input.
    if go_button :
        with st.spinner("Processing..."):
            handle_userinput(user_question,username)

    st.write("")

    st.markdown("#### Chat history for {}".format(st.session_state.username))

    write_chat_history()

    with st.sidebar:
        st.subheader("User")
        username = st.selectbox( 'Upload document relevant to the user', user_list)

        st.subheader("Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", type="pdf", accept_multiple_files=False)

        # If the user clicks the "Process" button, the following code is executed:
        # i. raw_text = get_pdf_text(pdf_docs): retrieves the text content from the uploaded PDF documents.
        # ii. text_chunks = get_text_chunks(raw_text): splits the text content into smaller chunks for efficient processing.
        # iii. vectorstore = get_vectorstore(text_chunks): creates a vector store that stores the vector representations of the text chunks.
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                store_vector(username,text_chunks)

                # create conversation chain
                st.session_state.conversation = get_qa_chain()

                st.success('PDF uploaded successfully!', icon="✅")

    with st.sidebar:
        st.divider()

    st.sidebar.markdown(
    """
    ### Sample questions to get started:
    1. What is Amazon Aurora?
    2. How can I migrate from PostgreSQL to Aurora and the other way around?
    3. What is my professional experience ? 
    4. What is my total professional experience ? 
    5. What is my educational background?
    6. What was my last work experience ?
    7. What is Aurora Standard and Aurora I/O-Optimized?
    """
)

if __name__ == '__main__':

    # This function loads the environment variables from a .env file.
    load_dotenv()

    # Define the Bedrock client.
    BEDROCK_CLIENT = boto3.client("bedrock-runtime")
    bedrock_embeddings = BedrockEmbeddings(model_id= "amazon.titan-embed-text-v1", client=BEDROCK_CLIENT)

    # Gather the connection information and create a PostgreSQL connection string
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='apgpg-pgvector-secret')
    database_secrets = json.loads(response['SecretString'])
    dbhost = database_secrets['host']
    dbport = database_secrets['port']
    dbuser = database_secrets['username']
    dbpass = database_secrets['password']
    dbconnstring = "postgresql://{}:{}@{}:{}".format(dbuser,dbpass,dbhost,dbport)

    main()
