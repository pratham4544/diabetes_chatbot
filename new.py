import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS Decrepeted 
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate, LLMChain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Google API key
GoogleGenerativeAIEmbeddings.api_key = api_key
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# Function to load and split the text from a PDF file
def get_file_text():
    text = ""
    pdf_reader = PdfReader('data/Bhagavad-Gita As It Is.pdf')
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to load or create a vector store from the text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate   
from langchain_groq import ChatGroq

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Function to create the conversational chain
def get_conversational_chain(question):

    retriever = db.as_retriever()

    model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key='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')

    template = """Ayou are helpful ai assistant had having greate knowledge in medical 
    domain as special in diabetes disease 
    user ask question {question} provide answer as per user ask if you do not know 
    the answer say i do not have knowledge do not make the answer {context}'
    
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    # model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key='AIzaSyCasdO-AH2Q4479mhfCGPkNQlUhLqI18P4')


    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    chain = setup_and_retrieval | prompt | model | output_parser
    
    response = chain.invoke(question)    
    return response



# Function to predict the next question
def predict_next_question(user_question):
    # Create a prompt template for predicting the next question
    prompt_template = """
    Based on the user's question: {user_question}, predict the next question the user might ask.
    """

    # Define the language model
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)
    
    # Create the prompt and LLM chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["user_question"])
    chain = prompt | model | StrOutputParser()
    # chain = LLMChain(llm=model, prompt=prompt)
    
    # Use the chain to predict the next question
    next_question = chain.invoke(user_question)
    
    return next_question


def text_to_img_prompt(response):
    # Create a prompt template for predicting the next question
    prompt_template = """
    Based on the response extract the Example text from the: {response}, write a prompt convert text-to-image generation llm diffusion model
    """

    # Define the language model
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)
    
    # Create the prompt and LLM chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["response"])
    chain = prompt | model | StrOutputParser()
    # chain = LLMChain(llm=model, prompt=prompt)
    
    # Use the chain to predict the next question
    text_to_img = chain.invoke(response)
    
    return text_to_img

def img_generator(img_prompt):
    import requests

    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    image_bytes = query({
        "inputs": img_prompt,
    })
    # You can access the image with PIL.Image for example
    import io
    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes))
    return image


# Function to handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the vector store
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform a similarity search and retrieve relevant documents as context
    docs = new_db.similarity_search(user_question)
    
    # Get the conversational chain
    chain = get_conversational_chain()
    
    # Run the chain with the retrieved context and the user's question
    response = chain.invoke({"context": docs , "question": user_question})

    # Create a container to display the response
    with st.container():
        st.write("Reply: ", response)
        st.write("Generationg img for you arjun...")
        img_prompt = text_to_img_prompt(response=response)
        
        # st.write(f"{img_prompt}")
        try:
            image = img_generator(img_prompt=img_prompt)
            st.image(image)
        except:
            st.error("Our image generation service is currently down due to server issues. We are working to resolve this as soon as possible. Thank you for your patience.")

    # Predict the next question
    next_question = predict_next_question(user_question)
    
    # Display the predicted next question in the sidebar
    st.sidebar.header("Predicted Next Question")
    st.sidebar.write(next_question)

# Main function
def main():
    st.set_page_config("Diabetes Chatbot")
    st.header("Diabetes Trained Chatbot🌟")
    # User input
    import random
    ques =  [   'is type 1 diabetes can cure?',
                'what is Signs and symptoms of diabetes?',
                'what is the history of diabetes?']
    
    selected_ques = random.choice(ques)
    if st.button('Generate Random Question'):
        st.write(selected_ques)
        user_question = selected_ques

        with st.spinner('Wait for it...'):
            if user_question:
                response = get_conversational_chain(user_question)
                st.write(response)
    
    user_question = st.text_input("Ask a Question to Bhagwan Krishna")
    if st.button("Submit"):
        with st.spinner('Wait for it...'):
            if user_question:
                response = get_conversational_chain(user_question)
                st.write(response)

    st.markdown("[For Suggestions & Messages](https://www.linkedin.com/in/prathameshshete/)")

if __name__ == "__main__":
    main()
