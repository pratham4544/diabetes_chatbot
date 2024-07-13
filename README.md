## Streamlit Chatbot with Bhagavad Gita Knowledge

This project implements a Streamlit application that allows users to chat with a virtual Lord Krishna, seeking guidance based on the Bhagavad Gita.

**Functionalities:**

* Users can ask questions related to life and goals.
* The chatbot responds with answers derived from the Bhagavad Gita, including:
    * Relevant shlokas (verses)
    * Explanations based on the Gita's teachings
    * Illustrative examples for better understanding
* The application predicts the next possible question based on the user's input (work in progress).
* Generates an image based on the response using a text-to-image generation model (work in progress).

**How to Use:**

1. **Prerequisites:**
    * Python 3.x
    * Streamlit (`pip install streamlit`)
    * Langchain (`pip install langchain`)
    * PyPDF2 (`pip install pypdf2`)
    * Transformers (`pip install transformers`)
    * dotenv (`pip install python-dotenv`)
2. **Download Models (replace placeholders with download links):**
    * Create a folder named `models` in the project directory.
    * Download the pre-trained language model:
        ```bash
        curl https://huggingface.co/docs/hub/en/models-downloading > models/gemini-1.5-pro-latest
        ```
    * Download the pre-trained embedding model:
        ```bash
        curl https://discuss.huggingface.co/t/is-there-a-way-to-download-embedding-model-files-and-load-from-local-folder-which-supports-langchain-vectorstore-embeddings/52518 > models/embedding-001
        ```
3. **Data:**
    * Place the Bhagavad Gita text file named "Bhagavad-Gita As It Is.pdf" in the `data` folder.
4. **API Key:**
    * Create a file named `.env` in the main directory and add your Google API key:
      ```
      GOOGLE_API_KEY=YOUR_API_KEY
      ```
5. **Run the application:**
    ```bash
    streamlit run app.py
    ```

**Disclaimer:**

* The text-to-image generation and next question prediction functionalities are under development.
* Ensure you have downloaded the required libraries and models before running the application.

**Further Enhancements:**

* Integrate a more sophisticated question prediction model.
* Implement sentiment analysis to provide more tailored responses.
* Allow users to explore specific chapters or verses of the Bhagavad Gita.
