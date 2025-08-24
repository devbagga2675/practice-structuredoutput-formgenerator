# zero_shot_form_generator.py
# --- Installation ---
# pip install streamlit langchain langchain-google-genai pydantic

import streamlit as st
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
# --- Pydantic Schemas for Structured Output ---
# This defines the exact structure we want the AI to return.
class Question(BaseModel):
    question_text: str = Field(description="The full text of the question.")
    question_type: Literal['multiple_choice', 'short_answer', 'paragraph', 'checkboxes', 'linear_scale']
    options: List[str] = Field(default=[], description="An array of STRINGS for choice-based or scale questions. For example, for a scale of 1-5, this must be [\"1\", \"2\", \"3\", \"4\", \"5\"].") 
    correct_answer: Optional[str] = Field(default=None, description="The correct answer for quiz questions.")

class FormStructure(BaseModel):
    title: str = Field(description="A short, engaging title for the form.")
    description: str = Field(description="A brief description of the form's purpose.")
    questions: List[Question] = Field(description="A list of question objects for the form.")

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Zero-Shot Form Generator", layout="wide")
st.title("📄 Zero-Shot Form Generator")
st.write(
    """
    This app showcases how a detailed system prompt can be used to generate structured JSON output from an LLM. 
    Fill in the details below to generate a form.
    """
)

# --- API Key Management ---
try:
    gemini_api_key = st.secrets["GOOGLE_API_KEY"]
    # gemini_api_key = os.environ.get("GOOGLE_API_KEY")
except KeyError:
    st.error("GOOGLE_API_KEY is not set. Please add it to your Streamlit secrets.")
    st.stop()

# --- Form Generation Logic ---
async def generate_form(purpose: str, audience: str, num_questions: int, context: str):
    """Generates a form using a LangChain chain with a JSON output parser."""
    
    # 1. Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=gemini_api_key)
    
    # 2. Define the Pydantic model for the output parser
    parser = JsonOutputParser(pydantic_object=FormStructure)

    # 3. Create the detailed "zero-shot" prompt template
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert form creator. Generate a structured JSON object for a form based on the following requirements.

    USER REQUIREMENTS:
    - Primary Purpose/Topic: {purpose}
    - Target Audience: {target_audience}
    - Number of Questions to Generate: {num_questions}
    - Additional Instructions: {additional_context}
    
    JSON FORMAT INSTRUCTIONS:
    {format_instructions}
    """)

    # 4. Create the LangChain chain
    chain = prompt_template | llm | parser

    # 5. Invoke the chain asynchronously
    with st.spinner("Generating your form..."):
        response_dict = await chain.ainvoke({
            "purpose": purpose,
            "target_audience": audience,
            "num_questions": num_questions,
            "additional_context": context or "None",
            "format_instructions": parser.get_format_instructions(),
        })
    
    return FormStructure(**response_dict)

# --- Streamlit Form for User Input ---
with st.form("generation_form"):
    st.subheader("Form Details")
    purpose_input = st.text_input("What is the form's purpose or topic?", "A quiz about the solar system")
    audience_input = st.text_input("Who is the target audience?", "5th-grade students")
    num_questions_input = st.slider("Number of questions?", min_value=3, max_value=15, value=5)
    context_input = st.text_area("Additional context (optional)", "Include at least one question about dwarf planets.")
    
    submitted = st.form_submit_button("✨ Generate Form")

if submitted:
    # Use asyncio.run() to execute the async function
    generated_form = asyncio.run(generate_form(purpose_input, audience_input, num_questions_input, context_input))
    
    st.divider()
    st.subheader("🎉 Generated Form")
    
    # Display the form in a structured, user-friendly way
    st.markdown(f"**Title:** {generated_form.title}")
    st.markdown(f"**Description:** {generated_form.description}")
    
    for i, question in enumerate(generated_form.questions):
        with st.expander(f"**Question {i+1}:** {question.question_text}"):
            st.markdown(f"**Type:** `{question.question_type}`")
            if question.options:
                st.markdown("**Options:**")
                for option in question.options:
                    st.markdown(f"- {option}")
            if question.correct_answer:
                st.success(f"**Correct Answer:** {question.correct_answer}")

    st.divider()
    st.subheader("Raw JSON Output")
    st.json(generated_form.dict())