import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

template = """\
For the following text, extract the following \
information:

sentiment: Is the customer happy with the product? 
Answer Positive if yes, Negative if \
not, Neutral if either of them, or Unknown if unknown.

delivery_days: How many days did it take \
for the product to arrive? If this \
information is not found, output No information about this.

price_perception: How does it feel the customer about the price? 
Answer Expensive if the customer feels the product is expensive, 
Cheap if the customer feels the product is cheap,
not, Neutral if either of them, or Unknown if unknown.

Format the output as bullet-points text with the \
following keys:
- Sentiment
- How long took it to deliver?
- How was the price perceived?

Input example:
This dress is pretty amazing. It arrived in two days, just in time for my wife's anniversary present. It is cheaper than the other dresses out there, but I think it is worth it for the extra features.

Output example:
- Sentiment: Positive
- How long took it to deliver? 2 days
- How was the price perceived? Cheap

text: {review}
"""

#PromptTemplate variables definition
prompt = PromptTemplate(
    input_variables=["review"],
    template=template,
)

# ---- Streamlit Page Config ----
st.set_page_config(page_title="Extract Info from Product Reviews")
st.title("🛍️ Extract Key Information from Product Reviews")

# ---- Layout ----
col1, col2 = st.columns(2)

with col1:
    st.markdown("### What this app extracts:")
    st.markdown("- Sentiment\n- Delivery time\n- Price perception")

with col2:
    st.write("📺 Follow [Learning Logic](https://www.youtube.com/@LearningLogic_official)")

# ---- Groq API Key Input ----
groq_api_key = st.text_input(
    label="🔑 Enter Groq API Key",
    placeholder="Ex: gsk-live-xxxxxxxxxxxxxxxxx",
    type="password"
)

def load_LLM(api_key: str, model_name: str = "llama3-70b-8192"):
    if not api_key or not api_key.startswith("gsk_"):
        return None
    return ChatGroq(groq_api_key=api_key, model_name=model_name)

# ---- Review Input ----
st.markdown("### ✍️ Enter the Product Review")
review_input = st.text_area(
    label="",
    placeholder="Type or paste a product review here...",
    key="review_input",
    height=200
)

if len(review_input.split()) > 700:
    st.warning("⚠️ Please limit your review to 700 words.")
    st.stop()

# ---- Process and Output ----
if review_input and groq_api_key:
    llm = load_LLM(groq_api_key.strip())
    
    if llm is None:
        st.error("❌ Please enter a valid Groq API key.")
        st.stop()
    
    with st.spinner("Analyzing review..."):
        formatted_prompt = prompt.format(review=review_input)
        try:
            result = llm.invoke(formatted_prompt)
            st.markdown("### ✅ Key Data Extracted:")
            st.success(result.content)
        except Exception as e:
            st.error(f"Error: {e}")