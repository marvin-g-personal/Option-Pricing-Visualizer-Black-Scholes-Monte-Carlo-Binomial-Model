import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Options Pricing Visualizer",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Container styling */
div.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Title container */
.title-container {
    position: absolute;
    top: 80%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 100%;
    text-align: center;
}

/* Typing Animation Styles */
.typing-line1 {
    font-size: 2.5rem;
    color: white;
    margin: 0 auto;
    text-align: center;
    font-weight: bold;
    padding: 10px;
    white-space: pre-line;
    position: relative;
    top: 260px;
}
            
/* Typing Animation Styles */
.typing-line2 {
    font-size: 2.5rem;
    color: white;
    font-weight: 500;
    margin: 0 auto;
    text-align: center; 
    white-space: nowrap;
    padding: 5px;
    white-space: pre-line;
    position: relative;
    top: 250px;
}

/* Button styling */
.stButton {
    display: flex;
    justify-content: center;
    margin-top: 270px;
    width: 100%;
}

.stButton > button {
    background-color: black;
    color: white;
    border: none;
    padding: 20px 40px;
    border-radius: 25px;
    font-size: 40px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: white;
    box-shadow: 0 0 10px #B22222;
    color: black;
    transform: scale(1.1);
}

/* Footer styling */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #2F4F4F;
    color: white;
    padding: 10px 0;
}

.footer-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    padding: 0 20px;
}

.footer-section {
    flex: 1;
    text-align: center;
    white-space: nowrap;
}

.footer-section.left {
    text-align: left;
}

.footer-section.right {
    text-align: right;
}

.footer a {
    color: #ADD8E6;
    text-decoration: none;
    margin: 0 10px;
}

.footer a:hover {
    text-decoration: underline;
}

/* Title and Button Container Styling */
.title-button-container {
    position: absolute;
    top: 70%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 100%;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Center container for content
st.markdown('<div class="title-button-container">', unsafe_allow_html=True)

# Title lines
st.markdown('<div class="typing-line1">Options Pricing Visualizer:</div>', unsafe_allow_html=True)
st.markdown('<div class="typing-line2">Black-Scholes, Monte Carlo & Binomial Models</div>', unsafe_allow_html=True)

# Button with navigation
if st.button("Try it now!"):
    st.session_state.example_params = True
    st.switch_page("pages/Main.py") 

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-content">
        <div class="footer-section left">© marvin-g-personal</div>
        <div class="footer-section">
            Created By: Marvin Gandhi, Bennett Franciosi, Elaine Zou, Vaibhav Singh & Tafari Darosa-Levy
        </div>
        <div class="footer-section right">
            <a href="https://dataclub.northeastern.edu/" target="_blank">DATA @ NEU</a>
            <a href="https://linkedin.com/in/marvin-gandhi" target="_blank">LinkedIn</a>
            <a href="https://github.com/marvin-g-personal/Option-Pricing-Visualizer-Black-Scholes-Monte-Carlo-Binomial-Model/tree/main" target="_blank">GitHub</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
