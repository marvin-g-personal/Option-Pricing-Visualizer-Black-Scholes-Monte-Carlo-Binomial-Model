import streamlit as st

st.set_page_config(
    page_title="Options Pricing Visualizer",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* hide menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

html, body {
    height: 100%;
    margin: 0;
}

/* background overlay */
body {
    background: linear-gradient(-45deg, #1e3c72, #2a5298);
    background-size: 400% 400%;
    animation: gradientBackground 15s ease infinite;
    cursor: url('https://cdn-icons-png.flaticon.com/512/25/25231.png'), crosshair; 
}

div.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* background motion */
.geometric-pattern {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%; /* Ensure it covers the full height */
    background: linear-gradient(45deg, #2c3e50 25%, transparent 25%), 
                linear-gradient(-45deg, #2c3e50 25%, transparent 25%), 
                linear-gradient(45deg, transparent 75%, #2c3e50 75%), 
                linear-gradient(-45deg, transparent 75%, #2c3e50 75%);
    background-size: 50px 50px;
    background-position: 0 0, 0 25px, 25px -25px, -25px 0px;
    animation: movePattern 10s linear infinite;
}

@keyframes movePattern {
    0% { background-position: 0 0, 0 25px, 25px -25px, -25px 0px; }
    100% { background-position: 50px 50px, 50px 75px, 75px 25px, 25px 50px; }
}

.title-button-container {
    position: absolute;
    top: 80%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 80%;
    text-align: center;
    animation: fadeIn 3s ease-in-out;
    background: rgba(0, 0, 0, 0.5); 
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
}

.typing-line1 {
    font-size: 2.5rem;
    color: white; 
    margin: 0 auto;
    text-align: center;
    font-weight: bold;
    padding: 10px;
    white-space: pre-line;
    position: relative;
    top: 255px;
    opacity: 0; 
    animation: slideIn 2s ease-out forwards; 
    text-shadow: 
        0 0 10px #000000, 
        0 0 10px #000000, 
        0 0 15px #000000; 
}

.typing-line2 {
    font-size: 2rem;
    color: white; 
    font-style: italic;
    font-weight: 500;
    margin: 0 auto;
    text-align: center; 
    white-space: nowrap;
    padding: 5px;
    white-space: pre-line;
    position: relative;
    top: 245px;
    opacity: 0; 
    animation: slideIn 2s ease-out 2s forwards; 
    text-shadow: 
        0 0 10px #000000, 
        0 0 10px #000000, 
        0 0 15px #000000; 
}

.stButton {
    display: flex;
    justify-content: center;
    margin-top: 265px;
    width: 100%;
    opacity: 0; 
    animation: fadeIn 2s ease-in-out 4s forwards; 
}

.stButton > button {
    background-color: black;
    color: white;
    border: 1px solid white; 
    padding: 20px 40px;
    border-radius: 25px;
    font-size: 40px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.stButton > button:hover {
    background-color: #b94242; 
    box-shadow: 0 0 10px #B22222;
    color: white;
    transform: scale(1.1);
}

.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #485162;
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
    transition: color 0.3s ease;
}

.footer a:hover {
    color: #9e5ed5;
    text-decoration: underline;
}

@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}

@keyframes slideIn {
    from { transform: translateY(-20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

.custom-line-bottom {
    height: 50px;
    background-color: black;
    width: 100%; 
    position: fixed;
    bottom: 45px;
    left: 0; 
    z-index: 1000;
    pointer-events: none;
}
            
.custom-line-top {
    height: 150px;
    background-color: black;
    width: 100%;
    position: fixed;
    bottom: 875px;
    left: 0; 
    z-index: 1000;
    pointer-events: none;
}

@media (max-width: 768px) {
    .typing-line1 {
        font-size: 2rem;
    }
    
    .typing-line2 {
        font-size: 1.5rem;
    }
    
    .stButton > button {
        font-size: 18px;
        padding: 10px 25px;
    }
}
.footer-section a {
    display: inline; 
    margin: 0;
    padding: 0;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-line-top"></div>', unsafe_allow_html=True)

st.markdown('<div class="geometric-pattern"></div>', unsafe_allow_html=True)

st.markdown('<div class="title-button-container">', unsafe_allow_html=True)

st.markdown('<div class="typing-line1">Options Pricing Dashboard:</div>', unsafe_allow_html=True)
st.markdown('<div class="typing-line2">A Visualizer for Black-Scholes, Monte Carlo & Binomial Models</div>', unsafe_allow_html=True)

if st.button("Run Visualizations"):
    st.session_state.example_params = True
    st.switch_page("pages/Main.py") 

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <div class="footer-content">
        <div class="footer-section left" style="flex: 0 0 auto;">
            <a href="https://www.instagram.com/dataclubnu/" target="_blank">DATA @ NEU</a> | 
            <a href="https://mail.google.com/mail/?view=cm&fs=1&to=gandhi.mar@northeastern.edu&su=Contact%20Request&body=Hello%2C%0A%0AI%20would%20like%20to%20get%20in%20touch%20with%20you." target="_blank">Contact</a>
        </div>
        <div class="footer-section" style="flex: 1; text-align: center; position: absolute; left: 50%; transform: translateX(-50%);">
            Created By:
            <a href="https://www.linkedin.com/in/marvin-gandhi/" target="_blank">Marvin Gandhi</a>,
            <a href="https://www.linkedin.com/in/bennett-franciosi-5b8454324/" target="_blank">Bennett Franciosi</a>,
            <a href="https://www.linkedin.com/in/elaine-zou/" target="_blank">Elaine Zou</a>,
            <a href="https://www.linkedin.com/in/vaibhav-singh-548b861b0/" target="_blank">Vaibhav Singh</a>,
            <a href="https://www.linkedin.com/in/tafari-darosa-levy-337b65262/" target="_blank">Tafari Darosa-Levy</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)



st.markdown('<div class="custom-line-bottom"></div>', unsafe_allow_html=True)

