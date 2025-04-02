from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_ollama import OllamaLLM
import concurrent.futures
import numpy as np
import cv2
from task import (
    detect_defects_loftr,
    detect_defects_sift,
    compare_histograms,
    compare_images_strict,
    detect_defect_autoencoder,
    compare_images_orb
)

# ==============================
# Define LangChain Tools
# ==============================
tools = [
    Tool(
        name="LoFTR Matching",
        func=lambda good, test: detect_defects_loftr(good, test),
        description="Performs defect detection using LoFTR matching."
    ),
    Tool(
        name="SIFT Matching",
        func=lambda good, test: detect_defects_sift(good, test),
        description="Performs defect detection using SIFT matching."
    ),
    Tool(
        name="Histogram Comparison",
        func=lambda good, test: compare_histograms(good, test),
        description="Performs defect detection using histogram comparison."
    ),
    Tool(
        name="SSIM Comparison",
        func=lambda good, test: compare_images_strict(good, test),
        description="Performs defect detection using Structural Similarity Index (SSIM)."
    ),
    Tool(
        name="Autoencoder Comparison",
        func=lambda test: detect_defect_autoencoder(test),
        description="Performs defect detection using autoencoder-based image reconstruction."
    ),
    Tool(
        name="ORB Matching",
        func=lambda good, test: compare_images_orb(good, test),
        description="Performs defect detection using ORB-based feature matching."
    )
]

# ==============================
# Set up LLM
# ==============================
llm = OllamaLLM(model="mistral")

def warm_up_llm():
    """Preload the Ollama LLM model with a dummy request."""
    try:
        print("🔥 Warming up the Ollama LLM model...")
        response = llm.invoke("Warm-up query: Respond with 'OK'")
        print(f"✅ LLM Warm-up Response: {response}")
    except Exception as e:
        print(f"⚠️ LLM Warm-up Failed: {str(e)}")

warm_up_llm()

# ==============================
# Agent Prompt (Decision Rules)
# ==============================
agent_prompt = """You are an AI agent responsible for detecting defects in machine parts by analyzing images. 
You receive results from multiple defect detection methods:
1️⃣ LoFTR Matching (feature-based)
2️⃣ SIFT Matching (feature-based)
3️⃣ Histogram Comparison (color-based)
4️⃣ SSIM Comparison (structural)
5️⃣ Autoencoder Comparison (deep learning-based anomaly detection)
6️⃣ ORB Matching (feature-based)

**Rules:**
- Use all methods equally and consider their results collectively.
- Analyze the outputs and make a final classification as 'GOOD' or 'DEFECTIVE'.
- Explain your decision based on the results.
- Provide the final output in the format:

Final Classification: [GOOD/DEFECTIVE]
Explanation: [Brief justification based on methods]
"""

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    description=agent_prompt,
    handle_parsing_errors=True
)

# ==============================
# Utility Functions
# ==============================
def run_method(tool, good_img_path, test_img_path):
    """Run a single defect detection method and handle errors."""
    method_name = tool.name
    print(f"🔍 Running {method_name}...")

    try:
        result = tool.func(good_img_path, test_img_path) if method_name != "Autoencoder Comparison" else tool.func(test_img_path)
        print(f"✅ {method_name} Result: {result}")
        return method_name, result
    except Exception as e:
        print(f"❌ Error in {method_name}: {str(e)}")
        return method_name, "ERROR"

# ==============================
# Main Defect Detection Pipeline
# ==============================
def detect_defects(good_img_path, test_img_path):
    """Run all defect detection methods and let the AI agent decide."""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_results = {executor.submit(run_method, tool, good_img_path, test_img_path): tool for tool in tools}
        
        for future in concurrent.futures.as_completed(future_results):
            method_name, result = future.result()
            results[method_name] = result
            print(f"🔍 {method_name} detected: {result}")
    
    ai_input = f"Here are the results: {results}. Classify the image as 'GOOD' or 'DEFECTIVE' with a reason."
    print("\n🤖 AI Agent's Final Analysis:\n")
    
    try:
        prediction = ""
        for chunk in agent.stream(ai_input):
            print(chunk, end="", flush=True)
            prediction += chunk
    except Exception as e:
        print(f"\n⚠️ AI Parsing Error: {str(e)}")

# ==============================
# Run Example
# ==============================
if __name__ == "__main__":
    good_image = "C:/Users/dharn/Desktop/quality_check/latest/good.jpg"
    test_image = "C:/Users/dharn/Desktop/quality_check/latest/test1.jpg"
    print("\n===== Running Defect Detection =====\n")
    detect_defects(good_image, test_image)
