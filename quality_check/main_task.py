from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_ollama import OllamaLLM
import concurrent.futures
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import img_to_array
from task import (
    detect_defects_loftr,
    detect_defects_sift,
    compare_histograms,
    compare_images_strict,
    detect_defect_autoencoder,
    compare_images_orb
)

# ==============================
# Autoencoder Model Definition
# ==============================
def build_autoencoder():
    input_img = Input(shape=(224, 224, 3))

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return Model(input_img, decoded)

# ==============================
# Image Preprocessing
# ==============================
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load {image_path}")
        return None
    
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_to_array(image) / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

# ==============================
# Train Autoencoder on a Good Image
# ==============================
def train_autoencoder(autoencoder, good_img_path, epochs=10):
    print("\n🔄 Training Autoencoder...")
    
    good_image = load_and_preprocess_image(good_img_path)
    if good_image is None:
        return None

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(good_image, good_image, epochs=epochs, verbose=1)
    
    print("✅ Training Completed!")
    return autoencoder

# ==============================
# LangChain Tools
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
        func=lambda autoencoder, test: detect_defect_autoencoder(autoencoder, test),  
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
    try:
        print("\U0001F525 Warming up LLM...")
        response = llm.invoke("Warm-up query: Respond with 'OK'")
        print(f"✅ LLM Warm-up Response: {response}")
    except Exception as e:
        print(f"⚠️ LLM Warm-up Failed: {str(e)}")

warm_up_llm()

# ==============================
# Utility Functions
# ==============================
def normalize_result(result):
    if isinstance(result, str):
        return result.upper()
    elif isinstance(result, (float, np.float64)):
        return "DEFECTIVE" if result < 0.5 else "GOOD"
    return "UNKNOWN"

def run_method(tool, good_img_path, test_img_path, autoencoder=None):
    method_name = tool.name
    print(f"🔍 Running {method_name}...")

    try:
        if method_name == "Autoencoder Comparison":
            if autoencoder is None:
                raise ValueError("Autoencoder model is missing.")
            result = tool.func(autoencoder, test_img_path)
        else:
            result = tool.func(good_img_path, test_img_path)

        normalized_result = normalize_result(result)
        print(f"✅ {method_name} Result: {normalized_result} ({result})")
        return method_name, normalized_result
    except Exception as e:
        print(f"     Error in {method_name}: {str(e)}")
        return method_name, "ERROR"

# ==============================
# Main Detection Pipeline
# ==============================
def stream_defect_detections(good_img_path, test_img_path, autoencoder):
    results = {}

    orb_tool = next(tool for tool in tools if tool.name == "ORB Matching")
    other_tools = [tool for tool in tools if tool.name != "ORB Matching"]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_results = {
            executor.submit(run_method, tool, good_img_path, test_img_path, autoencoder): tool 
            for tool in other_tools
        }

        for future in concurrent.futures.as_completed(future_results):
            method_name, result = future.result()
            results[method_name] = result
            print(f"🔍 {method_name} detected: {result}")

    ai_input = f"Here are the results: {results}. Classify the image as 'GOOD' or 'DEFECTIVE'."
    print("\n🤖 AI Agent's Initial Analysis:\n")

    try:
        prediction = ""
        for chunk in agent.stream(ai_input):
            print(chunk, end="", flush=True)
            prediction += chunk
    except Exception as e:
        print(f" AI Parsing Error: {str(e)}")

    print("\n🔄 Running ORB Matching after AI decision...\n")
    orb_name, orb_result = run_method(orb_tool, good_img_path, test_img_path)
    results[orb_name] = orb_result

    print(f"\n🔍 ORB Matching Final Result: {orb_result}")

    decision_counts = {
        "GOOD": sum(1 for r in results.values() if r == "GOOD"),
        "DEFECTIVE": sum(1 for r in results.values() if r == "DEFECTIVE"),
        "ERROR": sum(1 for r in results.values() if r == "ERROR"),
    }

    final_classification = "DEFECTIVE" if decision_counts["DEFECTIVE"] >= decision_counts["GOOD"] else "GOOD"

    print("\n **FINAL CLASSIFICATION** :")
    print(f"Classification: {final_classification}")

# ==============================
# Run Example
# ==============================
if __name__ == "__main__":
    good_image = "C:/Users/dharn/Desktop/quality_check/latest/good.jpg"
    test_image = "C:/Users/dharn/Desktop/quality_check/latest/test2.jpg"

    print("\n===== Training Autoencoder =====\n")
    autoencoder_model = build_autoencoder()
    autoencoder_model = train_autoencoder(autoencoder_model, good_image, epochs=50)

    print("\n===== Running Defect Detection =====\n")
    stream_defect_detections(good_image, test_image, autoencoder_model)
