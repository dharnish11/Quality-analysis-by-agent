from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_ollama import OllamaLLM
import concurrent.futures
from langchain.output_parsers import RegexParser
from langchain.schema import SystemMessage

# Define the tool functions as you already have
from task import (
    detect_defects_loftr,
    detect_defects_sift,
    compare_histograms,
    compare_images_strict,
    detect_defect_autoencoder,
    compare_images_orb,
    train_autoencoder,
    load_and_preprocess_image,
    build_autoencoder
)

# Create the list of tools (without lambda wrapper, using the actual functions directly)
tools = [
    Tool(name="LoFTR Matching", func=detect_defects_loftr, description="Performs defect detection using LoFTR matching."),
    Tool(name="SIFT Matching", func=detect_defects_sift, description="Performs defect detection using SIFT matching."),
    Tool(name="Histogram Comparison", func=compare_histograms, description="Performs defect detection using histogram comparison."),
    Tool(name="SSIM Comparison", func=compare_images_strict, description="Performs defect detection using Structural Similarity Index (SSIM)."),
    Tool(name="Autoencoder Comparison", func=detect_defect_autoencoder, description="Performs defect detection using autoencoder-based image reconstruction."),
    Tool(name="ORB Matching", func=compare_images_orb, description="Performs defect detection using ORB-based feature matching."),
]

# Define parsing rules for agent output
strict_parser = RegexParser(regex="(GOOD|DEFECTIVE|UNCERTAIN)", output_keys=["classification"])

# Initialize LLM model for decision making
llm = OllamaLLM(model="mistral")

# Define strict rules for classification
strict_instructions = """
You are a strict AI defect detection agent. Follow these rules:
1. If more than 50% of the methods classify the image as 'DEFECTIVE', classify it as 'DEFECTIVE'.
2. If more than 50% classify it as 'GOOD', classify it as 'GOOD'.
3. If there is a tie, classify it as 'UNCERTAIN'.
4. You MUST respond with only one of these words: GOOD, DEFECTIVE, or UNCERTAIN. No extra words, no explanations.
"""

# Initialize the agent with system message and tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    system_message=SystemMessage(content=strict_instructions),
    handle_parsing_errors=True,
    output_parser=strict_parser
)

# Define main function for defect detection streaming
def stream_defect_detections(good_img_path, test_img_path, autoencoder):
    results = {}

    # Run detection methods concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_results = {
            executor.submit(tool.func, good_img_path, test_img_path): tool 
            for tool in tools if tool.name != "Autoencoder Comparison"
        }

        for future in concurrent.futures.as_completed(future_results):
            tool = future_results[future]
            try:
                result = future.result()
                results[tool.name] = result
            except Exception as e:
                print(f"Error in {tool.name}: {str(e)}")
                results[tool.name] = "ERROR"

    # Run Autoencoder separately (if provided)
    if autoencoder:
        try:
            # Ensure both the autoencoder and test image path are passed correctly
            autoencoder_result = detect_defect_autoencoder(autoencoder, good_img_path, test_img_path)
            results["Autoencoder Comparison"] = autoencoder_result
        except Exception as e:
            print(f"Error in Autoencoder Comparison: {str(e)}")
            results["Autoencoder Comparison"] = "ERROR"

    # Prepare input for agent classification
    ai_input = f"Here are the results: {results}. What is the final classification?"
    try:
        final_decision = agent.invoke(ai_input).strip().upper()
    except Exception as e:
        print(f"Error in AI classification: {str(e)}")
        final_decision = "ERROR"

    print(final_decision)  # Output final classification decision
    return final_decision


# Example usage
if __name__ == "__main__":
    good_image = "C:/Users/dharn/Desktop/quality_check/latest/good.jpg"
    test_image = "C:/Users/dharn/Desktop/quality_check/latest/test2.jpg"
    autoencoder_model = build_autoencoder()
    autoencoder_model = train_autoencoder(autoencoder_model, good_image, epochs=50)

    # Stream defect detection and get final result
    stream_defect_detections(good_image, test_image, autoencoder_model)
