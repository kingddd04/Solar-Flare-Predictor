import streamlit as st
import subprocess
import sys

# Configure the web page layout, title, and favicon (the little icon in the browser tab)
st.set_page_config(page_title="Solar Flare Predictor", page_icon="☀️", layout="centered")

def run_script(script_name):
    """
    Executes a Python script located in the 'src' folder.
    Captures both normal print statements (stdout) and system logs/errors (stderr).
    """
    try:
        # Run the script using the current Python environment
        result = subprocess.run(
            [sys.executable, f"{script_name}"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Combine standard output and system error logs into a single string
        total_output = result.stdout + "\n" + result.stderr
        
        # Check if the script ran but produced absolutely no text
        if not total_output.strip():
            total_output = "⚠️ The script finished, but generated no text output."
            
        return True, total_output
        
    except subprocess.CalledProcessError as e:
        # If the script crashes, catch the error message so the UI doesn't crash
        error_msg = e.stdout + "\n" + e.stderr
        return False, error_msg
        
    except FileNotFoundError:
        # Handle the case where the file doesn't exist in the src folder
        return False, f"Cannot find the file src/{script_name}."

def extract_solar_class(text):
    """
    Scans the verbose logs (like TensorFlow loading bars) to find and extract
    ONLY the final inference result line containing 'Solar Class:'.
    """
    # Split the massive log text into individual lines
    for line in text.split('\n'):
        # If the line contains our target keyword, return just that line
        if "Solar Class:" in line:
            return line.strip() 
            
    # Fallback message if the keyword wasn't found in any line
    return "⚠️ Result not found in the logs."

def main():
    # --- UI HEADER ---
    st.title("☀️ Solar Flare Predictor")
    st.write("Welcome to the Machine Learning pipeline control panel. Choose a step to execute.")
    
    st.divider() # Visual separator

    # --- INDIVIDUAL PIPELINE BUTTONS ---
    # Create 3 equal-width columns for the single-step buttons
    col1, col2, col3 = st.columns(3)

    # Column 1: Features Pipeline
    with col1:
        if st.button("📊 Features Only", use_container_width=True):
            with st.spinner("Processing features..."):
                success, output = run_script('features_pipeline_main.py')
            
            if success:
                st.success("Features completed! ✅")
                # Hide verbose logs inside a collapsible expander
                with st.expander("Show logs", expanded=False):
                    st.code(output, language="text")
            else:
                st.error("❌ Error during Features:")
                st.code(output, language="text")

    # Column 2: Training Pipeline
    with col2:
        if st.button("🧠 Training Only", use_container_width=True):
            with st.spinner("Training the model..."):
                success, output = run_script('training_main.py')
                
            if success:
                st.success("Training completed! ✅")
                with st.expander("Show logs", expanded=False):
                    st.code(output, language="text")
            else:
                st.error("❌ Error during Training:")
                st.code(output, language="text")

    # Column 3: Inference Pipeline
    with col3:
        if st.button("🔮 Inference Only", use_container_width=True):
            with st.spinner("Running predictions..."):
                success, output = run_script('inference_main.py')
                
            if success:
                st.success("Inference completed! ✅")
                
                # Extract the clean result using our custom function
                clean_result = extract_solar_class(output)
                
                st.write("### 📊 Inference Results:")
                # Display the clean result prominently
                st.info(f"**{clean_result}**") 
                
                # Keep the messy TensorFlow logs available but hidden
                with st.expander("Show system logs (TensorFlow)", expanded=False):
                    st.code(output, language="text")
            else:
                st.error("❌ Error during Inference:")
                st.code(output, language="text")

    st.divider()

    # --- RUN ALL BUTTON ---
    # A prominent button to run the whole pipeline sequentially
    if st.button("🔄 Run ALL (Features -> Training -> Inference)", type="primary", use_container_width=True):
        with st.spinner("Running the entire pipeline. This might take a while..."):
            
            # Step 1: Features
            st.toast("Starting Features...") # Small popup notification
            success1, out1 = run_script('features_pipeline_main.py')
            if not success1:
                st.error("❌ Pipeline stopped at Features:")
                st.code(out1, language="text")
                st.stop() # Halt execution if this step fails
                
            # Step 2: Training
            st.toast("Starting Training...")
            success2, out2 = run_script('training_main.py')
            if not success2:
                st.error("❌ Pipeline stopped at Training:")
                st.code(out2, language="text")
                st.stop()
                
            # Step 3: Inference
            st.toast("Starting Inference...")
            success3, out3 = run_script('inference_main.py')
            if not success3:
                st.error("❌ Pipeline stopped at Inference:")
                st.code(out3, language="text")
                st.stop()
            
        # Trigger celebration animation
        st.balloons()
        st.success("Entire pipeline completed successfully! 🎉")
        
        # Extract and show the final result from the Run ALL process
        final_clean_result = extract_solar_class(out3)
        
        st.write("### 📊 Final Inference Results:")
        st.info(f"**{final_clean_result}**") 
        
        with st.expander("Show complete system logs", expanded=False):
            st.code(out3, language="text")

# Standard boilerplate to call the main() function
if __name__ == "__main__":
    main()