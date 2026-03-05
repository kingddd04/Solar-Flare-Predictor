import subprocess
import sys

def run_script(script_name):
    """Executes a Python script using the subprocess module."""
    print(f"\n{'='*40}")
    print(f"🚀 STARTING: {script_name}")
    print(f"{'='*40}\n")
    
    try:
        # sys.executable ensures the current Python environment (e.g., venv) is used
        subprocess.run([sys.executable, f"src/{script_name}"], check=True)
        return True # Returns True if execution was successful
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: The execution of {script_name} failed with code {e.returncode}.")
        return False # Returns False so the main menu knows it failed
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: Cannot find the file src/{script_name}.")
        return False

def main():
    print("\nWelcome to the SOLAR FLARE PREDICTOR ☀️")
    
    # Infinite loop to keep the menu active
    while True:
        print("\n" + "="*45)
        print("Choose which part of the pipeline you want to run:")
        print("1. 📊 Features Only (features_pipeline_main.py)")
        print("2. 🧠 Training Only (training_main.py)")
        print("3. 🔮 Inference Only (inference_main.py)")
        print("4. 🔄 Run ALL in the correct order (Features -> Training -> Inference)")
        print("0. ❌ Exit")
        print("="*45)
        
        choice = input("\nEnter the number of your choice (0-4): ").strip()
        
        if choice == '1':
            if run_script('features_pipeline_main.py'):
                print("✅ Features completed!")
                
        elif choice == '2':
            if run_script('training_main.py'):
                print("✅ Training completed!")
                
        elif choice == '3':
            if run_script('inference_main.py'):
                print("✅ Inference completed!")
                
        elif choice == '4':
            # Logical order of an ML pipeline.
            # The "if" ensures that if a step fails, the next ones do NOT start.
            print("\n🔄 Starting the full pipeline...")
            if run_script('features_pipeline_main.py'):
                if run_script('training_main.py'):
                    if run_script('inference_main.py'):
                        print("\n✅ Entire pipeline completed successfully!")
                        
        elif choice == '0':
            print("\n👋 Exiting. See you next time!")
            break # Breaks the loop and exits the program
            
        else:
            print("\n⚠️ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()