import os
import json
from config import Config
from datetime import datetime

class State_Manager:
    def __init__(self):
        self.config = Config()
        self.state_filepath = os.path.join(self.config.root_dir, "state", "state.json")

    def load_state(self):
        with open(self.state_filepath) as f: 
            state = json.load(f) 
        return state 


    def save_state(self):
        state = { 
            "last_update": datetime.now().isoformat() 
            } 
        with open(self.state_filepath, "w") as f: 
            json.dump(state, f)
        print("Saved state file:", self.state_filepath)


s = State_Manager()
s.save_state()


        
