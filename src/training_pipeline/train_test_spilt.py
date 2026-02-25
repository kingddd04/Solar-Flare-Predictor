"""Utilities for splitting training and test sets."""

import numpy as np
from sklearn.model_selection import train_test_split

class Train_Test_Split:
    
    @staticmethod
    def split_training(x_train, y_train, test_size=0.2, random_state=42, shuffle=False):
        
        x_tr, x_te, y_tr, y_te = train_test_split(
            x_train, y_train,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle
        )
        return x_tr, x_te, y_tr, y_te
