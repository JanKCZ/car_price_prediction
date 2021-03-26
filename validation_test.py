import copy

class Validation():
    best_val_error = 0
    break_training = False
    init_error = True
    best_model = 0
    
    def __init__(self, penalty_point = 0):
        self.penalty_point = penalty_point
    
    def validate_testing(self, model, val_error, patience = 5):
        if self.init_error:
            self.best_val_error = val_error
            self.init_error = False
            self.best_model = model
        if val_error < self.best_val_error:
            self.best_val_error = val_error
            self.best_model = model
        else:
            self.penalty_point += 1
            if patience == self.penalty_point:
                print(f"actual error {val_error} is bigger than the smallest of {patience} previous errors: {self.best_val_error}")
                self.break_training = True

        return self.best_model, self.best_val_error, self.break_training