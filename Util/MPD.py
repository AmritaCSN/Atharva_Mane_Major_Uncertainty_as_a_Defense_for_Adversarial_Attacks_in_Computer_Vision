import torch
import numpy as np
class MPD_Evaluator:
    def __init__(self, ensemble_models, num_classes, ensemble_count):
        self.ensemble_models = ensemble_models
        self.num_classes = num_classes
        self.ensemble_count = ensemble_count
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_probabilities(self, model, x_test):
        model.to(self.device)
        model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # No need to compute gradients during inference
            images = x_test.to(self.device)  # Move all test images to the device (GPU if available)
            outputs = model(images)  # Forward pass
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        return probabilities
    
    def evaluate(self,x_test):
        self.x_test = torch.from_numpy(x_test)  # Using torch.from_numpy()\
        self.total_probs = []

        for model in self.ensemble_models:
            # Get probabilities for the test images from the current model
            model_probs = self.get_probabilities(model, self.x_test)
            self.total_probs.append(model_probs)

        # Convert the list of probabilities into a numpy array
        self.total_probs = np.array(self.total_probs)
        mpd = np.full((len(self.x_test)), 100)        
        for i in range(self.num_classes):
            probs_i = self.total_probs[:, :, i]

            U_i = (probs_i - 1) ** 2
            U_i = U_i.sum(axis=0)
            U_i = np.sqrt(U_i / self.ensemble_count)
            mpd = np.minimum(mpd, U_i)  
        return mpd
