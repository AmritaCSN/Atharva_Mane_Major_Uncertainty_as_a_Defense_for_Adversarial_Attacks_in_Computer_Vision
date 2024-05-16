import numpy as np
from art.attacks.evasion import AdversarialPatchPyTorch,AdversarialPatch
class AE_Finder:

    def __init__(self, classifier, x_test, y_test):

        self.x_test = x_test
        self.y_test = y_test
        self.classifier=classifier
        
    def finder(self):
        self.max_successful_attacks = 100
        self.max_current_successful_attacks = 0
        self.best_rotation_max = None
        self.best_learning_rate = None
        self.best_max_iter = None
        self.rotation_max_range = np.arange(10, 51, 10)  # Range of max_rotation values
        self.learning_rate_range = np.arange(0.01, 5, 0.05)  # Range of learning rate values
        self.max_iter_range = np.arange(100, 501, 100)  # Range of max_iterations values
        def get_random_samples(test_set, test_labels, num_samples):
            indices = np.random.choice(len(self.x_test), num_samples, replace=False)
            x_sample = self.x_test[indices]
            y_sample = self.y_test[indices]
            return x_sample, y_sample

        # Assuming your test set is named test_set and test labels are named test_labels
        x_sample, y_sample = get_random_samples(self.x_test, self.y_test, num_samples=100)
        stop_search = False  # Flag variable to control loop termination
        for rotation in self.rotation_max_range:
            if stop_search:
                break
            for lr in self.learning_rate_range:
                if stop_search:
                    break
                for max_iter in self.max_iter_range:
                    print(f"rotation: {rotation}, learning_rate: {lr}, max_iter: {max_iter}")
                    if self.max_current_successful_attacks >= self.max_successful_attacks:
                        stop_search = True
                        break
                    self.attack=AdversarialPatchPyTorch(
                            estimator=self.classifier,
                            rotation_max=rotation,
                            learning_rate= lr,
                            max_iter= max_iter,
                            batch_size= 1000,
                            patch_shape = (1, 28, 28),
                            patch_type= 'square',
                            optimizer = 'Adam',
                            targeted=False,
                            verbose = True
                        )
                    self.patch = self.attack.generate(x_sample)
                    adv_examples=self.attack.apply_patch(x_sample,scale=0.5,patch_external=self.patch)
                    # Evaluate the classifier on adversarial examples
                    preds_adv = self.classifier.predict(adv_examples)

                    # Check where the attack is successful (predictions change)
                    self.successful_indices = np.where(np.argmax(preds_adv, axis=1) != y_sample)[0]
                    self.num_successful_attacks = len(self.successful_indices)

                    # Update if current combination is better
                    if self.num_successful_attacks > self.max_current_successful_attacks:
                        self.max_current_successful_attacks = self.num_successful_attacks
                        self.best_rotation_max = rotation
                        self.best_learning_rate = lr
                        self.best_max_iter = max_iter

                    elif self.num_successful_attacks == self.max_current_successful_attacks:
                        pass
                        print("Number of successful attacks is not strictly greater, skipping update.")
                    
                    print(f"Successful attacks: {self.num_successful_attacks}")
        print(f"Best rotation: {self.best_rotation_max}, Best learning_rate: {self.best_learning_rate}, Best max_iter: {self.best_max_iter}, Maximum successful attacks: {self.max_current_successful_attacks}")
        
