import numpy as np
from art.attacks.evasion import AutoProjectedGradientDescent,ProjectedGradientDescent,AdversarialPatch,BasicIterativeMethod,CarliniL2Method,CarliniLInfMethod

class AE_Finder:

    def __init__(self, classifier, attack_name, x_test, y_test):

        self.attack_name = attack_name
        self.x_test = x_test
        self.y_test = y_test
        self.classifier=classifier
        
    def finder(self):
        self.max_successful_attacks = 100
        self.max_current_successful_attacks = 0
        self.best_eps = None
        self.best_eps_step = None
        self.best_max_iter = None
        self.best_adv_examples = None
        self.eps_range = np.arange(0.01, 1.0, 0.01)  # Range of eps values
        self.eps_step_range = np.arange(0.01, 0.1, 0.01)  # Range of eps_step values
        self.max_iter_range = [10,  20,  30,  40,  50,  60,  70,  80,  90, 100]  # Range of max_iterations values
        def get_random_samples(test_set, test_labels, num_samples):
            indices = np.random.choice(len(self.x_test), num_samples, replace=False)
            x_sample = self.x_test[indices]
            y_sample = self.y_test[indices]
            return x_sample, y_sample

        # Assuming your test set is named test_set and test labels are named test_labels
        x_sample, y_sample = get_random_samples(self.x_test, self.y_test, num_samples=100)
        stop_search = False  # Flag variable to control loop termination
        for eps in self.eps_range:
            if stop_search:
                break
            for eps_step in self.eps_step_range:
                if stop_search:
                    break
                for max_iter in self.max_iter_range:
                    print(f"eps: {eps}, eps_step: {eps_step}, max_iter: {max_iter}")
                    if self.max_current_successful_attacks >= self.max_successful_attacks:
                        stop_search = True
                        break
                    if self.attack_name == 'bim':
                        self.attack = BasicIterativeMethod(self.classifier, eps=eps, eps_step=eps_step, max_iter=max_iter, verbose=False, batch_size=10000)
                    elif self.attack_name == 'pgdlinf':
                        self.attack = ProjectedGradientDescent(estimator=self.classifier, norm='inf', eps=eps, eps_step=eps_step, max_iter=max_iter, verbose=False, batch_size=10000)
                    elif self.attack_name == 'pgdl1':
                        self.attack = ProjectedGradientDescent(estimator=self.classifier, norm=1, eps=eps, eps_step=eps_step, max_iter=max_iter, verbose=False, batch_size=10000)
                    elif self.attack_name == 'pgdl2':
                        self.attack = ProjectedGradientDescent(estimator=self.classifier, norm=2, eps=eps, eps_step=eps_step, max_iter=max_iter, verbose=False, batch_size=10000)
                    elif self.attack_name == 'apgdlinf':
                        self.attack = AutoProjectedGradientDescent(self.classifier, norm=np.inf, eps=eps, eps_step=eps_step, max_iter=max_iter, verbose=False, batch_size=10000)
                    elif self.attack_name == 'apgdl1':
                        self.attack = AutoProjectedGradientDescent(self.classifier, norm=1, eps=eps, eps_step=eps_step, max_iter=max_iter, verbose=False, batch_size=10000)
                    elif self.attack_name == 'apgdl2':
                        self.attack = AutoProjectedGradientDescent(self.classifier, norm=2, eps=eps, eps_step=eps_step, max_iter=max_iter, verbose=False, batch_size=10000)
                    elif self.attack_name == 'cwl2':
                        self.attack = CarliniL2Method(self.classifier, eps=eps, eps_step=eps_step, batch_size=10000)
                    elif self.attack_name == 'cwlinf':
                        self.attack = CarliniLInfMethod(self.classifier, confidence=0, targeted=False, learning_rate=eps, max_iter=max_iter, verbose=False, batch_size=10000)
                    adv_examples = self.attack.generate(x=x_sample, y=y_sample)

                    # Evaluate the classifier on adversarial examples
                    preds_adv = self.classifier.predict(adv_examples)

                    # Check where the attack is successful (predictions change)
                    self.successful_indices = np.where(np.argmax(preds_adv, axis=1) != y_sample)[0]
                    self.num_successful_attacks = len(self.successful_indices)

                    # Update if current combination is better
                    if self.num_successful_attacks > self.max_current_successful_attacks:
                        self.max_current_successful_attacks = self.num_successful_attacks
                        self.best_eps = eps
                        self.best_eps_step = eps_step
                        self.best_max_iter = max_iter

                    elif self.num_successful_attacks == self.max_current_successful_attacks:
                        pass
                        print("Number of successful attacks is not strictly greater, skipping update.")
                    
                    print(f"Successful attacks: {self.num_successful_attacks}")
        print(f"Best eps: {self.best_eps}, Best eps_step: {self.best_eps_step}, Best max_iter: {self.best_max_iter}, Maximum successful attacks: {self.max_current_successful_attacks}")
