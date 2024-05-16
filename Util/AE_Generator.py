import numpy as np
from art.attacks.evasion import AutoProjectedGradientDescent,ProjectedGradientDescent,AdversarialPatch,BasicIterativeMethod,CarliniL2Method,CarliniLInfMethod

class AEGenerator:
    def __init__(self,classifier, attack_name, x_test, y_test,eps,eps_step,max_iter):

        self.attack_name = attack_name
        self.x_test = x_test
        self.y_test = y_test
        self.classifier=classifier
        self.eps=eps
        self.eps_step=eps_step
        self.max_iter=max_iter
        if self.attack_name == 'bim':
            self.attack = BasicIterativeMethod(self.classifier, eps=self.eps, eps_step=self.eps_step, max_iter=self.max_iter, verbose=False, batch_size=1000)
        elif self.attack_name == 'pgdlinf':
            self.attack = ProjectedGradientDescent(estimator=self.classifier, norm='inf', eps=self.eps, eps_step=self.eps_step, max_iter=self.max_iter, verbose=False, batch_size=1000)
        elif self.attack_name == 'pgdl1':
            self.attack = ProjectedGradientDescent(estimator=self.classifier, norm=1, eps=self.eps, eps_step=self.eps_step, max_iter=self.max_iter, verbose=False, batch_size=1000)
        elif self.attack_name == 'pgdl2':
            self.attack = ProjectedGradientDescent(estimator=self.classifier, norm=2, eps=self.eps, eps_step=self.eps_step, max_iter=self.max_iter, verbose=False, batch_size=1000)
        elif self.attack_name == 'apgdlinf':
            self.attack = AutoProjectedGradientDescent(self.classifier, norm=np.inf, eps=self.eps, eps_step=self.eps_step, max_iter=self.max_iter, verbose=False, batch_size=1000)
        elif self.attack_name == 'apgdl1':
            self.attack = AutoProjectedGradientDescent(self.classifier, norm=1, eps=self.eps, eps_step=self.eps_step, max_iter=self.max_iter, verbose=False, batch_size=1000)
        elif self.attack_name == 'apgdl2':
            self.attack = AutoProjectedGradientDescent(self.classifier, norm=2, eps=self.eps, eps_step=self.eps_step, max_iter=self.max_iter, verbose=False, batch_size=1000)
        elif self.attack_name == 'cwl2':
            self.attack = CarliniL2Method(self.classifier, eps=self.eps, eps_step=self.eps_step, batch_size=1000)
        elif self.attack_name == 'cwlinf':
            self.attack = CarliniLInfMethod(self.classifier, confidence=0, targeted=False, learning_rate=self.eps, max_iter=self.max_iter, verbose=False, batch_size=1000)


        # Generate adversarial examples
        self.x_test_adv = self.attack.generate(self.x_test)

        # Evaluate the ART classifier on the adversarial test examples
        self.adv_preds = self.classifier.predict(self.x_test_adv)

        # Getting the individual predictions and their indices
        self.ind_pred = np.argmax(self.adv_preds, axis=1) == self.y_test
        self.false_indices = np.where(~self.ind_pred)[0].tolist()

        # Calculating the accuracy
        self.acc_adv = np.sum(np.argmax(self.adv_preds, axis=1)== self.y_test) / len(self.y_test)
        print("Accuracy on {} adversarial test examples: {}%".format(self.attack_name.upper(), self.acc_adv * 100))
        print()
        print(f"The number of {self.attack_name.upper()} samples that successfully evaded the model are - {len(self.false_indices)}")