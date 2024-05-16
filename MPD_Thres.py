
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.metrics import f1_score,precision_score,recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
        #Function for class distribution
def class_dist(labels):
    # get unique class labels and their counts
    unique_classes, class_counts= np.unique(labels, return_counts=True)

    # Print the class distribution
    for label, count in zip(unique_classes, class_counts):
        print(f"Class {label}: {count} samples")
    
    
class MPD_Threshold_Calculator:
    def __init__(self, classifier,mpd_detector,attack_name,x_common_samples_cln,x_common_samples_adv, y_common_samples,x_test_stratified_clean,y_test_stratified_clean,max_mpd_scores):
        
        self.attack_name = attack_name
        self.x_common_samples_adv = x_common_samples_adv
        self.y_common_samples = y_common_samples
        self.x_common_samples_cln=x_common_samples_cln
        percent=1-(2500/len(x_common_samples_adv)) # Calculate the percentage for creating 2500 samples
        
        # Use train_test_split with stratify to create new variables with 25% CLEAN stratified data
        self.x_test_stratified_cln, _, self.y_test_stratified_cln, _ = train_test_split(self.x_common_samples_cln, self.y_common_samples, test_size=percent, stratify=self.y_common_samples, random_state=333)
        
   
        
        # Use train_test_split with stratify to create new variables with 25% AE stratified data
        
        self.x_test_stratified_adv, _, self.y_test_stratified_adv, _ = train_test_split(self.x_common_samples_adv, self.y_common_samples, test_size=percent, stratify=self.y_common_samples, random_state=333)

        self.x_test_stratified_adv=self.x_test_stratified_adv[:2500]
        self.y_test_stratified_adv=self.y_test_stratified_adv[:2500]
        
        # Now, x_test_stratified_adv and y_test_stratified_adv contain 25% of the stratified data from x_common_samples and y_common_samples
        print()
        print()
        print("Shape of x_test_stratified_bim:", self.x_test_stratified_adv.shape)
        print("Shape of y_test_stratified_bim:", self.y_test_stratified_adv.shape)
        print()
        print()
        print(f"Class Distribution for {self.attack_name.upper()} :")
        class_dist(self.y_test_stratified_adv)
        print()
        print()
        
        #Evaluate the ART classifier on benign test examples

        clean_predictions = classifier.predict(self.x_test_stratified_cln)
        clean_predictions=np.argmax(clean_predictions, axis=1)
        clean_accuracy = np.sum(clean_predictions == self.y_test_stratified_cln) / len(self.y_test_stratified_cln)
        print(f"Accuracy on the Clean Samples: {clean_accuracy*100}%")
        
        
        #Evaluate the ART classifier on AE test examples

        ae_predictions = classifier.predict(self.x_test_stratified_adv)
        ae_predictions=np.argmax(ae_predictions, axis=1)
        ae_accuracy = np.sum(ae_predictions == self.y_test_stratified_cln) / len(self.y_test_stratified_cln)
        print(f"Accuracy on the Adversarial Samples: {ae_accuracy*100}%")
        
        # Vertically stack the arrays to create a single set
        self.x_adv_testbed = np.vstack([x_test_stratified_clean, self.x_test_stratified_adv])
        #convert the labels as a binary problem i.e - AE or not AE
        self.y_adv_testbed = np.concatenate([np.zeros_like(y_test_stratified_clean), np.ones_like(self.y_test_stratified_adv)])

        # Now, x_testbed and y_testbed contain the combined set
        print(f"Shape of x_adv_testbed for {self.attack_name.upper()}:", self.x_adv_testbed.shape)
        print("Shape of y_adv_testbed:", self.y_adv_testbed.shape)
        print()
        print()
        print(f"Class Distribution for {self.attack_name} :")
        class_dist(self.y_adv_testbed)
        
        self.thvalues= np.linspace(0.01, max_mpd_scores[self.attack_name],int(max_mpd_scores[self.attack_name]*100))
        
        #mpd scores on the Adversarial Testbed
        self.mpd_scores_adv_testbed = mpd_detector.evaluate(self.x_adv_testbed)
        
        self.score_list=pd.DataFrame(columns=["Threshold_value","TP","FP","FN","TN", "recall", 'f1_score', 'accuracy'])

        for idx, t in enumerate(self.thvalues) :
            mpd_preds = (self.mpd_scores_adv_testbed>t).astype(int)

            cmvalues= (confusion_matrix(self.y_adv_testbed, mpd_preds).ravel()).tolist()
            values=[t.tolist(),]
            values.extend(cmvalues)
            values.append(recall_score(self.y_adv_testbed,mpd_preds))  
            values.append(f1_score(self.y_adv_testbed,mpd_preds))
            values.append(accuracy_score(self.y_adv_testbed,mpd_preds))

            self.score_list.loc[idx] = values
            
        print()
        print()
        #print(self.score_list)
        self.tvalue = self.score_list[self.score_list.f1_score==self.score_list.f1_score.max()]
        print()
        print()
        print(self.tvalue)
            
            #taking the best threshold value 
        self.best_t = self.tvalue.Threshold_value.values[0]
        self.mpd_pred = self.mpd_scores_adv_testbed.copy()
        self.mpd_pred[self.mpd_pred > self.best_t ] = 1
        self.mpd_pred[self.mpd_pred <= self.best_t] = 0
                # Extract indices of False Positive (FP) and False Negative (FN) samples
        self.FP_indices = np.where((self.mpd_pred == 1) & (self.y_adv_testbed == 0))[0]
        self.FN_indices = np.where((self.mpd_pred == 0) & (self.y_adv_testbed == 1))[0]

        # Extract FP and FN samples
        self.FP_samples = self.x_adv_testbed[self.FP_indices]
        self.FN_samples = self.x_adv_testbed[self.FN_indices]
        # Extract labels for FP and FN samples
        self.FP_labels = self.y_adv_testbed[self.FP_indices]
        self.FN_labels = self.y_adv_testbed[self.FN_indices]
        # Print the shapes of FP and FN samples
        print("Shape of False Positive (FP) samples:", self.FP_samples.shape)
        print("Shape of False Negative (FN) samples:", self.FN_samples.shape)
        # Print the shapes of FP and FN labels
        print("Shape of False Positive (FP) labels:", self.FP_labels.shape)
        print("Shape of False Negative (FN) labels:", self.FN_labels.shape)
        print()
        print()
        print()
        print()
        
        self.c = ConfusionMatrixDisplay(confusion_matrix(self.y_adv_testbed,self.mpd_pred), display_labels=['Non-Adversarial','Adversarial'])
        fig, ax = plt.subplots(figsize=(10,6))
        plt.grid(False)
        self.c.plot(ax = ax, cmap='OrRd', xticks_rotation = 0,values_format='d')