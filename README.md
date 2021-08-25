# iMul-kSite

This repository contains the necessary codes of iMul-kSite: Computational Identification of Multiple Lysine PTM Sites by Analyzing the Instance Hardness and Feature Importance.

Identification of post-translational modifications (PTM) is significant in the study of computational proteomics, cell biology, pathogenesis, and drug development due to its role in many bio-molecular mechanisms. Though there are several computational tools to identify individual PTMs, only three predictors have been established to predict multiple PTMs at the same lysine residue. Furthermore, detailed analysis and assessment on dataset balancing and the significance of different feature encoding techniques for a suitable multi-PTM prediction model are still lacking. This study introduces a computational method named 'iMul-kSite' for predicting acetylation, crotonylation, methylation, succinylation, and glutarylation, from an unrecognized peptide sample with one, multiple, or no modifications. After successfully eliminating the redundant data samples from the majority class by analyzing the hardness of the sequence-coupling information, feature representation has been optimized by adopting the combination of ANOVA F-Test and incremental feature selection approach. The proposed predictor predicts multi-label PTM sites with 92.83\% accuracy using the top 100 features. It has also achieved an 93.36\% aiming rate and 96.23\% coverage rate, which are much better than the existing state-of-the-art predictors on the validation test. This performance indicates that 'iMul-kSite' can be used as a supportive tool for further K-PTM study. For the convenience of the experimental scientists, 'iMul-kSite' has been deployed as a user-friendly web-server at http://103.99.176.239/iMul-kSite.

### Installation guide

  1. `git clone https://github.com/Sabit-Ahmed/iMul-kSite.git`
  2. 'cd iMul-kSite`
  3. 'pip install -r requirements.txt`
  4. `cd model/IFS_implementation` and run `python Com_IFS.py` for incremental feature selection experiment.
  
Or
 
  4. `cd model/Main` and run `python Main.py` for the main experiment.
  5. Select any option between 0 to 9 depending on your need. Each number corresponds to different feature extraction methods, such as, AAF, BE, CKSAAP, Sequence-Coupling etc.
  6. Go to 'iMul-kSite/performance' to find out the performance of the desired methods.
