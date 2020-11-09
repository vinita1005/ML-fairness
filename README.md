# ML-fairness

# Abstract

In this project, we have come up with a fairness system similar to that of COMPAS system. The
main goal of the project was to implement and choose such an algorithm which will improve the
current system in terms of accuracy or cost and will give unbiased results accross various racial groups.
We have implemented the various fairness algorithms which are used as a postprocessing methods in
various models and compared the results. After careful consideration, we have come up with solution
presented in this report.

# Introduction

COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) is the system
commonly used by many states to assess a criminal defendant's likelihood of committing a crime. It
was developed in 1998 and since then, it has been used to assess more than 1 million criminals. Many
people argue that such a system is highly biased towards a certain race than others. ProPublica's
study shows that the COMPAS system was highly biased with the white race than the black race. So
much so that black convicts were termed "risky" almost twice as the white convicts.
This study was then countered by Northpointe saying that ProPublica's study lacked a thorough
analysis and COMPAS system adopted a racial bias even when there was no data of the person being
of a certain race. Even so, it is not a bad idea to use risk assessment systems such as COMPAS. The
skills of decision making in humans is sometimes not that accurate at all. You would be surprised
to know that a study showed parole boards were more likely to let the convict go free if the judges had
a meal break just before the verdict!

# Proposed Solution

A machine learning risk analysis system could discover such inconsistencies stated above. That is
the reason we have created one such system to replace the COMPAS. We have tried to address the
problems of racial bias using a Group fairness approach [2].
Details of the choosen Fairness Market model:
Market Model of Choice: SVM
Algorithm of choice: Equal Opportunity
Secondary optimization criteria: Cost

Cost of the system based on market model:-
Cost on Training data: $-628,702,592
Cost on Test data: $-142,856,824
Accuracy of the system based on market model:-
Accuracy on Training data: 0.6378390911198701
Accuracy on Test data: 0.6497909893172318

# Problems addressed by the system
Mainly, unfair bias arises in machine learning system due to selection, sampling, reporting bias in
the data set or bias in the objective function.[3] Even if the data does not explicitly contain biases, it
may have some features known as sensitive features.
These features may affect the way the machine learning algorithm treats certain people having those
sensitive features. These features/attributes are generaly the data such as gender, sexuality, age etc.
Although the machine learning algorithm may not contain biases, it may happen that due to such biases
in the data itself, our algorithm might turn towards a certain criteria and increase the unfair bias.

# Impact of our solution

Our algorithm has a "Group fairness approach". Which means that every group will get an equal
opportunity. To achieve equal opportunity, we pick per-group thresholds such that the fraction of
low-risk group members is the same across all groups. i.e Equal TPR. This ensures that all the groups
are treated equally since every group will have a similar TPR. By using the secondary optimization
criteria as cost, we are not only getting the optimal cost but also reducing the false negative rate.

