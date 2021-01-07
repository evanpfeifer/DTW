# NFL Big Data Bowl 2021
To determine the effectiveness of various defensive plays, one can simply use existing data fields such as offensive formation or defenders in the box. However, these fields contain very coarse information and may not offer key insight into play efficacy.

I teamed up with my dad (John Pfeifer, PhD computer science candidate at the University of Sydney) to contribute a novel data mining method that automatically determines a set of offensive and defensive play type categories based on the movement of the players and the football. My dad largely taught me about the research topics that were utilized (Gonzalez clustering, Dynamic Time Warp, linear trajectory simplification), and I coded all of the implementations and performed the exploratory data analysis and project writeup.

Each offensive and defensive play cluster is assigned a unique ID, which are used to populate the offensive play category and defensive play category in the Plays table. Therefore, play categories represent clusters of similar plays.

We use the [Gonzalez (1985)](https://pdf.sciencedirectassets.com/271538/1-s2.0-S0304397500X05245/1-s2.0-0304397585902245/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEHEaCXVzLWVhc3QtMSJIMEYCIQDGfxpaGACnccvlqmpBdBRqDVHIiZ8X3lCBJ4S%2FDgNAeAIhAK8%2FMI4ZT7dq9aFBS%2FlC36OKGKp8AiF4uO%2B5XVWJ7eo5KrQDCDoQAxoMMDU5MDAzNTQ2ODY1Igxf0fq1SH3J5xaye7EqkQOu%2BpyLPBjl121xWljCYSHdsxvc0puui3sbthzaX5t7s9eiPR0FSXnng4KjDMnKlIWU5p6bmaS%2Bu1%2B97sP1BXUNBoqEKehWGfbau%2B3xVNHOVfPpac%2BHmLlQmVjHAlvulFKJcrJFcaPCHTMyXbqtN%2FsuKMB%2FHAbLp2cyAkTfFo4pn4WY3h1RXE1rlkXDMdoAlC2P6TNcmGDcy%2FxR2ZeptaTO9IBsnzrYBX%2F%2BCWfeE2oYC9tydFlOb2FwgaLRtL110CQZNuUezxmncqlXJoCEb2RFpcYkHfqIqNeLZeYXLaYpsJZ6hDuohZgImZUfSu%2FMBLMkqIJ%2BEm5z9GC3C4A3wOmCsvfv2Ae1TLr6v8vogCd6fOWsEH8xSgE1et5QOQBCQIShhxX8oC28lvh7S9NondhhOaNzWvpfwVc27PH2Jyy%2FlVIhZsvRG%2BLtooN4vC2AxA5Fd3QS1fixVEF7XM3pMeeGd8Owwt7Bc%2FfHL572evpNFT7hm79wfeU4SLJXqUSW68J17nGnGZYq%2FF6nU90J%2BhNmFjC8t9n%2FBTrqAZjuSa4ZByPcfgFkJyh6gOR5DhGjmFVfUfXqBSx%2FuXUrXCu143G2aIZ0ZghC%2B33G0sCGQf3LyMRCi7P%2BsNCoka6WSnl7rJ17DHoq94hF1JcbMqjc9rwJJPEZeBFubpfESACS%2FgqL5Ly8U2bCuTbXDiZBCSM7QBQfIJeKubnvrPjZ8jsMB9xuqIO1URnZGN2wyJKIniBXHeRWtbjOesN5lxQLBEOUqaXklQIVseZsTndy4KRPXr9fBUulb53YmGlhsGGVosI4j9R%2FodsnInv9X%2BFrXNgqiO78PZSLzSSmAHbYtQ58Phy1KpoILA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20210107T012913Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYTJKIFFML%2F20210107%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=304481e7a552dad5232e21f4abe2b3e59bb0832ecb045f4d21826150b69d3439&hash=696736a59b490f986429d934a6ab3ca98ddd680ca2a0944690a1f5db659a0c09&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=0304397585902245&tid=spdf-bc85b58f-90ee-4086-8401-9dbeecb72201&sid=66ebbdbe8fe9294e38791ed7a4f0aa32eb98gxrqa&type=client) clustering algorithm, since it produces clusters that have radii at most two times the optimal cluster radii. We use the Dynamic Time Warp (DTW) [Keogh et al. (2005)](https://www.researchgate.net/publication/225230134_Exact_indexing_of_dynamic_time_warping) distance measure to compute distances between trajectories since it can smooth outlier movement.

## Files

The link to the competition submission on Kaggle is [here](https://www.kaggle.com/evanpfeifer/uncovering-playbooks-automatic-play-clustering). Code for the Gonzalez clustering, Dynamic Time Warp, and linear trajectory simplification are found in `preprocessing.py` and `cluster.py`. 

As the clustering code takes awhile to run, you can find the clustered data [here](https://drive.google.com/drive/folders/1ZVGIW-JvhbQ6BtuBfnXWq-oLQ-OdiSMj?usp=sharing). For this example, we clustered until the maximum radius of any cluster was below 1000 (DTW distance). The DTW algorithm runtime is quadratic in the number of trajectory vertices (frames), which can be computationally expensive for larger trajectories. We simplified the trajectories (reduce the vertices) using a straightforward linear-time algorithm by [Driemel et al. (2012)](https://www.researchgate.net/publication/221589940_An_algorithmic_framework_for_segmenting_trajectories_based_on_spatio-temporal_criteria). Our simplification error for the experiments is 3 yards, which results in much smaller trajectories that still describe the movement well.

Check out the competition on [Kaggle](https://www.kaggle.com/c/nfl-big-data-bowl-2021).
