## Customer Segmentation Overview

Customer segmentation is the practice of dividing a company's customers into groups that reflect similarity among customers in each group.

<p align="center">
   <img src="1.png">
</p>


## Customer-Segmentation

Perform cluster analysis and dimensionality reduction to segment customers.
I have used both hierarchical and flat clustering techniques, ultimately focusing on the K-means algorithm. Along the way, I have also visualized the data appropriately to understand the methods. Ultimately, I employed Principal Components Analysis (PCA) through the scikit-learn package. Finally, combined the two (PCA+Kmeans) models to obtain a better segmentation. 

<p align="center">
   <img src="img/2.png">
</p>

<p align="center">
   <img src="img/3.png">
</p>

<p align="center">
   <img src="img/6.png">
</p>

<p align="center">
   <img src="img/7.png">
</p>

<p align="center">
   <img src="img/8.png">
</p>

<p align="center">
   <img src="img/9.png">
</p>



## Segment Interpretation
Once segmented, customers’ behavior will require some interpretation. And I have used the descriptive statistics by brand and by segment and visualized the findings. Through the descriptive analysis, I formed hypotheses about the segments, thus ultimately setting the ground for the subsequent modeling.

<p align="center">
   <img src="img/10.png">
</p>

<p align="center">
   <img src="img/11.png">
</p>

<p align="center">
   <img src="img/12.png">
</p>

<p align="center">
   <img src="img/13.png">
</p>

<p align="center">
   <img src="img/14.png">
</p>


## Elastic Modeling
In this step, I have done elastic modeling by calculating purchase probability elasticity, brand choice own price elasticity, brand choice cross-price elasticity, and purchase quantity elasticity. We will employ linear regressions and logistic regressions. 

## Predict Future Behavior
Finally, I leveraged the power of Deep Learning to predict future behavior using Feed Forward Neural Network.




