## Customer Segmentation Overview

Customer segmentation is the practice of dividing a company's customers into groups that reflect similarity among customers in each group.

<p align="center">
   <img src="1.png">
</p>


## Customer-Segmentation

Perform cluster analysis and dimensionality reduction to segment customers.
I have used both hierarchical and flat clustering techniques, ultimately focusing on the K-means algorithm. Along the way, I have also visualized the data appropriately to understand the methods. Ultimately, I employed Principal Components Analysis (PCA) through the scikit-learn package. Finally, combined the two (PCA+Kmeans) models to obtain a better segmentation. 

![2](img/2.png)

![3](img/3.png)

![6](img/6.png)

![7](img/7.png)

![8](img/8.png)

![9](img/9.png)


## Segment Interpretation
Once segmented, customers’ behavior will require some interpretation. And I have used the descriptive statistics by brand and by segment and visualized the findings. Through the descriptive analysis, I formed hypotheses about the segments, thus ultimately setting the ground for the subsequent modeling.

![10](img/10.png)

![11](img/11.png)

![12](img/12.png)

![13](img/13.png)

![14](img/14.png)


## Elastic Modeling
In this step, I have done elastic modeling by calculating purchase probability elasticity, brand choice own price elasticity, brand choice cross-price elasticity, and purchase quantity elasticity. We will employ linear regressions and logistic regressions. 

## Predict Future Behavior
Finally, I leveraged the power of Deep Learning to predict future behavior using Feed Forward Neural Network.




