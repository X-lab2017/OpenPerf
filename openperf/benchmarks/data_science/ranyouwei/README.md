# Developer Role Classification in GitHub Open Source Community
## Project Overview
The aim of this project is to classify the roles of developers in the GitHub open source community. Based on the developer's behavior and influence in the project, their roles can be roughly divided into four categories: observer, contributor, maintainer, and leader. This project involves constructing a dataset and building a classification model to divide the role of the developer. Additionally, the classification algorithm will be compared with other models, and a deep analysis of behavior patterns based on classification results will be conducted to understand the collaboration mechanism and open source ecology.

# Methodology
## Data Collection:

Data is collected from various GitHub repositories.
Key features include stars, forks, watches, issues, pull_requests, projects, commits, branches, packages, releases, contributors, and license.
# Data Cleaning:

Convert values such as 2.1k to 2100.
Remove commas and quotes from numeric fields.
One-hot encode the license field, expanding it into multiple columns.
Reason for Dataset Construction
The chosen features reflect the developer's activities and influence within the project. By cleaning and processing these features, we ensure that the dataset accurately represents the developer's behavior and influence.

# Classification Models
##Model Selection
MLP + GMM (Gaussian Mixture Model):

The dataset is first processed through a Multi-Layer Perceptron (MLP) to extract meaningful features.
These features are then used in a Gaussian Mixture Model (GMM) to classify developers into four roles.
## Comparison with Other Models:

K-Means Clustering: A simple and commonly used clustering algorithm.
Hierarchical Clustering: Builds a hierarchy of clusters.
DBSCAN (Density-Based Spatial Clustering of Applications with Noise): Identifies clusters based on the density of data points.

# Evaluation Metric
Silhouette Score: Used to evaluate the quality of clustering. It measures how similar an object is to its own cluster compared to other clusters.

# Results and Analysis
Model	Silhouette Score
MLP + GMM	0.55
Behavioral Patterns Analysis
Based on the classification results, the behavior patterns of different developer roles can be analyzed to gain insights into the collaboration mechanism and open source ecology. This involves examining how developers in each role contribute to the project, their frequency of interactions, and their overall impact on the project's success.

# Conclusion
This project provides a comprehensive approach to classify developer roles in the GitHub open source community. By analyzing the behavior patterns based on classification results, we can gain valuable insights into the collaboration mechanism and open source ecology. The comparison of different models highlights the effectiveness of the chosen approach.
