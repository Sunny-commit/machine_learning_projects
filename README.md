# 🖥️ Machine Learning Projects - Comprehensive Collection

A **collection of diverse machine learning projects** covering regression, classification, clustering, and NLP applications.

## 🎯 Overview

This portfolio includes:
- ✅ Regression models
- ✅ Classification systems
- ✅ Clustering analysis
- ✅ Dimensionality reduction
- ✅ Ensemble methods
- ✅ Model evaluation
- ✅ Real-world datasets

## 📊 Regression Models

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

class RegressionModels:
    """Collection of regression approaches"""
    
    @staticmethod
    def linear_regression(X_train, y_train, X_test):
        """Basic linear regression"""
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        predictions = lr.predict(X_test)
        coefficients = pd.Series(lr.coef_, index=X_train.columns)
        
        return {
            'model': lr,
            'predictions': predictions,
            'coefficients': coefficients,
            'intercept': lr.intercept_
        }
    
    @staticmethod
    def ridge_regression(X_train, y_train, X_test, alpha=1.0):
        """Ridge regression with L2 regularization"""
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        
        predictions = ridge.predict(X_test)
        
        return predictions
    
    @staticmethod
    def lasso_regression(X_train, y_train, X_test, alpha=0.1):
        """Lasso with feature selection"""
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train, y_train)
        
        predictions = lasso.predict(X_test)
        selected_features = X_train.columns[lasso.coef_ != 0]
        
        return {
            'predictions': predictions,
            'selected_features': selected_features
        }
    
    @staticmethod
    def random_forest_regression(X_train, y_train, X_test):
        """Random Forest regressor"""
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        
        feature_importance = pd.Series(
            rf.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return {
            'predictions': predictions,
            'feature_importance': feature_importance
        }
    
    @staticmethod
    def gradient_boosting_regression(X_train, y_train, X_test):
        """Gradient Boosting regressor"""
        gb = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        )
        
        gb.fit(X_train, y_train)
        predictions = gb.predict(X_test)
        
        return predictions
```

## 🎯 Classification Systems

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClassificationModels:
    """Classification approaches"""
    
    @staticmethod
    def logistic_regression(X_train, y_train, X_test, y_test):
        """Logistic regression classifier"""
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        
        y_pred = lr.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
    
    @staticmethod
    def svm_classifier(X_train, y_train, X_test, y_test):
        """Support Vector Machine"""
        svm = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm.fit(X_train, y_train)
        
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    @staticmethod
    def random_forest_classifier(X_train, y_train, X_test, y_test):
        """Random Forest classifier"""
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'feature_importance': pd.Series(
                rf.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)
        }
        
        return metrics
    
    @staticmethod
    def gradient_boosting_classifier(X_train, y_train, X_test, y_test):
        """Gradient Boosting classifier"""
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=42
        )
        
        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_test)
        
        return accuracy_score(y_test, y_pred)
```

## 🔄 Clustering Analysis

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

class ClusteringModels:
    """Unsupervised clustering"""
    
    @staticmethod
    def kmeans_clustering(X, n_clusters=3):
        """K-Means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        silhouette = silhouette_score(X, labels)
        
        return {
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette
        }
    
    @staticmethod
    def optimal_clusters(X, max_k=10):
        """Find optimal number of clusters"""
        inertias = []
        silhouettes = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))
        
        # Elbow method
        optimal_k = np.argmax(silhouettes) + 2
        
        return {
            'inertias': inertias,
            'silhouette_scores': silhouettes,
            'optimal_k': optimal_k
        }
    
    @staticmethod
    def dbscan_clustering(X, eps=0.5, min_samples=5):
        """DBSCAN density-based clustering"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise_points': n_noise
        }
    
    @staticmethod
    def hierarchical_clustering(X, n_clusters=3):
        """Agglomerative hierarchical clustering"""
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        labels = hierarchical.fit_predict(X)
        
        return labels
```

## 🔍 Dimensionality Reduction

```python
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

class DimensionalityReduction:
    """Reduce feature dimensions"""
    
    @staticmethod
    def pca_reduction(X, n_components=2):
        """Principal Component Analysis"""
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        
        explained_variance = pca.explained_variance_ratio_.sum()
        
        return {
            'reduced_data': X_reduced,
            'explained_variance': explained_variance,
            'components': pca.components_
        }
    
    @staticmethod
    def pca_optimal_components(X, variance_threshold=0.95):
        """Find optimal PCA dimensions"""
        pca = PCA()
        pca.fit(X)
        
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= variance_threshold) + 1
        
        return n_components
    
    @staticmethod
    def tsne_visualization(X, n_components=2, perplexity=30):
        """t-SNE for visualization"""
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        X_embedded = tsne.fit_transform(X)
        
        return X_embedded
```

## 📈 Model Evaluation & Comparison

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

class ModelEvaluator:
    """Compare and evaluate models"""
    
    @staticmethod
    def cross_validation_scores(model, X, y, cv=5):
        """K-fold cross validation"""
        scores = cross_val_score(model, X, y, cv=cv)
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    @staticmethod
    def model_comparison(models_dict, X_test, y_test):
        """Compare multiple models"""
        results = {}
        
        for name, model in models_dict.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
        
        best_model = max(results, key=results.get)
        
        return {
            'accuracies': results,
            'best_model': best_model
        }
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=None):
        """Visualize confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
```

## 💡 Interview Talking Points

**Q: Bias-Variance tradeoff?**
```
Answer:
- Underfit: high bias, low variance
- Overfit: low bias, high variance
- Regularization reduces variance
- Ensemble methods help balance
- Cross-validation essential
```

**Q: Feature scaling importance?**
```
Answer:
- Distance-based models (KNN, SVM)
- Gradient descent convergence
- Neural networks stability
- Tree models scale-invariant
- Normalization vs standardization
```

## 🌟 Portfolio Value

✅ Regression models
✅ Classification systems
✅ Clustering analysis
✅ Dimensionality reduction
✅ Model comparison
✅ Cross-validation
✅ End-to-end ML pipelines

---

**Technologies**: Scikit-learn, Pandas, NumPy

