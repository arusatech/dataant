#classification models
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
#Ensemble models
# from sklearn.ensemble import (
#     GradientBoostingClassifier,
#     AdaBoostClassifier,
#     XGBClassifier,
#     LightGBM
# )
#Clustering models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

#Generic models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  # or any other model
from jsonpath_nz import log, jprint
import pandas as pd
import time
from collections import deque
from datetime import datetime

class ModelTrainer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelTrainer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.model = None
            self.scaler = StandardScaler()
            self._initialized = True
            self.is_trained = False
            self.feature_columns = None
            self.last_training_shape = None
            self.training_history = deque(maxlen=100)  # Keep last 100 training times
    
    def calculate_metrics(self, y_true, y_pred_proba):
        """Calculate ROC and Precision-Recall metrics"""
        try:
            # ROC Curve metrics
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall Curve metrics
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = average_precision_score(y_true, y_pred_proba)
            
            return {
                'fpr': fpr,
                'tpr': tpr,
                'auc_roc': roc_auc,
                'precision': precision,
                'recall': recall,
                'auc_pr': pr_auc
            }
            
        except Exception as e:
            log.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train the model if needed"""
        current_shape = X.shape
        start_time = time.time()
        
        try:
            # Check if model needs retraining
            if (self.is_trained and 
                self.last_training_shape == current_shape and 
                self.feature_columns == list(X.columns)):
                log.info("Model already trained with same features and data shape")
                return self.last_metrics
            
            log.info(f"Training model with shape {current_shape}")
            self.feature_columns = list(X.columns)
            self.last_training_shape = current_shape
            
            # Check if we have at least two classes
            unique_classes = np.unique(y)
            n_classes = len(unique_classes)
            log.info(f"Training with {n_classes} classes: {unique_classes}")
            
            if n_classes < 2:
                raise ValueError(f"Need at least 2 classes for classification. Found classes: {unique_classes}")
            
            # Split and scale the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Fit and transform the training data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train the model
            self.model = LogisticRegression(random_state=random_state)
            self.model.fit(X_train_scaled, y_train)
            
            # Get prediction probabilities
            y_scores_train = self.model.predict_proba(X_train_scaled)[:, 1]
            y_scores_test = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_scores_test)
            metrics.update({
                'y_scores_train': y_scores_train,
                'y_scores_test': y_scores_test,
                'y_train': y_train,
                'y_test': y_test
            })
            
            # Calculate and store training time
            training_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.training_history.append({
                'timestamp': datetime.now(),
                'training_time': training_time,
                'data_size': len(X),
                'n_features': X.shape[1]
            })
            
            log.info(f"Model training completed in {training_time:.2f}ms")
            
            self.is_trained = True
            self.last_metrics = metrics
            return metrics
            
        except Exception as e:
            # Also track failed training attempts
            training_time = (time.time() - start_time) * 1000
            self.training_history.append({
                'timestamp': datetime.now(),
                'training_time': training_time,
                'data_size': len(X),
                'n_features': X.shape[1],
                'error': str(e)
            })
            log.error(f"Error in train_model: {str(e)}")
            self.is_trained = False
            raise
    
    def predict(self, X):
        """Make predictions on new data"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")
        
        if list(X.columns) != self.feature_columns:
            raise ValueError("Feature columns don't match training data")
            
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)[:, 1]
        except Exception as e:
            log.error(f"Error in predict: {str(e)}")
            raise
        

class ModelSelector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Initialize all models
        self.classification_models = {
            'logistic': LogisticRegression(random_state=random_state),
            'svm': SVC(random_state=random_state),
            'decision_tree': DecisionTreeClassifier(random_state=random_state),
            'random_forest': RandomForestClassifier(random_state=random_state),
            'knn': KNeighborsClassifier()
        }
        
        self.regression_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(random_state=random_state),
            'lasso': Lasso(random_state=random_state),
            'random_forest': RandomForestRegressor(random_state=random_state),
            'svr': SVR(),
        }
        self.clustering_models = {
            'kmeans': KMeans(random_state=random_state),
            'dbscan': DBSCAN(random_state=random_state),
            'hierarchical': AgglomerativeClustering(random_state=random_state)
        }
        # self.ensemble_models = {
        #     'gradient_boosting': GradientBoostingClassifier(random_state=random_state),
        #     'adaboost': AdaBoostClassifier(random_state=random_state),
        #     'xgboost': XGBClassifier(random_state=random_state),
        #     'lightgbm': LightGBM(random_state=random_state)
            
        # }
