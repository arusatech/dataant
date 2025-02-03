from shinywidgets import render_plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from jsonpath_nz import log, jprint
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from datetime import datetime

def plot_model_metrics(model_data, metric_type='roc'):
    """Create ROC or Precision-Recall curves plot"""
    try:
        fig = go.Figure()
        
        # One hot encode the labels
        y_train_onehot = pd.get_dummies(model_data['y_train'], columns=model_data['classes'])
        y_test_onehot = pd.get_dummies(model_data['y_test'], columns=model_data['classes'])
        
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        for i in range(model_data['y_scores_train'].shape[1]):
            if metric_type == 'roc':
                fpr_train, tpr_train, _ = roc_curve(y_train_onehot.iloc[:, i], 
                                                   model_data['y_scores_train'][:, i])
                auc_train = roc_auc_score(y_train_onehot.iloc[:, i], 
                                        model_data['y_scores_train'][:, i])
                
                fpr_test, tpr_test, _ = roc_curve(y_test_onehot.iloc[:, i], 
                                                 model_data['y_scores_test'][:, i])
                auc_test = roc_auc_score(y_test_onehot.iloc[:, i], 
                                       model_data['y_scores_test'][:, i])
                
                name_train = f"Train {y_train_onehot.columns[i]} (AUC={auc_train:.2f})"
                name_test = f"Test {y_test_onehot.columns[i]} (AUC={auc_test:.2f})"
                
                fig.add_trace(go.Scatter(x=fpr_train, y=tpr_train, name=name_train, mode='lines'))
                fig.add_trace(go.Scatter(x=fpr_test, y=tpr_test, name=name_test, 
                                       mode='lines', line=dict(dash='dash')))
                
                fig.update_layout(
                    xaxis=dict(title='False Positive Rate'),
                    yaxis=dict(title='True Positive Rate'),
                    title='ROC Curves (Training vs Test)'
                )
                
            elif metric_type == 'pr':
                precision_train, recall_train, _ = precision_recall_curve(
                    y_train_onehot.iloc[:, i], model_data['y_scores_train'][:, i])
                auc_train = average_precision_score(y_train_onehot.iloc[:, i], 
                                                 model_data['y_scores_train'][:, i])
                
                precision_test, recall_test, _ = precision_recall_curve(
                    y_test_onehot.iloc[:, i], model_data['y_scores_test'][:, i])
                auc_test = average_precision_score(y_test_onehot.iloc[:, i], 
                                                model_data['y_scores_test'][:, i])
                
                name_train = f"Train {y_train_onehot.columns[i]} (AUC={auc_train:.2f})"
                name_test = f"Test {y_test_onehot.columns[i]} (AUC={auc_test:.2f})"
                
                fig.add_trace(go.Scatter(x=recall_train, y=precision_train, 
                                       name=name_train, mode='lines'))
                fig.add_trace(go.Scatter(x=recall_test, y=precision_test, 
                                       name=name_test, mode='lines', line=dict(dash='dash')))
                
                fig.update_layout(
                    xaxis=dict(title='Recall'),
                    yaxis=dict(title='Precision'),
                    title='Precision-Recall Curves (Training vs Test)'
                )

        fig.update_layout(
            width=700,
            height=500,
            showlegend=True
        )
        return fig
        
    except Exception as e:
        log.error(f"Error in plot_model_metrics: {str(e)}")
        return go.Figure()

def plot_score_distribution(model_data):
    """Create a distribution plot using plotly"""
    try:
        train_df = pd.DataFrame({
            'Score': model_data['y_scores_train'][:, 1],
            'Set': 'Training',
            'True_Label': model_data['y_train']
        })
        
        test_df = pd.DataFrame({
            'Score': model_data['y_scores_test'][:, 1],
            'Set': 'Testing',
            'True_Label': model_data['y_test']
        })
        
        plot_df = pd.concat([train_df, test_df])
        
        # Get the latest training time from history
        latest_training = model_data['training_history'][-1]
        training_time_ms = latest_training['training_time']
        
        fig = px.histogram(
            plot_df,
            x='Score',
            color='Set',
            marginal='violin',
            opacity=0.7,
            title=f'Model Score Distribution (Training Time: {training_time_ms:.1f}ms)'
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        
        return fig
        
    except Exception as e:
        log.error(f"Error in plot_score_distribution: {str(e)}")
        return go.Figure()

def plot_default_metric(selected_field1, selected_field2, plot_df, index):
    """Create a scatter plot of the metric using plotly"""
    try:
        fig = px.scatter_3d(
                    plot_df,
                    x=selected_field1,
                    y=selected_field2,
                    z=index,
                    color=index
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        return fig
    except Exception as e:
        log.error(f"Error in plot_default_metric: {str(e)}")
        log.traceback(e)
        return go.Figure()
    
def plot_api_response(model_data):
    """Create a time series plot using plotly"""
    plot_df = pd.DataFrame(model_data['training_history'])
    
    fig = px.scatter(
        plot_df,
        x='timestamp',
        y='training_time',
        size='data_size',
        title='Model Training Time History (in ms)'
    )
    
    # Add error indicators if any exist
    if 'error' in plot_df.columns:
        error_points = plot_df[plot_df['error'].notna()]
        if len(error_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=error_points['timestamp'],
                    y=error_points['training_time'],
                    mode='markers',
                    marker=dict(symbol='x', color='red', size=10),
                    name='Errors'
                )
            )
    
    fig.update_layout(
        template='plotly_white',
        xaxis_tickangle=45,
        height=400
    )
    
    return fig 

def plot_production_score_distribution(combined_data):
    """Create a distribution plot comparing training and production scores"""
    try:
        # Log the input data structure
        log.info("Combined data keys: " + str(combined_data.keys()))
        log.info("Training data keys: " + str(combined_data.get('training_data', {}).keys()))
        log.info("Production data keys: " + str(combined_data.get('production_data', {}).keys()))
        
        # Extract data with validation
        training_data = combined_data.get('training_data', {})
        production_data = combined_data.get('production_data', {})
        metadata = combined_data.get('metadata', {})
        
        # Safely get scores
        if 'y_scores_test' in training_data:
            training_scores = training_data['y_scores_test'][:, 1]
            log.info(f"Training scores shape: {training_data['y_scores_test'].shape}")
        else:
            log.error("Missing y_scores_test in training_data")
            return go.Figure()
            
        if 'y_scores' in production_data:
            production_scores = production_data['y_scores'][:, 1]
            log.info(f"Production scores shape: {production_data['y_scores'].shape}")
        else:
            log.error("Missing y_scores in production_data")
            return go.Figure()
        
        # Create DataFrames for plotting
        train_df = pd.DataFrame({
            'Score': training_scores,
            'Type': 'Training',
        })
        
        prod_df = pd.DataFrame({
            'Score': production_scores,
            'Type': 'Production',
        })
        
        plot_df = pd.concat([train_df, prod_df])
        
        # Create plot
        fig = px.histogram(
            plot_df,
            x='Score',
            color='Type',
            barmode='overlay',
            opacity=0.7,
            marginal='violin',
            title=f'Score Distribution Comparison\n' +
                  f'Model: {metadata.get("model_name", "Unknown")} | Time: {metadata.get("timestamp", "Unknown")}'
        )
        
        # Add mean lines
        fig.add_vline(x=training_scores.mean(), 
                     line_dash="dash", 
                     line_color="blue",
                     annotation_text="Train Mean")
        fig.add_vline(x=production_scores.mean(), 
                     line_dash="dash", 
                     line_color="red",
                     annotation_text="Prod Mean")
        
        fig.update_layout(
            template='plotly_white',
            xaxis_range=[0, 1],
            height=400,
            bargap=0.1
        )
        
        return fig
        
    except Exception as e:
        log.error(f"Error in plot_production_score_distribution: {str(e)}")
        log.traceback(e)
        return go.Figure()
    