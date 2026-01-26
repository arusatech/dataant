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
        
        # Check if target is numeric (regression problem) - ROC not applicable
        y_train = pd.Series(model_data['y_train']) if not isinstance(model_data['y_train'], pd.Series) else model_data['y_train']
        y_test = pd.Series(model_data['y_test']) if not isinstance(model_data['y_test'], pd.Series) else model_data['y_test']
        
        # Check if target is numeric/continuous
        is_numeric = pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() > 10
        
        if is_numeric:
            # For regression problems, ROC is not applicable
            fig.add_annotation(
                text="ROC curves are for classification tasks only.<br>Current target appears to be numeric/continuous.<br>Please select a categorical target column.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            fig.update_layout(
                width=700,
                height=500,
                title="ROC Curves (Training vs Test) - Not Applicable",
                xaxis=dict(title='False Positive Rate', range=[0, 1]),
                yaxis=dict(title='True Positive Rate', range=[0, 1])
            )
            return fig
        
        # Get the number of classes from scores shape
        n_classes = model_data['y_scores_train'].shape[1]
        
        # Get classes from model (this is the correct order)
        if 'classes' in model_data:
            classes = model_data['classes']
        else:
            # Fallback: infer from unique values
            all_classes = pd.concat([y_train, y_test]).unique()
            classes = np.sort(all_classes)
        
        # Create one-hot encoding aligned with model classes
        y_train_onehot = pd.DataFrame(index=y_train.index)
        y_test_onehot = pd.DataFrame(index=y_test.index)
        
        for i, class_label in enumerate(classes):
            y_train_onehot[i] = (y_train == class_label).astype(int)
            y_test_onehot[i] = (y_test == class_label).astype(int)
        
        # Add diagonal reference line for ROC
        if metric_type == 'roc':
            fig.add_shape(
                type='line', 
                line=dict(dash='dash', color='gray', width=1),
                x0=0, x1=1, y0=0, y1=1
            )

        for i in range(n_classes):
            if metric_type == 'roc':
                # Get the class label
                class_label = classes[i] if i < len(classes) else f"Class {i}"
                
                # Get binary indicators for this class
                y_train_binary = y_train_onehot[i].values
                y_test_binary = y_test_onehot[i].values
                
                # Check if class has positive samples in at least one set
                has_positive_train = y_train_binary.sum() > 0
                has_positive_test = y_test_binary.sum() > 0
                
                if not has_positive_train and not has_positive_test:
                    log.warning(f"Skipping class {class_label} - no positive samples in train or test")
                    continue
                
                try:
                    # Plot training curve if it has positive samples
                    if has_positive_train:
                        fpr_train, tpr_train, _ = roc_curve(y_train_binary, 
                                                           model_data['y_scores_train'][:, i])
                        try:
                            auc_train = roc_auc_score(y_train_binary, 
                                                    model_data['y_scores_train'][:, i])
                        except ValueError:
                            # Handle case where only one class present
                            auc_train = 0.5
                            log.warning(f"Class {class_label} train: Only one class present, AUC set to 0.5")
                        
                        name_train = f"Train {class_label} (AUC={auc_train:.2f})"
                        fig.add_trace(go.Scatter(
                            x=fpr_train, 
                            y=tpr_train, 
                            name=name_train, 
                            mode='lines',
                            line=dict(width=2, color='blue')
                        ))
                    
                    # Plot test curve if it has positive samples
                    if has_positive_test:
                        fpr_test, tpr_test, _ = roc_curve(y_test_binary, 
                                                         model_data['y_scores_test'][:, i])
                        try:
                            auc_test = roc_auc_score(y_test_binary, 
                                                   model_data['y_scores_test'][:, i])
                        except ValueError:
                            # Handle case where only one class present
                            auc_test = 0.5
                            log.warning(f"Class {class_label} test: Only one class present, AUC set to 0.5")
                        
                        name_test = f"Test {class_label} (AUC={auc_test:.2f})"
                        fig.add_trace(go.Scatter(
                            x=fpr_test, 
                            y=tpr_test, 
                            name=name_test, 
                            mode='lines', 
                            line=dict(dash='dash', width=2, color='red')
                        ))
                except Exception as e:
                    log.error(f"Error plotting ROC for class {class_label}: {str(e)}")
                    continue
                
            elif metric_type == 'pr':
                # Get the class label
                class_label = classes[i] if i < len(classes) else f"Class {i}"
                
                # Get binary indicators for this class
                y_train_binary = y_train_onehot[i].values
                y_test_binary = y_test_onehot[i].values
                
                # Check if class has positive samples in at least one set
                has_positive_train = y_train_binary.sum() > 0
                has_positive_test = y_test_binary.sum() > 0
                
                if not has_positive_train and not has_positive_test:
                    log.warning(f"Skipping class {class_label} - no positive samples in train or test")
                    continue
                
                try:
                    # Plot training curve if it has positive samples
                    if has_positive_train:
                        precision_train, recall_train, _ = precision_recall_curve(
                            y_train_binary, model_data['y_scores_train'][:, i])
                        try:
                            auc_train = average_precision_score(y_train_binary, 
                                                             model_data['y_scores_train'][:, i])
                        except ValueError:
                            auc_train = 0.0
                            log.warning(f"Class {class_label} train: Only one class present")
                        
                        name_train = f"Train {class_label} (AP={auc_train:.2f})"
                        fig.add_trace(go.Scatter(
                            x=recall_train, 
                            y=precision_train, 
                            name=name_train, 
                            mode='lines',
                            line=dict(width=2, color='blue')
                        ))
                    
                    # Plot test curve if it has positive samples
                    if has_positive_test:
                        precision_test, recall_test, _ = precision_recall_curve(
                            y_test_binary, model_data['y_scores_test'][:, i])
                        try:
                            auc_test = average_precision_score(y_test_binary, 
                                                            model_data['y_scores_test'][:, i])
                        except ValueError:
                            auc_test = 0.0
                            log.warning(f"Class {class_label} test: Only one class present")
                        
                        name_test = f"Test {class_label} (AP={auc_test:.2f})"
                        fig.add_trace(go.Scatter(
                            x=recall_test, 
                            y=precision_test, 
                            name=name_test, 
                            mode='lines', 
                            line=dict(dash='dash', width=2, color='red')
                        ))
                except Exception as e:
                    log.error(f"Error plotting PR curve for class {class_label}: {str(e)}")
                    continue

        # Update layout with correct axis labels and title based on metric type
        if metric_type == 'roc':
            fig.update_layout(
                width=700,
                height=500,
                showlegend=True,
                xaxis=dict(title='False Positive Rate', range=[0, 1]),
                yaxis=dict(title='True Positive Rate', range=[0, 1]),
                title='ROC Curves (Training vs Test)'
            )
        elif metric_type == 'pr':
            fig.update_layout(
                width=700,
                height=500,
                showlegend=True,
                xaxis=dict(title='Recall', range=[0, 1]),
                yaxis=dict(title='Precision', range=[0, 1]),
                title='Precision-Recall Curves (Training vs Test)'
            )
        else:
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

def plot_default_metric(selected_field1, selected_field2, plot_df, selected_target):
    """Create a 3D scatter plot with selectable Field 1, Field 2, and Target"""
    try:
        # Ensure all columns exist
        if selected_field1 not in plot_df.columns or selected_field2 not in plot_df.columns or selected_target not in plot_df.columns:
            log.error(f"One or more selected fields not found in dataframe")
            return go.Figure()
        
        # Extract columns as Series to ensure 1D data
        col1 = plot_df[selected_field1]
        col2 = plot_df[selected_field2]
        col3 = plot_df[selected_target]
        
        # Handle case where column might be a DataFrame (select first column)
        if isinstance(col1, pd.DataFrame):
            col1 = col1.iloc[:, 0]
        if isinstance(col2, pd.DataFrame):
            col2 = col2.iloc[:, 0]
        if isinstance(col3, pd.DataFrame):
            col3 = col3.iloc[:, 0]
        
        # Create a clean dataframe with the three columns
        plot_df_clean = pd.DataFrame({
            selected_field1: col1,
            selected_field2: col2,
            selected_target: col3
        }).dropna()
        
        if len(plot_df_clean) == 0:
            log.warning("No valid data points after removing NaN values")
            return go.Figure()
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            plot_df_clean,
            x=selected_field1,
            y=selected_field2,
            z=selected_target,
            color=selected_target,
            title=f"3D Plot: {selected_field1} vs {selected_field2} vs {selected_target}",
            labels={
                selected_field1: selected_field1,
                selected_field2: selected_field2,
                selected_target: selected_target
            },
            color_continuous_scale='Viridis'
        )
        
        # Update layout for better visualization
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=30),
            scene=dict(
                xaxis_title=selected_field1,
                yaxis_title=selected_field2,
                zaxis_title=selected_target,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600
        )
        
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
    