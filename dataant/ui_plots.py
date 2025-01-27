from plotnine import (
    theme,
    ggplot,
    geom_density,
    theme_minimal,
    labs,
    aes,
    geom_line,
    geom_abline,
    geom_point,
    scale_x_continuous,
    scale_y_continuous,
    element_text,
    element_line,
    element_rect
)

import numpy as np
import pandas as pd
from jsonpath_nz import log, jprint
import json
import matplotlib.pyplot as plt

def parse_metrics(metrics_json):
    """Convert string arrays to numpy arrays"""
    metrics = json.loads(metrics_json) if isinstance(metrics_json, str) else metrics_json
    
    return {
        'fpr': np.array(json.loads(metrics['fpr'].replace('\n', '')) if isinstance(metrics['fpr'], str) 
                       else metrics['fpr']),
        'tpr': np.array(json.loads(metrics['tpr'].replace('\n', '')) if isinstance(metrics['tpr'], str) 
                       else metrics['tpr']),
        'precision': np.array(json.loads(metrics['precision'].replace('\n', '')) if isinstance(metrics['precision'], str) 
                            else metrics['precision']),
        'recall': np.array(json.loads(metrics['recall'].replace('\n', '')) if isinstance(metrics['recall'], str) 
                          else metrics['recall']),
        'auc_roc': float(metrics['auc_roc']),  # Changed from 'auc'
        'auc_pr': float(metrics['auc_pr'])     # Changed from 'average_precision'
    }

def plot_roc_curve(metrics_data):
    """
    Create ROC curve plot using plotnine
    """
    metrics = parse_metrics(metrics_data)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'False Positive Rate': metrics['fpr'],
        'True Positive Rate': metrics['tpr']
    })
    
    # Create the plot
    plot = (
        ggplot(plot_df, aes(x='False Positive Rate', y='True Positive Rate'))
        + geom_line(color='#2C85B2', size=1.2)  # Blue line
        + geom_abline(slope=1, intercept=0, linetype='dashed', color='gray', size=0.8)
        + labs(
            title=f'ROC Curve (AUC = {metrics["auc_roc"]:.3f})',
            x='False Positive Rate',
            y='True Positive Rate'
        )
        + theme_minimal()
        + theme(
            plot_title=element_text(size=12, face="bold"),
            axis_title=element_text(size=10),
            panel_grid_major=element_line(color='lightgray', size=0.5),
            panel_grid_minor=element_line(color='lightgray', size=0.25)
        )
        + scale_x_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.2))
        + scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.2))
    )
    
    return plot

def plot_precision_recall_curve(metrics_data):
    """
    Create Precision-Recall curve plot using plotnine
    """
    metrics = parse_metrics(metrics_data)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Recall': metrics['recall'],
        'Precision': metrics['precision']
    })
    
    # Create the plot
    plot = (
        ggplot(plot_df, aes(x='Recall', y='Precision'))
        + geom_line(color='#B22C2C', size=1.2)  # Red line
        + labs(
            title=f'Precision-Recall Curve (AP = {metrics["auc_pr"]:.3f})',
            x='Recall',
            y='Precision'
        )
        + theme_minimal()
        + theme(
            plot_title=element_text(size=12, face="bold"),
            axis_title=element_text(size=10),
            panel_grid_major=element_line(color='lightgray', size=0.5),
            panel_grid_minor=element_line(color='lightgray', size=0.25)
        )
        + scale_x_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.2))
        + scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.2))
    )
    
    return plot

def plot_score_distribution(metrics):
    """
    Create a distribution plot of model scores using plotnine
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing model metrics including 'y_scores_train' and 'y_scores_test'
    """
    try:
        # Create a DataFrame with the scores
        train_df = pd.DataFrame({
            'Score': metrics['y_scores_train'],
            'Set': 'Training',
            'True_Label': metrics['y_train']
        })
        
        test_df = pd.DataFrame({
            'Score': metrics['y_scores_test'],
            'Set': 'Testing',
            'True_Label': metrics['y_test']
        })
        
        # Combine the datasets
        plot_df = pd.concat([train_df, test_df])
        
        # Create the distribution plot
        plot = (
            ggplot(plot_df, aes(x='Score', fill='Set')) +
            geom_density(alpha=0.5) +
            theme_minimal() +
            labs(
                title='Model Score Distribution',
                x='Prediction Score',
                y='Density'
            ) +
            theme(
                figure_size=(10, 6),
                plot_background=element_rect(fill='white'),
                panel_grid_major=element_line(color='lightgray'),
                panel_grid_minor=element_line(color='lightgray'),
                legend_title=element_text(size=10),
                legend_text=element_text(size=8)
            )
        )
        
        return plot
        
    except Exception as e:
        log.error(f"Error in plot_score_distribution: {str(e)}")
        return ggplot()  

def plot_metrics(metrics):
    """Create ROC and Precision-Recall curves plot"""
    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        ax1.plot(
            metrics['fpr'], 
            metrics['tpr'], 
            color='darkorange', 
            lw=2, 
            label=f"ROC curve (AUC = {metrics['auc_roc']:.2f})"  # Changed from 'auc' to 'auc_roc'
        )
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC)')
        ax1.legend(loc="lower right")
        ax1.grid(True)
        
        # Precision-Recall Curve
        ax2.plot(
            metrics['recall'], 
            metrics['precision'], 
            color='darkgreen', 
            lw=2, 
            label=f"PR curve (AUC = {metrics['auc_pr']:.2f})"  # Changed from 'average_precision' to 'auc_pr'
        )
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        log.error(f"Error in plot_metrics: {str(e)}")
        # Return an empty figure on error
        return plt.figure()

# def plot_auc_curve(df: DataFrame, true_col: str, pred_col: str):
#     fpr, tpr, _ = roc_curve(df[true_col], df[pred_col])
#     roc_auc = auc(fpr, tpr)

#     roc_df = DataFrame({"fpr": fpr, "tpr": tpr})

#     plot = (
#         ggplot(roc_df, aes(x="fpr", y="tpr"))
#         + geom_line(color="darkorange", size=1.5, show_legend=True, linetype="solid")
#         + geom_abline(intercept=0, slope=1, color="navy", linetype="dashed")
#         + labs(
#             title="Receiver Operating Characteristic (ROC)",
#             subtitle=f"AUC: {roc_auc.round(2)}",
#             x="False Positive Rate",
#             y="True Positive Rate",
#         )
#         + theme_minimal()
#     )

#     return plot


# def plot_precision_recall_curve(df: DataFrame, true_col: str, pred_col: str):
#     precision, recall, _ = precision_recall_curve(df[true_col], df[pred_col])

#     pr_df = DataFrame({"precision": precision, "recall": recall})

#     plot = (
#         ggplot(pr_df, aes(x="recall", y="precision"))
#         + geom_line(color="darkorange", size=1.5, show_legend=True, linetype="solid")
#         + labs(
#             title="Precision-Recall Curve",
#             x="Recall",
#             y="Precision",
#         )
#         + theme_minimal()
#     )

#     return plot

def plot_production_score_distribution(metrics):
    """Create a distribution plot of model scores using plotnine"""
    plot_df = pd.DataFrame({
        'Score': metrics,
        'Type': 'Production'
    })
    
    plot = (
        ggplot(plot_df, aes(x='Score'))
        + geom_density(fill='#2CB270', alpha=0.5)  # Green fill
        + theme_minimal()
        + labs(
            title='Production Score Distribution',
            x='Model Score',
            y='Density'
        )
        + theme(
            figure_size=(8, 4),
            plot_title=element_text(size=12, face="bold"),
            axis_title=element_text(size=10),
            panel_grid_major=element_line(color='lightgray', size=0.5),
            panel_grid_minor=element_line(color='lightgray', size=0.25)
        )
        + scale_x_continuous(limits=[0, 1])
    )
    
    return plot
                
def plot_api_response(history):
    """Create a distribution plot of model scores using plotnine"""
    # Convert history to DataFrame
    plot_df = pd.DataFrame(history)
    
    # Create the plot
    plot = (
        ggplot(plot_df, aes(x='timestamp', y='training_time'))
        + geom_line(color='#2C85B2', size=1)
        + geom_point(aes(size='data_size'), color='#2C85B2', alpha=0.5)
        + theme_minimal()
        + labs(
            title='Model Training Time History',
            x='Time',
            y='Training Time (ms)',
            size='Data Size'
        )
        + theme(
            figure_size=(8, 4),
            plot_title=element_text(size=12, face="bold"),
            axis_title=element_text(size=10),
            axis_text_x=element_text(angle=45, hjust=1),
            panel_grid_major=element_line(color='lightgray', size=0.5),
            panel_grid_minor=element_line(color='lightgray', size=0.25)
        )
    )
    
    # Add error indicators if any exist
    if 'error' in plot_df.columns:
        error_points = plot_df[plot_df['error'].notna()]
        if len(error_points) > 0:
            plot = plot + geom_point(
                data=error_points,
                mapping=aes(x='timestamp', y='training_time'),
                color='red',
                size=3,
                shape='x'
            )
    
    return plot
