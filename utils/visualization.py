import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

def plot_risk_distribution(df):
    """Plot student risk distribution"""
    fig = px.histogram(
        df, 
        x='dropout_risk', 
        color='risk_category',
        color_discrete_map={
            'Low': '#4CAF50',
            'Medium': '#FFC107',
            'High': '#FF5722',
            'Extreme': '#D32F2F'
        },
        title='Student Dropout Risk Distribution',
        labels={'dropout_risk': 'Dropout Risk Score', 'count': 'Number of Students'},
        marginal='rug'
    )
    
    fig.update_layout(bargap=0.1)
    return fig

def plot_feature_importance(feature_importance):
    """Plot feature importance bars"""
    # Sort features by importance
    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    features = list(sorted_features.keys())
    importance = list(sorted_features.values())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    bars = ax.barh(features, importance, color='#1E88E5')
    
    # Add values to bars
    for i, v in enumerate(importance):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    # Customize plot
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    ax.set_xlim(0, max(importance) * 1.2)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm):
    """Plot confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Normal', 'Anomaly'],
        yticklabels=['Normal', 'Anomaly'],
        ax=ax
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    return fig

def plot_roc_curve(y_true, y_score):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(
        fpr, 
        tpr, 
        color='darkorange',
        lw=2, 
        label=f'ROC curve (area = {roc_auc:.2f})'
    )
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Customize plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    
    return fig

def plot_evidence_weights(evidence_weights):
    """Plot evidence weights for anomaly detection methods"""
    # Create dataframe from weights
    weights_df = pd.DataFrame({
        'Evidence Source': list(evidence_weights.keys()),
        'Weight': list(evidence_weights.values())
    })
    
    # Sort by weight
    weights_df = weights_df.sort_values('Weight', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        weights_df,
        y='Evidence Source',
        x='Weight',
        orientation='h',
        title='Evidence Source Weights',
        color='Weight',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    
    return fig

def plot_belief_uncertainty(df):
    """Plot belief vs uncertainty scatter plot"""
    if 'combined_belief_anomaly' not in df.columns or 'belief_uncertainty' not in df.columns:
        return None
    
    fig = px.scatter(
        df,
        x='combined_belief_anomaly',
        y='belief_uncertainty',
        color='combined_anomaly',
        color_discrete_map={0: '#3498db', 1: '#e74c3c'},
        hover_data=['student_id', 'dropout_risk'],
        title='Belief vs. Uncertainty in Anomaly Detection',
        labels={
            'combined_belief_anomaly': 'Belief in Anomaly',
            'belief_uncertainty': 'Uncertainty',
            'combined_anomaly': 'Anomaly Detected'
        }
    )
    
    # Add quadrant lines
    fig.add_shape(
        type="line",
        x0=0.5, y0=0, x1=0.5, y1=1,
        line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash")
    )
    
    fig.add_shape(
        type="line",
        x0=0, y0=0.3, x1=1, y1=0.3,
        line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash")
    )
    
    # Add quadrant annotations
    fig.add_annotation(
        x=0.75, y=0.15,
        text="High Belief, Low Uncertainty",
        showarrow=False,
        font=dict(color="red")
    )
    
    fig.add_annotation(
        x=0.25, y=0.15,
        text="Low Belief, Low Uncertainty",
        showarrow=False,
        font=dict(color="green")
    )
    
    fig.add_annotation(
        x=0.75, y=0.65,
        text="High Belief, High Uncertainty",
        showarrow=False,
        font=dict(color="orange")
    )
    
    fig.add_annotation(
        x=0.25, y=0.65,
        text="Low Belief, High Uncertainty",
        showarrow=False,
        font=dict(color="blue")
    )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def plot_feedback_metrics(df):
    """Plot feedback metrics analysis"""
    if 'feedback_response_time' not in df.columns:
        return None
        
    # Create scatter plot of feedback metrics
    fig = px.scatter(
        df,
        x='feedback_response_time',
        y='feedback_implementation_rate' if 'feedback_implementation_rate' in df.columns else 'feedback_ignored',
        color='dropout_risk',
        color_continuous_scale='RdYlGn_r',
        hover_data=['student_id', 'quiz_accuracy', 'video_completion_rate'],
        title='Feedback Metrics Analysis',
        labels={
            'feedback_response_time': 'Response Time (min)',
            'feedback_implementation_rate': 'Implementation Rate',
            'feedback_ignored': 'Feedback Ignored Rate',
            'dropout_risk': 'Dropout Risk'
        }
    )
    
    # Add quadrant lines to identify problem areas
    fig.add_shape(
        type="line",
        x0=15, y0=0, x1=15, y1=1,
        line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash")
    )
    
    if 'feedback_implementation_rate' in df.columns:
        fig.add_shape(
            type="line",
            x0=0, y0=0.6, x1=60, y1=0.6,
            line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash")
        )
    else:
        fig.add_shape(
            type="line",
            x0=0, y0=0.4, x1=60, y1=0.4,
            line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash")
        )
    
    # Add annotations for quadrants
    if 'feedback_implementation_rate' in df.columns:
        fig.add_annotation(
            x=30, y=0.3,
            text="High Risk Zone",
            showarrow=False,
            font=dict(color="red")
        )
        
        fig.add_annotation(
            x=7, y=0.8,
            text="Low Risk Zone",
            showarrow=False,
            font=dict(color="green")
        )
    
    return fig

def plot_belief_plausibility(df):
    """Plot belief and plausibility measures with uncertainty"""
    if 'combined_belief_anomaly' not in df.columns or 'plausibility_anomaly' not in df.columns:
        return None
        
    # Create a copy with the data we need
    plot_df = df[['student_id', 'combined_belief_anomaly', 'plausibility_anomaly', 
                  'belief_uncertainty', 'combined_anomaly', 'dropout_risk']].copy()
    
    # Add column for plotting the uncertainty range
    plot_df['uncertainty_range'] = plot_df['plausibility_anomaly'] - plot_df['combined_belief_anomaly']
    
    # Sort by belief for better visualization
    plot_df = plot_df.sort_values('combined_belief_anomaly', ascending=False).reset_index(drop=True)
    
    # Create the figure
    fig = go.Figure()
    
    # Add belief points
    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df['combined_belief_anomaly'],
        mode='markers',
        name='Belief (Lower Bound)',
        marker=dict(color='blue', size=8),
        hovertemplate='Student ID: %{text}<br>Belief: %{y:.3f}<extra></extra>',
        text=plot_df['student_id']
    ))
    
    # Add plausibility points
    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df['plausibility_anomaly'],
        mode='markers',
        name='Plausibility (Upper Bound)',
        marker=dict(color='red', size=8),
        hovertemplate='Student ID: %{text}<br>Plausibility: %{y:.3f}<extra></extra>',
        text=plot_df['student_id']
    ))
    
    # Add uncertainty range
    for i, row in plot_df.iterrows():
        fig.add_shape(
            type="line",
            x0=i, y0=row['combined_belief_anomaly'],
            x1=i, y1=row['plausibility_anomaly'],
            line=dict(color="rgba(0,0,0,0.3)", width=2)
        )
    
    # Add threshold line
    fig.add_shape(
        type="line",
        x0=0, y0=0.5, x1=len(plot_df), y1=0.5,
        line=dict(color="red", width=1, dash="dash")
    )
    
    # Update layout
    fig.update_layout(
        title='Belief-Plausibility Analysis of Anomalies',
        xaxis_title='Student Index',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1.05]),
        showlegend=True,
        hovermode='closest'
    )
    
    return fig