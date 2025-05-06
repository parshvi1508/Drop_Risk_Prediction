import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model_performance(model, X_test, y_test):
    """
    Evaluate model performance using standard metrics
    
    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: True labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Calculate standard metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Calculate ROC AUC if probabilities are available
    if hasattr(model, 'predict_proba'):
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        
        # Calculate precision-recall curve and average precision
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        metrics['average_precision'] = average_precision_score(y_test, y_prob)
        metrics['pr_curve'] = {'precision': precision, 'recall': recall}
    
    return metrics

def perform_cross_validation(model, X, y, cv=5, scoring='accuracy'):
    """
    Perform cross-validation on a model
    
    Args:
        model: Model to evaluate
        X: Features
        y: Target labels
        cv: Number of cross-validation folds
        scoring: Scoring metric to use
        
    Returns:
        Dictionary with cross-validation scores
    """
    # Set up stratified k-fold cross-validation
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring)
    
    # Calculate metrics
    cv_results = {
        'cv_scores': cv_scores,
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std(),
        'min_score': cv_scores.min(),
        'max_score': cv_scores.max()
    }
    
    return cv_results

def compare_model_performance(models, X, y, test_size=0.3, random_state=42):
    """
    Compare multiple model performances using the same dataset
    
    Args:
        models: Dictionary of model names and model objects
        X: Features
        y: Target labels
        test_size: Test split proportion
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with model performance metrics
    """
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Results dataframe
    results = pd.DataFrame(columns=[
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC', 'CV Mean'
    ])
    
    # Evaluate each model
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if hasattr(model, 'predict_proba') else 0.5
        
        # Cross validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = cv_scores.mean()
        
        # Store results
        results = pd.concat([
            results, 
            pd.DataFrame({
                'Model': [name],
                'Accuracy': [accuracy],
                'Precision': [precision],
                'Recall': [recall],
                'F1': [f1],
                'ROC AUC': [roc_auc],
                'CV Mean': [cv_mean]
            })
        ], ignore_index=True)
        
        # Print detailed classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
    
    # Find best model
    best_model_idx = results['F1'].idxmax()
    best_model_name = results.loc[best_model_idx, 'Model']
    best_model_f1 = results.loc[best_model_idx, 'F1']
    
    print(f"\nBest model by F1 score: {best_model_name} with F1 = {best_model_f1:.4f}")
    
    return results

def evaluate_anomaly_detection(ground_truth, predictions, scores=None):
    """
    Evaluate anomaly detection performance
    
    Args:
        ground_truth: True anomaly labels (1 for anomaly, 0 for normal)
        predictions: Predicted anomaly labels
        scores: Anomaly scores (if available)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Calculate standard metrics
    metrics = {
        'accuracy': accuracy_score(ground_truth, predictions),
        'precision': precision_score(ground_truth, predictions),
        'recall': recall_score(ground_truth, predictions),
        'f1': f1_score(ground_truth, predictions),
        'confusion_matrix': confusion_matrix(ground_truth, predictions)
    }
    
    # Calculate ROC AUC if scores are provided
    if scores is not None:
        metrics['roc_auc'] = roc_auc_score(ground_truth, scores)
    
    # Print metrics
    print(f"Anomaly detection metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    return metrics

def evaluate_evidence_model(df):
    """
    Evaluate the evidence-based model results
    
    Args:
        df: DataFrame with evidence-based model results
        
    Returns:
        Dictionary of evidence-based evaluation metrics
    """
    metrics = {}
    
    # Calculate average belief and uncertainty
    if 'combined_belief_anomaly' in df.columns:
        metrics['avg_anomaly_belief'] = df['combined_belief_anomaly'].mean()
    
    if 'belief_uncertainty' in df.columns:
        metrics['avg_uncertainty'] = df['belief_uncertainty'].mean()
        metrics['max_uncertainty'] = df['belief_uncertainty'].max()
    
    # Calculate conflict metrics
    if 'K' in df.columns:  # Conflict from Dempster's rule
        metrics['avg_conflict'] = df['K'].mean()
        metrics['max_conflict'] = df['K'].max()
    
    # Calculate agreement between different detection methods
    if 'is_anomaly' in df.columns and 'rule_based_anomaly' in df.columns:
        agreement = (df['is_anomaly'] == df['rule_based_anomaly']).mean()
        metrics['method_agreement'] = agreement
        
        # Calculate metrics for each method if ground truth is available
        if 'true_anomaly' in df.columns:
            stat_precision = precision_score(df['true_anomaly'], df['is_anomaly'])
            stat_recall = recall_score(df['true_anomaly'], df['is_anomaly'])
            rule_precision = precision_score(df['true_anomaly'], df['rule_based_anomaly'])
            rule_recall = recall_score(df['true_anomaly'], df['rule_based_anomaly'])
            combined_precision = precision_score(df['true_anomaly'], df['combined_anomaly'])
            combined_recall = recall_score(df['true_anomaly'], df['combined_anomaly'])
            
            metrics['statistical_precision'] = stat_precision
            metrics['statistical_recall'] = stat_recall
            metrics['rule_precision'] = rule_precision
            metrics['rule_recall'] = rule_recall
            metrics['combined_precision'] = combined_precision
            metrics['combined_recall'] = combined_recall
            
            # Calculate relative improvement
            if stat_precision > 0:
                metrics['precision_improvement'] = (combined_precision - stat_precision) / stat_precision
            if stat_recall > 0:
                metrics['recall_improvement'] = (combined_recall - stat_recall) / stat_recall
    
    return metrics

def visualize_confusion_matrix(conf_matrix, output_path=None):
    """
    Visualize confusion matrix
    
    Args:
        conf_matrix: Confusion matrix to visualize
        output_path: Path to save the visualization (optional)
    
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Normal', 'Anomaly'],
        yticklabels=['Normal', 'Anomaly']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def visualize_roc_curve(y_true, y_score, output_path=None):
    """
    Visualize ROC curve
    
    Args:
        y_true: True labels
        y_score: Predicted scores
        output_path: Path to save the visualization (optional)
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def generate_evaluation_report(metrics, output_path=None):
    """
    Generate a comprehensive evaluation report
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_path: Path to save the report (optional)
    
    Returns:
        Report as a string
    """
    report = "# Model Evaluation Report\n\n"
    
    # Add classification metrics
    report += "## Classification Metrics\n\n"
    report += f"- Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\n"
    report += f"- Precision: {metrics.get('precision', 'N/A'):.4f}\n"
    report += f"- Recall: {metrics.get('recall', 'N/A'):.4f}\n"
    report += f"- F1 Score: {metrics.get('f1', 'N/A'):.4f}\n"
    if 'roc_auc' in metrics:
        report += f"- ROC AUC: {metrics['roc_auc']:.4f}\n"
    if 'average_precision' in metrics:
        report += f"- Average Precision: {metrics['average_precision']:.4f}\n"
    
    # Add cross-validation results if available
    if 'cv_scores' in metrics:
        report += "\n## Cross-Validation Results\n\n"
        report += f"- Mean CV Score: {metrics['mean_score']:.4f}\n"
        report += f"- Standard Deviation: {metrics['std_score']:.4f}\n"
        report += f"- Min Score: {metrics['min_score']:.4f}\n"
        report += f"- Max Score: {metrics['max_score']:.4f}\n"
    
    # Add evidence-based metrics if available
    if 'avg_anomaly_belief' in metrics:
        report += "\n## Evidence-Based Metrics\n\n"
        report += f"- Average Anomaly Belief: {metrics['avg_anomaly_belief']:.4f}\n"
        if 'avg_uncertainty' in metrics:
            report += f"- Average Uncertainty: {metrics['avg_uncertainty']:.4f}\n"
        if 'method_agreement' in metrics:
            report += f"- Method Agreement: {metrics['method_agreement']:.4f}\n"
        if 'precision_improvement' in metrics:
            report += f"- Precision Improvement: {metrics['precision_improvement']:.2%}\n"
        if 'recall_improvement' in metrics:
            report += f"- Recall Improvement: {metrics['recall_improvement']:.2%}\n"
    
    # Write report to file if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report

def evaluate_ds_model(df):
    """Evaluate the effectiveness of the Dempster-Shafer evidence combination"""
    if 'true_anomaly' not in df.columns:
        # For datasets without ground truth, we can't fully evaluate
        return {
            'avg_belief': df['combined_belief_anomaly'].mean(),
            'avg_uncertainty': df['belief_uncertainty'].mean(),
            'avg_conflict': df['K'].mean() if 'K' in df.columns else 'N/A',
            'high_confidence_pct': (df['belief_uncertainty'] < 0.3).mean()
        }
    
    # Calculate metrics for each evidence source individually
    evidence_sources = ['statistical', 'rule_based', 'quiz', 'video', 'feedback']
    source_metrics = {}
    
    for source in evidence_sources:
        if f'{source}_bel_anomaly' in df.columns:
            # Use 0.5 as classification threshold for individual sources
            source_preds = (df[f'{source}_bel_anomaly'] > 0.5).astype(int)
            source_metrics[source] = {
                'accuracy': (source_preds == df['true_anomaly']).mean(),
                'false_pos': ((source_preds == 1) & (df['true_anomaly'] == 0)).sum() / len(df),
                'false_neg': ((source_preds == 0) & (df['true_anomaly'] == 1)).sum() / len(df)
            }
    
    # Calculate metrics for combined model
    combined_metrics = {
        'accuracy': (df['combined_anomaly'] == df['true_anomaly']).mean(),
        'false_pos': ((df['combined_anomaly'] == 1) & (df['true_anomaly'] == 0)).sum() / len(df),
        'false_neg': ((df['combined_anomaly'] == 0) & (df['true_anomaly'] == 1)).sum() / len(df)
    }
    
    # Calculate improvement over best individual source
    best_source_acc = max([m['accuracy'] for m in source_metrics.values()]) if source_metrics else 0
    improvement = (combined_metrics['accuracy'] - best_source_acc) / best_source_acc if best_source_acc > 0 else 0
    
    results = {
        'source_metrics': source_metrics,
        'combined_metrics': combined_metrics,
        'improvement': improvement,
        'avg_belief': df['combined_belief_anomaly'].mean(),
        'avg_uncertainty': df['belief_uncertainty'].mean(),
        'avg_conflict': df['K'].mean() if 'K' in df.columns else 'N/A',
        'high_confidence_pct': (df['belief_uncertainty'] < 0.3).mean()
    }
    
    return results
