import pandas as pd
import numpy as np
import os

def generate_dataset(n_samples=100):
    """Generate synthetic dataset with all required features"""
    np.random.seed(42)
    
    # Create normal students (70%)
    normal_count = int(n_samples * 0.7)
    normal_students = pd.DataFrame({
        'student_id': range(1, normal_count + 1),
        'video_completion_rate': np.random.uniform(70, 100, normal_count),
        'quiz_accuracy': np.random.uniform(60, 100, normal_count),
        'avg_time_per_video': np.random.uniform(10, 30, normal_count),
        'feedback_response_time': np.random.uniform(1, 10, normal_count),  # Minutes to respond
        'feedback_implementation_rate': np.random.uniform(0.7, 1.0, normal_count),  # % of feedback implemented
        'feedback_quality': np.random.uniform(3, 5, normal_count),  # Rating out of 5
        'feedback_engagement_pattern': np.random.uniform(0.7, 1.0, normal_count),  # Consistency in engagement
        'num_course_views': np.random.randint(30, 100, normal_count),
        'location_change': np.random.randint(0, 3, normal_count),
        'feedback_ignored': np.random.uniform(0, 0.3, normal_count),
        'ip_address_changes': np.random.randint(0, 2, normal_count),
        'multi_device_logins': np.random.randint(1, 2, normal_count),
        'question_time_anomalies': np.random.uniform(0, 0.2, normal_count),
        'flagged_forum_posts': np.zeros(normal_count)
    })
    
    # Create at-risk students (30%)
    at_risk_count = n_samples - normal_count
    at_risk_students = pd.DataFrame({
        'student_id': range(normal_count + 1, n_samples + 1),
        'video_completion_rate': np.random.uniform(20, 60, at_risk_count),
        'quiz_accuracy': np.random.uniform(20, 40, at_risk_count),
        'avg_time_per_video': np.random.uniform(35, 60, at_risk_count),
        'feedback_response_time': np.random.uniform(15, 60, at_risk_count),  # Slower response
        'feedback_implementation_rate': np.random.uniform(0.1, 0.5, at_risk_count),  # Lower implementation
        'feedback_quality': np.random.uniform(1, 3, at_risk_count),  # Lower quality
        'feedback_engagement_pattern': np.random.uniform(0.1, 0.6, at_risk_count),  # Inconsistent
        'num_course_views': np.random.randint(5, 30, at_risk_count),
        'location_change': np.random.randint(4, 8, at_risk_count),
        'feedback_ignored': np.random.uniform(0.6, 1.0, at_risk_count),
        'ip_address_changes': np.random.randint(3, 8, at_risk_count),
        'multi_device_logins': np.random.randint(2, 5, at_risk_count),
        'question_time_anomalies': np.random.uniform(0.5, 1.0, at_risk_count),
        'flagged_forum_posts': np.random.binomial(1, 0.3, at_risk_count)
    })
    
    # Combine datasets
    df = pd.concat([normal_students, at_risk_students], ignore_index=True)
    
    # Shuffle rows
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def load_sample_data(dataset_name):
    """Load a sample dataset based on name"""
    if dataset_name == "OULAD sample data":
        return load_oulad_sample()
    elif dataset_name == "EdNet sample data":
        return load_ednet_sample()
    else:
        # Default to synthetic data
        return generate_dataset(n_samples=200)

def load_oulad_sample():
    """Load a sample from the Open University Learning Analytics Dataset"""
    # This would normally load from a file, but for this example we'll generate synthetic data
    # with characteristics similar to OULAD
    np.random.seed(43)
    n_samples = 200
    
    # Create student data with OULAD-like features
    data = pd.DataFrame({
        'student_id': range(1, n_samples + 1),
        'video_completion_rate': np.random.uniform(0, 100, n_samples),
        'quiz_accuracy': np.random.uniform(0, 100, n_samples),
        'avg_time_per_video': np.random.uniform(0, 60, n_samples),
        'feedback_response_time': np.random.gamma(5, 2, n_samples),
        'feedback_implementation_rate': np.random.beta(3, 2, n_samples),
        'feedback_quality': np.random.uniform(1, 5, n_samples),
        'feedback_engagement_pattern': np.random.beta(3, 2, n_samples),
        'num_course_views': np.random.poisson(30, n_samples),
        'location_change': np.random.geometric(0.5, n_samples),
        'feedback_ignored': np.random.beta(2, 5, n_samples),
        'question_time_anomalies': np.random.beta(2, 5, n_samples)
    })
    
    # Create correlations to make the data more realistic
    noise = np.random.normal(0, 10, n_samples)
    data['quiz_accuracy'] = 100 - data['avg_time_per_video'] + noise
    data['quiz_accuracy'] = data['quiz_accuracy'].clip(0, 100)
    
    noise = np.random.normal(0, 15, n_samples)
    data['video_completion_rate'] = data['quiz_accuracy'] * 0.8 + noise
    data['video_completion_rate'] = data['video_completion_rate'].clip(0, 100)
    
    return data

def load_ednet_sample():
    """Load a sample from the EdNet Dataset"""
    # This would normally load from a file, but for this example we'll generate synthetic data
    # with characteristics similar to EdNet
    np.random.seed(44)
    n_samples = 200
    
    # Create student data with EdNet-like features
    data = pd.DataFrame({
        'student_id': range(1, n_samples + 1),
        'video_completion_rate': np.random.beta(5, 2, n_samples) * 100,
        'quiz_accuracy': np.random.beta(5, 3, n_samples) * 100,
        'avg_time_per_video': np.random.gamma(5, 5, n_samples),
        'feedback_response_time': np.random.gamma(3, 3, n_samples),
        'feedback_implementation_rate': np.random.beta(4, 2, n_samples),
        'feedback_quality': np.random.beta(4, 2, n_samples) * 5,
        'feedback_engagement_pattern': np.random.beta(4, 2, n_samples),
        'num_course_views': np.random.poisson(40, n_samples),
        'location_change': np.random.poisson(2, n_samples),
        'feedback_ignored': np.random.beta(2, 5, n_samples),
        'ip_address_changes': np.random.poisson(1, n_samples),
        'multi_device_logins': np.random.poisson(1, n_samples) + 1,
        'question_time_anomalies': np.random.beta(2, 5, n_samples),
        'flagged_forum_posts': np.random.binomial(1, 0.05, n_samples)
    })
    
    return data

def validate_data(df):
    """Validate and preprocess uploaded data"""
    required_columns = [
        'student_id', 'video_completion_rate', 'quiz_accuracy', 
        'avg_time_per_video', 'feedback_response_time', 'num_course_views', 
        'location_change'
    ]
    
    # Check required columns
    if not all(col in df.columns for col in required_columns):
        return None
    
    # Add missing columns with default values if needed
    if 'feedback_implementation_rate' not in df.columns:
        df['feedback_implementation_rate'] = 0.5
    
    if 'feedback_quality' not in df.columns:
        df['feedback_quality'] = 3.0
        
    if 'feedback_engagement_pattern' not in df.columns:
        df['feedback_engagement_pattern'] = 0.5
    
    if 'feedback_ignored' not in df.columns:
        df['feedback_ignored'] = 0.0
    
    if 'ip_address_changes' not in df.columns:
        df['ip_address_changes'] = 0
    
    if 'multi_device_logins' not in df.columns:
        df['multi_device_logins'] = 1
    
    if 'question_time_anomalies' not in df.columns:
        df['question_time_anomalies'] = 0.0
    
    if 'flagged_forum_posts' not in df.columns:
        df['flagged_forum_posts'] = 0
    
    # Ensure numeric data types
    numeric_columns = [
        'video_completion_rate', 'quiz_accuracy', 'avg_time_per_video',
        'feedback_response_time', 'feedback_implementation_rate', 'feedback_quality',
        'feedback_engagement_pattern', 'num_course_views', 'location_change',
        'feedback_ignored', 'ip_address_changes', 'multi_device_logins',
        'question_time_anomalies', 'flagged_forum_posts'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN values
    df = df.fillna({
        'video_completion_rate': 0,
        'quiz_accuracy': 0,
        'avg_time_per_video': 0,
        'feedback_response_time': 10,
        'feedback_implementation_rate': 0.5,
        'feedback_quality': 3.0,
        'feedback_engagement_pattern': 0.5,
        'num_course_views': 0,
        'location_change': 0,
        'feedback_ignored': 0,
        'ip_address_changes': 0,
        'multi_device_logins': 1,
        'question_time_anomalies': 0,
        'flagged_forum_posts': 0
    })
    
    return df