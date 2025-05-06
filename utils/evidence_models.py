import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_evidence_model(df):
    """Train a model on the student data with evidence theory integration"""
    features = ['video_completion_rate', 'quiz_accuracy', 'avg_time_per_video',
              'forum_activity', 'num_course_views', 'location_change']
    
    # Add advanced features if available
    advanced_features = ['feedback_ignored', 'ip_address_changes', 
                      'multi_device_logins', 'question_time_anomalies']
    for feature in advanced_features:
        if feature in df.columns:
            features.append(feature)
    
    # Add new feedback metrics if available
    feedback_metrics = ['feedback_response_time', 'feedback_implementation_rate', 
                       'feedback_quality', 'feedback_engagement_pattern']
    for feature in feedback_metrics:
        if feature in df.columns:
            features.append(feature)
    
    # Create X (features) - ensure all features exist
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    
    # Create target y - using low quiz scores as an indicator of at-risk students
    y = (df['quiz_accuracy'] < 50).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate evidence weights
    evidence_weights = {
        'statistical': 0.8,  # Isolation Forest weight
        'rule_based': 0.7,   # Rule-based detection weight
        'quiz_performance': 0.9,  # Quiz performance weight
        'video_engagement': 0.8,  # Video engagement weight
    }
    
    # Determine which engagement metric to use based on available columns
    if any(col in df.columns for col in ['feedback_response_time', 'feedback_implementation_rate']):
        evidence_weights['feedback_metrics'] = 0.85  # New feedback metrics weight
    elif 'feedback_ignored' in df.columns:
        evidence_weights['feedback_response'] = 0.75  # Legacy feedback metric weight
    else:
        evidence_weights['forum_participation'] = 0.6  # Forum participation weight
    
    # Add weights for advanced features if available
    if 'ip_address_changes' in df.columns or 'multi_device_logins' in df.columns:
        evidence_weights['network_behavior'] = 0.65
    if 'question_time_anomalies' in df.columns:
        evidence_weights['question_timing'] = 0.7
    
    # Calculate metrics
    metrics = {
        'accuracy': model.score(X_test, y_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'feature_importance': dict(zip(available_features, model.feature_importances_)),
        'evidence_weights': evidence_weights,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    # Save model for later use
    try:
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/evidence_model.pkl')
    except Exception as e:
        print(f"Could not save model: {str(e)}")
    
    # Return the model and metrics
    return model, metrics

def detect_anomalies(df):
    """Detect anomalies using evidence theory approach"""
    # Make a copy of the dataframe to avoid warnings
    df = df.copy()
    
    # Define rule-based anomaly conditions
    rule_conditions = [
        ((df['avg_time_per_video'] > 40) & (df['quiz_accuracy'] < 50)),  # High video time + low scores
        ((df['video_completion_rate'] > 80) & (df['quiz_accuracy'] < 40)),  # Video binging with low retention
        (df['location_change'] > 5)  # Excessive location hopping
    ]
    
    # Add forum activity condition if no feedback metrics are available
    if all(col not in df.columns for col in ['feedback_response_time', 'feedback_implementation_rate']):
        rule_conditions.append(df['forum_activity'] < 2)  # Low forum activity
    
    # Add feedback-based conditions if available
    if 'feedback_response_time' in df.columns:
        rule_conditions.append(df['feedback_response_time'] > 20)  # Slow feedback response
    
    if 'feedback_implementation_rate' in df.columns:
        rule_conditions.append(df['feedback_implementation_rate'] < 0.5)  # Low implementation rate
    
    if 'feedback_quality' in df.columns:
        rule_conditions.append(df['feedback_quality'] < 3)  # Poor feedback quality
    
    # Add legacy feedback condition if new metrics aren't available
    elif 'feedback_ignored' in df.columns:
        rule_conditions.append(df['feedback_ignored'] > 0.7)  # High feedback ignored rate
    
    # Add other advanced conditions if columns exist
    if 'ip_address_changes' in df.columns:
        rule_conditions.append(df['ip_address_changes'] > 3)  # Multiple IP changes
    
    if 'multi_device_logins' in df.columns:
        rule_conditions.append(df['multi_device_logins'] > 2)  # Logins from multiple devices
    
    if 'question_time_anomalies' in df.columns:
        rule_conditions.append(df['question_time_anomalies'] > 0.5)  # Question time anomalies
    
    if 'flagged_forum_posts' in df.columns:
        rule_conditions.append(df['flagged_forum_posts'] > 0)  # Flagged forum posts
    
    # Combine conditions with OR
    rule_based_anomalies = df[np.logical_or.reduce(rule_conditions)]
    
    # Features for Isolation Forest
    features = ['video_completion_rate', 'quiz_accuracy', 'avg_time_per_video',
               'num_course_views', 'location_change']
    
    # Add forum activity if no feedback metrics are available
    if all(col not in df.columns for col in ['feedback_response_time', 'feedback_implementation_rate']):
        features.append('forum_activity')
    
    # Add feedback metrics if available
    feedback_metrics = ['feedback_response_time', 'feedback_implementation_rate', 
                       'feedback_quality', 'feedback_engagement_pattern', 'feedback_ignored']
    for feature in feedback_metrics:
        if feature in df.columns:
            features.append(feature)
    
    # Add other advanced features if available
    advanced_features = ['ip_address_changes', 'multi_device_logins', 'question_time_anomalies']
    for feature in advanced_features:
        if feature in df.columns:
            features.append(feature)
    
    # Ensure all features exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[available_features])
    
    # Apply Isolation Forest
    model = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly_score'] = model.fit_predict(scaled_data)
    
    # Mark anomalies (Isolation Forest returns -1 for anomalies)
    df['is_anomaly'] = (df['anomaly_score'] == -1).astype(int)
    
    # Add rule-based anomaly flag
    df['rule_based_anomaly'] = df['student_id'].isin(rule_based_anomalies['student_id']).astype(int)
    
    # Evidence theory integration
    df = evidence_theory_combination(df)
    
    # Calculate dropout risk with enhanced features
    df['dropout_risk'] = calculate_dropout_risk(df)
    
    # Create anomaly dataframe
    anomaly_df = df[df['combined_anomaly'] == 1].copy()
    
    return df, anomaly_df

def evidence_theory_combination(df):
    """
    Implement Dempster-Shafer evidence theory for combining multiple anomaly detection methods
    This allows for uncertainty quantification and more robust anomaly detection
    """
    # Create a frame of discernment: {normal, anomaly}
    # In DS theory, we assign belief masses to each possible subset of this frame
    
    # Define evidence reliability weights (these determine how much we trust each source)
    reliability = {
        'statistical': 0.8,  # Isolation Forest reliability
        'rule_based': 0.7,   # Rule-based detection reliability
        'quiz': 0.9,         # Quiz performance reliability
        'video': 0.8,        # Video engagement reliability
        'feedback': 0.85,    # Feedback metrics reliability (previously forum)
        'network': 0.65      # Network behavior reliability (if available)
    }
    
    # Calculate basic probability assignments (BBAs) for each evidence source
    
    # Statistical model evidence (Isolation Forest)
    df['stat_bel_anomaly'] = df['is_anomaly'] * reliability['statistical']
    df['stat_bel_normal'] = (1 - df['is_anomaly']) * reliability['statistical']
    df['stat_uncertainty'] = 1 - df['stat_bel_anomaly'] - df['stat_bel_normal']
    
    # Rule-based evidence
    df['rule_bel_anomaly'] = df['rule_based_anomaly'] * reliability['rule_based']
    df['rule_bel_normal'] = (1 - df['rule_based_anomaly']) * reliability['rule_based']
    df['rule_uncertainty'] = 1 - df['rule_bel_anomaly'] - df['rule_bel_normal']
    
    # Quiz performance evidence
    df['quiz_bel_anomaly'] = (df['quiz_accuracy'] < 50).astype(int) * reliability['quiz']
    df['quiz_bel_normal'] = (df['quiz_accuracy'] >= 50).astype(int) * reliability['quiz']
    df['quiz_uncertainty'] = 1 - df['quiz_bel_anomaly'] - df['quiz_bel_normal']
    
    # Video engagement evidence
    df['video_bel_anomaly'] = (df['video_completion_rate'] < 60).astype(int) * reliability['video']
    df['video_bel_normal'] = (df['video_completion_rate'] >= 60).astype(int) * reliability['video']
    df['video_uncertainty'] = 1 - df['video_bel_anomaly'] - df['video_bel_normal']
    
    # Feedback evidence (replacing forum activity)
    # Combine multiple feedback metrics into a single evidence source
    if all(col in df.columns for col in ['feedback_response_time', 'feedback_implementation_rate']):
        # Consider multiple feedback metrics for more robust evidence
        feedback_issues = (
            (df['feedback_response_time'] > 20) |  # Slow response
            (df['feedback_implementation_rate'] < 0.5) |  # Low implementation
            (df['feedback_quality'] < 3 if 'feedback_quality' in df.columns else False) |
            (df['feedback_engagement_pattern'] < 0.6 if 'feedback_engagement_pattern' in df.columns else False)
        )
        df['feedback_bel_anomaly'] = feedback_issues.astype(int) * reliability['feedback']
    elif 'feedback_ignored' in df.columns:
        # Fall back to legacy feedback_ignored if new metrics aren't available
        df['feedback_bel_anomaly'] = (df['feedback_ignored'] > 0.6).astype(int) * reliability['feedback']
    else:
        # Fall back to forum activity if no feedback metrics are available
        df['feedback_bel_anomaly'] = (df['forum_activity'] < 3).astype(int) * 0.6  # Lower reliability for forum
        
    df['feedback_bel_normal'] = (1 - df['feedback_bel_anomaly']) * reliability['feedback']
    df['feedback_uncertainty'] = 1 - df['feedback_bel_anomaly'] - df['feedback_bel_normal']
    
    # Network behavior evidence (if available)
    if 'ip_address_changes' in df.columns or 'multi_device_logins' in df.columns:
        if 'ip_address_changes' in df.columns and 'multi_device_logins' in df.columns:
            df['network_bel_anomaly'] = ((df['ip_address_changes'] > 3) | 
                                         (df['multi_device_logins'] > 2)).astype(int) * reliability['network']
        elif 'ip_address_changes' in df.columns:
            df['network_bel_anomaly'] = (df['ip_address_changes'] > 3).astype(int) * reliability['network']
        else:
            df['network_bel_anomaly'] = (df['multi_device_logins'] > 2).astype(int) * reliability['network']
            
        df['network_bel_normal'] = (1 - df['network_bel_anomaly']) * reliability['network']
        df['network_uncertainty'] = 1 - df['network_bel_anomaly'] - df['network_bel_normal']
    
    # Improved Dempster's rule of combination with conflict tracking
    def combine_evidence(bel_a1, bel_a2, bel_n1, bel_n2, unc1, unc2):
        """
        Implements Dempster's rule of combination with proper handling of conflict
        Returns combined belief, combined disbelief, combined uncertainty, and conflict K
        """
        # Calculate conflict (K)
        K = bel_a1 * bel_n2 + bel_n1 * bel_a2
        
        # Avoid division by zero
        normalization = 1 - K
        if normalization <= 0:
            # High conflict case - use Murphy's average combination rule instead
            avg_bel_anomaly = (bel_a1 + bel_a2) / 2
            avg_bel_normal = (bel_n1 + bel_n2) / 2
            avg_uncertainty = (unc1 + unc2) / 2
            return avg_bel_anomaly, avg_bel_normal, avg_uncertainty, 1.0
            
        # Calculate combined beliefs
        combined_bel_anomaly = (bel_a1 * bel_a2 + bel_a1 * unc2 + unc1 * bel_a2) / normalization
        combined_bel_normal = (bel_n1 * bel_n2 + bel_n1 * unc2 + unc1 * bel_n2) / normalization
        combined_uncertainty = (unc1 * unc2) / normalization
        
        return combined_bel_anomaly, combined_bel_normal, combined_uncertainty, K
    
    # Track conflict for analysis
    df['K'] = 0.0
    
    # Combine statistical and rule-based evidence
    df['temp_bel_anomaly'], df['temp_bel_normal'], df['temp_uncertainty'], df['K'] = zip(
        *df.apply(
            lambda x: combine_evidence(
                x['stat_bel_anomaly'], x['rule_bel_anomaly'],
                x['stat_bel_normal'], x['rule_bel_normal'],
                x['stat_uncertainty'], x['rule_uncertainty']
            ),
            axis=1
        )
    )
    
    # Combine with quiz evidence
    df['temp_bel_anomaly2'], df['temp_bel_normal2'], df['temp_uncertainty2'], K2 = zip(
        *df.apply(
            lambda x: combine_evidence(
                x['temp_bel_anomaly'], x['quiz_bel_anomaly'],
                x['temp_bel_normal'], x['quiz_bel_normal'],
                x['temp_uncertainty'], x['quiz_uncertainty']
            ),
            axis=1
        )
    )
    df['K'] = df['K'] + pd.Series(K2)
    
    # Combine with video evidence
    df['temp_bel_anomaly3'], df['temp_bel_normal3'], df['temp_uncertainty3'], K3 = zip(
        *df.apply(
            lambda x: combine_evidence(
                x['temp_bel_anomaly2'], x['video_bel_anomaly'],
                x['temp_bel_normal2'], x['video_bel_normal'],
                x['temp_uncertainty2'], x['video_uncertainty']
            ),
            axis=1
        )
    )
    df['K'] = df['K'] + pd.Series(K3)
    
    # Combine with feedback evidence (previously forum)
    df['temp_bel_anomaly4'], df['temp_bel_normal4'], df['temp_uncertainty4'], K4 = zip(
        *df.apply(
            lambda x: combine_evidence(
                x['temp_bel_anomaly3'], x['feedback_bel_anomaly'],
                x['temp_bel_normal3'], x['feedback_bel_normal'],
                x['temp_uncertainty3'], x['feedback_uncertainty']
            ),
            axis=1
        )
    )
    df['K'] = df['K'] + pd.Series(K4)
    
    # Rename final combined values
    df['combined_belief_anomaly'] = df['temp_bel_anomaly4']
    df['combined_belief_normal'] = df['temp_bel_normal4']
    df['belief_uncertainty'] = df['temp_uncertainty4']
    
    # Add network evidence if available
    if 'network_bel_anomaly' in df.columns:
        df['combined_belief_anomaly'], df['combined_belief_normal'], df['belief_uncertainty'], K5 = zip(
            *df.apply(
                lambda x: combine_evidence(
                    x['combined_belief_anomaly'], x['network_bel_anomaly'],
                    x['combined_belief_normal'], x['network_bel_normal'],
                    x['belief_uncertainty'], x['network_uncertainty']
                ),
                axis=1
            )
        )
        df['K'] = df['K'] + pd.Series(K5)
    
    # Calculate plausibility (upper probability bound)
    df['plausibility_anomaly'] = df['combined_belief_anomaly'] + df['belief_uncertainty']
    df['plausibility_normal'] = df['combined_belief_normal'] + df['belief_uncertainty']
    
    # Mark combined anomalies based on belief threshold
    df['combined_anomaly'] = (df['combined_belief_anomaly'] > 0.5).astype(int)
    
    # Clean up temporary columns
    temp_cols = [col for col in df.columns if col.startswith('temp_')]
    df.drop(temp_cols, axis=1, inplace=True)
    
    return df

def calculate_dropout_risk(df):
    """Calculate dropout risk score using enhanced features and evidence theory"""
    # Base risk calculation
    risk = (
        (100 - df['quiz_accuracy']) * 0.45 +  # Low quiz scores increase risk
        (100 - df['video_completion_rate']) * 0.25  # Low completion increases risk
    )
    
    # Use feedback metrics instead of forum participation if available
    if all(col in df.columns for col in ['feedback_response_time', 'feedback_implementation_rate']):
        risk += (df['feedback_response_time'] > 20).astype(int) * 15  # Slow response time
        risk += (df['feedback_implementation_rate'] < 0.5).astype(int) * 20  # Low implementation
        
        if 'feedback_quality' in df.columns:
            risk += (df['feedback_quality'] < 3).astype(int) * 15  # Low quality feedback
            
        if 'feedback_engagement_pattern' in df.columns:
            risk += (df['feedback_engagement_pattern'] < 0.6).astype(int) * 10  # Poor engagement pattern
    else:
        # Fall back to forum activity if no feedback metrics are available
        risk += (df['forum_activity'] < 3).astype(int) * 15  # Low forum activity increases risk
        
        # Include legacy feedback metrics if available
        if 'feedback_ignored' in df.columns:
            risk += (df['feedback_ignored'] > 0.6).astype(int) * 20  # High feedback ignored rate
    
    # Add additional behavioral factors
    risk += (df['avg_time_per_video'] > 30).astype(int) * 10  # Long video times increase risk
    
    # Add advanced factors if available
    if 'ip_address_changes' in df.columns:
        risk += (df['ip_address_changes'] > 3).astype(int) * 15  # Multiple IP addresses
    
    if 'multi_device_logins' in df.columns:
        risk += (df['multi_device_logins'] > 2).astype(int) * 10  # Multiple device logins
    
    if 'question_time_anomalies' in df.columns:
        risk += (df['question_time_anomalies'] > 0.5).astype(int) * 15  # Question time anomalies
    
    if 'flagged_forum_posts' in df.columns:
        risk += (df['flagged_forum_posts'] > 0).astype(int) * 10  # Flagged forum posts
    
    # Scale to 0-1 range
    risk = risk / 100.0
    
    # Incorporate evidence theory results if available
    if 'combined_belief_anomaly' in df.columns and 'belief_uncertainty' in df.columns:
        # Adjust risk based on belief and uncertainty
        # Higher belief in anomaly with low uncertainty increases risk
        # Higher uncertainty reduces the impact of belief
        confidence_factor = 1 - df['belief_uncertainty']
        evidence_impact = df['combined_belief_anomaly'] * confidence_factor
        
        # Blend original risk with evidence-based assessment
        risk = risk * 0.7 + evidence_impact * 0.3
    
    # Cap at 1.0
    risk = risk.clip(0, 1.0)
    
    # Enhance risk score for detected anomalies
    if 'combined_anomaly' in df.columns:
        anomaly_idx = df[df['combined_anomaly'] == 1].index
        risk.loc[anomaly_idx] = risk.loc[anomaly_idx].apply(lambda x: min(0.95, x * 1.25))
    
    return risk

def identify_at_risk_students(df, threshold=0.7):
    """Identify students at high risk of dropping out using evidence-based approach"""
    # Calculate weighted risk score based on available metrics
    risk_factors = [
        (df['quiz_accuracy'] < 50).astype(int) * 0.35,  # Low quiz scores
        (df['video_completion_rate'] < 60).astype(int) * 0.25,  # Low video completion
        (df['avg_time_per_video'] > 35).astype(int) * 0.1,  # High time on videos
        (df['location_change'] > 4).astype(int) * 0.1  # Frequent location changes
    ]
    
    # Use feedback metrics if available, otherwise use forum activity
    if all(col in df.columns for col in ['feedback_response_time', 'feedback_implementation_rate']):
        risk_factors.append((df['feedback_response_time'] > 20).astype(int) * 0.15)  # Slow response
        risk_factors.append((df['feedback_implementation_rate'] < 0.5).astype(int) * 0.15)  # Low implementation
        
        if 'feedback_quality' in df.columns:
            risk_factors.append((df['feedback_quality'] < 3).astype(int) * 0.1)  # Low quality
    else:
        risk_factors.append((df['forum_activity'] < 3).astype(int) * 0.2)  # Low forum engagement
        
        # Include legacy feedback metrics if available
        if 'feedback_ignored' in df.columns:
            risk_factors.append((df['feedback_ignored'] > 0.6).astype(int) * 0.2)  # High feedback ignored
    
    # Add other advanced factors if available
    if 'ip_address_changes' in df.columns:
        risk_factors.append((df['ip_address_changes'] > 3).astype(int) * 0.15)  # Multiple IP addresses
    
    if 'multi_device_logins' in df.columns:
        risk_factors.append((df['multi_device_logins'] > 2).astype(int) * 0.1)  # Multiple device logins
    
    if 'question_time_anomalies' in df.columns:
        risk_factors.append((df['question_time_anomalies'] > 0.5).astype(int) * 0.15)  # Question time anomalies
    
    # Sum all risk factors to get risk score
    risk_score = pd.Series(0, index=df.index)
    for factor in risk_factors:
        risk_score += factor
    
    # Incorporate evidence theory if available
    if 'combined_belief_anomaly' in df.columns:
        # Adjust risk score based on belief in anomaly
        evidence_factor = df['combined_belief_anomaly'] * 0.2
        risk_score += evidence_factor
    
    # Identify high-risk students
    at_risk = df[risk_score >= threshold].copy()
    at_risk['risk_score'] = risk_score[risk_score >= threshold]
    
    return at_risk.sort_values('risk_score', ascending=False)