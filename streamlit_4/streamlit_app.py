import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="RP Field Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .subject-input {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 RP Field Recommendation System</h1>
    <p>Get personalized field recommendations based on your academic performance</p>
</div>
""", unsafe_allow_html=True)

# Load the pre-trained model and preprocessing objects
@st.cache_resource
def load_model():
    """Load the pre-trained model and preprocessing objects"""
    try:
        # Try to load the neural network model first
        try:
            model = keras.models.load_model('field_recommendation_model.h5')
            model_type = 'neural_network'
        except:
            # Fall back to tree-based model
            model = joblib.load('field_recommendation_model.pkl')
            model_type = 'random_forest' if hasattr(model, 'n_estimators') else 'decision_tree'
        
        # Load preprocessing objects
        subject_scaler = joblib.load('subject_scaler.pkl')
        field_encoder = joblib.load('field_encoder.pkl')
        board_encoder = joblib.load('board_encoder.pkl')
        combination_encoder = joblib.load('combination_encoder.pkl')
        board_ohe = joblib.load('board_ohe.pkl')
        combination_ohe = joblib.load('combination_ohe.pkl')
        subject_columns = joblib.load('subject_columns.pkl')
        
        return {
            'model': model,
            'model_type': model_type,
            'subject_scaler': subject_scaler,
            'field_encoder': field_encoder,
            'board_encoder': board_encoder,
            'combination_encoder': combination_encoder,
            'board_ohe': board_ohe,
            'combination_ohe': combination_ohe,
            'subject_columns': subject_columns
        }
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

class FieldRecommendationSystem:
    def __init__(self, model, board_ohe, combination_ohe, subject_scaler, field_encoder, 
                 board_encoder, combination_encoder, subject_columns, model_type='neural_network'):
        self.model = model
        self.board_ohe = board_ohe
        self.combination_ohe = combination_ohe
        self.subject_scaler = subject_scaler
        self.field_encoder = field_encoder
        self.board_encoder = board_encoder
        self.combination_encoder = combination_encoder
        self.subject_columns = subject_columns
        self.model_type = model_type
    
    def predict(self, examination_board, combination, marks_dict):
        """
        Predict field based on examination board, combination, and subject scores
        
        Parameters:
        - examination_board: The examination board (REB or RTB)
        - combination: The student's combination (e.g., 'SWD', 'NIT')
        - marks_dict: Dictionary with subject names as keys and marks as values
        
        Returns:
        - Dictionary with prediction details and confidence
        """
        # Encode the examination board
        try:
            board_encoded = self.board_encoder.transform([examination_board])[0]
        except ValueError:
            # If board is unknown, use the most common one (REB)
            board_encoded = 0
        
        # One-hot encode the board
        board_ohe = self.board_ohe.transform([[board_encoded]])
        
        # Encode the combination
        try:
            combination_encoded = self.combination_encoder.transform([combination])[0]
        except ValueError:
            # If combination is unknown, use the most common one
            combination_encoded = 0
        
        # One-hot encode the combination
        combination_ohe = self.combination_ohe.transform([[combination_encoded]])
        
        # Create a feature vector from the marks
        subject_features = np.zeros(len(self.subject_columns))
        
        for subject, mark in marks_dict.items():
            if subject in self.subject_columns:
                idx = self.subject_columns.index(subject)
                subject_features[idx] = mark
        
        # Scale the subject features
        subject_features_scaled = self.subject_scaler.transform(subject_features.reshape(1, -1))
        
        # Combine all features
        combined_features = np.concatenate([board_ohe, combination_ohe, subject_features_scaled], axis=1)
        
        # Make prediction based on model type
        if self.model_type == 'neural_network':
            prediction = self.model.predict(combined_features, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]
        else:  # tree-based models
            prediction = self.model.predict_proba(combined_features)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]
        
        field_name = self.field_encoder.inverse_transform([predicted_class])[0]
        
        # Get top 3 predictions with confidence scores
        if self.model_type == 'neural_network':
            all_predictions = self.model.predict(combined_features, verbose=0)[0]
            top_3_indices = np.argsort(all_predictions)[-3:][::-1]
            top_3_fields = self.field_encoder.inverse_transform(top_3_indices)
            top_3_confidences = all_predictions[top_3_indices]
        else:
            all_predictions = self.model.predict_proba(combined_features)[0]
            top_3_indices = np.argsort(all_predictions)[-3:][::-1]
            top_3_fields = self.field_encoder.inverse_transform(top_3_indices)
            top_3_confidences = all_predictions[top_3_indices]
        
        recommendations = list(zip(top_3_fields, top_3_confidences))
        
        return {
            'predicted_field': field_name,
            'confidence': float(confidence),
            'method': self.model_type,
            'examination_board': examination_board,
            'combination': combination,
            'top_subjects': self.get_top_subjects(subject_features),
            'recommendations': recommendations
        }
    
    def get_top_subjects(self, features):
        """Get the subjects with the highest marks"""
        subject_scores = list(zip(self.subject_columns, features))
        subject_scores.sort(key=lambda x: x[1], reverse=True)
        return [subject for subject, score in subject_scores if score > 0][:5]

def load_data():
    """Load and prepare the dataset"""
    try:
        df = pd.read_json("../dataset/rp_merged_dataset_cleaned_marks_to_80_where_was_1.json")
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure the JSON file is in the correct location.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def prepare_data(df):
    """Prepare and clean the dataset for analysis"""
    if df is None:
        return None, None, None, None, None, None, None
    
    # Data cleaning
    subject_columns = []
    for idx, row in df.iterrows():
        if isinstance(row['marks'], dict):
            subject_columns.extend(list(row['marks'].keys()))
        elif isinstance(row['marks'], str):
            try:
                marks_dict = json.loads(row['marks'])
                subject_columns.extend(list(marks_dict.keys()))
            except:
                continue

    subject_columns = list(set(subject_columns))
    
    # Create cleaned DataFrame
    df_clean = df[['examinationBoard', 'combination', 'department', 'field', 'yearStudy', 'average_score']].copy()
    
    # Extract marks into separate columns
    for subject in subject_columns:
        df_clean[subject] = np.nan
    
    # Fill in the marks data
    for idx, row in df.iterrows():
        marks_data = None
        if isinstance(row['marks'], dict):
            marks_data = row['marks']
        elif isinstance(row['marks'], str):
            try:
                marks_data = json.loads(row['marks'])
            except:
                continue
        
        if marks_data:
            for subject, score in marks_data.items():
                if subject in df_clean.columns and pd.notna(score):
                    df_clean.at[idx, subject] = float(score)
    
    # Handle 'Synthetic Course'
    df_clean['field'] = df_clean['field'].apply(
        lambda x: 'Not Recommended' if x == 'Synthetic Course' else x
    )
    
    # Fill missing values
    subject_cols = [col for col in df_clean.columns if col not in 
                   ['examinationBoard', 'combination', 'department', 'field', 'yearStudy', 'average_score']]
    
    for col in subject_cols:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    df_clean = df_clean.dropna(subset=['field'])
    
    # Remove fields with very few samples
    field_counts = df_clean['field'].value_counts()
    valid_fields = field_counts[field_counts >= 5].index
    df_clean = df_clean[df_clean['field'].isin(valid_fields)]
    
    return df_clean, subject_cols, subject_columns, field_counts, df_clean['examinationBoard'].unique(), df_clean['combination'].unique(), df_clean['department'].unique()

def analyze_combination_subjects(df):
    """Analyze the dataset to create a mapping of combinations to their actual subjects"""
    combination_subject_mapping = {}
    
    if df is None:
        return combination_subject_mapping
    
    # Get all subject columns
    subject_columns = []
    for idx, row in df.iterrows():
        if isinstance(row['marks'], dict):
            subject_columns.extend(list(row['marks'].keys()))
        elif isinstance(row['marks'], str):
            try:
                marks_dict = json.loads(row['marks'])
                subject_columns.extend(list(marks_dict.keys()))
            except:
                continue
    
    subject_columns = list(set(subject_columns))
    
    # For each unique combination, find subjects that are consistently taken
    unique_combinations = df[['examinationBoard', 'combination']].drop_duplicates()
    
    for _, row in unique_combinations.iterrows():
        board = row['examinationBoard']
        combination = row['combination']
        key = f"{board}_{combination}"
        
        # Filter students with this combination
        combo_students = df[(df['examinationBoard'] == board) & (df['combination'] == combination)]
        
        if len(combo_students) == 0:
            continue
            
        # Analyze which subjects are most commonly taken by students in this combination
        subject_scores = {}
        
        for _, student in combo_students.iterrows():
            marks_data = None
            if isinstance(student['marks'], dict):
                marks_data = student['marks']
            elif isinstance(student['marks'], str):
                try:
                    marks_data = json.loads(student['marks'])
                except:
                    continue
            
            if marks_data:
                for subject, score in marks_data.items():
                    if pd.notna(score) and score > 0:  # Valid score
                        if subject not in subject_scores:
                            subject_scores[subject] = []
                        subject_scores[subject].append(float(score))
        
        # Filter subjects that are taken by at least 30% of students in this combination
        min_students = max(1, len(combo_students) * 0.3)
        relevant_subjects = []
        
        for subject, scores in subject_scores.items():
            if len(scores) >= min_students:
                relevant_subjects.append(subject)
        
        combination_subject_mapping[key] = sorted(relevant_subjects)
    
    return combination_subject_mapping

def get_combination_subject_mapping(df):
    """Get the combination-subject mapping"""
    return analyze_combination_subjects(df)

def get_subjects_for_combination(df, board, combination, subject_mapping):
    """Get subjects that are actually taken in the selected combination"""
    key = f"{board}_{combination}"
    
    if key in subject_mapping:
        return subject_mapping[key]
    else:
        return []

def get_combinations_for_board(df, board):
    """Get available combinations for a specific examination board"""
    if df is None:
        return []
    return sorted(df[df['examinationBoard'] == board]['combination'].unique().tolist())

# Main application
def main():
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df = None
        st.session_state.processed_data = None
        st.session_state.model_loaded = False
        st.session_state.recommendation_system = None

    # Load model only once
    if not st.session_state.model_loaded:
        with st.spinner("Loading pre-trained model..."):
            model_data = load_model()
            if model_data is not None:
                st.session_state.recommendation_system = FieldRecommendationSystem(
                    model=model_data['model'],
                    board_ohe=model_data['board_ohe'],
                    combination_ohe=model_data['combination_ohe'],
                    subject_scaler=model_data['subject_scaler'],
                    field_encoder=model_data['field_encoder'],
                    board_encoder=model_data['board_encoder'],
                    combination_encoder=model_data['combination_encoder'],
                    subject_columns=model_data['subject_columns'],
                    model_type=model_data['model_type']
                )
                st.session_state.model_loaded = True
            else:
                st.error("Failed to load the pre-trained model. Please ensure all model files are available.")
                st.stop()

    # Load data only once
    if not st.session_state.data_loaded:
        with st.spinner("Loading dataset..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True
        
        if st.session_state.df is None:
            st.stop()
        
        # Prepare data
        with st.spinner("Preparing data for analysis..."):
            processed = prepare_data(st.session_state.df)
            if processed[0] is not None:
                df_clean, subject_cols, subject_columns, field_counts, boards, combinations, departments = processed
                subject_mapping = get_combination_subject_mapping(st.session_state.df)
                
                st.session_state.processed_data = {
                    'df_clean': df_clean,
                    'subject_cols': subject_cols,
                    'subject_columns': subject_columns,
                    'field_counts': field_counts,
                    'boards': boards,
                    'combinations': combinations,
                    'departments': departments,
                    'subject_mapping': subject_mapping
                }
            else:
                st.error("Failed to process the data.")
                st.stop()
    
    # Get processed data from session state
    data = st.session_state.processed_data
    if data is None:
        st.error("Failed to process the data. Please check your dataset.")
        st.stop()
    
    # Sidebar for model info
    with st.sidebar:
        st.header("📊 Model Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(st.session_state.df))
            st.metric("Available Fields", len(data['field_counts']))
        with col2:
            st.metric("Exam Boards", len(data['boards']))
            st.metric("Subjects", len(data['subject_columns']))
        
        # Show field distribution using Streamlit native chart
        st.subheader("Field Distribution")
        top_10_fields = data['field_counts'].head(10)
        st.bar_chart(top_10_fields)
        
        # Show combination-subject mapping in sidebar
        st.subheader("📋 Curriculum Structure")
        sample_mappings = list(data['subject_mapping'].items())[:3]  # Show first 3 as examples
        for key, subjects in sample_mappings:
            board, combination = key.split('_', 1)
            with st.expander(f"{board} - {combination}"):
                if subjects:
                    st.write("**Subjects:**")
                    for subject in subjects[:8]:  # Show first 8 subjects
                        st.write(f"• {subject}")
                    if len(subjects) > 8:
                        st.write(f"... and {len(subjects) - 8} more")
                else:
                    st.write("No subjects mapped")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Get Your Field Recommendation")
        
        # Step 1: Select examination board
        st.subheader("Step 1: Select Your Examination Board")
        selected_board = st.selectbox(
            "Choose your examination board:",
            data['boards'],
            key="board_select"
        )
        
        if selected_board:
            # Step 2: Select combination
            st.subheader("Step 2: Select Your Combination")
            available_combinations = get_combinations_for_board(data['df_clean'], selected_board)
            
            if available_combinations:
                selected_combination = st.selectbox(
                    "Choose your combination:",
                    available_combinations,
                    key="combination_select"
                )
                
                if selected_combination:
                    # Step 3: Enter subject marks
                    st.subheader("Step 3: Enter Your Subject Marks")
                    
                    available_subjects = get_subjects_for_combination(
                        data['df_clean'], selected_board, selected_combination, data['subject_mapping']
                    )
                    
                    if available_subjects:
                        # Show statistics about the combination
                        combination_stats = data['df_clean'][
                            (data['df_clean']['examinationBoard'] == selected_board) & 
                            (data['df_clean']['combination'] == selected_combination)
                        ]
                        
                        st.success(f"📚 {selected_combination} combination includes {len(available_subjects)} subjects")
                        st.info(f"👥 Based on curriculum analysis of {len(combination_stats)} students")
                        
                        # Show the subjects that will be displayed
                        with st.expander("📖 View all subjects in this combination"):
                            cols_preview = st.columns(3)
                            for i, subject in enumerate(available_subjects):
                                with cols_preview[i % 3]:
                                    st.write(f"• {subject}")
                        
                        subject_scores = {}
                        
                        # Show subjects in a more organized way
                        st.write("**Enter your marks for each subject:**")
                        
                        # Create columns for better layout - adjust based on number of subjects
                        num_cols = min(3, len(available_subjects)) if len(available_subjects) > 6 else 2
                        cols = st.columns(num_cols)
                        
                        for i, subject in enumerate(available_subjects):
                            with cols[i % num_cols]:
                                # Get average score for this subject in this combination for context
                                subject_key = f"{selected_board}_{selected_combination}"
                                combo_data = st.session_state.df[
                                    (st.session_state.df['examinationBoard'] == selected_board) & 
                                    (st.session_state.df['combination'] == selected_combination)
                                ]
                                
                                # Calculate average from actual marks data
                                avg_scores = []
                                for _, row in combo_data.iterrows():
                                    marks_data = None
                                    if isinstance(row['marks'], dict):
                                        marks_data = row['marks']
                                    elif isinstance(row['marks'], str):
                                        try:
                                            marks_data = json.loads(row['marks'])
                                        except:
                                            continue
                                    
                                    if marks_data and subject in marks_data:
                                        score = marks_data[subject]
                                        if pd.notna(score) and score > 0:
                                            avg_scores.append(float(score))
                                
                                if avg_scores:
                                    avg_score_for_subject = np.mean(avg_scores)
                                    help_text = f"Average score: {avg_score_for_subject:.1f} | Students who took this: {len(avg_scores)}"
                                else:
                                    help_text = "Enter your score for this subject"
                                
                                score = st.number_input(
                                    f"**{subject}**",
                                    min_value=0,
                                    max_value=100,
                                    value=75,
                                    step=1,
                                    key=f"subject_{subject}",
                                    help=help_text
                                )
                                subject_scores[subject] = score
                        
                        # Predict button
                        if st.button("🎯 Get Field Recommendations", type="primary"):
                            with st.spinner("Analyzing your academic profile..."):
                                result = st.session_state.recommendation_system.predict(
                                    selected_board, selected_combination, subject_scores
                                )
                                
                                st.session_state['recommendations'] = result['recommendations']
                                st.session_state['avg_score'] = np.mean(list(subject_scores.values()))
                    else:
                        st.warning(f"⚠️ No subjects found for the {selected_combination} combination under {selected_board}.")
                        st.info("This might indicate:")
                        st.write("- Limited data for this specific combination")
                        st.write("- This combination might not be commonly offered")
                        st.write("- Please try selecting a different combination")
                        
                        # Show what combinations are available for reference
                        st.write("**Available combinations for this board:**")
                        for combo in available_combinations:
                            combo_count = len(data['df_clean'][
                                (data['df_clean']['examinationBoard'] == selected_board) & 
                                (data['df_clean']['combination'] == combo)
                            ])
                            st.write(f"• {combo} ({combo_count} students)")
            else:
                st.warning(f"No combinations available for {selected_board} examination board.")
    
    with col2:
        st.header("📈 Your Results")
        
        if 'recommendations' in st.session_state:
            recommendations = st.session_state['recommendations']
            avg_score = st.session_state.get('avg_score', 0)
            
            # Display average score
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Your Average Score</h4>
                <h2>{avg_score:.1f}/100</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Display recommendations
            st.subheader("🎯 Top Field Recommendations")
            
            for i, (field, confidence) in enumerate(recommendations, 1):
                confidence_pct = confidence * 100
                
                # Color coding based on confidence
                if confidence_pct >= 70:
                    color = "#28a745"  # Green
                elif confidence_pct >= 50:
                    color = "#ffc107"  # Yellow
                else:
                    color = "#dc3545"  # Red
                
                st.markdown(f"""
                <div style="background-color: {color}; color: white; padding: 1rem; 
                            border-radius: 8px; margin-bottom: 0.5rem;">
                    <h4 style="margin: 0;">#{i} {field}</h4>
                    <p style="margin: 0; font-size: 1.2em;">
                        Confidence: {confidence_pct:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization of recommendations using Streamlit native chart
            if len(recommendations) > 1:
                fields = [rec[0][:20] + "..." if len(rec[0]) > 20 else rec[0] for rec in recommendations]
                confidences = [rec[1] * 100 for rec in recommendations]
                
                chart_data = pd.DataFrame({
                    'Field': fields,
                    'Confidence': confidences
                })
                st.bar_chart(chart_data.set_index('Field'))
        
        else:
            st.info("👆 Complete the form above to get your personalized field recommendations!")
    
    # Additional information
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("ℹ️ How it Works")
        st.write("""
        1. Select your examination board
        2. Choose your academic combination
        3. Enter your subject marks
        4. Get AI-powered field recommendations
        """)
    
    with col2:
        st.subheader("🎯 Accuracy")
        st.write("""
        Our recommendation system uses machine learning 
        trained on thousands of student records to provide 
        accurate field predictions based on academic performance.
        """)
    
    with col3:
        st.subheader("📞 Support")
        st.write("""
        Need help? Contact the academic advisory team 
        for personalized guidance and additional 
        information about our programs.
        """)

if __name__ == "__main__":
    main()