import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

def clean_word(text):
    """Clean text while preserving more meaningful content"""
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'\<.*?\>', '', text)
    
    # Replace punctuation with space (except for meaningful ones like $ % etc.)
    text = re.sub(r'[^\w\s\$\%\&]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class FakeNewsDetector:
    def __init__(self, fake_path='Fake.csv', real_path='True.csv'):
        # Initialize attributes that will be used later
        self.last_article_text = ""
        self.last_prediction = None
        self.last_credibility = None
        
        # Load and prepare data
        try:
            self.fake_data = pd.read_csv(fake_path)
            self.real_data = pd.read_csv(real_path)
            
            # Print dataset sizes to check balance
            print(f"Fake news samples: {len(self.fake_data)}")
            print(f"Real news samples: {len(self.real_data)}")
            
            # Add labels
            self.fake_data['label'] = 0
            self.real_data['label'] = 1
            
            # Combine datasets
            self.data = pd.concat([self.fake_data, self.real_data], ignore_index=True)
            
            # Clean data
            self.data.drop_duplicates(inplace=True)
            try:
                self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce', format='mixed')
                self.data.dropna(subset=['date'], inplace=True)
            except Exception as e:
                print(f"Warning: Issue processing dates - {e}")
            
            # Create combined title and text field for better features
            self.data['content'] = self.data['title'].fillna('') + ' ' + self.data['text'].fillna('')
            
            # Clean text
            self.data['content'] = self.data['content'].apply(clean_word)
            
            # Ensure label is integer type
            self.data['label'] = self.data['label'].astype(int)
            
            # Create a separate dataframe for user feedback (with higher weight)
            self.user_feedback_data = pd.DataFrame({'content': [], 'label': [], 'weight': []})
            
            # Check for empty content
            empty_content = self.data['content'].isna() | (self.data['content'] == '')
            if empty_content.any():
                print(f"Warning: {empty_content.sum()} rows have empty content after cleaning")
                self.data = self.data[~empty_content].reset_index(drop=True)
            
            # Initialize vectorizer with more features and add bigrams
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english',
                use_idf=True,
                smooth_idf=True
            )
            
            # Use Random Forest with calibration for better probability estimates
            base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                class_weight='balanced'
            )
            self.model = CalibratedClassifierCV(base_model, cv=5)
            
            # Prepare training data
            self.x = self.data['content']
            self.y = self.data['label']
            
            # Train initial model
            self.train_model()
            
        except Exception as e:
            print(f"Error initializing detector: {e}")
            import traceback
            traceback.print_exc()
        
    def train_model(self):
        try:
            # Prepare base dataset
            x_data = self.data['content']
            y_data = self.data['label']
            
            # Calculate class weights
            n_samples = len(y_data)
            n_fake = sum(y_data == 0)
            n_real = sum(y_data == 1)
            
            # Base weights (inversely proportional to class frequencies)
            fake_weight = n_samples / (2.0 * n_fake) if n_fake > 0 else 1.0
            real_weight = n_samples / (2.0 * n_real) if n_real > 0 else 1.0
            
            # Apply weights to all samples
            sample_weight = np.ones(len(self.data))
            sample_weight[y_data == 0] = fake_weight
            sample_weight[y_data == 1] = real_weight
            
            # Add user feedback data if available
            if not self.user_feedback_data.empty:
                # Combine data
                x_data = pd.concat([x_data, self.user_feedback_data['content']], ignore_index=True)
                y_data = pd.concat([y_data, self.user_feedback_data['label']], ignore_index=True)
                
                # Ensure label is integer type
                y_data = y_data.astype(int)
                
                # Add user feedback weights (higher than base weights)
                user_weights = self.user_feedback_data['weight'].values
                sample_weight = np.concatenate([sample_weight, user_weights])
            
            # Split data - use simple split if stratify would cause issues (when few samples in a class)
            try:
                self.x_train, self.x_test, self.y_train, self.y_test, self.train_weights, self.test_weights = train_test_split(
                    x_data, y_data, sample_weight, random_state=45, test_size=0.3, stratify=y_data
                )
            except ValueError:
                # Fall back to non-stratified split if we get value error
                self.x_train, self.x_test, self.y_train, self.y_test, self.train_weights, self.test_weights = train_test_split(
                    x_data, y_data, sample_weight, random_state=45, test_size=0.3
                )
            
            # Vectorize text
            self.xv_train = self.vectorizer.fit_transform(self.x_train)
            self.xv_test = self.vectorizer.transform(self.x_test)
            
            # Convert to explicit numpy arrays to ensure right type
            y_train_array = np.array(self.y_train).astype(int)
            
            # Train model with sample weights
            self.model.fit(self.xv_train, y_train_array, sample_weight=self.train_weights)
            
            # Get predictions for evaluation
            y_pred = self.model.predict(self.xv_test)
            
            # Calculate accuracy and confusion matrix
            self.accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            # Print detailed evaluation
            print(f"\nModel trained with accuracy: {self.accuracy:.4f}")
            print("\nConfusion Matrix:")
            print(f" {'FAKE':^10}|{'REAL':^10}")
            print("-" * 21)
            print(f"FAKE | {conf_matrix[0][0]:^10}|{conf_matrix[0][1]:^10}")
            print(f"REAL | {conf_matrix[1][0]:^10}|{conf_matrix[1][1]:^10}")

            tpr = conf_matrix[1][1] / (conf_matrix[1][0] + conf_matrix[1][1]) if (conf_matrix[1][0] + conf_matrix[1][1]) > 0 else 0
            tnr = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1]) if (conf_matrix[0][0] + conf_matrix[0][1]) > 0 else 0
            
            print(f"\nReal News Detection Rate: {tpr:.2%}")
            print(f"Fake News Detection Rate: {tnr:.2%}\n")

            if self.last_article_text:
                self.evaluate_retraining_effect()
                
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
        
    def predict_article(self, article_text):

        self.last_article_text = article_text
        
        try:

            cleaned_text = clean_word(article_text)

            article_vector = self.vectorizer.transform([cleaned_text])

            prediction = self.model.predict(article_vector)[0]

            probabilities = self.model.predict_proba(article_vector)[0]

            self.last_prediction = prediction

            if prediction == 1:
                credibility = probabilities[1] * 100
                result = "TRUE"
            else:
                credibility = probabilities[0] * 100
                result = "FAKE"
                
            self.last_credibility = credibility
            
            print(f"\nPrediction: {result}")
            print(f"Credibility Score: {credibility:.2f}%")
            
            return prediction, credibility
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            import traceback
            traceback.print_exc()
            return None, 0
        
    def add_article_to_training(self, article_text, label):
        """Add a new article to the training data and retrain the model"""
        try:
            label = int(label)

            cleaned_text = clean_word(article_text)

            if self.last_prediction is not None and label != self.last_prediction:
                weight = 20.0
                print("\nThis feedback corrects a wrong prediction. Applying higher weight.")
            else:
                weight = 10.0
            
            new_row = pd.DataFrame({
                'content': [cleaned_text],
                'label': [label],
                'weight': [weight]
            })
            
            new_row['label'] = new_row['label'].astype(int)
            
            if self.user_feedback_data.empty:
                self.user_feedback_data = new_row
            else:
                self.user_feedback_data = pd.concat([self.user_feedback_data, new_row], ignore_index=True)
                self.user_feedback_data['label'] = self.user_feedback_data['label'].astype(int)
            
            self.train_model()
            print("Model has been retrained with the new article.")
            
        except Exception as e:
            print(f"Error adding article to training: {e}")
            import traceback
            traceback.print_exc()
        
    def evaluate_retraining_effect(self):
        """Check if retraining had an effect on the previous prediction"""
        try:
            cleaned_text = clean_word(self.last_article_text)
            
            article_vector = self.vectorizer.transform([cleaned_text])
            
            new_prediction = self.model.predict(article_vector)[0]
            
            new_probabilities = self.model.predict_proba(article_vector)[0]
            
            if new_prediction == 1:
                new_credibility = new_probabilities[1] * 100
                new_result = "TRUE"
            else:
                new_credibility = new_probabilities[0] * 100
                new_result = "FAKE"
                
            if new_prediction != self.last_prediction:
                print(f"\nRetraining effect: Prediction changed from {['FAKE', 'TRUE'][self.last_prediction]} to {new_result}")
            else:
                print(f"\nRetraining effect: Prediction remains {new_result}")
                
            credibility_change = new_credibility - self.last_credibility
            print(f"Credibility Score: {new_credibility:.2f}% ({'+' if credibility_change > 0 else ''}{credibility_change:.2f}%)")
            
            return new_prediction, new_credibility
            
        except Exception as e:
            print(f"Error evaluating retraining effect: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":

    try:
        print("\n=== Initializing Fake News Detector ===\n")
        detector = FakeNewsDetector()
        
        while True:
            print("\n==== Fake News Detector ====")
            print("1. Test an article")
            print("2. Exit")
            choice = input("Enter your choice (1-2): ")
            
            if choice == '1':
                print("\nEnter the article text (press Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if line:
                        lines.append(line)
                    else:
                        break
                article_text = '\n'.join(lines)
                
                if not article_text.strip():
                    print("No article text provided.")
                    continue
                
                prediction, credibility = detector.predict_article(article_text)
                
                feedback = input("\nWould you like to provide feedback for training? (y/n): ")
                if feedback.lower() == 'y':
                    true_label = input("Was this article real (1) or fake (0)? ")
                    if true_label in ['0', '1']:
                        detector.add_article_to_training(article_text, int(true_label))
                    else:
                        print("Invalid input. Feedback not recorded.")
                
            elif choice == '2':
                print("Exiting program.")
                break
            else:
                print("Invalid choice, please try again.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()