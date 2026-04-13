import streamlit as st

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def predict(self, input_data):
        # Make predictions using the model
        return self.model.predict(input_data)

    def evaluate(self, test_data, test_labels):
        # Evaluate the model on test data
        predictions = self.predict(test_data)
        accuracy = (predictions == test_labels).mean()
        return accuracy

    def visualize_predictions(self, input_data, predictions):
        # Visualize predictions using Streamlit
        st.write("### Predictions")
        for i in range(len(input_data)):
            st.write(f"Input: {input_data[i]}, Prediction: {predictions[i]}")

    def run_app(self):
        # Streamlit app for the model evaluation
        st.title("Model Evaluation")
        input_data = st.text_input("Input data (comma separated):")
        if st.button("Evaluate"):
            # Convert input to a suitable format
            input_data_list = [float(x) for x in input_data.split(",")]
            prediction = self.predict(input_data_list)
            st.write(f"Prediction: {prediction}")
            st.write(f"Accuracy: {self.evaluate(test_data, test_labels)}")