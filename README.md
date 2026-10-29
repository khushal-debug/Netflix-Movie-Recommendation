# Netflix-Movie-Recommendation

A content-based recommendation system that suggests similar movies and TV shows using the Netflix titles dataset.

Technologies Used

1. Python

2. Pandas

3. Scikit-learn

4. TF-IDF Vectorization

5. Cosine Similarity

6. Streamlit

#How It Works

-Loads the Netflix titles dataset.

-Combines available metadata into text features.

-Converts text into numerical vectors using TF-IDF.

-Calculates similarity between titles using cosine similarity.

-Displays recommended titles through a Streamlit interface.

-Run Locally

Install the dependencies:

pip install -r requirements.txt

Run the application:

streamlit run Movie_Recomandation_system.py

#Dataset

The application uses the Netflix titles dataset. The CSV file must be available at the path expected by the Python script
