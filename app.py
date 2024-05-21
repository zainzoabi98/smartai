import datetime

import openai
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# הגדר את מפתח ה-API שלך
openai.api_key = 'sk-proj-fTVGTJUI8prbxXSTMKCzT3BlbkFJZwnPvHrQTQrXSnIU0IHN'

def edit_post_description(description):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert content editor."},
                {"role": "user", "content": f"Please edit the following post description to make it more engaging:\n\n{description}"}
            ],
            max_tokens=150
        )
        edited_description = response.choices[0].message['content'].strip()
        print(f"Original description: {description}")
        print(f"Edited description: {edited_description}")
        return edited_description
    except Exception as e:
        print(f"Error editing description: {e}")
        return description

def analyze_and_recommend(data):
    try:
        print("Initial data:")
        print(data)

        data['timestamp'] = pd.to_datetime(data['timestamp'])
        twelve_hours_ago = datetime.datetime.now() - datetime.timedelta(hours=12)
        data = data[data['timestamp'] >= twelve_hours_ago]

        print("Filtered data (last 12 hours):")
        print(data)

        if not data.empty:
            # שמירת העמודה 'description' לפני הטרנספורמציה
            descriptions = data['description'].copy()

            data = pd.get_dummies(data, columns=['contentType'], drop_first=True)
            print("Data after get_dummies:")
            print(data)

            # Remove non-numeric columns מלבד 'description'
            data = data.drop(columns=['postID', 'timestamp', 'fileUploaded'])

            data = data.dropna(subset=['impressions'])

            X = data.drop(columns=['impressions'])
            y = data['impressions']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            predicted_impressions = model.predict(X)
            data['predicted_impressions'] = predicted_impressions

            print("Data with predictions:")
            print(data)

            recommendations = []
            for index, row in data.iterrows():
                recommendation = {'id': row['id'], 'recommendation': '', 'additional_recommendation': '', 'edited_description': None}
                if isinstance(descriptions.loc[index], str):  # בדוק אם התיאור הוא מחרוזת
                    recommendation['edited_description'] = descriptions.loc[index]  # החזר את התיאור המקורי
                    recommendation['recommendation'] = 'Could not process description due to non-numeric value.'
                else:
                    if row['predicted_impressions'] < row['impressions']:
                        recommendation['recommendation'] = 'Increase advertisement to additional target groups'
                        recommendation['edited_description'] = edit_post_description(descriptions.loc[index])
                    else:
                        recommendation['recommendation'] = 'Improve the quality of the advertisement'

                    if row['clicks'] > 1000:
                        recommendation['additional_recommendation'] = 'Consider using new advertising formats such as sidebar ads and product displays.'
                    elif row['views'] < 500:
                        recommendation['additional_recommendation'] = 'Use more images and videos that are exciting and engaging.'
                    else:
                        recommendation['additional_recommendation'] = 'Update the posts and increase their utilization throughout the day and week.'

                recommendations.append(recommendation)

            print("Recommendations:")
            for rec in recommendations:
                print(rec)

            return recommendations
        else:
            print("No data after filtering.")
            return []
    except Exception as e:
        print(f"Error in analyze_and_recommend: {e}")
        return []

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        json_data = request.get_json(force=True)  # Force parsing as JSON
        if 'posts' not in json_data:
            return jsonify({"error": "Invalid input, 'posts' key not found"}), 400

        data = pd.DataFrame(json_data['posts'])
        print("Received data:")
        print(data)

        recommendations = analyze_and_recommend(data)
        return jsonify({'recommendations': recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
