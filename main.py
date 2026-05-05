from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load data and model once at startup
data = pd.read_csv("Cleaned_data.csv")
# Drop the unnamed index column if it exists
if data.columns[0].startswith("Unnamed"):
    data = data.drop(columns=[data.columns[0]])

pipe = pickle.load(open("RidgeModel.pkl", "rb"))


def format_indian_price(lakhs: float) -> str:
    """Convert a number in lakhs to a clean Indian price string."""
    if lakhs >= 100:
        crores = lakhs / 100
        return f"\u20b9 {crores:.2f} Cr"
    return f"\u20b9 {lakhs:.2f} Lakhs"


def find_comparables(location, sqft, bhk, max_results=3):
    """Progressively loosen filters until we find at least some comparable properties."""
    # Strict: same location + same BHK + sqft within +/- 20%
    similar = data[
        (data['location'] == location) &
        (data['bhk'] == bhk) &
        (data['total_sqft'].between(sqft * 0.8, sqft * 1.2))
    ]
    if len(similar) > 0:
        return similar.head(max_results), "Same location, same BHK, similar size"

    # Loosen 1: same location + same BHK (any sqft)
    similar = data[(data['location'] == location) & (data['bhk'] == bhk)]
    if len(similar) > 0:
        return similar.head(max_results), "Same location and BHK"

    # Loosen 2: same location only
    similar = data[data['location'] == location]
    if len(similar) > 0:
        return similar.head(max_results), "Same location, different BHK"

    # Loosen 3: same BHK + similar size, any location
    similar = data[
        (data['bhk'] == bhk) &
        (data['total_sqft'].between(sqft * 0.7, sqft * 1.3))
    ]
    if len(similar) > 0:
        return similar.head(max_results), "Similar BHK and size, different locations"

    return data.iloc[0:0], "No comparables found"


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get('location')
        sqft = float(request.form.get('total_sqft'))
        bath = int(request.form.get('bath'))
        bhk = int(request.form.get('bhk'))

        # Basic input validation
        if not location:
            return "Please select a location.", 400
        if sqft <= 0 or bath <= 0 or bhk <= 0:
            return "Please enter valid positive numbers.", 400
        if sqft / bhk < 300:
            return "Square footage seems too small for the BHK count.", 400

        input_df = pd.DataFrame(
            [[location, sqft, bath, bhk]],
            columns=['location', 'total_sqft', 'bath', 'bhk']
        )
        prediction_lakhs = float(pipe.predict(input_df)[0])

        # Guard against negative predictions on edge cases
        if prediction_lakhs < 0:
            prediction_lakhs = 0

        # Find comparables with progressive fallback
        similar, match_type = find_comparables(location, sqft, bhk)

        comparables = []
        for _, row in similar.iterrows():
            comparables.append({
                'sqft': int(row['total_sqft']),
                'bhk': int(row['bhk']),
                'bath': int(row['bath']),
                'actual_price': format_indian_price(row['price']),
                'location': row['location'],
            })

        formatted_price = format_indian_price(prediction_lakhs)
        price_per_sqft = (prediction_lakhs * 100000) / sqft

        result = {
            'predicted_price': formatted_price,
            'price_per_sqft': f"\u20b9 {price_per_sqft:,.0f} per sq.ft",
            'comparable_count': len(comparables),
            'match_type': match_type,
            'comparables': comparables,
        }
        return jsonify(result)

    except ValueError:
        return "Invalid input \u2014 please check your numbers.", 400
    except Exception as e:
        return f"Prediction failed: {str(e)}", 500


@app.route('/insights')
def insights():
    """Bonus endpoint \u2014 returns market insights from the dataset."""
    insights_data = {
        'total_properties': len(data),
        'avg_price_lakhs': round(data['price'].mean(), 2),
        'median_price_lakhs': round(data['price'].median(), 2),
        'top_5_expensive_areas': (
            data.groupby('location')['price'].mean()
                .sort_values(ascending=False).head(5).round(2).to_dict()
        ),
        'top_5_affordable_areas': (
            data.groupby('location')['price'].mean()
                .sort_values(ascending=True).head(5).round(2).to_dict()
        ),
        'bhk_distribution': data['bhk'].value_counts().to_dict(),
    }
    return jsonify(insights_data)


if __name__ == "__main__":
    # Port 5000 (was 500 - port below 1024 needs admin permission and would fail)
    app.run(debug=True, port=5000)