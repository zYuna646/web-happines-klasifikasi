from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

model = None  
accuracy = 0

@app.route('/')
def index():
    return render_template('train.html', accuracy=accuracy)

@app.route('/train', methods=['POST'])
def train_model():
    global model, accuracy

    uploaded_file = request.files['csv_file']
    if not uploaded_file:
        return redirect(url_for('index'))

    test_size = float(request.form['test_size']) 

    df = pd.read_csv(uploaded_file)

    # Assuming 'Kelayakan' is the target variable
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=12345)

    # Clean 'Penghasilan Orang Tua Perbulan' column
    train_data['Penghasilan Orang Tua Perbulan'] = train_data['Penghasilan Orang Tua Perbulan'].str.replace(',', '').astype(float)
    test_data['Penghasilan Orang Tua Perbulan'] = test_data['Penghasilan Orang Tua Perbulan'].str.replace(',', '').astype(float)

    features = ['Tempat Tinggal', 'Pekerjaan Orang Tua', 'Penghasilan Orang Tua Perbulan',
                'Tanggungan Orang Tua', 'Kendaraan']

    model = DecisionTreeClassifier()
    model.fit(train_data[features], train_data['Kelayakan'])

    predictions = model.predict(test_data[features])

    accuracy = accuracy_score(test_data['Kelayakan'], predictions)
    accuracy = "{:.2f}".format(accuracy * 100)

    return redirect(url_for('index'))

@app.route('/classify', methods=['GET', 'POST'])
def classify_passenger():
    if request.method == 'POST':
        tempat_tinggal = int(request.form['tempat_tinggal'])
        pekerjaan_orang_tua = int(request.form['pekerjaan_orang_tua'])
        penghasilan_orang_tua = float(request.form['penghasilan_orang_tua'].replace(',', ''))  # Clean and convert to float
        tanggungan_orang_tua = int(request.form['tanggungan_orang_tua'])
        kendaraan = int(request.form['kendaraan'])

        user_data = pd.DataFrame({
            'Tempat Tinggal': [tempat_tinggal],
            'Pekerjaan Orang Tua': [pekerjaan_orang_tua],
            'Penghasilan Orang Tua Perbulan': [penghasilan_orang_tua],
            'Tanggungan Orang Tua': [tanggungan_orang_tua],
            'Kendaraan': [kendaraan]
        })

        if model:
            prediction = model.predict(user_data)
            result = "Tidak" if prediction[0] == 0 else "Ya"
            return render_template('classify.html', result=result)
        else:
            return "Model belum dilatih. Silakan latih model terlebih dahulu."

    return render_template('classify.html')

if __name__ == '__main__':
    app.run(debug=True)
