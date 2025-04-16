import os

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'f8d7a86c35c24dba92f9b32b4af2b9c1'  # Needed for session management

# Dummy user (you can connect DB later)
USER = {'username': 'admin', 'password': '1234'}

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] == USER['username'] and request.form['password'] == USER['password']:
            session['user'] = request.form['username']
            return redirect(url_for('home'))
        else:
            error = 'Invalid Credentials'
    return render_template('login.html', error=error)

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route("/forecasting", methods=["GET", "POST"])
def forecasting():
    if request.method == "POST":
        try:
            # Retrieve form data
            input_data = pd.DataFrame([{
                "SalesAmount": float(request.form.get("sales_amount", 0)),
                "PurchaseAmount": float(request.form.get("purchase_amount", 0)),
                "TaxSlab": float(request.form.get("tax_slab", 0)),
                "InflationRate": float(request.form.get("inflation_rate", 0)),
                "ProfitMargin": float(request.form.get("profit_margin", 0)),
                "CapitalExpenditure": float(request.form.get("capital_expenditure", 0)),
                "RevenueGrowth": float(request.form.get("revenue_growth", 0)),
                "InterestRate": float(request.form.get("interest_rate", 0)),
                "GDPGrowthRate": float(request.form.get("gdp_growth_rate", 0)),
                "IndustryType": request.form.get("industry_type", "Manufacturing")
            }])

            input_data = input_data.reindex(columns=feature_columns, fill_value=0)

            prediction = model.predict(input_data)[0]
            return render_template("forecasting.html", prediction=prediction, error=None)
        except Exception as e:
            return render_template("forecasting.html", prediction=None, error=f"Error: {e}")

    return render_template("forecasting.html", prediction=None, error=None)

@app.route("/fraudulent", methods=["GET", "POST"])
def fraudulent():
    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template("fraudulent.html", result=None, error="No file chosen. Please upload a valid file.", fraud_count=None)

        try:
            uploads_folder = "uploads"
            os.makedirs(uploads_folder, exist_ok=True)
            file_path = os.path.join(uploads_folder, file.filename)
            file.save(file_path)

            # Read uploaded file
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file.filename.endswith(".xlsx"):
                df = pd.read_excel(file_path)
            else:
                return render_template("fraudulent.html", result=None, error="Unsupported file format.", fraud_count=None)

            if df.empty:
                return render_template("fraudulent.html", result=None, error="The uploaded file is empty.", fraud_count=None)

            # Analyze fraud
            fraud_result = analyze_gst_fraud(df)
            fraud_count = len(fraud_result)

            return render_template("fraudulent.html", result=fraud_result.to_dict(orient="records"), error=None, fraud_count=fraud_count)

        except ValueError as ve:
            return render_template("fraudulent.html", result=None, error=str(ve), fraud_count=None)
        except Exception as e:
            return render_template("fraudulent.html", result=None, error=f"Error processing file: {e}", fraud_count=None)

    return render_template("fraudulent.html", result=None, error=None, fraud_count=None)

def analyze_gst_fraud(df):
    required_columns = [
        "GSTIN", "Business Name", "PAN Number", "Business Type", "Industry Type",
        "Invoice Number", "Invoice Date", "Invoice Amount", "Taxable Amount", "GST Rate (%)",
        "CGST Amount", "SGST Amount", "IGST Amount", "Total GST Amount",
        "Input Tax Credit (ITC) Claimed", "Output Tax Payable", "ITC Utilization",
        "Net Tax Paid", "GSTR-1 Filed Date", "GSTR-3B Filed Date"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Define fraud logic
    fraud_conditions = (
        (df["Input Tax Credit (ITC) Claimed"] > df["Output Tax Payable"]) |
        (df["GSTR-1 Filed Date"].isnull() | df["GSTR-3B Filed Date"].isnull()) |
        (df["Total GST Amount"] != df["CGST Amount"] + df["SGST Amount"] + df["IGST Amount"])
    )

    df["Fraudulent"] = fraud_conditions.astype(int)
    df["Fraud Status"] = df["Fraudulent"].map({1: "Fraudulent", 0: "Genuine"})

    fraud_df = df[df["Fraudulent"] == 1]

    return fraud_df[[
        "GSTIN", "Business Name", "Invoice Number", "Invoice Amount",
        "Input Tax Credit (ITC) Claimed", "Output Tax Payable", "Net Tax Paid",
        "GSTR-1 Filed Date", "GSTR-3B Filed Date", "Fraud Status"
    ]]
@app.route("/fraudulent-info")
def fraudulent_info():
    return render_template("fraudulent_info.html")

@app.route("/forecasting-info")
def forecasting_info():
    return render_template("forecasting_info.html")

@app.route("/about")
def about():
    return render_template("about.html")














 

@app.route("/download", methods=["GET"])
def download():
    try:
        # Path to the report file (example: "fraudulent_report.csv")
        report_path = "outputs/fraudulent_report.csv"
        
        # Ensure the file exists before sending it
        if not os.path.exists(report_path):
            return "Report not found. Please generate it first.", 404

        return send_file(report_path, as_attachment=True)
    except Exception as e:
        return f"Error occurred: {e}", 500
    pass





@app.route("/generate-report", methods=["POST"])
def generate_report():
    try:
        # File path for the report
        report_path = "outputs/fraudulent_report.csv"

        # Example: Write fraudulent data to CSV file
        fraud_data = session.get("fraud_result")  # Store fraudulent results in session during analysis
        
        if fraud_data:
            pd.DataFrame(fraud_data).to_csv(report_path, index=False)
            return "Report generated successfully!", 200
        else:
            return "No fraudulent data available to generate the report.", 400
    except Exception as e:
        return f"Error occurred: {e}", 500
    

@app.route("/analysis", methods=["GET"])
def analysis():
    try:
        # Retrieve fraudulent data from session or database
        fraud_data = session.get("fraud_result")
        
        if fraud_data:
            return render_template("analysis.html", result=fraud_data)
        else:
            return "No analysis data available.", 400
    except Exception as e:
        return f"Error occurred: {e}", 500



if __name__ == "__main__":
    app.run(debug=True)
