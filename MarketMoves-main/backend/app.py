from flask import Flask, render_template, request
from newsapi1 import analyze_stock

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    error_message = None
    results = None

    if request.method == "POST":
        stock_ticker = request.form.get("ticker")  # Change this line
        # Analyze the stock ticker using the analyze_stock function
        results, error_message = analyze_stock(stock_ticker)

        # If there's an error message, display it to the user without crashing
        if error_message:
            return render_template("index.html", error_message=error_message)

    return render_template("index.html", results=results, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True, port = 5511)