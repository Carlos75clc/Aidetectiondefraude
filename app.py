from flask import Flask, render_template, jsonify
import os
import json
import plotly.graph_objs as go
import numpy as np

app = Flask(__name__)

# Charger les résultats depuis un fichier JSON
def load_results():
    with open('results.json', 'r') as f:
        results = json.load(f)
    return results

@app.route('/')
def index():
    results = load_results()
    # Calcul des précisions
    avg_acc = np.mean(results['val_acc']) * 100
    best_acc = np.max(results['val_acc']) * 100

    avg_acc = round(avg_acc, 2)
    best_acc = round(best_acc, 2)
    return render_template('dashboard.html', avg_acc=avg_acc, best_acc=best_acc)

@app.route('/data')
def get_data():
    results = load_results()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)