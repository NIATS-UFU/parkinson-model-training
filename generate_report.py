"""Interactive HTML Report Generator for Parkinson Baseline Training.

Generates a gorgeous, self-contained, interactive HTML dashboard to visualise
epoch-by-epoch training/validation loss curves and cross-validation metrics.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_html_report(history_json_path: Path | str, output_html_path: Path | str) -> None:
    """Read baseline training history JSON and generate a premium interactive HTML/JS report."""
    with open(history_json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Convert paths to strings for output
    history_json_path = str(history_json_path)
    output_html_path = str(output_html_path)

    # Embed data directly in the HTML as a JSON string
    data_str = json.dumps(data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parkinson Drawing CV Baseline Dashboard</title>

    <!-- Modern Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">

    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 28, 45, 0.6);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-primary: #8b5cf6; /* Vibrant violet */
            --accent-primary-glow: rgba(139, 92, 246, 0.3);
            --accent-secondary: #06b6d4; /* Sleek cyan */
            --accent-secondary-glow: rgba(6, 182, 212, 0.3);
            --success: #10b981;
            --danger: #ef4444;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', sans-serif;
        }}

        body {{
            background-color: var(--bg-color);
            background-image:
                radial-gradient(at 10% 10%, rgba(139, 92, 246, 0.1) 0px, transparent 50%),
                radial-gradient(at 90% 90%, rgba(6, 182, 212, 0.1) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-primary);
            padding: 2.5rem 1.5rem;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        /* Header Styling */
        header {{
            margin-bottom: 2.5rem;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1.5rem;
        }}

        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.25rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            background: linear-gradient(135deg, #fff 30%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .subtitle {{
            color: var(--text-secondary);
            font-size: 0.95rem;
            font-weight: 400;
        }}

        .meta-badge {{
            background: rgba(139, 92, 246, 0.15);
            border: 1px solid rgba(139, 92, 246, 0.3);
            padding: 0.4rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            color: #c084fc;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }}

        .metric-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 1rem;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
            opacity: 0;
            transition: opacity 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
            border-color: rgba(255, 255, 255, 0.15);
            box-shadow: 0 10px 20px -10px rgba(0, 0, 0, 0.5), 0 0 15px var(--accent-primary-glow);
        }}

        .metric-card:hover::before {{
            opacity: 1;
        }}

        .metric-title {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}

        .metric-value {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 0.25rem;
        }}

        .metric-sub {{
            font-size: 0.75rem;
            color: var(--success);
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }}

        /* Main Dashboard Section */
        .dashboard-main {{
            display: grid;
            grid-template-columns: 2fr 1.25fr;
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }}

        @media (max-width: 900px) {{
            .dashboard-main {{
                grid-template-columns: 1fr;
            }}
        }}

        .panel {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 1rem;
            padding: 1.75rem;
            backdrop-filter: blur(12px);
        }}

        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }}

        .panel-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 600;
            color: #fff;
        }}

        /* Selector Styling */
        .control-group {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        label {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-weight: 500;
        }}

        select {{
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--card-border);
            color: #fff;
            padding: 0.4rem 1.2rem 0.4rem 0.6rem;
            border-radius: 0.5rem;
            font-size: 0.85rem;
            font-weight: 500;
            outline: none;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        select:hover {{
            border-color: var(--accent-primary);
        }}

        /* Chart Canvas wrapper */
        .chart-wrapper {{
            position: relative;
            width: 100%;
            height: 350px;
        }}

        /* Model Configuration Panel */
        .config-card {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}

        .config-item {{
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 0.75rem;
        }}

        .config-item:last-child {{
            border-bottom: none;
            padding-bottom: 0;
        }}

        .config-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        .config-value {{
            font-size: 0.85rem;
            font-weight: 600;
            color: #fff;
        }}

        /* Accordion fold summary lists */
        .fold-summary-list {{
            margin-top: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .fold-summary-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0.75rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 0.5rem;
            font-size: 0.8rem;
        }}

        .fold-summary-name {{
            font-weight: 500;
            color: var(--text-secondary);
        }}

        .fold-summary-val {{
            font-weight: 600;
            color: var(--accent-secondary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div>
                <h1>Parkinson drawing cross-validation baseline</h1>
                <p class="subtitle">Leakage-Safe Patient-Stratified Group K-Fold Performance Report</p>
            </div>
            <div>
                <span class="meta-badge" id="model-badge">Model: Simple CNN</span>
            </div>
        </header>

        <!-- Metric Cards -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div>
                    <p class="metric-title">Accuracy</p>
                    <p class="metric-value" id="val-accuracy">0.0%</p>
                </div>
                <div class="metric-sub">Average over K Folds</div>
            </div>
            <div class="metric-card">
                <div>
                    <p class="metric-title">Balanced Acc</p>
                    <p class="metric-value" id="val-balanced-acc">0.0%</p>
                </div>
                <div class="metric-sub">Primary Metric</div>
            </div>
            <div class="metric-card">
                <div>
                    <p class="metric-title">Sensitivity (PD)</p>
                    <p class="metric-value" id="val-sensitivity">0.0%</p>
                </div>
                <div class="metric-sub">True Positive Rate</div>
            </div>
            <div class="metric-card">
                <div>
                    <p class="metric-title">Specificity (HC)</p>
                    <p class="metric-value" id="val-specificity">0.0%</p>
                </div>
                <div class="metric-sub">True Negative Rate</div>
            </div>
            <div class="metric-card">
                <div>
                    <p class="metric-title">ROC-AUC</p>
                    <p class="metric-value" id="val-roc-auc">0.0%</p>
                </div>
                <div class="metric-sub">Discriminative Ability</div>
            </div>
        </div>

        <!-- Dashboard Charts and Config -->
        <div class="dashboard-main">
            <!-- Loss Curves Panel -->
            <div class="panel">
                <div class="panel-header">
                    <h2 class="panel-title">Training vs Validation Loss curves</h2>
                    <div class="control-group">
                        <label for="fold-select">Fold Selection:</label>
                        <select id="fold-select">
                            <option value="average">Average Folds</option>
                            <option value="1">Fold 1</option>
                            <option value="2">Fold 2</option>
                            <option value="3">Fold 3</option>
                            <option value="4">Fold 4</option>
                            <option value="5">Fold 5</option>
                        </select>
                    </div>
                </div>
                <div class="chart-wrapper">
                    <canvas id="lossChart"></canvas>
                </div>
            </div>

            <!-- Configuration and Folds Overview -->
            <div class="panel">
                <div class="panel-header">
                    <h2 class="panel-title">Run Configuration & Summary</h2>
                </div>
                <div class="config-card">
                    <div class="config-item">
                        <span class="config-label">Model Type:</span>
                        <span class="config-value" id="config-model">simple-cnn</span>
                    </div>
                    <div class="config-item">
                        <span class="config-label">Epochs:</span>
                        <span class="config-value" id="config-epochs">5</span>
                    </div>
                    <div class="config-item">
                        <span class="config-label">Batch Size:</span>
                        <span class="config-value" id="config-batch">32</span>
                    </div>
                    <div class="config-item">
                        <span class="config-label">Learning Rate:</span>
                        <span class="config-value" id="config-lr">0.001</span>
                    </div>
                    <div class="config-item">
                        <span class="config-label">Total Folds:</span>
                        <span class="config-value" id="config-folds">5</span>
                    </div>

                    <h3 class="panel-title" style="margin-top: 1rem; font-size: 1rem;">Fold Balanced Acc</h3>
                    <div class="fold-summary-list" id="fold-summary-container">
                        <!-- Populated by JS -->
                    </div>
                </div>
            </div>
        </div>

        <!-- Folds Comparison Panel -->
        <div class="panel" style="margin-bottom: 2.5rem;">
            <div class="panel-header">
                <h2 class="panel-title">Cross-Validation Metrics Comparison Across Folds</h2>
            </div>
            <div class="chart-wrapper">
                <canvas id="metricsChart"></canvas>
            </div>
        </div>
    </div>

    <!-- Script to Load Data and Draw Charts -->
    <script>
        const runData = {data_str};

        // Populate UI text
        document.getElementById('model-badge').textContent = 'Model: ' + runData.model_type;
        document.getElementById('val-accuracy').textContent = (runData.consolidated.accuracy * 100).toFixed(1) + '%';
        document.getElementById('val-balanced-acc').textContent = (runData.consolidated.balanced_accuracy * 100).toFixed(1) + '%';
        document.getElementById('val-sensitivity').textContent = (runData.consolidated.sensitivity * 100).toFixed(1) + '%';
        document.getElementById('val-specificity').textContent = (runData.consolidated.specificity * 100).toFixed(1) + '%';
        document.getElementById('val-roc-auc').textContent = (runData.consolidated.roc_auc * 100).toFixed(1) + '%';

        document.getElementById('config-model').textContent = runData.model_type + (runData.frozen_backbone ? ' (frozen)' : '');
        document.getElementById('config-epochs').textContent = runData.epochs;
        document.getElementById('config-batch').textContent = runData.batch_size;
        document.getElementById('config-lr').textContent = runData.lr;
        document.getElementById('config-folds').textContent = runData.num_folds;

        // Populate fold summaries
        const foldList = document.getElementById('fold-summary-container');
        runData.folds.forEach(f => {{
            const item = document.createElement('div');
            item.className = 'fold-summary-item';
            item.innerHTML = `
                <span class="fold-summary-name">Fold ${{f.fold}}</span>
                <span class="fold-summary-val">${{(f.best_metrics.balanced_accuracy * 100).toFixed(1)}}%</span>
            `;
            foldList.appendChild(item);
        }});

        // Generate select options dynamically
        const foldSelect = document.getElementById('fold-select');
        foldSelect.innerHTML = '<option value="average">Average Folds</option>';
        runData.folds.forEach(f => {{
            foldSelect.innerHTML += `<option value="${{f.fold}}">Fold ${{f.fold}}</option>`;
        }});

        // ----------------------------------------------------
        // Chart 1: Loss curves
        // ----------------------------------------------------
        const ctxLoss = document.getElementById('lossChart').getContext('2d');
        let lossChart;

        function getLossData(foldValue) {{
            let labels = [];
            let trainLoss = [];
            let valLoss = [];

            if (foldValue === 'average') {{
                const maxEpochs = runData.epochs;
                for (let e = 1; e <= maxEpochs; e++) {{
                    labels.push('Epoch ' + e);
                    let sumTrain = 0, sumVal = 0, count = 0;
                    runData.folds.forEach(f => {{
                        const epochData = f.epochs.find(x => x.epoch === e);
                        if (epochData) {{
                            sumTrain += epochData.train_loss;
                            sumVal += epochData.val_loss;
                            count++;
                        }}
                    }});
                    trainLoss.push(count > 0 ? sumTrain / count : 0);
                    valLoss.push(count > 0 ? sumVal / count : 0);
                }}
            }} else {{
                const fIndex = parseInt(foldValue) - 1;
                const fold = runData.folds[fIndex];
                fold.epochs.forEach(e => {{
                    labels.push('Epoch ' + e.epoch);
                    trainLoss.push(e.train_loss);
                    valLoss.push(e.val_loss);
                }});
            }}

            return {{ labels, trainLoss, valLoss }};
        }}

        function drawLossChart(foldValue) {{
            const {{ labels, trainLoss, valLoss }} = getLossData(foldValue);

            if (lossChart) {{
                lossChart.destroy();
            }}

            lossChart = new Chart(ctxLoss, {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: [
                        {{
                            label: 'Train Loss',
                            data: trainLoss,
                            borderColor: '#8b5cf6',
                            backgroundColor: 'rgba(139, 92, 246, 0.1)',
                            borderWidth: 3,
                            tension: 0.35,
                            fill: true
                        }},
                        {{
                            label: 'Val Loss',
                            data: valLoss,
                            borderColor: '#06b6d4',
                            backgroundColor: 'rgba(6, 182, 212, 0.1)',
                            borderWidth: 3,
                            tension: 0.35,
                            fill: true
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{ color: '#e5e7eb', font: {{ family: 'Inter', weight: 500 }} }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                            ticks: {{ color: '#9ca3af', font: {{ family: 'Inter' }} }}
                        }},
                        x: {{
                            grid: {{ display: false }},
                            ticks: {{ color: '#9ca3af', font: {{ family: 'Inter' }} }}
                        }}
                    }}
                }}
            }});
        }}

        foldSelect.addEventListener('change', (e) => drawLossChart(e.target.value));
        drawLossChart('average');

        // ----------------------------------------------------
        // Chart 2: Folds Metrics Comparison Bar Chart
        // ----------------------------------------------------
        const ctxMetrics = document.getElementById('metricsChart').getContext('2d');
        const metricLabels = runData.folds.map(f => 'Fold ' + f.fold);

        const accData = runData.folds.map(f => f.best_metrics.accuracy);
        const balAccData = runData.folds.map(f => f.best_metrics.balanced_accuracy);
        const sensData = runData.folds.map(f => f.best_metrics.sensitivity);
        const specData = runData.folds.map(f => f.best_metrics.specificity);
        const aucData = runData.folds.map(f => f.best_metrics.roc_auc);

        new Chart(ctxMetrics, {{
            type: 'bar',
            data: {{
                labels: metricLabels,
                datasets: [
                    {{
                        label: 'Accuracy',
                        data: accData,
                        backgroundColor: '#8b5cf6',
                        borderRadius: 4
                    }},
                    {{
                        label: 'Balanced Acc',
                        data: balAccData,
                        backgroundColor: '#06b6d4',
                        borderRadius: 4
                    }},
                    {{
                        label: 'Sensitivity (PD)',
                        data: sensData,
                        backgroundColor: '#10b981',
                        borderRadius: 4
                    }},
                    {{
                        label: 'Specificity (HC)',
                        data: specData,
                        backgroundColor: '#f59e0b',
                        borderRadius: 4
                    }},
                    {{
                        label: 'ROC-AUC',
                        data: aucData,
                        backgroundColor: '#ec4899',
                        borderRadius: 4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'top',
                        labels: {{ color: '#e5e7eb', font: {{ family: 'Inter', weight: 500 }} }}
                    }}
                }},
                scales: {{
                    y: {{
                        min: 0,
                        max: 1.0,
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{
                            color: '#9ca3af',
                            callback: function(value) {{ return (value * 100) + '%'; }},
                            font: {{ family: 'Inter' }}
                        }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#9ca3af', font: {{ family: 'Inter' }} }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
