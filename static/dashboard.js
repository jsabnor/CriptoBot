// ============================================================================
// DUAL BOT DASHBOARD - JavaScript
// ============================================================================

// Configuration
const SYMBOLS = ['ETH', 'XRP', 'BNB', 'SOL'];
const REFRESH_INTERVAL = 30000; // 30 seconds
let currentView = 'combined';
let currentADXSymbol = 'ETH';
let currentEMASymbol = 'ETH';

// Loading Spinner Helpers
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (!element) return;

    // Check if overlay already exists
    if (element.querySelector('.spinner-overlay')) return;

    const overlay = document.createElement('div');
    overlay.className = 'spinner-overlay';
    overlay.innerHTML = '<div class="spinner"></div>';
    element.appendChild(overlay);
}

function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (!element) return;

    const overlay = element.querySelector('.spinner-overlay');
    if (overlay) {
        overlay.remove();
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function () {
    console.log('Dual Bot Dashboard initialized');

    // Setup tab navigation
    setupTabs();

    // Setup symbol selectors
    setupSymbolSelectors();

    // Setup optimizer
    setupOptimizerView();

    // Initial update
    updateDashboard();

    // Auto-refresh
    setInterval(updateDashboard, REFRESH_INTERVAL);
});

// ============================================================================
// TAB NAVIGATION
// ============================================================================

function setupTabs() {
    const tabs = document.querySelectorAll('.tab');

    tabs.forEach(tab => {
        tab.addEventListener('click', function () {
            const viewName = this.getAttribute('data-view');
            switchView(viewName);
        });
    });
}

function switchView(viewName) {
    // Update current view
    currentView = viewName;

    // Hide all views
    document.querySelectorAll('.view').forEach(view => {
        view.classList.remove('active');
    });

    // Show selected view
    document.getElementById(`${viewName}-view`).classList.add('active');

    // Update active tab
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelector(`[data-view="${viewName}"]`).classList.add('active');

    // Update view-specific data
    if (viewName === 'adx') {
        updateADXView();
    } else if (viewName === 'ema') {
        updateEMAView();
    } else if (viewName === 'comparison') {
        updateComparisonView();
    } else if (viewName === 'optimizer') {
        loadLastOptimizerResults();
    }
}

// ============================================================================
// SYMBOL SELECTORS
// ============================================================================

function setupSymbolSelectors() {
    // ADX symbol selector
    const adxSelect = document.getElementById('adx-symbol-select');
    if (adxSelect) {
        adxSelect.addEventListener('change', function () {
            currentADXSymbol = this.value;
            updateADXChart(currentADXSymbol);
        });
    }

    // EMA symbol selector
    const emaSelect = document.getElementById('ema-symbol-select');
    if (emaSelect) {
        emaSelect.addEventListener('change', function () {
            currentEMASymbol = this.value;
            updateEMAChart(currentEMASymbol);
        });
    }
}

// ============================================================================
// MAIN UPDATE FUNCTION
// ============================================================================

function updateDashboard() {
    updateDualStatus();
    updateLastUpdateTime();

    // Update view-specific data based on current view
    if (currentView === 'combined') {
        // Combined view updates automatically with dual status
    } else if (currentView === 'adx') {
        updateADXView();
    } else if (currentView === 'ema') {
        updateEMAView();
    } else if (currentView === 'comparison') {
        updateComparisonView();
    }
}

// ============================================================================
// COMBINED VIEW
// ============================================================================

function updateDualStatus() {
    fetch('/api/dual_status')
        .then(r => r.json())
        .then(data => {
            const combined = data.combined;

            // Total Equity
            document.getElementById('combined-equity').textContent =
                `$${combined.total_equity.toFixed(2)}`;

            // Individual equities
            document.getElementById('combined-adx-equity').textContent =
                `$${combined.adx_equity.toFixed(2)}`;
            document.getElementById('combined-ema-equity').textContent =
                `$${combined.ema_equity.toFixed(2)}`;

            // Combined ROI
            const roiElement = document.getElementById('combined-roi');
            const roiValue = combined.combined_roi.toFixed(2);
            roiElement.textContent = `${roiValue > 0 ? '+' : ''}${roiValue}%`;
            roiElement.className = `metric-value ${roiValue >= 0 ? 'profit' : 'loss'}`;

            // Positions
            document.getElementById('combined-positions').textContent =
                `${combined.total_positions}/8`;
            document.getElementById('combined-adx-pos').textContent =
                combined.adx_positions;
            document.getElementById('combined-ema-pos').textContent =
                combined.ema_positions;

            // Distribution percentages
            document.getElementById('combined-adx-pct').textContent =
                `${combined.adx_percentage.toFixed(1)}%`;
            document.getElementById('combined-ema-pct').textContent =
                `${combined.ema_percentage.toFixed(1)}%`;

            // Update distribution chart
            renderDistributionChart(combined);
        })
        .catch(error => console.error('Error updating dual status:', error));
}

function renderDistributionChart(data) {
    const chartData = [{
        values: [data.adx_equity, data.ema_equity],
        labels: ['Bot ADX', 'Bot EMA'],
        type: 'pie',
        marker: {
            colors: ['#3498db', '#e74c3c']
        },
        textinfo: 'label+percent',
        hoverinfo: 'label+value+percent',
        hole: 0.4
    }];

    const layout = {
        height: 300,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ecf0f1' },
        showlegend: false,
        margin: { t: 0, b: 0, l: 0, r: 0 }
    };

    Plotly.newPlot('distribution-chart', chartData, layout, { responsive: true });
}

// ============================================================================
// ADX VIEW
// ============================================================================

function updateADXView() {
    // Update status
    fetch('/api/bot/adx/status')
        .then(r => r.json())
        .then(data => {
            document.getElementById('adx-equity').textContent = `$${data.total_equity.toFixed(2)}`;

            const roiElement = document.getElementById('adx-roi');
            const roiValue = data.roi.toFixed(2);
            roiElement.textContent = `${roiValue > 0 ? '+' : ''}${roiValue}%`;
            roiElement.className = `metric-value ${roiValue >= 0 ? 'profit' : 'loss'}`;

            document.getElementById('adx-positions').textContent = `${data.open_positions}/4`;
        })
        .catch(error => console.error('Error updating ADX status:', error));

    // Update chart
    updateADXChart(currentADXSymbol);

    // Update trades
    fetch('/api/bot/adx/trades')
        .then(r => r.json())
        .then(trades => {
            renderTradesTable(trades, 'adx-trades-table');
        })
        .catch(error => console.error('Error updating ADX trades:', error));
}

function updateADXChart(symbol) {
    const containerId = 'adx-chart';
    showLoading(containerId);

    fetch(`/api/chart/${symbol}`)
        .then(r => r.json())
        .then(data => {
            hideLoading(containerId);
            renderChart(data, containerId, 'ADX');
        })
        .catch(error => {
            hideLoading(containerId);
            console.error(`Error updating ADX chart for ${symbol}:`, error);
        });
}

// ============================================================================
// EMA VIEW
// ============================================================================

function updateEMAView() {
    // Update status
    fetch('/api/bot/ema/status')
        .then(r => r.json())
        .then(data => {
            document.getElementById('ema-equity').textContent = `$${data.total_equity.toFixed(2)}`;

            const roiElement = document.getElementById('ema-roi');
            const roiValue = data.roi.toFixed(2);
            roiElement.textContent = `${roiValue > 0 ? '+' : ''}${roiValue}%`;
            roiElement.className = `metric-value ${roiValue >= 0 ? 'profit' : 'loss'}`;

            document.getElementById('ema-positions').textContent = `${data.open_positions}/4`;
        })
        .catch(error => console.error('Error updating EMA status:', error));

    // Update chart
    updateEMAChart(currentEMASymbol);

    // Update trades
    fetch('/api/bot/ema/trades')
        .then(r => r.json())
        .then(trades => {
            renderTradesTable(trades, 'ema-trades-table');
        })
        .catch(error => console.error('Error updating EMA trades:', error));
}

function updateEMAChart(symbol) {
    const containerId = 'ema-chart';
    showLoading(containerId);

    fetch(`/api/chart/${symbol}`)
        .then(r => r.json())
        .then(data => {
            hideLoading(containerId);
            renderChart(data, containerId, 'EMA');
        })
        .catch(error => {
            hideLoading(containerId);
            console.error(`Error updating EMA chart for ${symbol}:`, error);
        });
}

// ============================================================================
// COMPARISON VIEW
// ============================================================================

function updateComparisonView() {
    fetch('/api/comparison')
        .then(r => r.json())
        .then(data => {
            renderROIComparison(data);
            renderWinRateComparison(data);
            renderComparisonTable(data);
        })
        .catch(error => console.error('Error updating comparison view:', error));
}

function renderROIComparison(data) {
    const chartData = [{
        x: ['Bot ADX', 'Bot EMA'],
        y: [data.adx.roi, data.ema.roi],
        type: 'bar',
        marker: {
            color: ['#3498db', '#e74c3c']
        },
        text: [
            `${data.adx.roi.toFixed(2)}%`,
            `${data.ema.roi.toFixed(2)}%`
        ],
        textposition: 'auto'
    }];

    const layout = {
        title: 'ROI (%)',
        height: 300,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ecf0f1' },
        yaxis: { title: 'ROI %' }
    };

    Plotly.newPlot('roi-comparison-chart', chartData, layout, { responsive: true });
}

function renderWinRateComparison(data) {
    const chartData = [{
        x: ['Bot ADX', 'Bot EMA'],
        y: [data.adx.win_rate, data.ema.win_rate],
        type: 'bar',
        marker: {
            color: ['#3498db', '#e74c3c']
        },
        text: [
            `${data.adx.win_rate.toFixed(1)}%`,
            `${data.ema.win_rate.toFixed(1)}%`
        ],
        textposition: 'auto'
    }];

    const layout = {
        title: 'Win Rate (%)',
        height: 300,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ecf0f1' },
        yaxis: { title: 'Win Rate %', range: [0, 100] }
    };

    Plotly.newPlot('winrate-comparison-chart', chartData, layout, { responsive: true });
}

function renderComparisonTable(data) {
    const tbody = document.getElementById('comparison-table-body');
    if (!tbody) return;

    const metrics = [
        { name: 'Equity', adx: data.adx.equity, ema: data.ema.equity, format: '$' },
        { name: 'ROI', adx: data.adx.roi, ema: data.ema.roi, format: '%' },
        { name: 'Total Trades', adx: data.adx.total_trades, ema: data.ema.total_trades, format: '' },
        { name: 'Wins', adx: data.adx.wins, ema: data.ema.wins, format: '' },
        { name: 'Losses', adx: data.adx.losses, ema: data.ema.losses, format: '' },
        { name: 'Win Rate', adx: data.adx.win_rate, ema: data.ema.win_rate, format: '%' },
        { name: 'Total PnL', adx: data.adx.total_pnl, ema: data.ema.total_pnl, format: '$' }
    ];

    tbody.innerHTML = metrics.map(m => {
        const diff = m.adx - m.ema;
        const diffClass = diff > 0 ? 'profit' : (diff < 0 ? 'loss' : '');
        const diffText = diff > 0 ? `+${diff.toFixed(2)}` : diff.toFixed(2);

        return `
            <tr>
                <td>${m.name}</td>
                <td class="adx-text">${m.format === '$' ? '$' : ''}${m.adx.toFixed(2)}${m.format === '%' ? '%' : ''}</td>
                <td class="ema-text">${m.format === '$' ? '$' : ''}${m.ema.toFixed(2)}${m.format === '%' ? '%' : ''}</td>
                <td class="${diffClass}">${m.format === '$' ? '$' : ''}${diffText}${m.format === '%' ? '%' : ''}</td>
            </tr>
        `;
    }).join('');
}

// ============================================================================
// CHART RENDERING (Shared)
// ============================================================================

function renderChart(data, containerId, botType) {
    const candles = data.candles;

    // Separate closed candles from current candle
    const closedCandles = candles.filter(c => !c.is_current);
    const currentCandle = candles.find(c => c.is_current);

    // Calculate range for last 50 candles to reduce clutter
    let xrange = null;
    if (closedCandles.length > 50) {
        const start = closedCandles[closedCandles.length - 50].timestamp;
        const end = closedCandles[closedCandles.length - 1].timestamp;
        xrange = [start, end];
    }

    // Layout configuration
    const layout = {
        height: 500,
        margin: { b: 80, r: 50, t: 30, l: 50 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ecf0f1' },
        xaxis: {
            rangeslider: { visible: false },
            anchor: 'y',
            range: xrange,
            tickformat: '%d/%m %H:%M',
            tickangle: -45,
            automargin: true,
            tickmode: 'auto',
            nticks: 10
        },
        yaxis: { domain: [0, 1] }, // Full height for main chart
        grid: { rows: 1, columns: 1, pattern: 'independent' },
        showlegend: true,
        legend: { orientation: 'h', y: 1.02, x: 0.5, xanchor: 'center' }
    };

    // Base traces (Candlesticks)
    const traces = [];

    // 1. Candlestick Trace
    traces.push({
        x: closedCandles.map(c => c.timestamp),
        open: closedCandles.map(c => c.open),
        high: closedCandles.map(c => c.high),
        low: closedCandles.map(c => c.low),
        close: closedCandles.map(c => c.close),
        type: 'candlestick',
        name: data.symbol,
        increasing: { line: { color: '#26a69a' } },
        decreasing: { line: { color: '#ef5350' } }
    });

    // 2. Current Candle (Ghost)
    if (currentCandle) {
        traces.push({
            x: [currentCandle.timestamp],
            open: [currentCandle.open],
            high: [currentCandle.high],
            low: [currentCandle.low],
            close: [currentCandle.close],
            type: 'candlestick',
            name: 'Actual (en progreso)',
            increasing: { line: { color: '#26a69a', width: 1, dash: 'dot' }, fillcolor: 'rgba(38, 166, 154, 0.3)' },
            decreasing: { line: { color: '#ef5350', width: 1, dash: 'dot' }, fillcolor: 'rgba(239, 83, 80, 0.3)' },
            showlegend: false
        });
    }

    // Update Indicator Panels (Last Closed Candle)
    if (closedCandles.length > 0) {
        const last = closedCandles[closedCandles.length - 1];
        updateIndicatorsPanel(botType, last);
    }

    // Strategy Specific Chart Indicators
    if (botType === 'ADX') {
        // MA 50
        traces.push({
            x: closedCandles.map(c => c.timestamp),
            y: closedCandles.map(c => c.ma),
            type: 'scatter',
            mode: 'lines',
            name: 'MA 50',
            line: { color: '#f39c12', width: 1.5 }
        });

        // MA 200
        traces.push({
            x: closedCandles.map(c => c.timestamp),
            y: closedCandles.map(c => c.long_ma),
            type: 'scatter',
            mode: 'lines',
            name: 'MA 200',
            line: { color: '#e74c3c', width: 1.5 }
        });

    } else if (botType === 'EMA') {
        // EMA 15
        traces.push({
            x: closedCandles.map(c => c.timestamp),
            y: closedCandles.map(c => c.ema_fast),
            type: 'scatter',
            mode: 'lines',
            name: 'EMA 15',
            line: { color: '#3498db', width: 1.5 }
        });

        // EMA 30
        traces.push({
            x: closedCandles.map(c => c.timestamp),
            y: closedCandles.map(c => c.ema_slow),
            type: 'scatter',
            mode: 'lines',
            name: 'EMA 30',
            line: { color: '#9b59b6', width: 1.5 }
        });
    }

    Plotly.newPlot(containerId, traces, layout, { responsive: true });
}

function updateIndicatorsPanel(botType, candle) {
    if (botType === 'ADX') {
        // ADX Value
        const adxEl = document.getElementById('adx-val');
        if (adxEl) {
            adxEl.textContent = candle.adx ? candle.adx.toFixed(2) : '--';
            adxEl.className = `indicator-value ${candle.adx > 25 ? 'bullish' : 'bearish'}`;
        }

        // Trend (MA50)
        const trendEl = document.getElementById('adx-trend');
        if (trendEl) {
            const isBullish = candle.close > candle.ma;
            trendEl.textContent = isBullish ? 'BULLISH' : 'BEARISH';
            trendEl.className = `indicator-value ${isBullish ? 'bullish' : 'bearish'}`;
        }

        // ATR
        const atrEl = document.getElementById('adx-atr');
        if (atrEl) {
            atrEl.textContent = candle.atr ? candle.atr.toFixed(2) : '--';
            atrEl.className = 'indicator-value neutral';
        }

    } else if (botType === 'EMA') {
        // EMA Fast
        const fastEl = document.getElementById('ema-fast-val');
        if (fastEl) fastEl.textContent = candle.ema_fast ? candle.ema_fast.toFixed(2) : '--';

        // EMA Slow
        const slowEl = document.getElementById('ema-slow-val');
        if (slowEl) slowEl.textContent = candle.ema_slow ? candle.ema_slow.toFixed(2) : '--';

        // Signal
        const signalEl = document.getElementById('ema-signal');
        if (signalEl) {
            const isBuy = candle.ema_fast > candle.ema_slow;
            signalEl.textContent = isBuy ? 'BUY ZONE' : 'SELL ZONE';
            signalEl.className = `indicator-value ${isBuy ? 'bullish' : 'bearish'}`;
        }
    }
}

function renderTradesTable(trades, elementId) {
    const container = document.getElementById(elementId);
    if (!container) return;

    if (trades.length === 0) {
        container.innerHTML = '<p class="no-data">No hay trades registrados</p>';
        return;
    }

    let html = `
        <table>
            <thead>
                <tr>
                    <th>Fecha</th>
                    <th>Tipo</th>
                    <th>Precio</th>
                    <th>Cant.</th>
                    <th>PnL</th>
                </tr>
            </thead>
            <tbody>
    `;

    trades.forEach(trade => {
        const date = new Date(trade.timestamp).toLocaleString('es-ES');
        const typeClass = trade.type === 'buy' ? 'positive' : 'negative';
        const pnlClass = trade.pnl > 0 ? 'positive' : (trade.pnl < 0 ? 'negative' : '');
        const pnlText = trade.pnl ? `$${trade.pnl.toFixed(2)}` : '-';

        html += `
            <tr>
                <td>${date}</td>
                <td class="${typeClass}">${trade.type.toUpperCase()}</td>
                <td>$${trade.price.toFixed(2)}</td>
                <td>${trade.qty.toFixed(4)}</td>
                <td class="${pnlClass}">${pnlText}</td>
            </tr>
        `;
    });

    html += `</tbody></table>`;
    container.innerHTML = html;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function updateLastUpdateTime() {
    const now = new Date();
    const element = document.getElementById('last-update');
    if (element) {
        element.textContent = now.toLocaleTimeString('es-ES');
    }
}

// ============================================================================
// OPTIMIZER FUNCTIONS
// ============================================================================

function setupOptimizerView() {
    const runBtn = document.getElementById('run-optimizer-btn');
    if (runBtn) {
        runBtn.addEventListener('click', runOptimization);
    }
}

function getSelectedSymbols() {
    const checkboxes = document.querySelectorAll('.symbol-checkbox:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

async function runOptimization() {
    const strategy = document.getElementById('opt-strategy').value;
    const symbols = getSelectedSymbols();

    if (symbols.length === 0) {
        alert('Por favor selecciona al menos un símbolo');
        return;
    }

    // Disable button and show progress
    const runBtn = document.getElementById('run-optimizer-btn');
    runBtn.disabled = true;
    runBtn.textContent = '⏳ Ejecutando...';

    document.getElementById('optimizer-config').style.display = 'none';
    document.getElementById('optimizer-results').style.display = 'none';
    document.getElementById('optimizer-progress').style.display = 'block';

    // Simulate progress (since we can't get real progress from sync call)
    const progressFill = document.getElementById('progress-fill');
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 2;
        if (progress <= 90) {
            progressFill.style.width = progress + '%';
        }
    }, 1000);

    try {
        const response = await fetch('/api/optimizer/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                strategy: strategy,
                symbols: symbols
            })
        });

        clearInterval(progressInterval);
        progressFill.style.width = '100%';

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Hide progress, show results
        setTimeout(() => {
            document.getElementById('optimizer-progress').style.display = 'none';
            displayResults(data);
        }, 500);

    } catch (error) {
        clearInterval(progressInterval);
        console.error('Error running optimization:', error);
        alert('Error ejecutando optimización: ' + error.message);

        // Reset UI
        document.getElementById('optimizer-progress').style.display = 'none';
        document.getElementById('optimizer-config').style.display = 'block';
    } finally {
        runBtn.disabled = false;
        runBtn.textContent = '🚀 Iniciar Optimización';
    }
}

function displayResults(data) {
    const resultsContainer = document.getElementById('optimizer-results');
    const resultsInfo = document.getElementById('results-info');

    // Update info
    const timestamp = new Date(data.timestamp).toLocaleString('es-ES');
    resultsInfo.textContent = `Optimización ${data.strategy.toUpperCase()} completada el ${timestamp}. Total de configuraciones probadas: ${data.total_configs}`;

    // Render tables
    renderResultsTable('results-table-score', data.top_score, data.strategy);
    renderResultsTable('results-table-roi', data.top_roi, data.strategy);

    // Show results
    resultsContainer.style.display = 'block';
    document.getElementById('optimizer-config').style.display = 'block';
}

function renderResultsTable(tableId, results, strategy) {
    const table = document.getElementById(tableId);
    const tbody = table.querySelector('tbody');
    tbody.innerHTML = '';

    results.forEach((result, index) => {
        const row = document.createElement('tr');

        // Rank
        const rankCell = document.createElement('td');
        rankCell.textContent = index + 1;
        row.appendChild(rankCell);

        // Parameters
        const paramCell = document.createElement('td');
        paramCell.className = 'param-cell';
        if (strategy === 'ema') {
            paramCell.textContent = `EMA(${result.fast},${result.slow})`;
            if (result.use_filter) paramCell.textContent += ' +Filter';
            if (result.rsi_filter) paramCell.textContent += ' +RSI';
            if (result.atr_sl) paramCell.textContent += ` +ATR(${result.atr_mult})`;
        } else {
            paramCell.textContent = `ROC(${result.period},${result.ma_period})`;
            if (result.min_roc > 0) paramCell.textContent += ` minROC:${result.min_roc}`;
            if (result.use_trend_filter) paramCell.textContent += ' +Trend';
            if (result.rsi_filter) paramCell.textContent += ' +RSI';
            if (result.atr_sl) paramCell.textContent += ` +ATR(${result.atr_mult})`;
        }
        row.appendChild(paramCell);

        // ROI
        const roiCell = document.createElement('td');
        roiCell.textContent = result.avg_roi.toFixed(2) + '%';
        roiCell.className = result.avg_roi > 0 ? 'positive' : 'negative';
        row.appendChild(roiCell);

        // Win Rate
        const wrCell = document.createElement('td');
        wrCell.textContent = result.avg_win_rate.toFixed(1) + '%';
        row.appendChild(wrCell);

        // Drawdown
        const ddCell = document.createElement('td');
        ddCell.textContent = result.avg_drawdown.toFixed(2) + '%';
        ddCell.className = 'negative';
        row.appendChild(ddCell);

        // Sharpe
        const sharpeCell = document.createElement('td');
        sharpeCell.textContent = result.avg_sharpe.toFixed(2);
        row.appendChild(sharpeCell);

        // Score
        const scoreCell = document.createElement('td');
        scoreCell.textContent = result.score.toFixed(2);
        scoreCell.className = 'positive';
        row.appendChild(scoreCell);

        tbody.appendChild(row);
    });
}

async function loadLastOptimizerResults() {
    const strategy = document.getElementById('opt-strategy').value;

    try {
        const response = await fetch(`/api/optimizer/last-results?strategy=${strategy}`);

        if (response.ok) {
            const data = await response.json();
            displayResults(data);
        }
    } catch (error) {
        console.log('No previous results found');
    }
}
