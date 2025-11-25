// ============================================================================
// DUAL BOT DASHBOARD - JavaScript
// ============================================================================

// Configuration
const SYMBOLS = ['ETH', 'XRP', 'BNB', 'SOL'];
const REFRESH_INTERVAL = 30000; // 30 seconds
let currentView = 'combined';
let currentADXSymbol = 'ETH';
let currentEMASymbol = 'ETH';

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function () {
    console.log('Dual Bot Dashboard initialized');

    // Setup tab navigation
    setupTabs();

    // Setup symbol selectors
    setupSymbolSelectors();

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
        updateADXChart(currentADXSymbol);
    } else if (viewName === 'ema') {
        updateEMAChart(currentEMASymbol);
    } else if (viewName === 'comparison') {
        updateComparisonView();
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

            // Render distribution chart
            renderDistributionChart(combined.adx_equity, combined.ema_equity);
        })
        .catch(err => console.error('Error updating dual status:', err));
}

function renderDistributionChart(adxEquity, emaEquity) {
    const data = [{
        values: [adxEquity, emaEquity],
        labels: ['Bot ADX', 'Bot EMA'],
        type: 'pie',
        marker: {
            colors: ['#3498db', '#e74c3c']
        },
        textinfo: 'label+percent+value',
        texttemplate: '%{label}<br>$%{value:.2f}<br>%{percent}',
        hovertemplate: '%{label}<br>$%{value:.2f}<br>%{percent}<extra></extra>'
    }];

    const layout = {
        height: 400,
        showlegend: true,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ecf0f1' }
    };

    Plotly.newPlot('distribution-chart', data, layout, { responsive: true });
}

// ============================================================================
// ADX VIEW
// ============================================================================

function updateADXView() {
    // Update ADX status
    fetch('/api/bot/adx/status')
        .then(r => r.json())
        .then(data => {
            document.getElementById('adx-equity').textContent =
                `$${data.total_equity.toFixed(2)}`;

            const roiElement = document.getElementById('adx-roi');
            const roiValue = data.roi.toFixed(2);
            roiElement.textContent = `${roiValue > 0 ? '+' : ''}${roiValue}%`;
            roiElement.className = `metric-value ${roiValue >= 0 ? 'profit' : 'loss'}`;

            document.getElementById('adx-positions').textContent =
                `${data.open_positions}/${data.total_pairs}`;
        })
        .catch(err => console.error('Error updating ADX status:', err));

    // Update ADX chart
    updateADXChart(currentADXSymbol);

    // Update ADX trades
    updateADXTrades();
}

function updateADXChart(symbol) {
    fetch(`/api/chart/${symbol}`)
        .then(r => r.json())
        .then(data => {
            renderChart(data, 'adx-chart', 'ADX');
        })
        .catch(err => console.error('Error updating ADX chart:', err));
}

function updateADXTrades() {
    fetch('/api/bot/adx/trades')
        .then(r => r.json())
        .then(trades => {
            renderTradesTable(trades, 'adx-trades-table');
        })
        .catch(err => console.error('Error updating ADX trades:', err));
}

// ============================================================================
// EMA VIEW
// ============================================================================

function updateEMAView() {
    // Update EMA status
    fetch('/api/bot/ema/status')
        .then(r => r.json())
        .then(data => {
            document.getElementById('ema-equity').textContent =
                `$${data.total_equity.toFixed(2)}`;

            const roiElement = document.getElementById('ema-roi');
            const roiValue = data.roi.toFixed(2);
            roiElement.textContent = `${roiValue > 0 ? '+' : ''}${roiValue}%`;
            roiElement.className = `metric-value ${roiValue >= 0 ? 'profit' : 'loss'}`;

            document.getElementById('ema-positions').textContent =
                `${data.open_positions}/${data.total_pairs}`;
        })
        .catch(err => console.error('Error updating EMA status:', err));

    // Update EMA chart
    updateEMAChart(currentEMASymbol);

    // Update EMA trades
    updateEMATrades();
}

function updateEMAChart(symbol) {
    fetch(`/api/chart/${symbol}`)
        .then(r => r.json())
        .then(data => {
            renderChart(data, 'ema-chart', 'EMA');
        })
        .catch(err => console.error('Error updating EMA chart:', err));
}

function updateEMATrades() {
    fetch('/api/bot/ema/trades')
        .then(r => r.json())
        .then(trades => {
            renderTradesTable(trades, 'ema-trades-table');
        })
        .catch(err => console.error('Error updating EMA trades:', err));
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
        .catch(err => console.error('Error updating comparison:', err));
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
    const trades = data.trades || [];

    // Separate closed candles from current candle
    const closedCandles = candles.filter(c => !c.is_current);
    const currentCandle = candles.find(c => c.is_current);

    // Candlestick trace (closed candles)
    const candlestickTrace = {
        x: closedCandles.map(c => c.timestamp),
        open: closedCandles.map(c => c.open),
        high: closedCandles.map(c => c.high),
        low: closedCandles.map(c => c.low),
        close: closedCandles.map(c => c.close),
        type: 'candlestick',
        name: data.symbol,
        increasing: { line: { color: '#26a69a' } },
        decreasing: { line: { color: '#ef5350' } }
    };

    const traces = [candlestickTrace];

    // Current candle (if exists)
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
            decreasing: { line: { color: '#ef5350', width: 1, dash: 'dot' }, fillcolor: 'rgba(239, 83, 80, 0.3)' }
        });
    }

    // Add MA lines
    traces.push({
        x: closedCandles.map(c => c.timestamp),
        y: closedCandles.map(c => c.ma),
        type: 'scatter',
        mode: 'lines',
        name: 'MA 50',
        line: { color: '#f39c12', width: 1 }
    });

    const layout = {
        height: 500,
        xaxis: { rangeslider: { visible: false } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ecf0f1' }
    };

    Plotly.newPlot(containerId, traces, layout, { responsive: true });
}

// ============================================================================
// TRADES TABLE RENDERING (Shared)
// ============================================================================

function renderTradesTable(trades, containerId) {
    const container = document.getElementById(containerId);

    if (!trades || trades.length === 0) {
        container.innerHTML = '<p style="text-align:center; color:#95a5a6;">No hay trades registrados aún</p>';
        return;
    }

    const html = `
        <table>
            <thead>
                <tr>
                    <th>Fecha</th>
                    <th>Par</th>
                    <th>Tipo</th>
                    <th>Precio</th>
                    <th>Cantidad</th>
                    <th>PnL</th>
                    <th>Razón</th>
                </tr>
            </thead>
            <tbody>
                ${trades.reverse().map(t => `
                    <tr>
                        <td>${new Date(t.timestamp).toLocaleString('es-ES')}</td>
                        <td>${t.symbol}</td>
                        <td class="${t.type === 'buy' ? 'buy-badge' : 'sell-badge'}">${t.type.toUpperCase()}</td>
                        <td>$${parseFloat(t.price).toFixed(2)}</td>
                        <td>${parseFloat(t.qty).toFixed(6)}</td>
                        <td class="${parseFloat(t.pnl) >= 0 ? 'profit' : 'loss'}">
                            ${parseFloat(t.pnl) >= 0 ? '+' : ''}$${parseFloat(t.pnl).toFixed(2)}
                        </td>
                        <td>${t.reason || '-'}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;

    container.innerHTML = html;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function updateLastUpdateTime() {
    const now = new Date();
    document.getElementById('last-update').textContent =
        now.toLocaleTimeString('es-ES');
}
