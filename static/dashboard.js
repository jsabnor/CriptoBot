// Configuration
const SYMBOLS = ['ETH', 'XRP', 'BNB', 'SOL'];
const REFRESH_INTERVAL = 30000; // 30 seconds

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function () {
    console.log('Dashboard initialized');
    updateDashboard();
    setInterval(updateDashboard, REFRESH_INTERVAL);
});

// Main update function
function updateDashboard() {
    updateStatus();
    updateCharts();
    updateTrades();
    updateLastUpdateTime();
}

// Update bot status and metrics
function updateStatus() {
    fetch('/api/status')
        .then(r => r.json())
        .then(data => {
            // Total Equity
            document.getElementById('total-equity').textContent =
                `$${data.total_equity.toFixed(2)}`;

            // ROI with color
            const roiElement = document.getElementById('roi');
            const roiValue = data.roi.toFixed(2);
            roiElement.textContent = `${roiValue > 0 ? '+' : ''}${roiValue}%`;
            roiElement.className = `metric-value ${roiValue >= 0 ? 'profit' : 'loss'}`;

            // Positions
            document.getElementById('positions').textContent =
                `${data.open_positions}/${data.total_pairs}`;

            // Mode
            document.getElementById('bot-mode').textContent =
                data.mode.toUpperCase();
            const trades = data.trades;

            // Candlestick trace
            const candlestick = {
                x: candles.map(c => c.timestamp),
                open: candles.map(c => c.open),
                high: candles.map(c => c.high),
                low: candles.map(c => c.low),
                close: candles.map(c => c.close),
                type: 'candlestick',
                name: symbol,
                increasing: { line: { color: '#26a69a' } },
                decreasing: { line: { color: '#ef5350' } }
            };

            // MA50 line
            const ma50 = {
                x: candles.map(c => c.timestamp),
                y: candles.map(c => c.ma),
                type: 'scatter',
                mode: 'lines',
                name: 'MA50',
                line: { color: '#ff6b35', width: 1.5 },
                yaxis: 'y'
            };

            // MA200 line
            const ma200 = {
                x: candles.map(c => c.timestamp),
                y: candles.map(c => c.long_ma),
                type: 'scatter',
                mode: 'lines',
                name: 'MA200',
                line: { color: '#4ecdc4', width: 1.5 },
                yaxis: 'y'
            };

            const traces = [candlestick, ma50, ma200];

            // Buy markers
            const buys = trades.filter(t => t.type === 'buy');
            if (buys.length > 0) {
                traces.push({
                    x: buys.map(t => t.timestamp),
                    y: buys.map(t => t.price),
                    mode: 'markers',
                    name: 'Compra',
                    marker: {
                        color: '#26a69a',
                        size: 12,
                        symbol: 'triangle-up',
                        line: { color: '#fff', width: 1 }
                    },
                    yaxis: 'y'
                });
            }

            // Sell markers
            const sells = trades.filter(t => t.type === 'sell');
            if (sells.length > 0) {
                traces.push({
                    x: sells.map(t => t.timestamp),
                    y: sells.map(t => t.price),
                    mode: 'markers',
                    name: 'Venta',
                    marker: {
                        color: '#ef5350',
                        size: 12,
                        symbol: 'triangle-down',
                        line: { color: '#fff', width: 1 }
                    },
                    yaxis: 'y'
                });
            }

            const layout = {
                title: {
                    text: `${symbol}/USDT - 4h`,
                    font: { color: '#c9d1d9', size: 16 }
                },
                xaxis: {
                    rangeslider: { visible: false },
                    gridcolor: '#30363d',
                    color: '#8b949e'
                },
                yaxis: {
                    title: 'Price (USDT)',
                    gridcolor: '#30363d',
                    color: '#8b949e'
                },
                plot_bgcolor: '#0d1117',
                paper_bgcolor: '#161b22',
                font: { color: '#c9d1d9' },
                margin: { l: 50, r: 50, t: 40, b: 40 },
                legend: {
                    x: 0,
                    y: 1,
                    bgcolor: 'rgba(0,0,0,0.3)'
                },
                hovermode: 'x unified'
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['select2d', 'lasso2d', 'toggleSpikelines']
            };

            Plotly.newPlot(
                `chart-${symbol.toLowerCase()}`,
                traces,
                layout,
                config
            );
        }

// Update trades table
function updateTrades() {
                fetch('/api/trades')
                    .then(r => r.json())
                    .then(trades => {
                        if (trades.length === 0) {
                            return;
                        }

                        const tbody = document.getElementById('trades-body');
                        tbody.innerHTML = '';

                        // Update total trades count
                        const buyTrades = trades.filter(t => t.type === 'buy').length;
                        document.getElementById('total-trades').textContent = buyTrades;

                        // Show most recent trades first
                        trades.reverse().slice(0, 20).forEach(trade => {
                            const row = document.createElement('tr');

                            const typeClass = trade.type === 'buy' ? 'buy' : 'sell';
                            const pnlClass = trade.pnl >= 0 ? 'profit' : 'loss';

                            row.innerHTML = `
                    <td>${new Date(trade.timestamp).toLocaleString()}</td>
                    <td><strong>${trade.symbol.replace('/USDT', '')}</strong></td>
                    <td class="${typeClass}">${trade.type.toUpperCase()}</td>
                    <td>$${parseFloat(trade.price).toFixed(4)}</td>
                    <td>${parseFloat(trade.qty).toFixed(6)}</td>
                    <td>${trade.reason || '-'}</td>
                    <td class="${pnlClass}">
                        ${trade.pnl !== 0 ? (trade.pnl > 0 ? '+' : '') + trade.pnl.toFixed(2) : '-'}
                    </td>
                `;

                            tbody.appendChild(row);
                        });
                    })
                    .catch(err => console.error('Error updating trades:', err));
            }

// Update last update time
function updateLastUpdateTime() {
                const now = new Date();
                document.getElementById('last-update').textContent =
                    now.toLocaleTimeString();
            }
