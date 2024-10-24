async function fetchStockData() {
    const ticker = document.getElementById('ticker').value;
    const response = await fetch('/get_stock_data/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticker: ticker }),
    });
    const data = await response.json();
    if (response.ok) {
        document.getElementById('result').innerText = 
            `Current Price: ${data.currentPrice}, Open Price: ${data.openPrice}`;
    } else {
        document.getElementById('result').innerText = 'Error fetching data';
    }
}
