const predictionForm = document.getElementById('predictionForm');

predictionForm.addEventListener('submit', (event) => {
    event.preventDefault();

    const formData = new FormData(predictionForm);
    const data = {};

    for (const [key, value] of formData.entries()) {
        data[key] = value;
    }

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('prediction').textContent = `Predicted life expectancy: ${data.prediction}`;
    })
    .catch(error => {
        console.error('Error:', error);
        // Handle error, display error message to user
    });
});
