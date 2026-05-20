function summarize() {
    let text = document.getElementById("inputText").value;

    fetch('/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerText = data.summary;
    });
}