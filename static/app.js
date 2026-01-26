async function sendMessage() {
    const input = document.getElementById("user-input");
    const text = input.value.trim();
    if (!text) return;

    appendMessage("user", text);
    input.value = "";

    const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            question: text,
            k: 5,
            score_threshold: null,
            doc_type: null,
            filename: null
        })
    });

    const data = await res.json();
    appendMessage("ai", data.answer);
}

function appendMessage(role, text) {
    const chat = document.getElementById("chat-window");
    const msg = document.createElement("div");
    msg.className = `message ${role}`;
    msg.innerHTML = `<div class="bubble">${text}</div>`;
    chat.appendChild(msg);
    chat.scrollTop = chat.scrollHeight;
}
