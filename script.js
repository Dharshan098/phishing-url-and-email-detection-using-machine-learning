document.addEventListener("DOMContentLoaded", function () {
    const chatWindow = document.getElementById("chatWindow");
    const chatBody = document.getElementById("chatBody");
    const chatInput = document.getElementById("chatInput");

    // Function to toggle chat visibility
    window.toggleChat = function () {
        if (chatWindow.classList.contains("active")) {
            chatWindow.classList.remove("active");
            setTimeout(() => {
                chatWindow.style.display = "none";
                chatBody.innerHTML = '<div class="chat-message bot-message">Welcome! How can I help you?</div>';
            }, 300);
        } else {
            chatWindow.style.display = "flex";
            setTimeout(() => {
                chatWindow.classList.add("active");
            }, 10);
        }
    };
});

const API_KEY = "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"; 

// Function to send a message
async function sendMessage() {
    const chatInput = document.getElementById("chatInput");
    const chatBody = document.getElementById("chatBody");
    const message = chatInput.value.trim();

    if (!message) return;

    // Add user message to chat
    const userMessage = document.createElement("div");
    userMessage.className = "chat-message user-message";
    userMessage.textContent = message;
    chatBody.appendChild(userMessage);
    chatInput.value = ""; // Clear input

    // Scroll to the latest message
    chatBody.scrollTop = chatBody.scrollHeight;

    // Add a loading response from the bot
    const botMessage = document.createElement("div");
    botMessage.className = "chat-message bot-message";
    botMessage.textContent = "Thinking...";
    chatBody.appendChild(botMessage);
    chatBody.scrollTop = chatBody.scrollHeight;

    try {
        // Call OpenAI API
        const botResponse = await fetchOpenAIResponse(message);
        botMessage.textContent = botResponse; // Update bot message with response
    } catch (error) {
        botMessage.textContent = "Oops! Something went wrong. Please try again!";
    }

    chatBody.scrollTop = chatBody.scrollHeight;
}

// Function to fetch response from OpenAI API
async function fetchOpenAIResponse(userMessage) {
    const API_URL = "https://api.openai.com/v1/chat/completions";

    const requestOptions = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${API_KEY}`
        },
        body: JSON.stringify({
            model: "gpt-4o-mini",
            messages: [{ role: "user", content: userMessage }]
        })
    };

    const response = await fetch(API_URL, requestOptions);
    if (!response.ok) {
        throw new Error("Failed to fetch response from OpenAI");
    }
    const data = await response.json();
    return data.choices[0].message.content;
}

// Allow "Enter" key to send message
document.getElementById("chatInput").addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});

function googleCSESearch() {
    const query = document.getElementById("searchInput").value.trim();
    if (query) {
        // Uses the built-in search trigger
        google.search.cse.element.getElement('searchresults-only0').execute(query);
    }
}

// Optional: trigger search on Enter
document.getElementById("searchInput").addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
        googleCSESearch();
    }
});