// import { marked } from "https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js";

document.addEventListener('DOMContentLoaded', () => {
    const textarea = document.querySelector('.input-area textarea');
    const sendButton = document.querySelector("#send-btn");
    const newChatBtn = document.querySelector('.new-chat-btn');
    const chatDisplay = document.querySelector('.chat-display');
    const greetingContainer = document.querySelector('.greeting-container');
    const contextMenu = document.getElementById('custom-context-menu');
    
    let isNewLine = false;
    let currentChatId = null;
    let isWaitingForResponse = false;
    let targetChatId = null;

    const pathParts = window.location.pathname.split('/');
    if (pathParts.length > 2 && pathParts[1] === 'chat') {
        currentChatId = pathParts[2];
        if (greetingContainer) greetingContainer.style.display = 'none';
    }

    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            if (!e.shiftKey) {
                e.preventDefault();
                if (!isWaitingForResponse) sendMessage();
            } else {
                isNewLine = true;
            }
        }
    });

    textarea.addEventListener('input', function() {
        if (!isNewLine) return;
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 300) + 'px';
        this.style.overflowY = this.scrollHeight > 300 ? 'scroll' : 'hidden';
        isNewLine = false;
    });

    document.querySelectorAll('.chat-item').forEach(item => {
        item.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            targetChatId = item.getAttribute('href').split('/').pop();
            contextMenu.style.display = 'block';
            contextMenu.style.left = `${e.pageX}px`;
            contextMenu.style.top = `${e.pageY}px`;
        });
    });

    document.addEventListener('click', () => {
        if (contextMenu) contextMenu.style.display = 'none';
    });

    async function handleRename(cid) {
        const newName = prompt("Enter new chat name:");
        if (!newName) return;
        const res = await fetch(`/chat/${cid}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName })
        });
        if (res.ok) window.location.reload();
    }

    async function handleDelete(cid) {
        if (!confirm("Delete this conversation?")) return;
        const res = await fetch(`/chat/${cid}/delete`, { method: 'DELETE' });
        if (res.ok) window.location.href = '/';
    }

    document.querySelectorAll('.opt-rename').forEach(el => el.onclick = (e) => { e.preventDefault(); handleRename(el.closest('.chat-item').getAttribute('href').split('/').pop()); });
    document.querySelectorAll('.opt-delete').forEach(el => el.onclick = (e) => { e.preventDefault(); handleDelete(el.closest('.chat-item').getAttribute('href').split('/').pop()); });
    if(document.querySelector('.context-opt-rename')) document.querySelector('.context-opt-rename').onclick = () => handleRename(targetChatId);
    if(document.querySelector('.context-opt-delete')) document.querySelector('.context-opt-delete').onclick = () => handleDelete(targetChatId);

    async function sendMessage() {
        const message = textarea.value;
        if (isWaitingForResponse) return;

        if (greetingContainer) greetingContainer.style.display = 'none';

        appendMessage("user", message);
        
        textarea.value = '';
        textarea.style.height = 'auto';
        isWaitingForResponse = true;
        sendButton.disabled = true;

        const endpoint = currentChatId ? `/chat/${currentChatId}` : '/chat/new';
        
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            let aiMessageElement = appendMessage("assistant", "");
            const thinkingWrapper = aiMessageElement.querySelector('.thinking-wrapper');
            const thinkingArea = aiMessageElement.querySelector('.thinking-content');
            const contentArea = aiMessageElement.querySelector('.content-text');
            
            let currentThinking = "";
            let currentContent = "";
            let receivedChatId = false;

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (!line) continue;
                    try {
                        const data = JSON.parse(line);
                        
                        if (data.chat_id && !receivedChatId) {
                            currentChatId = data.chat_id;
                            receivedChatId = true;
                            window.history.pushState({}, '', `/chat/${currentChatId}`);
                        }

                        if (data.thinking) {
                            if (thinkingWrapper) thinkingWrapper.style.display = 'block';
                            currentThinking += data.thinking;
                            if (thinkingArea) {
                                thinkingArea.innerHTML = currentThinking.replace(/\n/g, '<br>')
                        }
                        }

                        if (data.content) {
                            if (contentArea.innerText === "Generating...") contentArea.innerText = "";
                            currentContent += data.content;
                        }
                        contentArea.innerHTML = currentContent.replace(/\n/g, '<br>')
                        
                        chatDisplay.scrollTop = chatDisplay.scrollHeight;
                    } catch (e) { console.error("JSON parse error", e); }
                }
            }
        } catch (error) {
            console.error(error);
            const errEl = appendMessage("assistant", "System Error: Failed to connect to the backend...");
            errEl.querySelector('.content-text').classList.add('error');
        } finally {
            isWaitingForResponse = false;
            sendButton.disabled = false;
            textarea.focus();
        }
    }

    function appendMessage(role, text) {
        const template = document.getElementById('message-template');
        const clone = template.content.cloneNode(true);
        const msgDiv = clone.querySelector('.message');
        const contentDiv = clone.querySelector('.content-text');
        
        msgDiv.classList.add(`${role}-message`);
        
        if (role === 'assistant') {
            contentDiv.innerHTML = text.replace(/\n/g, '<br>') ? text.replace(/\n/g, '<br>') : "Generating...";
        } else {
            if (!text) return;
            const cleanText = text.replace(/\n/g, '<br>');
            contentDiv.innerHTML = cleanText;
        }

        chatDisplay.appendChild(clone);
        chatDisplay.scrollTop = chatDisplay.scrollHeight;
        return chatDisplay.lastElementChild;
    }

    sendButton.addEventListener('click', sendMessage);
    if (newChatBtn) newChatBtn.addEventListener('click', () => window.location.href = '/');
});

const settingsToggle = document.getElementById('settings-toggle');
const settingsPanel = document.getElementById('settings-panel');

function closeSettings() {
    settingsPanel.classList.remove('active');
    settingsToggle.classList.remove('open');
    settingsToggle.innerText = '⚙';
    settingsToggle.style.background = '#363636';
}

function openSettings() {
    settingsPanel.classList.add('active');
    settingsToggle.classList.add('open');
    settingsToggle.innerText = '✕';
    settingsToggle.style.background = '#801c1c';
}

settingsToggle.addEventListener('click', (e) => {
    e.stopPropagation(); 
    
    if (settingsPanel.classList.contains('active')) {
        closeSettings();
    } else {
        openSettings();
    }
});

document.addEventListener('click', (e) => {
    if (settingsPanel.classList.contains('active')) {
        if (!settingsPanel.contains(e.target) && !settingsToggle.contains(e.target)) {
            closeSettings();
        }
    }
});

async function updateModelStatuses() {
    try {
        const response = await fetch('/api/models-status');
        if (!response.ok) return;
        const models = await response.json();

        models.forEach(model => {
            const container = document.querySelector(`.model-row[data-model="${model.name}"]`);
            
            if (container) {
                const statusEl = container.querySelector('.js-status');
                if (statusEl && statusEl.innerText !== model.state) {
                    statusEl.innerText = model.state;
                }
            }
        });
    } catch (err) {
        console.error("Status check failed. Pulse backend might be busy.");
    }
}

setInterval(updateModelStatuses, 500);