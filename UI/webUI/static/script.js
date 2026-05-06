document.addEventListener('DOMContentLoaded', () => {
    const sendButton = document.querySelector("#send-btn");
    const newChatBtn = document.querySelector('.new-chat-btn');
    const chatDisplay = document.querySelector('.chat-display');
    const greetingContainer = document.querySelector('.greeting-container');
    const contextMenu = document.getElementById('custom-context-menu');
    const textarea = document.getElementById('user-input');
    const para = document.querySelector(".paragraph-greeting");
    const tempToggle = document.querySelector('.temp-chat-btn');
    const deleteModal = document.getElementById('delete-modal');
    const confirmDeleteBtn = document.getElementById('confirm-delete');
    const cancelDeleteBtn = document.getElementById('cancel-delete');
    
    let isTempMode = false;
    let isNewLine = false;
    let currentChatId = null;
    let isWaitingForResponse = false;
    let targetChatId = null;

    const pathParts = window.location.pathname.split('/');
    if (pathParts.length > 2 && pathParts[1] === 'chat') {
        currentChatId = pathParts[2];
        if (greetingContainer) greetingContainer.style.display = 'none';
    }

    if (tempToggle) {
        tempToggle.addEventListener('click', (e) => {
            e.preventDefault();
            isTempMode = !isTempMode;
            tempToggle.classList.toggle('active', isTempMode);
            
            if (isTempMode) {
                tempToggle.innerText = "🗨️ Normal chat";
                textarea.placeholder = "Temporary Chat: Is there anything I can help with?";
                if (para) para.innerHTML = "<strong>Temporary Chat:</strong> History won't be saved.";
                document.title = "Temporary Chat";
                
            } else {
                textarea.placeholder = "Is there anything I can help with?";
                tempToggle.innerText = "💬 Temp chat";
                
                if (para) para.innerText = "How can I help you?";
                document.title = "New Chat";
                window.location.href = '/'
            }
            textarea.focus();
        });
    }

    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isWaitingForResponse) sendMessage();
        } else if (e.key === 'Enter' && e.shiftKey) {
            isNewLine = true;
        }
    });

    textarea.addEventListener('input', function() {
        if (!isNewLine) return;
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 300) + 'px';
        this.style.overflowY = this.scrollHeight > 300 ? 'scroll' : 'hidden';
        isNewLine = false;
    });

    document.addEventListener('contextmenu', (e) => {
        const item = e.target.closest('.chat-item');
        if (item) {
            e.preventDefault();
            targetChatId = item.getAttribute('href').split('/').pop();
            openMenu(e);
        }
    });

    document.addEventListener('click', (e) => {
        if (contextMenu && !contextMenu.contains(e.target)) {
            contextMenu.style.display = 'none';
        }

        if (e.target.classList.contains('menu-dots')) {
            e.preventDefault();
            e.stopPropagation();
            const item = e.target.closest('.chat-item');
            targetChatId = item.getAttribute('href').split('/').pop();
            openMenu(e);
        }

        if (e.target.classList.contains('context-opt-rename')) {
            handleRename(targetChatId);
        }
        if (e.target.classList.contains('context-opt-delete')) {
            openDeleteModal(targetChatId);
        }
    });

    function openMenu(e) {
        contextMenu.style.display = 'block';
        contextMenu.style.left = `${e.pageX}px`;
        contextMenu.style.top = `${e.pageY}px`;
    }

    async function handleRename(cid) {
        const chatItem = document.querySelector(`.chat-item[href*="${cid}"]`);
        if (!chatItem) return;
        const nameSpan = chatItem.querySelector('.chat-name');
        const originalName = nameSpan.innerText;

        const input = document.createElement('input');
        input.type = 'text';
        input.value = originalName;
        input.classList.add('rename-input');
        nameSpan.replaceWith(input);
        input.focus();
        input.select();
    
        const saveRename = async () => {
            const newName = input.value.trim();
            if (newName && newName !== originalName) {
                const res = await fetch(`/chat/${cid}/rename`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: newName })
                });
                if (res.ok) {
                    const newSpan = document.createElement('span');
                    newSpan.classList.add('chat-name');
                    newSpan.innerText = newName;
                    input.replaceWith(newSpan);
                    return;
                }
            }
            const oldSpan = document.createElement('span');
            oldSpan.classList.add('chat-name');
            oldSpan.innerText = originalName;
            input.replaceWith(oldSpan);
        };
    
        input.addEventListener('blur', saveRename);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') input.blur();
            if (e.key === 'Escape') { input.value = originalName; input.blur(); }
        });
    }

    function openDeleteModal(cid) {
        targetChatId = cid;
        deleteModal.style.display = 'flex';
    }

    cancelDeleteBtn.addEventListener('click', () => deleteModal.style.display = 'none');
    window.addEventListener('click', (e) => { if (e.target === deleteModal) deleteModal.style.display = 'none'; });

    confirmDeleteBtn.addEventListener('click', async () => {
        if (!targetChatId) return;
        confirmDeleteBtn.disabled = true;
        confirmDeleteBtn.innerText = "Deleting...";

        const res = await fetch(`/chat/${targetChatId}/delete`, { method: 'DELETE' });
        if (res.ok) {
            window.location.href = '/';
        } else {
            alert("Failed to delete chat.");
            confirmDeleteBtn.disabled = false;
            confirmDeleteBtn.innerText = "Delete";
            deleteModal.style.display = 'none';
        }
    });

    async function sendMessage() {
        const message = textarea.value.trim();
        if (isWaitingForResponse) return;

        if (greetingContainer) greetingContainer.style.display = 'none';

        if (message.trim()) appendMessage("user", message);

        textarea.value = '';
        textarea.style.height = 'auto';
        isWaitingForResponse = true;
        sendButton.disabled = true;

        let endpoint = currentChatId ? `/chat/${currentChatId}` : (isTempMode ? '/temp_chat' : '/chat/new');
        
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
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                buffer += chunk
                const lines = buffer.split('\n');

                buffer = lines.pop();
                
                for (const line of lines) {
                    if (!line) continue;
                    try {
                        const data = JSON.parse(line);
                        
                        if (data.chat_id && !receivedChatId) {
                            currentChatId = data.chat_id;
                            receivedChatId = true;
                            if (!isTempMode) {
                                window.history.pushState({}, '', `/chat/${currentChatId}`);
                                if (endpoint === '/chat/new') addChatToSidebar(currentChatId);
                            }
                        }

                        if (data.thinking) {
                            if (thinkingWrapper) thinkingWrapper.style.display = 'block';
                            currentThinking += data.thinking;
                            if (thinkingArea) thinkingArea.innerHTML = currentThinking.replace(/\n/g, '<br>');
                        }

                        if (data.content) {
                            if (contentArea.innerText === "Generating...") contentArea.innerText = "";
                            currentContent += data.content;
                            contentArea.innerHTML = currentContent.replace(/\n/g, '<br>');
                        }
                        chatDisplay.scrollTop = chatDisplay.scrollHeight;
                    } catch (e) {console.error(e)}
                }
            }
            if (buffer.trim()) {
                try {
                    const data = JSON.parse(buffer);
                    if (data.chat_id && !receivedChatId) {
                        currentChatId = data.chat_id;
                        receivedChatId = true;
                        if (!isTempMode) {
                            window.history.pushState({}, '', `/chat/${currentChatId}`);
                            if (endpoint === '/chat/new') addChatToSidebar(currentChatId);
                        }
                    }

                    if (data.thinking) {
                        if (thinkingWrapper) thinkingWrapper.style.display = 'block';
                        currentThinking += data.thinking;
                        if (thinkingArea) thinkingArea.innerHTML = currentThinking.replace(/\n/g, '<br>');
                    }

                    if (data.content) {
                        if (contentArea.innerText === "Generating...") contentArea.innerText = "";
                        currentContent += data.content;
                        contentArea.innerHTML = currentContent.replace(/\n/g, '<br>');
                    }
                    chatDisplay.scrollTop = chatDisplay.scrollHeight;
                } catch (e) {}
            }
        } catch (error) {
            appendMessage("assistant", "System Error: Failed to connect...");
        } finally {
            isWaitingForResponse = false;
            sendButton.disabled = false;
            textarea.focus();
        }
    }

    function addChatToSidebar(cid) {
        const chatList = document.querySelector('.chat-list');
        if (!chatList) return;
        document.querySelectorAll('.chat-item').forEach(el => el.classList.remove('active'));
    
        const newChatItem = document.createElement('a');
        newChatItem.href = `/chat/${cid}`;
        newChatItem.className = 'chat-item active';
        newChatItem.innerHTML = `<span class="chat-name">New Chat</span><button class="menu-dots">⋮</button>`;
        chatList.prepend(newChatItem);
    }

    function appendMessage(role, text) {
        const template = document.getElementById('message-template');
        const clone = template.content.cloneNode(true);
        const msgDiv = clone.querySelector('.message');
        const contentDiv = clone.querySelector('.content-text');
        
        msgDiv.classList.add(`${role}-message`);
        contentDiv.innerHTML = text ? text.replace(/\n/g, '<br>') : (role === 'assistant' ? "Generating..." : "");

        chatDisplay.appendChild(clone);
        chatDisplay.scrollTop = chatDisplay.scrollHeight;
        return chatDisplay.lastElementChild;
    }

    sendButton.addEventListener('click', sendMessage);
    if (newChatBtn) newChatBtn.addEventListener('click', () => window.location.href = '/');
});

const settingsToggle = document.getElementById('settings-toggle');
const settingsPanel = document.getElementById('settings-panel');

if (settingsToggle) {
    settingsToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        const isActive = settingsPanel.classList.toggle('active');
        settingsToggle.classList.toggle('open', isActive);
        settingsToggle.innerText = isActive ? '✕' : '⚙';
        settingsToggle.style.background = isActive ? '#801c1c' : '#363636';
    });
}

document.addEventListener('click', (e) => {
    if (settingsPanel && settingsPanel.classList.contains('active')) {
        if (!settingsPanel.contains(e.target) && !settingsToggle.contains(e.target)) {
            settingsPanel.classList.remove('active');
            settingsToggle.classList.remove('open');
            settingsToggle.innerText = '⚙';
            settingsToggle.style.background = '#363636';
        }
    }
});

class ToastSystem {
    constructor() {
        this.container = document.getElementById('toast-container');
        this.queue = [];
        this.isProcessing = false;
        this.MAX_VISIBLE_TOASTS = 3;
        this.SPAM_THRESHOLD = 400; // in ms

        this.eventMap = {
            "error": "error",
            "summarisation failed": "error",
            "warn": "warn",
            "success": "success",
            "initialised": "success",
            "session created": "success",
        };
    }

    show(eventConst, customMsg = null, typeOverride = null) {
        this.queue.push({ eventConst, customMsg, typeOverride });
        this.processQueue();
    }

    async processQueue() {
        if (this.isProcessing || this.queue.length === 0) return;
        this.isProcessing = true;

        while (this.queue.length > 0) {
            const { eventConst, customMsg, typeOverride } = this.queue.shift();
            
            if (this.queue.length > 10 && !typeOverride && !["error", "warn"].includes(this.eventMap[eventConst])) {
                continue; 
            }

            this.renderToast(eventConst, customMsg, typeOverride);
            
            await new Promise(resolve => setTimeout(resolve, this.SPAM_THRESHOLD));
        }

        this.isProcessing = false;
    }

    renderToast(eventConst, customMsg, typeOverride) {
        const type = typeOverride || this.eventMap[eventConst] || "info";
        const msg = customMsg || this.capitalize(eventConst);
        
        const visibleToasts = this.container.querySelectorAll('.toast:not(.hiding)');
        if (visibleToasts.length >= this.MAX_VISIBLE_TOASTS) {
            this.hide(visibleToasts[0]);
        }
        
        const icons = {
            info: 'ℹ️',
            error: '🟥',
            warn: '⚠️',
            success: '✅'
        };

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${icons[type]}</span>
            <div class="toast-content">
                <small style="opacity: 0.7; display: block; font-size: 0.7rem; text-transform: uppercase;">${eventConst}</small>
                <span class="toast-message">${msg}</span>
            </div>
        `;

        this.container.appendChild(toast);

        setTimeout(() => toast.classList.add('show'), 10);

        setTimeout(() => this.hide(toast), 3000);
    }

    hide(toast) {
        if (!toast || toast.classList.contains('hiding')) return;
        
        toast.classList.add('hiding');
        toast.classList.remove('show');
        
        toast.addEventListener('transitionend', () => {
            toast.remove();
        }, { once: true });
    }

    capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
}

const toasts = new ToastSystem();

const eventSource = new EventSource('/events/notifications');

eventSource.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        toasts.show(data.type, data.message);
    } catch (e) {
        console.error("Failed to parse event data", e);
    }
};

eventSource.onerror = () => {
    console.warn("EventSource failed. Server might be restarting...");
};

async function updateModelStatuses() {
    try {
        const response = await fetch('/api/models-status');
        if (!response.ok) return;
        const models = await response.json();
        models.forEach(model => {
            const container = document.querySelector(`.model-row[data-model="${model.name}"]`);
            if (container) {
                const statusEl = container.querySelector('.js-status');
                if (statusEl) statusEl.innerText = model.state;
            }
        });
    } catch (err) {console.error(err)}
}
setInterval(updateModelStatuses, 750);