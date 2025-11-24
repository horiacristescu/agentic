// Global state
let allMessages = [];

// Load messages from API
async function loadMessages() {
    const timeline = document.getElementById('timeline');
    timeline.innerHTML = '<div style="padding: 40px; text-align: center; color: #858585;">Loading messages...</div>';
    
    try {
        const response = await fetch('/api/messages');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        allMessages = await response.json();
        renderTimeline(allMessages);
    } catch (error) {
        console.error('Error loading messages:', error);
        timeline.innerHTML = `
            <div style="color: #f48771; padding: 20px;">
                <h3>Error loading messages</h3>
                <p>${error.message}</p>
                <p style="font-size: 12px; color: #858585; margin-top: 10px;">
                    Check the browser console (F12) for details.
                </p>
            </div>
        `;
    }
}

// Load agent state
async function loadState() {
    try {
        const response = await fetch('/api/state');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const state = await response.json();
        
        document.getElementById('turn-count').textContent = state.turn_count;
        document.getElementById('message-count').textContent = state.message_count;
        document.getElementById('tokens-used').textContent = state.tokens_used.toLocaleString();
        document.getElementById('tools').textContent = state.tools.join(', ');
    } catch (error) {
        console.error('Error loading state:', error);
    }
}

// Format tool call for display
function formatToolCall(tc) {
    const params = [];
    for (const [key, value] of Object.entries(tc.args)) {
        if (typeof value === 'string') {
            params.push(`${key}="${value}"`);
        } else {
            params.push(`${key}=${value}`);
        }
    }
    return `${tc.tool}(${params.join(', ')})`;
}

// Escape HTML for safe display
function escapeHtml(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// Render timeline of messages
function renderTimeline(messages) {
    console.log(`[DEBUG] renderTimeline called with ${messages.length} messages`);
    console.time('renderTimeline execution');
    
    const timeline = document.getElementById('timeline');
    const showJson = document.getElementById('json-toggle').checked;
    timeline.innerHTML = '';
    
    console.time('Building HTML');
    messages.forEach((msg, i) => {
        const div = document.createElement('div');
        div.className = `message ${msg.role}`;
        
        let html = `
            <div class="message-header">
                <span class="message-role">${msg.role}</span>
                <span class="tokens">${msg.tokens ? `${msg.tokens.in}/${msg.tokens.out} tokens` : ''}</span>
            </div>
        `;
        
        // Handle system prompt specially - simplified for performance
        if (msg.role === 'system') {
            if (showJson) {
                html += `<div class="message-content">${escapeHtml(msg.content)}</div>`;
            } else {
                // Simple summary instead of expensive parsing
                const toolCount = (msg.content.match(/Tool Name:/g) || []).length;
                html += `
                    <div class="message-content" style="color: #858585; font-size: 12px;">
                        System prompt loaded (${toolCount} tools available)
                    </div>
                `;
            }
        }
        // Handle assistant messages specially - parse JSON to show reasoning and result
        else if (msg.role === 'assistant') {
            if (showJson) {
                html += `<div class="message-content">${escapeHtml(msg.content)}</div>`;
            } else {
                try {
                    const parsed = JSON.parse(msg.content);
                    
                    // Show reasoning
                    if (parsed.reasoning) {
                        html += `<div class="message-content">💭 ${escapeHtml(parsed.reasoning)}</div>`;
                    }
                    
                    // Tool calls are now shown with their results, not here
                    
                    // Show final result if finished
                    if (parsed.is_finished && parsed.result) {
                        html += `
                            <div class="message-content" style="margin: 10px -15px 0 -15px; padding: 15px; background: rgba(78, 201, 176, 0.1); border-radius: 0;">✅ ${escapeHtml(parsed.result)}</div>
                        `;
                    }
                } catch (e) {
                    // Fallback to raw content if parsing fails
                    html += `<div class="message-content">${escapeHtml(msg.content)}</div>`;
                }
            }
        }
        // Show content for non-assistant messages (but skip tool messages, they're formatted below)
        else if (!msg.tool_call_id && (showJson || msg.role !== 'assistant')) {
            html += `<div class="message-content">${escapeHtml(msg.content)}</div>`;
        }
        
        if (msg.tool_call_id) {
            // Find the tool call from the previous assistant message
            let toolCallInfo = null;
            for (let j = i - 1; j >= 0; j--) {
                const prevMsg = messages[j];
                if (prevMsg.role === 'assistant' && prevMsg.tool_calls) {
                    toolCallInfo = prevMsg.tool_calls.find(tc => tc.id === msg.tool_call_id);
                    if (toolCallInfo) break;
                }
            }
            
            // Show tool call + result grouped together
            html += '<div class="tool-calls">';
            
            // Show the tool call first
            if (toolCallInfo) {
                const formatted = formatToolCall(toolCallInfo);
                html += `
                    <div class="tool-call-header">🔧 [${msg.tool_call_id}] ${escapeHtml(formatted)}</div>
                `;
            } else {
                html += `
                    <div class="tool-call-header">🔧 [${msg.tool_call_id}] ${msg.name}</div>
                `;
            }
            
            // Show the result
            html += `
                <div class="tool-call-header" style="margin-top: 8px;">📎 Result:</div>
                    <pre style="white-space: pre-wrap; margin: 8px 0; font-size: 12px; line-height: 1.4;">${escapeHtml(msg.content)}</pre>
            `;
            
            html += '</div>';
        }
        
        // Format aggregated tool results for user messages
        if (msg.role === 'user' && !showJson && msg.content.includes('call_')) {
            // Check if this looks like tool results
            if (msg.content.match(/call_\d+:/)) {
                html = `
                    <div class="message-header">
                        <span class="message-role">tool results</span>
                    </div>
                    <div class="tool-calls">
                `;
                
                // Split by call_X: pattern to get each tool result block
                const parts = msg.content.split(/(call_\d+:)/);
                for (let i = 1; i < parts.length; i += 2) {
                    const callId = parts[i].replace(':', ''); // "call_1"
                    const result = parts[i + 1] || '';
                    
                    // Show full multi-line result
                    html += `
                        <div class="tool-call">
                            <div class="tool-call-header">✓ [${callId}]</div>
                            <pre style="white-space: pre-wrap; margin: 8px 0; font-size: 12px; line-height: 1.4;">${escapeHtml(result.trim())}</pre>
                        </div>
                    `;
                }
                
                html += '</div>';
            }
        }
        
        if (msg.error_code) {
            html += `<div class="error">❌ Error: ${msg.error_code}</div>`;
        }
        
        div.innerHTML = html;
        timeline.appendChild(div);
    });
    console.timeEnd('Building HTML');
    
    console.time('Scrolling');
    timeline.scrollTop = timeline.scrollHeight;
    console.timeEnd('Scrolling');
    
    console.timeEnd('renderTimeline execution');
}

// Send message via SSE stream
async function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    const sendBtn = document.getElementById('send-btn');
    sendBtn.disabled = true;
    input.disabled = true;
    sendBtn.textContent = 'Thinking...';
    
    input.value = '';
    
    try {
        // Send message to start SSE stream
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        // Read SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const {done, value} = await reader.read();
            
            if (done) break;
            
            buffer += decoder.decode(value, {stream: true});
            
            // Process complete SSE messages
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep incomplete line in buffer
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    
                    if (data === '[DONE]') {
                        // Stream complete
                        await loadState(); // Update state counters
                        sendBtn.textContent = 'Send';
                        continue;
                    }
                    
                    try {
                        const msg = JSON.parse(data);
                        
                        if (msg.error) {
                            alert('Error: ' + msg.message);
                            break;
                        }
                        
                        // Add message to allMessages and re-render
                        allMessages.push(msg);
                        renderTimeline(allMessages);
                        
                        // Scroll to bottom
                        const timeline = document.getElementById('timeline');
                        timeline.scrollTop = timeline.scrollHeight;
                        
                    } catch (e) {
                        console.error('Failed to parse SSE message:', data, e);
                    }
                }
            }
        }
        
    } catch (error) {
        console.error('Error sending message:', error);
        alert('Error: ' + error.message);
    } finally {
        sendBtn.disabled = false;
        input.disabled = false;
        sendBtn.textContent = 'Send';
        input.focus();
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Re-render when toggle changes
    document.getElementById('json-toggle').addEventListener('change', () => {
        if (allMessages.length > 0) {
            renderTimeline(allMessages);
        }
    });
    
    // Handle Enter key (send) and Shift+Enter (new line)
    const input = document.getElementById('message-input');
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Send button click
    document.getElementById('send-btn').addEventListener('click', sendMessage);
    
    // Initial load
    console.log('[DEBUG] Starting initial load...');
    console.time('Total page load');
    loadMessages().then(() => {
        console.timeEnd('Total page load');
    });
    loadState();
    
    // Auto-refresh state every 2 seconds
    setInterval(loadState, 2000);
});


