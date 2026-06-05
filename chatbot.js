document.addEventListener("DOMContentLoaded", () => {
    // Inject FontAwesome if not present for icons
    if (!document.querySelector('link[href*="font-awesome"]')) {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css";
        document.head.appendChild(link);
    }

    // Create Chatbot UI Styles
    const style = document.createElement("style");
    style.innerHTML = `
      #chat-widget-container {
          position: fixed;
          bottom: 30px;
          right: 30px;
          z-index: 9999;
          font-family: inherit;
      }

      #chat-bubble {
          width: 65px;
          height: 65px;
          background: linear-gradient(135deg, var(--brand-copper, #c58255), #e09b6e);
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          box-shadow: 0 8px 25px rgba(197, 130, 85, 0.4);
          transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
      }

      #chat-bubble:hover {
          transform: scale(1.1);
      }

      #chat-bubble i {
          color: #fff;
          font-size: 28px;
          transition: transform 0.3s ease;
      }

      #chat-window {
          position: absolute;
          bottom: 85px;
          right: 0;
          width: 380px;
          height: 550px;
          background-color: var(--brand-darker, #000c0d);
          border: 1px solid rgba(197, 130, 85, 0.2);
          border-radius: 20px;
          box-shadow: 0 15px 40px rgba(0, 0, 0, 0.5);
          display: flex;
          flex-direction: column;
          overflow: hidden;
          opacity: 0;
          pointer-events: none;
          transform: translateY(20px);
          transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
      }

      #chat-window.active {
          opacity: 1;
          pointer-events: auto;
          transform: translateY(0);
      }

      .chat-header {
          background: linear-gradient(135deg, var(--brand-dark, #001a1c), #000c0d);
          padding: 20px;
          border-bottom: 1px solid rgba(197, 130, 85, 0.3);
          display: flex;
          align-items: center;
          justify-content: space-between;
      }

      .chat-header-info {
          display: flex;
          align-items: center;
          gap: 15px;
      }

      .chat-avatar {
          width: 45px;
          height: 45px;
          background: var(--brand-copper, #c58255);
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-size: 20px;
          box-shadow: 0 0 15px rgba(197, 130, 85, 0.5);
      }

      .chat-title h3 {
          color: #fff;
          margin: 0;
          font-size: 1.1rem;
          font-weight: 600;
      }

      .chat-title p {
          color: var(--brand-copper, #c58255);
          margin: 3px 0 0 0;
          font-size: 0.8rem;
          font-weight: 500;
      }

      .close-btn {
          background: none;
          border: none;
          color: rgba(255, 255, 255, 0.6);
          font-size: 20px;
          cursor: pointer;
          transition: color 0.3s;
      }

      .close-btn:hover {
          color: #fff;
      }

      .chat-messages {
          flex: 1;
          padding: 20px;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 15px;
          scroll-behavior: smooth;
      }

      .chat-messages::-webkit-scrollbar {
          width: 6px;
      }

      .chat-messages::-webkit-scrollbar-thumb {
          background: rgba(197, 130, 85, 0.3);
          border-radius: 6px;
      }

      .msg {
          max-width: 80%;
          padding: 12px 16px;
          border-radius: 18px;
          font-size: 0.95rem;
          line-height: 1.5;
          word-wrap: break-word;
          position: relative;
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
      }

      .msg.bot {
          align-self: flex-start;
          background: rgba(255, 255, 255, 0.05);
          color: #fff;
          border-bottom-left-radius: 4px;
          border: 1px solid rgba(255, 255, 255, 0.1);
      }

      .msg.user {
          align-self: flex-end;
          background: var(--brand-copper, #c58255);
          color: var(--brand-darker, #000c0d);
          font-weight: 500;
          border-bottom-right-radius: 4px;
      }

      .chat-input-area {
          padding: 20px;
          background: rgba(0,0,0,0.2);
          border-top: 1px solid rgba(197, 130, 85, 0.2);
          display: flex;
          gap: 10px;
          align-items: center;
      }

      .chat-input-area input {
          flex: 1;
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.1);
          color: #fff;
          padding: 14px 18px;
          border-radius: 30px;
          font-size: 0.95rem;
          outline: none;
          transition: border-color 0.3s;
      }

      .chat-input-area input::placeholder {
          color: rgba(255, 255, 255, 0.4);
      }

      .chat-input-area input:focus {
          border-color: var(--brand-copper, #c58255);
      }

      .send-btn {
          width: 45px;
          height: 45px;
          background: var(--brand-copper, #c58255);
          border: none;
          border-radius: 50%;
          color: var(--brand-darker, #000c0d);
          font-size: 18px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.3s ease;
          box-shadow: 0 4px 10px rgba(197, 130, 85, 0.3);
      }

      .send-btn:hover {
          transform: translateY(-2px);
          background: #d8966b;
      }
      
      .send-btn:disabled {
          background: #7a5439;
          cursor: not-allowed;
          transform: none;
      }

      .typing-indicator {
          display: flex;
          gap: 5px;
          padding: 12px 16px;
          align-self: flex-start;
          background: transparent;
      }

      .typing-indicator span {
          width: 8px;
          height: 8px;
          background-color: rgba(255,255,255,0.4);
          border-radius: 50%;
          animation: typing 1.4s infinite ease-in-out both;
      }

      .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
      .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }

      @keyframes typing {
          0%, 80%, 100% { transform: scale(0); }
          40% { transform: scale(1); }
      }
      
      /* Markdown basic styling for bot responses */
      .msg.bot code { background: rgba(0,0,0,0.3); padding: 2px 5px; border-radius: 4px; font-family: monospace; color: #e09b6e;}
      .msg.bot p { margin: 0 0 10px 0; }
      .msg.bot p:last-child { margin: 0; }
      .msg.bot ul { margin: 0; padding-left: 20px; }
      
      @media (max-width: 500px) {
          #chat-window {
              position: fixed;
              top: 0;
              left: 0;
              width: 100vw;
              height: 100vh;
              bottom: auto;
              right: auto;
              border-radius: 0;
              border: none;
          }
      }
  `;
    document.head.appendChild(style);

    // Parse basic markdown to HTML
    const parseMarkdown = (text) => {
        let html = text.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
        html = html.replace(/\\*(.*?)\\*/g, '<em>$1</em>');
        html = html.replace(/`(.*?)`/g, '<code>$1</code>');
        html = html.replace(/\\n\\n/g, '</p><p>');
        html = html.replace(/\\n/g, '<br/>');
        return '<p>' + html + '</p>';
    }

    // Inject HTML
    const container = document.createElement("div");
    container.id = "chat-widget-container";
    container.innerHTML = `
      <div id="chat-window">
          <div class="chat-header">
              <div class="chat-header-info">
                  <div class="chat-avatar"><i class="fa-solid fa-stethoscope"></i></div>
                  <div class="chat-title">
                      <h3>SignLink Healthcare</h3>
                      <p>AI Assistant</p>
                  </div>
              </div>
              <button class="close-btn" id="close-chat"><i class="fa-solid fa-xmark"></i></button>
          </div>
          <div class="chat-messages" id="chat-messages">
              <div class="msg bot">Hello! I am your SignLink Healthcare Assistant. Ask me about symptoms, basic treatments, or health conditions!</div>
          </div>
          <div class="chat-input-area">
              <input type="text" id="chat-input" placeholder="Type your health query..." autocomplete="off">
              <button class="send-btn" id="send-chat"><i class="fa-solid fa-paper-plane"></i></button>
          </div>
      </div>
      <div id="chat-bubble">
          <i class="fa-solid fa-message"></i>
      </div>
  `;
    document.body.appendChild(container);

    // Logic
    const bubble = document.getElementById("chat-bubble");
    const chatWindow = document.getElementById("chat-window");
    const closeBtn = document.getElementById("close-chat");
    const sendBtn = document.getElementById("send-chat");
    const chatInput = document.getElementById("chat-input");
    const chatMessages = document.getElementById("chat-messages");

    let isOpen = false;

    const toggleChat = () => {
        isOpen = !isOpen;
        if (isOpen) {
            chatWindow.classList.add("active");
            bubble.innerHTML = '<i class="fa-solid fa-xmark"></i>';
            chatInput.focus();
        } else {
            chatWindow.classList.remove("active");
            setTimeout(() => {
                bubble.innerHTML = '<i class="fa-solid fa-message"></i>';
            }, 200);
        }
    };

    bubble.addEventListener("click", toggleChat);
    closeBtn.addEventListener("click", toggleChat);

    const appendMessage = (text, sender) => {
        const msgDiv = document.createElement("div");
        msgDiv.className = \`msg \${sender}\`;
      if (sender === 'bot') {
          msgDiv.innerHTML = parseMarkdown(text);
      } else {
          msgDiv.textContent = text;
      }
      chatMessages.appendChild(msgDiv);
      chatMessages.scrollTop = chatMessages.scrollHeight;
  };

  const showTyping = () => {
      const typingDiv = document.createElement("div");
      typingDiv.className = "typing-indicator";
      typingDiv.id = "typing-indicator";
      typingDiv.innerHTML = "<span></span><span></span><span></span>";
      chatMessages.appendChild(typingDiv);
      chatMessages.scrollTop = chatMessages.scrollHeight;
  };

  const removeTyping = () => {
      const typing = document.getElementById("typing-indicator");
      if (typing) typing.remove();
  };

  const sendMessage = async () => {
      const text = chatInput.value.trim();
      if (!text) return;

      appendMessage(text, "user");
      chatInput.value = "";
      sendBtn.disabled = true;

      showTyping();

      try {
          const res = await fetch("/api/chat", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ message: text })
          });
          const data = await res.json();
          removeTyping();
          sendBtn.disabled = false;
          
          if (data.response) {
              appendMessage(data.response, "bot");
          } else if (data.error) {
              appendMessage("Error: " + data.error, "bot");
          }
      } catch (err) {
          removeTyping();
          sendBtn.disabled = false;
          appendMessage("Sorry, I could not connect to the server.", "bot");
      }
      chatInput.focus();
  };

  sendBtn.addEventListener("click", sendMessage);
  chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
          e.preventDefault();
          sendMessage();
      }
  });
});
