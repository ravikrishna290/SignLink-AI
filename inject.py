import glob
for f in glob.glob("*.html"):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    if 'chatbot.js' not in content:
        content = content.replace('</body>', '  <script src="chatbot.js"></script>\n</body>')
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Injected into {f}")
