from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static, RichLog, Label
from textual.containers import Horizontal
from textual.message import Message
from textual import work

from PIL import Image
import torch
import os


MODEL_ID = "google/medgemma-1.5-4b-it"


class MedGemmaApp(App):
    """MedGemma TUI Chat Interface."""

    TITLE = "MedGemma Chat"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear"),
    ]

    CSS = """
    #chat-log {
        height: 1fr;
        border: solid $primary;
        padding: 0 1;
        scrollbar-size: 1 1;
    }
    #status-bar {
        height: 1;
        dock: bottom;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }
    #chat-input {
        dock: bottom;
    }
    .user-msg {
        color: $success;
        margin: 0 0 0 4;
    }
    .assistant-msg {
        color: $text;
        margin: 0 0 0 0;
    }
    .system-msg {
        color: $warning;
        text-style: italic;
    }
    """

    class ModelLoaded(Message):
        pass

    class GenerationComplete(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class GenerationError(Message):
        def __init__(self, error: str) -> None:
            super().__init__()
            self.error = error

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.processor = None
        self.conversation: list[dict] = []
        self.staged_image: Image.Image | None = None
        self.generating = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="chat-log", wrap=True, markup=True)
        yield Label("Loading model...", id="status-bar")
        yield Input(placeholder="Type a message (or /help for commands)", id="chat-input", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.log_system("Loading MedGemma model... this may take a moment.")
        self.load_model()

    @work(thread=True)
    def load_model(self) -> None:
        from transformers import AutoProcessor, AutoModelForImageTextToText

        try:
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            self.model = model
            self.processor = processor
            self.post_message(self.ModelLoaded())
        except Exception as e:
            self.post_message(self.GenerationError(f"Failed to load model: {e}"))

    def on_med_gemma_app_model_loaded(self, message: ModelLoaded) -> None:
        self.log_system("Model loaded! You can start chatting.")
        self.query_one("#chat-input", Input).disabled = False
        self.query_one("#status-bar", Label).update("Ready")
        self.query_one("#chat-input", Input).focus()

    def on_med_gemma_app_generation_complete(self, message: GenerationComplete) -> None:
        self.log_assistant(message.text)
        self.generating = False
        inp = self.query_one("#chat-input", Input)
        inp.disabled = False
        inp.focus()
        status = "Ready"
        if self.staged_image:
            status = "Ready | Image attached"
        self.query_one("#status-bar", Label).update(status)

    def on_med_gemma_app_generation_error(self, message: GenerationError) -> None:
        self.log_system(f"Error: {message.error}")
        self.generating = False
        inp = self.query_one("#chat-input", Input)
        inp.disabled = False
        inp.focus()
        self.query_one("#status-bar", Label).update("Ready (error occurred)")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()

        if text.startswith("/"):
            self.handle_command(text)
            return

        self.send_message(text)

    def handle_command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/help":
            self.log_system(
                "Commands:\n"
                "  /image <path or URL>  — Attach an image (local path or http URL)\n"
                "  /clear         — Clear chat and reset conversation\n"
                "  /help          — Show this help\n"
                "  Ctrl+C         — Quit"
            )
        elif cmd == "/clear":
            self.action_clear_chat()
        elif cmd == "/image":
            if len(parts) < 2:
                self.log_system("Usage: /image <path or URL>")
                return
            source = parts[1].strip()
            if source.startswith(("http://", "https://")):
                try:
                    import requests
                    from io import BytesIO
                    resp = requests.get(source, timeout=30)
                    resp.raise_for_status()
                    self.staged_image = Image.open(BytesIO(resp.content)).convert("RGB")
                    name = source.split("/")[-1].split("?")[0] or "image"
                    self.log_system(f"Image attached: {name}")
                    self.query_one("#status-bar", Label).update(f"Image attached: {name}")
                except Exception as e:
                    self.log_system(f"Failed to fetch image: {e}")
            else:
                path = os.path.expanduser(source)
                if not os.path.isfile(path):
                    self.log_system(f"File not found: {path}")
                    return
                try:
                    self.staged_image = Image.open(path).convert("RGB")
                    self.log_system(f"Image attached: {os.path.basename(path)}")
                    self.query_one("#status-bar", Label).update(
                        f"Image attached: {os.path.basename(path)}"
                    )
                except Exception as e:
                    self.log_system(f"Failed to open image: {e}")
        else:
            self.log_system(f"Unknown command: {cmd}. Type /help for commands.")

    def send_message(self, text: str) -> None:
        content: list[dict] = []
        display_prefix = ""

        if self.staged_image:
            content.append({"type": "image", "image": self.staged_image})
            display_prefix = "[image] "
            self.staged_image = None

        content.append({"type": "text", "text": text})
        self.conversation.append({"role": "user", "content": content})

        self.log_user(f"{display_prefix}{text}")

        self.generating = True
        inp = self.query_one("#chat-input", Input)
        inp.disabled = True
        self.query_one("#status-bar", Label).update("Generating...")

        self.run_generation()

    @work(thread=True)
    def run_generation(self) -> None:
        try:
            inputs = self.processor.apply_chat_template(
                self.conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, max_new_tokens=2000, do_sample=False
                )
                generation = generation[0][input_len:]

            decoded = self.processor.decode(generation, skip_special_tokens=True)
            self.conversation.append(
                {"role": "assistant", "content": [{"type": "text", "text": decoded}]}
            )
            self.post_message(self.GenerationComplete(decoded))
        except Exception as e:
            # Remove the failed user message so conversation stays consistent
            if self.conversation and self.conversation[-1]["role"] == "user":
                self.conversation.pop()
            self.post_message(self.GenerationError(str(e)))

    def log_system(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[italic yellow]System:[/] {text}")

    def log_user(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold green]You:[/] {text}")

    def log_assistant(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold cyan]MedGemma:[/] {text}")

    def action_clear_chat(self) -> None:
        self.conversation.clear()
        self.staged_image = None
        self.query_one("#chat-log", RichLog).clear()
        self.query_one("#status-bar", Label).update("Ready")
        self.log_system("Conversation cleared.")


if __name__ == "__main__":
    app = MedGemmaApp()
    app.run()
