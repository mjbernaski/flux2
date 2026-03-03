#!/usr/bin/env -S .venv/bin/python
"""CLI Chat Client for vLLM with Conversation Branching.

A tree-structured conversation interface for local vLLM (OpenAI-compatible) servers.
Supports branching, navigation, persistence, streaming, and reasoning display.
"""

import json
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

# ─── Configuration ───────────────────────────────────────────────────────────

API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
CONVERSATIONS_DIR = Path(__file__).parent / "conversations"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# ANSI escape helpers
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


# ─── Data Model ──────────────────────────────────────────────────────────────

@dataclass
class ChatNode:
    id: str
    role: str  # "system" | "user" | "assistant"
    content: str
    parent: Optional[str] = None
    children: list[str] = field(default_factory=list)
    reasoning: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChatNode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ConversationTree:
    """Manages a tree of ChatNode objects with navigation and persistence."""

    def __init__(self):
        self.nodes: dict[str, ChatNode] = {}
        self.root_id: Optional[str] = None
        self.current_id: Optional[str] = None
        self.name: Optional[str] = None

    def new_conversation(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.nodes.clear()
        root = ChatNode(id=self._gen_id(), role="system", content=system_prompt)
        self.nodes[root.id] = root
        self.root_id = root.id
        self.current_id = root.id

    def add_message(self, role: str, content: str, reasoning: Optional[str] = None,
                    metadata: Optional[dict] = None) -> ChatNode:
        node = ChatNode(
            id=self._gen_id(),
            role=role,
            content=content,
            parent=self.current_id,
            reasoning=reasoning,
            metadata=metadata or {},
        )
        self.nodes[node.id] = node
        if self.current_id:
            self.nodes[self.current_id].children.append(node.id)
        self.current_id = node.id
        return node

    def get_branch_messages(self) -> list[dict]:
        """Walk from root to current node, returning OpenAI-format messages."""
        path = self._path_to_current()
        messages = []
        for nid in path:
            node = self.nodes[nid]
            messages.append({"role": node.role, "content": node.content})
        return messages

    def back(self, n: int = 1) -> Optional[ChatNode]:
        node = self.nodes.get(self.current_id)
        for _ in range(n):
            if node and node.parent:
                node = self.nodes.get(node.parent)
            else:
                break
        if node:
            self.current_id = node.id
        return node

    def goto(self, node_id: str) -> bool:
        if node_id in self.nodes:
            self.current_id = node_id
            return True
        return False

    def delete_node(self, node_id: str) -> bool:
        if node_id == self.root_id:
            return False
        if node_id not in self.nodes:
            return False
        # Collect all descendants
        to_delete = []
        stack = [node_id]
        while stack:
            nid = stack.pop()
            to_delete.append(nid)
            node = self.nodes.get(nid)
            if node:
                stack.extend(node.children)
        # Remove from parent's children list
        parent = self.nodes[node_id].parent
        if parent and parent in self.nodes:
            self.nodes[parent].children = [
                c for c in self.nodes[parent].children if c != node_id
            ]
        # Delete all
        for nid in to_delete:
            del self.nodes[nid]
        # If current was deleted, move to parent
        if self.current_id in to_delete:
            self.current_id = parent or self.root_id
        return True

    def branches_from(self, node_id: Optional[str] = None) -> list[ChatNode]:
        nid = node_id or self.current_id
        node = self.nodes.get(nid)
        if not node:
            return []
        return [self.nodes[c] for c in node.children if c in self.nodes]

    def _path_to_current(self) -> list[str]:
        path = []
        nid = self.current_id
        while nid:
            path.append(nid)
            nid = self.nodes[nid].parent
        return list(reversed(path))

    def history_display(self) -> list[ChatNode]:
        return [self.nodes[nid] for nid in self._path_to_current()]

    def tree_str(self) -> str:
        if not self.root_id:
            return "(empty)"
        lines = []
        self._tree_recurse(self.root_id, "", True, lines)
        return "\n".join(lines)

    def _tree_recurse(self, node_id: str, prefix: str, is_last: bool, lines: list):
        node = self.nodes[node_id]
        connector = "└── " if is_last else "├── "
        marker = " ◀" if node_id == self.current_id else ""
        preview = node.content[:60].replace("\n", " ")
        if len(node.content) > 60:
            preview += "..."
        tag = f"[{node.role}]"
        short_id = node.id[:8]
        lines.append(f"{prefix}{connector}{DIM}{short_id}{RESET} {tag} {preview}{YELLOW}{marker}{RESET}")
        extension = "    " if is_last else "│   "
        children = [c for c in node.children if c in self.nodes]
        for i, child_id in enumerate(children):
            self._tree_recurse(child_id, prefix + extension, i == len(children) - 1, lines)

    def save(self, name: Optional[str] = None):
        CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
        save_name = name or self.name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = save_name
        data = {
            "name": save_name,
            "root_id": self.root_id,
            "current_id": self.current_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }
        path = CONVERSATIONS_DIR / f"{save_name}.json"
        path.write_text(json.dumps(data, indent=2))
        return path

    def load(self, name: str) -> bool:
        path = CONVERSATIONS_DIR / f"{name}.json"
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        self.name = data["name"]
        self.root_id = data["root_id"]
        self.current_id = data["current_id"]
        self.nodes = {nid: ChatNode.from_dict(nd) for nid, nd in data["nodes"].items()}
        return True

    @staticmethod
    def list_saved() -> list[str]:
        CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
        return sorted(p.stem for p in CONVERSATIONS_DIR.glob("*.json"))

    @staticmethod
    def _gen_id() -> str:
        return uuid.uuid4().hex[:12]


# ─── CLI ─────────────────────────────────────────────────────────────────────

COMMANDS = [
    "/save", "/load", "/list", "/history", "/tree", "/branch", "/back",
    "/goto", "/fork", "/reset", "/delete", "/retry", "/system", "/model",
    "/quit", "/help",
]


class ChatCLI:
    def __init__(self):
        self.client = OpenAI(base_url=API_BASE, api_key="not-needed")
        self.tree = ConversationTree()
        self.model: Optional[str] = None
        self.interrupted = False
        completer = WordCompleter(COMMANDS, sentence=True)
        self.session = PromptSession(
            history=InMemoryHistory(),
            completer=completer,
        )

    def run(self):
        self._connect()
        self.tree.new_conversation()
        print(f"\n{BOLD}Chat started.{RESET} Model: {CYAN}{self.model}{RESET}")
        print(f"Type {GREEN}/help{RESET} for commands, {GREEN}/quit{RESET} or Ctrl+D to exit.\n")

        while True:
            try:
                user_input = self.session.prompt(f"{GREEN}You>{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{DIM}Goodbye.{RESET}")
                self._autosave()
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input.split(None, 1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                if self._handle_command(cmd, arg):
                    continue
                else:
                    break  # /quit
            else:
                self._send_message(user_input)

    def _connect(self):
        try:
            models = self.client.models.list()
            self.model = models.data[0].id if models.data else "unknown"
        except Exception as e:
            print(f"{RED}Failed to connect to vLLM at {API_BASE}: {e}{RESET}")
            print(f"{DIM}Make sure the server is running.{RESET}")
            sys.exit(1)

    def _send_message(self, text: str):
        self.tree.add_message("user", text)
        messages = self.tree.get_branch_messages()

        # Set up interrupt handler
        self.interrupted = False
        old_handler = signal.getsignal(signal.SIGINT)

        def on_interrupt(sig, frame):
            self.interrupted = True

        signal.signal(signal.SIGINT, on_interrupt)

        try:
            print(f"\n{CYAN}Assistant>{RESET} ", end="", flush=True)
            response_text = ""
            reasoning_text = ""

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )

            for chunk in stream:
                if self.interrupted:
                    print(f"\n{DIM}(interrupted){RESET}")
                    break

                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                # Check for reasoning content (Qwen thinking tokens)
                reasoning_content = getattr(delta, "reasoning_content", None)
                if reasoning_content:
                    if not reasoning_text:
                        print(f"{DIM}{ITALIC}", end="", flush=True)
                    reasoning_text += reasoning_content
                    print(reasoning_content, end="", flush=True)
                    continue

                if delta.content:
                    if reasoning_text and not response_text:
                        # Transition from reasoning to content
                        print(f"{RESET}\n\n{CYAN}Assistant>{RESET} ", end="", flush=True)
                    response_text += delta.content
                    print(delta.content, end="", flush=True)

            if reasoning_text and not response_text:
                print(RESET, end="")

            print("\n")

            self.tree.add_message(
                "assistant",
                response_text,
                reasoning=reasoning_text or None,
                metadata={"model": self.model},
            )
            self._autosave()

        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}\n")
            # Remove the dangling user message if assistant failed
            self.tree.back(1)
            user_node_id = [
                c for c in self.tree.nodes[self.tree.current_id].children
            ]
            if user_node_id:
                self.tree.delete_node(user_node_id[-1])
        finally:
            signal.signal(signal.SIGINT, old_handler)

    def _handle_command(self, cmd: str, arg: str) -> bool:
        """Returns True to continue REPL, False to quit."""
        handlers = {
            "/quit": self._cmd_quit,
            "/help": self._cmd_help,
            "/save": self._cmd_save,
            "/load": self._cmd_load,
            "/list": self._cmd_list,
            "/history": self._cmd_history,
            "/tree": self._cmd_tree,
            "/branch": self._cmd_branch,
            "/back": self._cmd_back,
            "/goto": self._cmd_goto,
            "/fork": self._cmd_fork,
            "/reset": self._cmd_reset,
            "/delete": self._cmd_delete,
            "/retry": self._cmd_retry,
            "/system": self._cmd_system,
            "/model": self._cmd_model,
        }
        handler = handlers.get(cmd)
        if handler:
            return handler(arg)
        print(f"{RED}Unknown command: {cmd}{RESET}. Type /help for available commands.")
        return True

    def _cmd_quit(self, _arg: str) -> bool:
        self._autosave()
        print(f"{DIM}Goodbye.{RESET}")
        return False

    def _cmd_help(self, _arg: str) -> bool:
        print(f"""
{BOLD}Commands:{RESET}
  {GREEN}/save [name]{RESET}    — Save conversation (auto-saves by default)
  {GREEN}/load <name>{RESET}    — Load a saved conversation
  {GREEN}/list{RESET}           — List saved conversations
  {GREEN}/history{RESET}        — Show current branch message history
  {GREEN}/tree{RESET}           — Show full conversation tree (ASCII)
  {GREEN}/branch{RESET}         — List branches from current node
  {GREEN}/back [n]{RESET}       — Go back n messages (default 1)
  {GREEN}/goto <id>{RESET}      — Jump to a node by ID (first 8 chars)
  {GREEN}/fork{RESET}           — Alias: go back 1 to create a branch point
  {GREEN}/reset{RESET}          — Reset to system prompt (root)
  {GREEN}/delete <id>{RESET}    — Delete a node and its descendants
  {GREEN}/retry{RESET}          — Regenerate last assistant response
  {GREEN}/system <text>{RESET}  — Set/change system prompt
  {GREEN}/model{RESET}          — Show current model info
  {GREEN}/quit{RESET}           — Exit (also Ctrl+D)
""")
        return True

    def _cmd_save(self, arg: str) -> bool:
        name = arg.strip() or None
        path = self.tree.save(name)
        print(f"{DIM}Saved to {path}{RESET}")
        return True

    def _cmd_load(self, arg: str) -> bool:
        name = arg.strip()
        if not name:
            print(f"{RED}Usage: /load <name>{RESET}")
            return True
        if self.tree.load(name):
            print(f"{DIM}Loaded conversation '{name}'{RESET}")
            # Show the last few messages for context
            history = self.tree.history_display()
            for node in history[-3:]:
                if node.role == "system":
                    continue
                tag = f"{GREEN}You>{RESET}" if node.role == "user" else f"{CYAN}Assistant>{RESET}"
                preview = node.content[:200]
                if len(node.content) > 200:
                    preview += "..."
                print(f"  {tag} {preview}")
            print()
        else:
            print(f"{RED}Conversation '{name}' not found.{RESET}")
        return True

    def _cmd_list(self, _arg: str) -> bool:
        saved = ConversationTree.list_saved()
        if not saved:
            print(f"{DIM}No saved conversations.{RESET}")
        else:
            print(f"{BOLD}Saved conversations:{RESET}")
            for name in saved:
                marker = " ◀ (current)" if name == self.tree.name else ""
                print(f"  {name}{YELLOW}{marker}{RESET}")
        return True

    def _cmd_history(self, _arg: str) -> bool:
        history = self.tree.history_display()
        print(f"\n{BOLD}Current branch ({len(history)} messages):{RESET}")
        for node in history:
            short_id = node.id[:8]
            if node.role == "system":
                print(f"  {DIM}{short_id} [system]{RESET} {node.content[:80]}")
            elif node.role == "user":
                print(f"  {DIM}{short_id}{RESET} {GREEN}You>{RESET} {node.content[:120]}")
            else:
                preview = node.content[:120]
                if len(node.content) > 120:
                    preview += "..."
                print(f"  {DIM}{short_id}{RESET} {CYAN}Asst>{RESET} {preview}")
        print()
        return True

    def _cmd_tree(self, _arg: str) -> bool:
        print(f"\n{BOLD}Conversation tree:{RESET}")
        print(self.tree.tree_str())
        print()
        return True

    def _cmd_branch(self, _arg: str) -> bool:
        branches = self.tree.branches_from()
        if not branches:
            print(f"{DIM}No branches from current node.{RESET}")
        else:
            print(f"{BOLD}Branches from current node:{RESET}")
            for b in branches:
                marker = " ◀" if b.id == self.tree.current_id else ""
                print(f"  {DIM}{b.id[:8]}{RESET} [{b.role}] {b.content[:80]}{YELLOW}{marker}{RESET}")
        return True

    def _cmd_back(self, arg: str) -> bool:
        n = 1
        if arg.strip():
            try:
                n = int(arg.strip())
            except ValueError:
                print(f"{RED}Usage: /back [n]{RESET}")
                return True
        node = self.tree.back(n)
        if node:
            print(f"{DIM}Moved to {node.id[:8]} [{node.role}]: {node.content[:80]}{RESET}")
        return True

    def _cmd_goto(self, arg: str) -> bool:
        target = arg.strip()
        if not target:
            print(f"{RED}Usage: /goto <id>{RESET}")
            return True
        # Support partial ID matching
        matches = [nid for nid in self.tree.nodes if nid.startswith(target)]
        if len(matches) == 1:
            self.tree.goto(matches[0])
            node = self.tree.nodes[matches[0]]
            print(f"{DIM}Jumped to {node.id[:8]} [{node.role}]: {node.content[:80]}{RESET}")
        elif len(matches) > 1:
            print(f"{RED}Ambiguous ID. Matches: {', '.join(m[:8] for m in matches)}{RESET}")
        else:
            print(f"{RED}Node not found: {target}{RESET}")
        return True

    def _cmd_fork(self, _arg: str) -> bool:
        node = self.tree.back(1)
        if node:
            print(f"{DIM}Moved back to {node.id[:8]} — type your message to create a new branch.{RESET}")
        return True

    def _cmd_reset(self, _arg: str) -> bool:
        if self.tree.root_id:
            self.tree.current_id = self.tree.root_id
            print(f"{DIM}Reset to root (system prompt).{RESET}")
        return True

    def _cmd_delete(self, arg: str) -> bool:
        target = arg.strip()
        if not target:
            print(f"{RED}Usage: /delete <id>{RESET}")
            return True
        matches = [nid for nid in self.tree.nodes if nid.startswith(target)]
        if len(matches) == 1:
            if self.tree.delete_node(matches[0]):
                print(f"{DIM}Deleted node {matches[0][:8]} and its descendants.{RESET}")
                self._autosave()
            else:
                print(f"{RED}Cannot delete root node.{RESET}")
        elif len(matches) > 1:
            print(f"{RED}Ambiguous ID. Matches: {', '.join(m[:8] for m in matches)}{RESET}")
        else:
            print(f"{RED}Node not found: {target}{RESET}")
        return True

    def _cmd_retry(self, _arg: str) -> bool:
        current = self.tree.nodes.get(self.tree.current_id)
        if not current or current.role != "assistant":
            print(f"{RED}Can only retry from an assistant message.{RESET}")
            return True
        # Get the user message that preceded this
        parent_id = current.parent
        # Delete the assistant response
        self.tree.delete_node(current.id)
        # current_id is now at parent (the user message)
        # Re-send
        if parent_id and parent_id in self.tree.nodes:
            self.tree.current_id = parent_id
            messages = self.tree.get_branch_messages()
            # Need to generate a new response
            self.interrupted = False
            old_handler = signal.getsignal(signal.SIGINT)

            def on_interrupt(sig, frame):
                self.interrupted = True

            signal.signal(signal.SIGINT, on_interrupt)

            try:
                print(f"\n{CYAN}Assistant>{RESET} ", end="", flush=True)
                response_text = ""
                reasoning_text = ""

                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )

                for chunk in stream:
                    if self.interrupted:
                        print(f"\n{DIM}(interrupted){RESET}")
                        break
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue
                    reasoning_content = getattr(delta, "reasoning_content", None)
                    if reasoning_content:
                        if not reasoning_text:
                            print(f"{DIM}{ITALIC}", end="", flush=True)
                        reasoning_text += reasoning_content
                        print(reasoning_content, end="", flush=True)
                        continue
                    if delta.content:
                        if reasoning_text and not response_text:
                            print(f"{RESET}\n\n{CYAN}Assistant>{RESET} ", end="", flush=True)
                        response_text += delta.content
                        print(delta.content, end="", flush=True)

                if reasoning_text and not response_text:
                    print(RESET, end="")
                print("\n")

                self.tree.add_message(
                    "assistant",
                    response_text,
                    reasoning=reasoning_text or None,
                    metadata={"model": self.model},
                )
                self._autosave()
            except Exception as e:
                print(f"\n{RED}Error: {e}{RESET}\n")
            finally:
                signal.signal(signal.SIGINT, old_handler)
        return True

    def _cmd_system(self, arg: str) -> bool:
        text = arg.strip()
        if not text:
            root = self.tree.nodes.get(self.tree.root_id)
            if root:
                print(f"{DIM}System prompt: {root.content}{RESET}")
            return True
        root = self.tree.nodes.get(self.tree.root_id)
        if root:
            root.content = text
            print(f"{DIM}System prompt updated.{RESET}")
        return True

    def _cmd_model(self, _arg: str) -> bool:
        print(f"{BOLD}Model:{RESET} {self.model}")
        print(f"{BOLD}API Base:{RESET} {API_BASE}")
        return True

    def _autosave(self):
        if self.tree.nodes:
            try:
                self.tree.save()
            except Exception:
                pass  # Don't crash on autosave failure


def main():
    cli = ChatCLI()
    cli.run()


if __name__ == "__main__":
    main()
