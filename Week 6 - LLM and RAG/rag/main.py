# main.py
import tkinter as tk
from tkinter import filedialog, scrolledtext
from agent import Agent

class AgentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Agent")

        # Datasource input
        tk.Label(root, text="Datasource (File path or URL):").pack(anchor="w", padx=5, pady=2)
        frame = tk.Frame(root)
        frame.pack(fill="x", padx=5, pady=2)

        self.datasource_entry = tk.Entry(frame, width=50)
        self.datasource_entry.pack(side="left", padx=5, pady=2)
        self.browse_btn = tk.Button(frame, text="Browse", command=self.browse_file)
        self.browse_btn.pack(side="left", padx=5, pady=2)

        # Prompt input
        tk.Label(root, text="\nEnter your question:").pack(anchor="w", padx=5)
        self.prompt_entry = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=5)
        self.prompt_entry.pack(padx=5, pady=5)

        # Run button
        self.run_btn = tk.Button(root, text="Run", command=self.run_agent)
        self.run_btn.pack(pady=5)

        # Output display
        tk.Label(root, text="\nAnswer:").pack(anchor="w", padx=5)
        self.output_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=10, state="disabled")
        self.output_area.pack(padx=5, pady=5)

    def browse_file(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.datasource_entry.delete(0, tk.END)
            self.datasource_entry.insert(0, filepath)

    def run_agent(self):
        datasource = self.datasource_entry.get().strip()
        user_prompt = self.prompt_entry.get("1.0", tk.END).strip()

        if not datasource or not user_prompt:
            self.display_output("⚠️ Please provide both datasource and question.")
            return

        try:
            # Just pass the datasource (could be URL or file)
            agent = Agent(datasource)
            answer = agent.run(user_prompt)
            result = answer.content if hasattr(answer, "content") else str(answer)
            self.display_output(result)
        except Exception as e:
            self.display_output(f"❌ Error: {e}")

    def display_output(self, text):
        self.output_area.config(state="normal")
        self.output_area.delete("1.0", tk.END)
        self.output_area.insert(tk.END, text)
        self.output_area.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = AgentApp(root)
    root.mainloop()
