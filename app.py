"""Tkinter desktop app: load the three slide images and get the blood group.

Run:
    python app.py
"""
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from src.detector import detect

REAGENTS = ["Anti-A", "Anti-B", "Anti-D"]
THUMB = 220
SAMPLES = Path(__file__).resolve().parent / "samples"


def to_photo(img, gray=False):
    rgb = img if gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail((THUMB, THUMB))
    return ImageTk.PhotoImage(pil)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Blood Group Detection - Image Segmentation")
        self.configure(padx=14, pady=14)
        self.images = {r: None for r in REAGENTS}
        self._photos = []  # keep references so Tk does not drop the images

        tk.Label(self, text="Blood Group Detection", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, columnspan=3)
        tk.Label(self, text="Load the slide image for each reagent, then press Detect.").grid(row=1, column=0, columnspan=3, pady=(0, 10))

        self.original, self.processed, self.status = {}, {}, {}
        for col, reagent in enumerate(REAGENTS):
            frame = tk.LabelFrame(self, text=reagent, padx=6, pady=6)
            frame.grid(row=2, column=col, padx=6, sticky="n")
            tk.Button(frame, text=f"Load {reagent} image", command=lambda r=reagent: self.load(r)).pack(fill="x")
            self.original[reagent] = tk.Label(frame, text="(no image)", width=30, height=12, relief="groove")
            self.original[reagent].pack(pady=4)
            self.processed[reagent] = tk.Label(frame, text="segmented clumps", width=30, height=12, relief="groove")
            self.processed[reagent].pack(pady=4)
            self.status[reagent] = tk.Label(frame, text="", font=("Segoe UI", 10, "bold"))
            self.status[reagent].pack()

        buttons = tk.Frame(self)
        buttons.grid(row=3, column=0, columnspan=3, pady=10)
        tk.Button(buttons, text="Load sample set…", command=self.load_sample).pack(side="left", padx=5)
        tk.Button(buttons, text="Detect blood group", command=self.run, bg="#b71c1c", fg="white",
                  font=("Segoe UI", 11, "bold")).pack(side="left", padx=5)

        self.result = tk.Label(self, text="", font=("Segoe UI", 22, "bold"), fg="#b71c1c")
        self.result.grid(row=4, column=0, columnspan=3)

    def set_image(self, reagent, path):
        img = cv2.imread(str(path))
        if img is None:
            messagebox.showerror("Error", f"Could not read {path}")
            return
        self.images[reagent] = img
        photo = to_photo(img)
        self._photos.append(photo)
        self.original[reagent].configure(image=photo, text="", width=THUMB, height=THUMB)

    def load(self, reagent):
        path = filedialog.askopenfilename(title=f"{reagent} slide image",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.set_image(reagent, path)

    def load_sample(self):
        folder = filedialog.askdirectory(title="Pick a sample folder (e.g. samples/A_pos)", initialdir=SAMPLES)
        if folder:
            for reagent in REAGENTS:
                self.set_image(reagent, Path(folder) / f"{reagent.lower().replace('-', '_')}.png")

    def run(self):
        if any(v is None for v in self.images.values()):
            messagebox.showwarning("Missing images", "Please load all three images first.")
            return
        try:
            group, results = detect(*(self.images[r] for r in REAGENTS))
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        for reagent, res in results.items():
            photo = to_photo(res.masks["clumps"], gray=True)
            self._photos.append(photo)
            self.processed[reagent].configure(image=photo, text="", width=THUMB, height=THUMB)
            verdict = "Agglutinated" if res.agglutinated else "No agglutination"
            self.status[reagent].configure(text=f"{verdict}\nclumps {res.clumps} | variation {res.variation}",
                                           fg="#b71c1c" if res.agglutinated else "#1b5e20")
        self.result.configure(text=f"Blood group: {group}")


if __name__ == "__main__":
    App().mainloop()
