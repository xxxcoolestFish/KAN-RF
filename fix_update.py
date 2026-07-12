import sys; sys.path.insert(0, ".")
with open("control/cdpn.py", encoding="utf-8") as f:
    content = f.read()

old = "    def update(self, wm, s_dataset=None, n_samples=500):\n        self.compute(wm, s_dataset, n_samples)"
new = '    def update(self, wm, s_dataset=None, n_samples=500):\n        self.a_fit = estimate_a_fit_from_wm(wm, s_dataset, device=self.device)\n        print(f"  [CausalBridge] a_fit updated: {self.a_fit:.2f}")\n        self.compute(wm, s_dataset, n_samples)'

assert old in content, "update() not found!"
content = content.replace(old, new)
with open("control/cdpn.py", "w", encoding="utf-8") as f:
    f.write(content)

import ast
ast.parse(content)
print("Bridge.update() fixed. Syntax OK")
