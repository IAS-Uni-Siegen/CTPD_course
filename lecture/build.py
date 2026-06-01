import fileinput
import os
from subprocess import run

os.makedirs("built", exist_ok=True)
latex_engine = os.environ.get("LATEX_ENGINE", "lualatex")
call_latex_l = [
    latex_engine,
    "-synctex=1",
    "-interaction=nonstopmode",
    "-halt-on-error",
    "-file-line-error",
    "-shell-escape",
    "main.tex",
]


def clear_tex_binaries():
    # clean misc files
    for file in os.listdir("."):
        if file.startswith("main"):
            if not file.endswith((".tex", ".pdf")):
                os.remove(file)


# build main pdf
clear_tex_binaries()
with fileinput.input("main.tex", inplace=True) as f:
    for line in f:
        if "includeonly{" in line:
            # comment out includeonly flag
            print(f"%{line}", end="")
        # check if the line including '\documentclass' has the parameter 'handout' - if not add it
        elif "]{beamer}" in line:
            if "handout" not in line:
                print(line.replace("]{beamer}", ", handout]{beamer}"), end="")
            else:
                print(line, end="")
        else:
            print(line, end="")

run(call_latex_l, check=True)
run(call_latex_l, check=True)


# go into the parent directory
os.chdir("..")
os.makedirs("built", exist_ok=True)
# take main.pdf from the exercise folder and move it to the parent folder
os.replace("lecture/main.pdf", os.path.join("built", "lecture.pdf"))