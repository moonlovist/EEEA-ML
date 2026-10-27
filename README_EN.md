# Student guide · English

<!-- current-session-index -->
> **Session index:** [Session 1](sessions/session01/README.md) · [Session 2](sessions/session02/README.md). This page describes the original Lab 00–08 practice collection. For the new Session 2, use its [English lab guide](sessions/session02/lab/en/README.md).

**Ting Tan · EEEA · Oct 2026 · L1–L2**

[Home](README.md) · [Français](README_FR.md)

## 1. Download and open the course

1. [Download the student pack](https://github.com/moonlovist/EEEA-ML/raw/refs/heads/main/downloads/ML_Labs_2026_student.zip).
2. In Windows File Explorer, open your personal folder and create `AI_Course`.
3. Right-click the ZIP → **Extract All**. Choose `AI_Course`.
4. In the extracted `ML_Labs_2026` folder, double-click **START_HERE.html**.
5. Choose **English**, then **Install and start**. The guide explains each step, with commands and expected results.

Keep the complete folder: the notebooks use the accompanying data and Python modules. `START_HERE.html` opens the course portal on your computer. Reading pages, hints and figures are available immediately after extraction.

On GitHub, **Code → Download ZIP** also provides the entire repository. After extraction, open its `ML_Labs_2026` subfolder to start.

## 2. Prepare Python and Jupyter

Follow the portal’s Windows guide. The target setup uses standard 64-bit Python 3.13 on Windows x64; Python 3.11 or 3.12 can also use the proposed setup. Initial installation requires a connection or packages prepared by the teacher. Allow 30–45 minutes in the first session.

After installing Python, open `ML_Labs_2026` in File Explorer. Click the address bar, type `cmd`, and press Enter. Run these commands one at a time, waiting for each command to finish:

```bat
RUN.cmd -m venv .venv
INSTALL_NOTEBOOK.cmd
CHECK_SETUP.cmd
START_NOTEBOOK.cmd
```

`INSTALL_NOTEBOOK.cmd` installs the numerical libraries and Jupyter. The diagnostic should display `status: READY`. `START_NOTEBOOK.cmd` starts Jupyter and normally opens a browser tab. Keep the terminal open. If needed, copy the complete local URL printed in the terminal into your browser.

The detailed guide explains expected messages and installation troubleshooting. Later sessions reuse the environment: double-click `START_NOTEBOOK.cmd`.

## 3. Complete your first exercise

1. In the **Jupyter** file browser, open **notebooks → en → Lab00_EN.ipynb**.
2. Read the text, then select the first code cell.
3. Predict the displayed value and press **Shift+Enter**. The result appears below the cell.
4. Continue in order. The symbol `[*]` indicates that a cell is running.
5. Change one value as instructed, then rerun that cell and the cells that depend on it.
6. Save with **Ctrl+S**. Then use **Kernel → Restart Kernel and Run All** to check the complete execution.

After Lab 00, open Lab 01. Each lab provides a worked example, a short exercise, two hints and one change to compare. `None` marks an answer to complete; the displayed feedback helps you continue. Use `REPORT_EN.txt` to write your report.

## 4. Recognise the three ways to open the material

| Entry point | Use |
| --- | --- |
| `.ipynb` link on GitHub | Read cells in GitHub’s preview |
| Notebook link in the local portal | Read the formatted page and open hints |
| File opened in Jupyter | Edit, execute, display results and save |

The `.ipynb` file stores cells in JSON format. If a browser displays that text, open `START_HERE.html` to read the lab or launch `START_NOTEBOOK.cmd` to work in Jupyter. GitHub’s **Raw** button displays the file contents.

Pre-computed results are in `examples/en`. Each report contains figures, measurements and the software versions used. Experiments run during your session save their results in `outputs/en`.

## 5. Course sequence

| TP / Lab | Français | English |
| --- | --- | --- |
| 00 | [Premiers pas en Python](ML_Labs_2026/notebooks/fr/Lab00_FR.ipynb) | [Python warm-up](ML_Labs_2026/notebooks/en/Lab00_EN.ipynb) |
| 01 | [Données, variables et cibles](ML_Labs_2026/notebooks/fr/Lab01_FR.ipynb) | [Data, features and targets](ML_Labs_2026/notebooks/en/Lab01_EN.ipynb) |
| 02 | [Régression linéaire et erreurs](ML_Labs_2026/notebooks/fr/Lab02_FR.ipynb) | [Linear regression and errors](ML_Labs_2026/notebooks/en/Lab02_EN.ipynb) |
| 03 | [Classification : kNN et arbres](ML_Labs_2026/notebooks/fr/Lab03_FR.ipynb) | [Classification: kNN and trees](ML_Labs_2026/notebooks/en/Lab03_EN.ipynb) |
| 04 | [Validation, surapprentissage et choix du modèle](ML_Labs_2026/notebooks/fr/Lab04_FR.ipynb) | [Validation, overfitting and model selection](ML_Labs_2026/notebooks/en/Lab04_EN.ipynb) |
| 05 | [K-means et standardisation](ML_Labs_2026/notebooks/fr/Lab05_FR.ipynb) | [K-means and scaling](ML_Labs_2026/notebooks/en/Lab05_EN.ipynb) |
| 06 | [Réseau de neurones, gradients et apprentissage](ML_Labs_2026/notebooks/fr/Lab06_FR.ipynb) | [Neural networks, gradients and learning](ML_Labs_2026/notebooks/en/Lab06_EN.ipynb) |
| 07 | [Prochain token et attention causale](ML_Labs_2026/notebooks/fr/Lab07_FR.ipynb) | [Next token and causal attention](ML_Labs_2026/notebooks/en/Lab07_EN.ipynb) |
| 08 | [Synthèse et évaluation finale](ML_Labs_2026/notebooks/fr/Lab08_FR.ipynb) | [Synthesis and final evaluation](ML_Labs_2026/notebooks/en/Lab08_EN.ipynb) |

Levels A and B form the core work. Level C offers an optional exploration. Explain the inputs, axes, units and meaning of the result alongside each execution.
