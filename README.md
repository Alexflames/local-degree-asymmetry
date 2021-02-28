# local-degree-asymmetry
Source code for paper "Friendship Paradox in Growth Networks: Analytical and Empirical Analysis".

Can be used to acquire friendship index dynamic for nodes in networks, acquire and analyze friendship index distribution.

## How to run
There are 3 python source files in the root of the repository with detailed descriptions on top.

1. Open `main.py` and edit `input_type_num` variable on line 29 to select which type of experiment you would like to run. The variable is an
index for `input_types` array on line 27.
2. Edit model parameters or select input filename. For models you may record trajectories of nodes specified in `focus_indices` array:
  - For real graph change filename in `experiment_file()` function on line 100;
  - For BA model change parameters in `experiment_ba()` function on lines 159-162;
  - For TC model change parameters in `experiment_triadic()` function on lines 280-282.

Output: raw node trajectories and histograms with friendship index distribution.

3. To average results for node trajectories after running `main.py` go to `process_output.py`, edit filename on line 3.
4. To evaluate friendship paradox use code in `analyze_hist.py` that accepts histograms produced by `main.py`

Tested on Windows 10, Python 3.7.6. Please, see next section on how to visualize output.

## How to visualize
Output histograms and averaged degree dynamics are created in the format, that is accepted by LaTeX Tikzpicture environment.

Example of code:
```
\begin{tikzpicture}\footnotesize
\begin{axis}[height = 1.3in, width=\linewidth,
       xmin=1.2,
       xmax=4.8,
       tick align = {outside},
       ymin=0,
       ymax=12,
       xlabel={$\log(\#\beta_i\ \mathrm{in} \ \mathrm{interval})$, BA model},
legend style = {cells = {anchor=west}, nodes = {scale=0.75}}, legend pos=south west
]
\addplot[blue, only marks, mark=*, mark options={scale=0.25}] table[x=lnt,y=lnb]{source_data/hist_out_ba_335000_3.txt};
\addlegendentry{$\log(\#\beta_i(t))$}
\addplot[red, smooth, thick] table[x=lnt,y=linreg]{source_data/hist_out_ba_335000_3.txt};
\addlegendentry{$-2.48\log t+C$}
\end{axis}
\end{tikzpicture}
```
Produces following image:

![log-log degree distribution](https://sun9-20.userapi.com/impf/kizMCXMXAITwrFSFeRSLObOCzrOPva49s4t9Pg/7hy9a9pOsRQ.jpg?size=437x201&quality=96&proxy=1&sign=21bb70f26895355c0a292b3e9b185c74&type=album)
