# Measuring-the-Variability-of-Local-Characteristics
Source code for paper: "Measuring the Variability of Local Characteristics in Complex Networks: Empirical and Analytical Analysis".

The program can be used to simulate Barabasi-Albert and triadic closure (by Holme and Kim) networks, acquire friendship index, average degree dynamics for nodes in networks, acquire friendship index distributions, degree-degree correlations in real and synthetic networks.

## How to run
There are 2 main python source files in the root of the repository.

1. `main-ui.py` handles a simple self-explanatory UI for running experiments. 
2. To manually run the program open `main.py` and follow detailed instuctions on top of the file. In short, edit `experiment_type_num` variable to select which type of experiment you would like to run. The variable is an index for `input_types`. Edit model parameters or select input filename. For models you may record trajectories of nodes specified in `focus_indices` array.

Output: histograms with value distributions, node trajectories: both raw and processed.

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
