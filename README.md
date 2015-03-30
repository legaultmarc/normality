This is a very short Python script to generate probability plots and histograms
to quickly verify (or gain insight) on data normality. It's use is
straightforward and unix-like:

The following example will extract the 4th column of a text file delimited by
semi columns and will plot the relevant histogram and probability plots.

```bash
./normality.py sample_text_file.txt -f 4 -d ';' --save demo.png
```

The script also works without from the `stdin`:
```bash
cut -f 4 -d ';' sample_text_file.txt | ./normality -
```

If seaborn is installed, the plots should look somewhat like this:

![Sample plot from the program](https://raw.githubusercontent.com/legaultmarc/normality/master/demo.png)
