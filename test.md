```python
# Initialize OK
from client.api.notebook import Notebook
ok = Notebook('lab10.ok')
```

    =====================================================================
    Assignment: Correlation
    OK, version v1.12.5
    =====================================================================
    


# Lab 10: Correlation

Welcome to Lab 10!

In today's lab, we will learn about ways to understand and quantify [the association between two variables](https://www.inferentialthinking.com/chapters/15/1/correlation.html).


```python
# Run this cell, but please don't change it.

# These lines import the Numpy and Datascience modules.
import numpy as np
from datascience import *

# These lines do some fancy plotting magic.
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')
import warnings
warnings.simplefilter('ignore', FutureWarning)

# These lines load the tests.
from client.api.notebook import Notebook
ok = Notebook('lab10.ok')
_ = ok.submit()
```

    =====================================================================
    Assignment: Correlation
    OK, version v1.12.5
    =====================================================================
    



    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>


    Saving notebook... Saved 'lab10.ipynb'.
    Submit... 0.0% complete
    Could not submit: The server could not verify that you are authorized to access the URL requested.  You either supplied the wrong credentials (e.g. a bad password), or your browser doesn't understand how to supply the credentials required.
    Backup... 0.0% complete
    Could not backup: The server could not verify that you are authorized to access the URL requested.  You either supplied the wrong credentials (e.g. a bad password), or your browser doesn't understand how to supply the credentials required.
    


# 1. How Faithful is Old Faithful? 

Old Faithful is a geyser in Yellowstone National Park that is famous for eruption on a fairly regular schedule. Run the cell below to see Old Faithful in action!


```python
# For the curious: this is how to display a YouTube video in a
# Jupyter notebook.  The argument to YouTubeVideo is the part
# of the URL (called a "query parameter") that identifies the
# video.  For example, the full URL for this video is:
#   https://www.youtube.com/watch?v=wE8NDuzt8eg
from IPython.display import YouTubeVideo
YouTubeVideo("wE8NDuzt8eg")
```





<iframe
    width="400"
    height="300"
    src="https://www.youtube.com/embed/wE8NDuzt8eg"
    frameborder="0"
    allowfullscreen
></iframe>




Some of Old Faithful's eruptions last longer than others.  Whenever there is a long eruption, it usually followed by an even longer wait before the next eruption.

If you visit Yellowstone, you might want to predict when the next eruption will happen, so you can see the rest of the park in the meantime instead of waiting by the geyser. Today, we will use a dataset on eruption durations and waiting times to see how closely these variables are related - if there is a strong relationship, we should be able to predict one from the other. You’ll learn more about this method of prediction in lecture tomorrow.

The dataset has one row for each observed eruption.  It includes the following columns:
- `duration`: Eruption duration, in minutes.
- `wait`: Time between this eruption and the next, also in minutes.

Run the next cell to load the dataset.


```python
faithful = Table.read_table("faithful.csv")
faithful
```




<table border="1" class="dataframe">
    <thead>
        <tr>
            <th>duration</th> <th>wait</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>3.6     </td> <td>79  </td>
        </tr>
        <tr>
            <td>1.8     </td> <td>54  </td>
        </tr>
        <tr>
            <td>3.333   </td> <td>74  </td>
        </tr>
        <tr>
            <td>2.283   </td> <td>62  </td>
        </tr>
        <tr>
            <td>4.533   </td> <td>85  </td>
        </tr>
        <tr>
            <td>2.883   </td> <td>55  </td>
        </tr>
        <tr>
            <td>4.7     </td> <td>88  </td>
        </tr>
        <tr>
            <td>3.6     </td> <td>85  </td>
        </tr>
        <tr>
            <td>1.95    </td> <td>51  </td>
        </tr>
        <tr>
            <td>4.35    </td> <td>85  </td>
        </tr>
    </tbody>
</table>
<p>... (262 rows omitted)</p>



Let’s first look at our data to see whether we can visually identify a linear relationship, which is what the correlation coefficient measures.

**Question 1.1.** Make a scatter plot of the data.  It's conventional to put the column we want to predict on the vertical axis and the other column on the horizontal axis.

<!--
BEGIN QUESTION
name: q1_1
-->


```python
faithful.plot('wait','duration')
plots.title('Duration v. wait (original units)');
```


![png](output_9_0.png)


**Question 1.2.** Are eruption duration and waiting time roughly linearly related based on the scatter plot above? Is this relationship positive?

<!--
BEGIN QUESTION
name: q1_2
-->

The relation is positive and roughly linearly related.

We're going to continue with the assumption that they are linearly related, and quantify the strength of this linear relationship.

We'd next like to plot the data in standard units. If you don't remember the definition of standard units, textbook section [14.2](https://www.inferentialthinking.com/chapters/14/2/Variability.html#standard-units) might help!

**Question 1.3.** Compute the mean and standard deviation of the eruption durations and waiting times.  **Then** create a table called `faithful_standard` containing the eruption durations and waiting times in standard units.  The columns should be named `duration (standard units)` and `wait (standard units)`.

<!--
BEGIN QUESTION
name: q1_3
-->


```python
np.mean(faithful.column('duration'))
```




    3.4877830882352936




```python
duration_mean = np.mean(faithful.column('duration'))
duration_std = np.std(faithful.column('duration'))
wait_mean = np.mean(faithful.column('wait'))
wait_std = np.std(faithful.column('wait'))

def find_standard(x):
    return((x - np.mean(x)))/np.std(x)
faithful_standard = Table().with_columns(
    "duration (standard units)", find_standard(faithful.column('duration')),
    "wait (standard units)", find_standard(faithful.column('wait')))
faithful_standard
```




<table border="1" class="dataframe">
    <thead>
        <tr>
            <th>duration (standard units)</th> <th>wait (standard units)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0.0984989                </td> <td>0.597123             </td>
        </tr>
        <tr>
            <td>-1.48146                 </td> <td>-1.24518             </td>
        </tr>
        <tr>
            <td>-0.135861                </td> <td>0.228663             </td>
        </tr>
        <tr>
            <td>-1.0575                  </td> <td>-0.655644            </td>
        </tr>
        <tr>
            <td>0.917443                 </td> <td>1.03928              </td>
        </tr>
        <tr>
            <td>-0.530851                </td> <td>-1.17149             </td>
        </tr>
        <tr>
            <td>1.06403                  </td> <td>1.26035              </td>
        </tr>
        <tr>
            <td>0.0984989                </td> <td>1.03928              </td>
        </tr>
        <tr>
            <td>-1.3498                  </td> <td>-1.46626             </td>
        </tr>
        <tr>
            <td>0.756814                 </td> <td>1.03928              </td>
        </tr>
    </tbody>
</table>
<p>... (262 rows omitted)</p>




```python
ok.grade("q1_3");
```

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Running tests
    
    ---------------------------------------------------------------------
    Test summary
        Passed: 3
        Failed: 0
    [ooooooooook] 100.0% passed
    


**Question 1.4.** Plot the data again, but this time in standard units.

<!--
BEGIN QUESTION
name: q1_4
-->


```python
faithful_standard.plot('duration (standard units)','wait (standard units)')
plots.title('Duration v. wait (standard units)');
```


![png](output_18_0.png)


You'll notice that this plot looks the same as the last one!  However, the data and axes are scaled differently.  So it's important to read the ticks on the axes.

**Question 1.5.** Among the following numbers, which would you guess is closest to the correlation between eruption duration and waiting time in this dataset?

1. -1
2. 0
3. 1

Assign `correlation` to the number corresponding to your guess.

<!--
BEGIN QUESTION
name: q1_5
-->


```python
correlation = 3
```


```python
ok.grade("q1_5");
```

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Running tests
    
    ---------------------------------------------------------------------
    Test summary
        Passed: 1
        Failed: 0
    [ooooooooook] 100.0% passed
    


**Question 1.6.** Compute the correlation `r`.  

*Hint:* Use `faithful_standard`.  Section [15.1](https://www.inferentialthinking.com/chapters/15/1/Correlation.html#calculating-r) explains how to do this.


<!--
BEGIN QUESTION
name: q1_6
-->


```python
faithful_standard.column('duration (standard units)')
```




    array([ 0.09849886, -1.48145856, -0.13586149, -1.05750332,  0.91744345,
           -0.53085085,  1.06402839,  0.09849886, -1.34979544,  0.75681445,
           -1.45249268,  0.37674691,  0.62515133, -1.52534627,  1.06402839,
           -1.1593228 , -1.52534627,  1.1518038 , -1.65700939,  0.66903904,
           -1.48145856, -1.52534627, -0.03316426, -0.36934409,  0.91744345,
            0.09849886, -1.33487362,  0.5224541 ,  0.31793739,  0.82966804,
            0.71292674,  0.85951168, -0.10601785,  0.47856639,  0.30301557,
           -1.29098592, -1.42264904,  1.18076969, -1.45249268,  1.13688198,
            0.75681445, -1.40860497,  0.94728709, -1.52534627,  0.91744345,
           -0.14990556,  0.30301557, -1.21813233,  1.00521886, -1.30590774,
            1.1518038 ,  1.07807246, -1.45249268,  1.18076969, -1.54026809,
            1.22465739,  0.20119609, -1.59819986,  0.94728709,  0.72784856,
           -1.10139103,  0.88847757, -1.52534627,  1.1518038 , -1.46653674,
            0.80070215,  0.59618544,  1.06402839, -1.24709821,  1.06402839,
            0.47856639, -1.33487362,  0.88847757,  0.44960051, -1.32082956,
            1.38616415, -1.29098592,  0.94728709,  0.34690327,  0.09849886,
            0.5663418 ,  0.74189263,  0.53737592, -0.75028938,  0.50841003,
            1.2685451 ,  0.4057128 ,  0.90339939, -1.1593228 ,  0.44960051,
           -1.13035691,  0.74189263, -1.42264904,  1.16672562, -1.45249268,
            0.71292674,  1.0350625 ,  0.23016197, -1.42264904,  1.23957921,
           -0.8819525 ,  0.77173627, -1.21813233,  0.88847757,  0.49348821,
           -1.42264904,  1.06402839, -1.49638038,  1.19569151,  0.17135245,
            1.09299428, -1.0425815 ,  1.23957921,  0.81562397, -1.56923397,
            1.00521886, -1.02765968,  0.97625298, -1.46653674,  0.81562397,
           -0.76433344,  0.50841003,  0.66903904, -1.33487362,  0.97625298,
            0.2450838 , -1.37876133,  0.88847757, -1.07154739,  1.02014068,
           -1.42264904,  0.59618544, -0.60370444,  0.74189263, -1.45249268,
            0.78578033, -1.40860497,  1.2685451 , -1.27694185,  0.21524015,
            0.65411722, -1.10139103,  0.91744345,  1.16672562,  0.74189263,
           -1.32082956,  1.00521886, -1.29098592,  1.41513004, -1.48145856,
            1.35632051,  0.44960051, -0.95480609,  0.97625298,  0.06953297,
            0.44960051,  0.88847757,  0.5224541 , -1.48145856,  0.42063462,
           -1.13035691,  0.58126362, -1.30590774,  0.30301557,  0.01072344,
            0.96133116, -0.98377197,  1.32735463, -1.36471726,  0.9911748 ,
           -1.37876133, -1.23305415,  0.96133116, -0.13586149,  0.59618544,
            0.74189263,  0.88847757, -0.93988427,  0.44960051,  0.59618544,
           -1.40860497,  0.96133116,  0.66903904,  0.2450838 , -1.27694185,
            0.82966804,  0.5224541 , -1.45249268,  0.81562397, -1.14527873,
            1.1518038 , -1.45249268,  1.1518038 ,  0.53737592,  0.41975687,
            0.65411722,  0.01072344,  0.77085851, -1.08646921,  1.0350625 ,
           -1.21813233,  0.75681445,  0.5663418 , -1.42264904,  0.97625298,
           -1.49638038,  0.77173627,  0.31793739, -1.36471726,  0.88847757,
           -0.96972791,  1.06402839, -1.42264904,  0.30301557, -0.06213015,
            0.65411722, -0.95480609,  1.1518038 , -1.30590774,  0.58126362,
           -1.42264904,  0.68396086, -1.52534627,  0.87355575,  0.44960051,
            0.55229774,  0.5224541 ,  0.68396086,  0.37674691,  0.93236527,
            0.5224541 , -0.93988427,  0.61022951, -1.11543509,  0.84458986,
           -1.40860497, -1.43757086,  0.69800492,  0.4057128 , -1.01361561,
            0.58126362, -0.99869379,  1.2685451 , -0.51592903,  0.96133116,
            0.30301557, -1.23305415,  0.77173627, -1.18916644,  0.75681445,
           -1.13035691,  0.84458986,  0.06953297,  0.88847757,  0.58126362,
            0.2889715 ,  0.37674691,  0.84458986, -1.30590774,  0.69800492,
            1.12283792,  0.91744345, -1.43757086,  0.66903904, -1.32082956,
           -1.08646921,  1.1079161 ,  0.55229774, -1.17424462,  0.81562397,
           -1.46653674,  0.85951168])




```python
faithful_standard('wait (standard units)')
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-139-949a729df462> in <module>
    ----> 1 faithful_standard('wait (standard units)')
    

    TypeError: 'Table' object is not callable



```python
r = np.mean(faithful_standard.column('duration (standard units)')*faithful_standard.column('wait (standard units)'))
r
```




    0.9008111683218132




```python
ok.grade("q1_6");
```

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Running tests
    
    ---------------------------------------------------------------------
    Test summary
        Passed: 1
        Failed: 0
    [ooooooooook] 100.0% passed
    


# 2. Cheese and Doctorates

We’ll now investigate the relationship between two unusual variables. For every year between 2000 and 2009 (inclusive), we have data on the per-capita consumption of mozzarella cheese in that year and the number of civil engineering doctorates awarded in that year. These are real data from the U.S. Department of Agriculture and the National Science Foundation.

We can perform the same process that we performed above to investigate whether there is an association between the cheese consumption in a certain year and the number of civil engineering degrees awarded in that year. Just because we can do something, though, doesn’t mean it is meaningful. While you carry out the following process, think about whether the analysis you are performing is meaningful or not.

**Question 2.1.** Run the next cell to load in the dataset

<!--
BEGIN QUESTION
name: q2_1
-->


```python
cheese_doctors = Table().with_columns(
    "Cheese Consumption", make_array(9.3, 9.7, 9.7, 9.7, 9.9, 10.2, 10.5, 11, 10.6, 10.6),
    "Civil Engineering Doctorates", make_array(480, 501, 540, 552, 547, 622, 655,701, 712, 708))
cheese_doctors
```




<table border="1" class="dataframe">
    <thead>
        <tr>
            <th>Cheese Consumption</th> <th>Civil Engineering Doctorates</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>9.3               </td> <td>480                         </td>
        </tr>
        <tr>
            <td>9.7               </td> <td>501                         </td>
        </tr>
        <tr>
            <td>9.7               </td> <td>540                         </td>
        </tr>
        <tr>
            <td>9.7               </td> <td>552                         </td>
        </tr>
        <tr>
            <td>9.9               </td> <td>547                         </td>
        </tr>
        <tr>
            <td>10.2              </td> <td>622                         </td>
        </tr>
        <tr>
            <td>10.5              </td> <td>655                         </td>
        </tr>
        <tr>
            <td>11                </td> <td>701                         </td>
        </tr>
        <tr>
            <td>10.6              </td> <td>712                         </td>
        </tr>
        <tr>
            <td>10.6              </td> <td>708                         </td>
        </tr>
    </tbody>
</table>



**Question 2.2.** Let’s visually inspect the relationship in the table. Make a scatter plot which displays mozzarella cheese consumption on the x-axis, and number of Civil Engineering degrees on the y-axis.

<!--
BEGIN QUESTION
name: q2_2
-->


```python
cheese_doctors.plot('Cheese Consumption','Civil Engineering Doctorates')
```


![png](output_32_0.png)


**Question 2.3.** Write a function called correlation_from_table that takes as arguments the name of a table and the names of two columns which contain numerical values. The function should return the correlation coefficient (a single number) between the two variables.

<!--
BEGIN QUESTION
name: q2_3
-->


```python
def correlation_from_table(table, col_x, col_y):
    data_x = table.column(col_x)
    data_y = table.column(col_y)
    standard_units_x = find_standard(data_x)

    standard_units_y = find_standard(data_y)
    correlation = np.mean(standard_units_x * standard_units_y)
    return correlation
correlation_from_table(cheese_doctors, 'Cheese Consumption', 'Civil Engineering Doctorates')
```




    0.9586477872804794




```python
ok.grade("q2_3");
```

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Running tests
    
    ---------------------------------------------------------------------
    Test summary
        Passed: 1
        Failed: 0
    [ooooooooook] 100.0% passed
    


**Question 2.4.** Call your correlation_from_table function on cheese_doctors to find the correlation coefficient between the two variables.

<!--
BEGIN QUESTION
name: q2_4
-->


```python
correlation_from_table(cheese_doctors, 'Cheese Consumption', 'Civil Engineering Doctorates')
```




    0.9586477872804794



You should have found a strong correlation (close to 1) between the two variables. But how should we interpret this value?

**Question 2.5.** Does the high degree of linear association between these two variables tell us anything about a causal relationship between mozzarella cheese consumption and civil engineering doctorates? If we knew the amount of mozzarella cheese eaten per capita in 2010, would it make sense to try to use this to predict the number of civil engineering graduates in that year?

<!--
BEGIN QUESTION
name: q2_5
-->

No, the high association between cheese and Engineering doctrate can't reflect any causation between this two variables.

What you’ve just seen is an example of a spurious correlation, in which a relationship between two variables is purely due to chance. There are many such examples at https://www.tylervigen.com/spurious-correlations, where we found these data. This is an amusing website to spend a few minutes on, and it serves as a reminder that we should not use correlation as justification or evidence that two variables are related in a mechanistic or causal way. As you’ve probably heard before, “correlation does not imply causation.”

The example above is silly, and makes this statement seem obvious, but it can be dangerous to forget the difference between a correlative link and a causal link.

Now, let’s investigate another situation when it’s important to recognize the limitations of correlations. Run the next cell to load a mystery dataset.


```python
mystery = Table.read_table("mystery.csv")
mystery
```




<table border="1" class="dataframe">
    <thead>
        <tr>
            <th>x</th> <th>y</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>5.31801 </td> <td>220.445 </td>
        </tr>
        <tr>
            <td>-46.9561</td> <td>2011.91 </td>
        </tr>
        <tr>
            <td>20.5586 </td> <td>60.7178 </td>
        </tr>
        <tr>
            <td>-25.1235</td> <td>1797.85 </td>
        </tr>
        <tr>
            <td>16.6883 </td> <td>-267.489</td>
        </tr>
        <tr>
            <td>-83.7817</td> <td>7357.47 </td>
        </tr>
        <tr>
            <td>-8.96097</td> <td>1077.6  </td>
        </tr>
        <tr>
            <td>-43.0446</td> <td>1953.85 </td>
        </tr>
        <tr>
            <td>31.4746 </td> <td>2446.56 </td>
        </tr>
        <tr>
            <td>-3.85866</td> <td>411.589 </td>
        </tr>
    </tbody>
</table>
<p>... (390 rows omitted)</p>



**Question 2.6.** We’ll start by committing a sin against data science: we’ll start calculating correlation before investigating our data. In the following cell, assign mystery_correlation to the value of the correlation coefficient between the two variables in the mystery table.

<!--
BEGIN QUESTION
name: q2_6
-->


```python
mystery_correlation = correlation_from_table(mystery, 'x', 'y')
mystery_correlation
```




    -0.024711531764254033




```python
ok.grade("q2_6");
```

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Running tests
    
    ---------------------------------------------------------------------
    Test summary
        Passed: 1
        Failed: 0
    [ooooooooook] 100.0% passed
    


**Question 2.7.** Based on the value of the correlation, what can we can say about the relationship between x and y?

<!--
BEGIN QUESTION
name: q2_7
-->

There is a weak negative correlation between x and y.

**Question 2.8.** Now let’s see why it’s valuable to investigate our data visually first. Create a scatter plot of the data in the mystery table.

<!--
BEGIN QUESTION
name: q2_8
-->


```python
mystery.plot('x', 'y')
```


![png](output_50_0.png)


**Question 2.9.** What do you see? Is there a linear relationship between x and y? Is there some other kind of relationship? Why did we get the value for the correlation coefficient that we got? Discuss your answers with a neighbor or TA, then summarize below.

<!--
BEGIN QUESTION
name: q2_8
-->

There is a non-linear relationship between x and y. The corelation coefficient only reflects linear association between two variables.

That's it! You've completed Lab 10.

Be sure to 
- **run all the tests** (the next cell has a shortcut for that), 
- **Save and Checkpoint** from the `File` menu,
- **run the last cell to submit your work**,
- and ask one of the staff members to check you off.


```python
# For your convenience, you can run this cell to run all the tests at once!
import os
print("Running all tests...")
_ = [ok.grade(q[:-3]) for q in os.listdir("tests") if q.startswith('q')]
print("Finished running all tests.")
```

    Running all tests...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Running tests
    
    ---------------------------------------------------------------------
    Test summary
        Passed: 1
        Failed: 0
    [ooooooooook] 100.0% passed
    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Running tests
    
    ---------------------------------------------------------------------
    Test summary
        Passed: 1
        Failed: 0
    [ooooooooook] 100.0% passed
    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Running tests
    
    ---------------------------------------------------------------------
    Test summary
        Passed: 3
        Failed: 0
    [ooooooooook] 100.0% passed
    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Running tests
    
    ---------------------------------------------------------------------
    Test summary
        Passed: 1
        Failed: 0
    [ooooooooook] 100.0% passed
    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Running tests
    
    ---------------------------------------------------------------------
    Test summary
        Passed: 1
        Failed: 0
    [ooooooooook] 100.0% passed
    
    Finished running all tests.



```python
# Run this cell to submit your work *after* you have passed all of the test cells.
# It's ok to run this cell multiple times. Only your final submission will be scored.

_ = ok.submit()
```


    <IPython.core.display.Javascript object>



```python
ok.auth(force=True)
```

    
    Open the following URL:
    
    https://okpy.org/client/login/
    
    After logging in, copy the code from the web page and paste it into the box.
    Then press the "Enter" key on your keyboard.
    
    Paste your code here: HXgQ0DUjy9NbdEwvlEqgivQTSvDtXP
    Successfully logged in as oldjack@berkeley.edu



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
