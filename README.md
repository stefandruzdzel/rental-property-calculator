# Rental Property Calculator
#### Video Demo:  https://youtu.be/Uxicc7Dljck
#### Description:

Summary:
For my final project I created a website application.  The user provides basic information in a form for a rental property such as purchase price, estimated rent, loan information.  Once they submit they are taken to a page that shows projected financial performance.  What seperates this rental property calculator from other ones available on the internet is that instead of using dumb averages, this one has 4 variables where the user can select a distribution of outcomes.  The final calculation then runs 100 iterations, and shows the range of possible outcomes.

Technologies:
The website uses Flask to run server side functions and to render templates.  The form the users fill out is made with Bootstrap.  There are 4 interactive charts, which were created with PyScript.  PyScript is a very new technology, and was annnounced less than 6 months ago.  It allows for coding in Python directly within an HTML file.  The logic to interpret this code is linked in with a reference link.  On the positive, this technology was very helpful because I was already familiar with Python plotting.  On the negative, since PyScript is still in Alpha this project may break as changes are made.  Also, the page takes a long time to load, perhaps using a Javascript solution would have been better.

Statistical Distributions:
Four of the variables allow the user to select custom distributions of outcome, these variables are Appreciation, Maintenance, Tenant Stay and Rent Growth.  Interactive charts update as the user changes values for these distributions.

Appreciation:  Uses a Normal Distribution, the mean and STD are pre-filled with the US average from the last 40 years.  The user can update these values and instantly see the graph of the distribution update.

Maintenance:  Beta Prime Distribution.  Alpha = 1, Beta is adjustable, I convert the users 1 through 10 rating into a reasonable Beta value.  This distribution is based off my intuitive understanding of how maintenance is likely to occure - I do not have actual data on this variable.

Tenant Stay: Same as maintenance except that Alpha = 10.  Again, based on intuition, as I do not have actual data on tenant stay distributions.

Rent Growth: Normal Distribution, mean and STD are pre-filled with the US average from the last 40 years

Other Features unique to my calculator:
Leasing Fee: Often times property managers charge the first months rent as a leasing fee.
Adjustable Rate Mortgages: follows a statistical distribution based on 40 years of historical data.  I did not make this distribution adjustable.

Output:
Flask runs 100 iterations.  The user is then shown two charts: Monthly cash flow and profit if sold for the next 30 years.  These charts show the median value, and also 80% and 95% confidence intervals.  The user is also shown a table with performance for the next 30 years, at the 20th, 50th and 80th percentiles.

