# LyricsGenerator

Song Lyrics Generation from scratch, without using any external machine learning libraries. We have used Recurrent Neural Networks (RNNs) to implement this project.

New songs lyrics are generated based on the chosen artist's previously released songs. The RNN stores the context and style of songs of the chosen artist and generates new song lyrics in a similar fashion.
# Dataset Used

https://www.kaggle.com/mousehead/songlyrics <br />
• This dataset contains lyrics of 55000+ English songs <br />
• The dataset contains four features - <br />
    1. Artist <br />
    2. Song Name <br />
    3. Link to the webpage with the song lyrics (for reference) <br />
    4. Lyrics of the song, unmodified <br />
• Number of Songs - 57650 <br />
• Number of Artists - 643

# Steps to run

1) To run the program, these are the steps need to be followed - 

2) The code has the following dependencies, which need to be installed before running this code:
	a) Python, Install version greater than 3.5
	b) Pandas, More details at: https://pandas.pydata.org/
	c) numpy, More details at: https://numpy.org/

3) Execute driver.py file

Note - Since data is to be downloaded from cloud, the pre-processing part of the program will take time.

# Future Enhancements

Add Long Short-Term Memory (LSTM) to the Recurrent Neural Network to obtain better results.
