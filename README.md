# twitter_hot_topic_Clustering


### How to run
this project need ```streamlit, pandas```
```streamlit run web.py```

###	What problem I tried to solve?
There is a huge stream of information on Twitter every day, and it is not easy to find out the hot topics or opinions (not the kind recommended by the tweeters), so it is a meaningful task to dig out the hot information from the large number of tweets.

###	What I did?
I built a web application to dig out the hot information from the large number of tweets. 
I obtained the desired results by vectorizing the tweets through text vectorization techniques and then clustering them using the DBSCAN algorithm.
Also, I developed a web application that was able to simplify the whole analysis process into a simple click-and-upload operation
 
[![g56TPK.png](https://z3.ax1x.com/2021/05/19/g56TPK.png)](https://imgtu.com/i/g56TPK)

###	Highlights about my project
1.	Implemented text vectorization representation of twitter text and successfully used DBSCAN algorithm to perform hotspot clustering with good results.
2.	Completed the visualization of the clustering results for analysis.
3.	Integrate this project into a web application, realize the visual interface, open the browser can be used at any time and anywhere in the PC and mobile, convenient and fast

Reference and explanations:
1.	Our demo twitter dataset comes from Kaggle, but not limited to it, You can also crawl through the twitter crawler on Github to get the desired data for analysis.
2.	Our web application is powered by streamlit. Streamlit turns data scripts into shareable web apps all in Python. Thanks for their excellent work
