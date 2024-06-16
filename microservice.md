# How to use microservice?

## Starting the microservice

Before running the microservice, please make sure that you have installed all necessary packages, listed in requirements.txt file.
To start the microservice, use following code:
```
python -m uvicorn microservice:app --reload
```
To switch it off, use Ctrl+C in a terminal with open microservice
## Using the microservice
### Check if microservice is working
To check if microservice is working, you can use the following command:
```
curl http://127.0.0.1:8000/status
```
If content of the return is {"status":"OK"}, the microservice has started up correctly

### Getting predictions
Microservice offers the ability to generate playlist in 6 different genres:
1. pop
2. trap
3. rap
4. rock
5. metal
6. hip hop
Number of tracks on the playlist can also be chosen by the user. You can also use 2 different models, neural networks and gradient boosting.

#### Sample calls
Generate pop playlist with 50 tracks using neural networks model:
```
curl http://127.0.0.1:8000/generate?genre=pop&num_tracks=50&model=NN
```
Please note, that sometimes it may be neccessary to add quotation marks around '&' symbols. In that case, the same call would look like this:
```
curl http://127.0.0.1:8000/generate?genre=pop"&"num_tracks=50"&"model=NN
```

Generate hip hop playlist with 30 tracks using gradient boosting model:
```
curl http://127.0.0.1:8000/generate?genre="hip hop""&"num_tracks=30"&"model=GB
```

### AB Test
AB Test generates playlists of given genre and given length using both models and returns them. User can later use generated playlists for their own comparisons and trials.
Sample call of an AB Test, generating metal playlists with 25 tracks each:
```
curl http://127.0.0.1:8000/abtest?genre=metal"&"num_tracks=25
```

## Logs
Logs from microservice are collected in microservice_log.txt file.

## Microservice tests
Tests for microservice can be found in microservice_test.py file.