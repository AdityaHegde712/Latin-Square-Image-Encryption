# Latin Square Image Encryption in Python

Reference paper link: https://dx.doi.org/10.14569/IJACSA.2021.0121216

# How to Run Web App in Development Mode

### Back-End

1. Make sure `python 3.x` is installed locally
2. Make sure these python libraries are installed in your environment:
   - flask
   - flask_cors
   - numpy
3. Go to back-end directory:

```
$ cd prod/crypto-proj-back-end
```

4. Run server:

```
$ python app.py
```

5. Note down the URL the server is running down. By default, should be http://127.0.0.1:5000

### Front-End

1. Make sure you have `npm` installed
2. Go to front-end directory on a different terminal:

```
$ cd prod/crypto-proj-front-end
```

3. Install dependencies:

```
$ npm install
```

4. Run server in dev mode:

```
$ npm run dev
```

Note: you may have to change the URL in `prod/crypto-proj-front-end/src/constants.js` to match the URL the flask server is running on
