# Embedding test

Small test script set for comparing the similarity difference between different vectors.

## Setup

Add an `.env` file with your `OPENAI_API_KEY` in it. Refer to the `.env.example`.

Then you can modify the test cases in `index.js`

## Run

```shell
node index.js
```

## Test cases

On first run, example test cases will be used and written to `test-cases.json`, you can then edit these test cases to run your own.


## Other

You'll get a warning of using punycode which messes with the structure. You can use node `20.5.1` with nvm to avoid this.